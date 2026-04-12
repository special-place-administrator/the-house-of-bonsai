#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn.cuh"

// InnerQ: update the fattn-side inverse scale array from host
void rq_innerq_update_fattn_scales(const float * scale_inv) {
    cudaMemcpyToSymbol(d_innerq_channel_scale_inv_fattn, scale_inv, 128 * sizeof(float));
}

void rq_innerq_init_fattn() {
    float ones[128];
    for (int i = 0; i < 128; i++) ones[i] = 1.0f;
    cudaMemcpyToSymbol(d_innerq_channel_scale_inv_fattn, ones, sizeof(ones));
}

// Q² calibration: host-side management
static int q_calibrate_state = 0; // 0=off, 1=collecting, 2=done

void rq_q_calibrate_init() {
    const char * env = getenv("RQ_Q_CALIBRATE");
    if (!env || atoi(env) != 1) return;

    double zeros[128] = {};
    int zero = 0, one = 1;
    cudaMemcpyToSymbol(d_q_channel_sq_fattn, zeros, sizeof(zeros));
    cudaMemcpyToSymbol(d_q_channel_count_fattn, &zero, sizeof(zero));
    cudaMemcpyToSymbol(d_q_calibrate_fattn, &one, sizeof(one));
    q_calibrate_state = 1;
    fprintf(stderr, "RQ_Q_CALIBRATE: collecting per-position Q² statistics\n");
}

void rq_q_calibrate_finalize() {
    if (q_calibrate_state != 1) return;

    int zero = 0;
    cudaMemcpyToSymbol(d_q_calibrate_fattn, &zero, sizeof(zero));

    double sq[128];
    int count;
    cudaMemcpyFromSymbol(sq, d_q_channel_sq_fattn, sizeof(sq));
    cudaMemcpyFromSymbol(&count, d_q_channel_count_fattn, sizeof(count));

    if (count == 0) {
        fprintf(stderr, "RQ_Q_CALIBRATE: no Q vectors seen, skipping\n");
        q_calibrate_state = 2;
        return;
    }

    // Compute E[Q²] per position and save as 128 float32 values
    float weights[128];
    fprintf(stderr, "RQ_Q_CALIBRATE: %d Q groups accumulated\n", count);
    double total = 0;
    for (int i = 0; i < 128; i++) {
        weights[i] = (float)(sq[i] / count);
        total += weights[i];
    }
    float mean = (float)(total / 128.0);
    float maxw = 0, minw = 1e30f;
    for (int i = 0; i < 128; i++) {
        if (weights[i] > maxw) maxw = weights[i];
        if (weights[i] < minw) minw = weights[i];
    }
    fprintf(stderr, "  E[Q²] mean=%.6f min=%.6f max=%.6f ratio=%.2f\n",
            mean, minw, maxw, maxw / (minw > 1e-10f ? minw : 1e-10f));

    const char * path = "/tmp/q_weights.bin";
    FILE * fp = fopen(path, "wb");
    if (fp) {
        fwrite(weights, sizeof(float), 128, fp);
        fclose(fp);
        fprintf(stderr, "  Saved Q weights to %s (128 floats)\n", path);
    }

    q_calibrate_state = 2;
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if ((turing_mma_available(cc) || amd_wmma_available(cc)) && Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || amd_wmma_available(cc) || Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }

        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// Context-adaptive V alpha: logarithmic scaling based on current KV occupancy.
// 3-bit: alpha = 1.1484 - 0.01443 * ln(n_kv), calibrated on Qwen3.5-27B decode-time KLD sweeps
//   at 8K/16K/32K. Optima: 8K→1.020, 16K→1.005, 32K→1.000.
// 2-bit: alpha = 0.8865 + 0.0195 * ln(n_kv), calibrated on fine-grained 0.005-step decode-time
//   KLD sweeps at 2K/7K/16K/32K on Qwen3.5-27B. Optima: 2K→1.030, 7K→1.065, 16K→1.090, 32K→1.075.
// Override with RQ_ISO_DECODE_ALPHA_V env var to force a static alpha (disables adaptive).
static float d_tcq_decode_alpha_v_static = 0.0f; // 0 = use adaptive, >0 = static override
static float d_tcq_decode_alpha_k = 1.0f;       // K decode alpha, static (default 1.0)
static bool d_tcq_decode_alpha_loaded = false;

static inline float tcq_compute_alpha_v(ggml_type v_type, int64_t n_kv) {
    if (d_tcq_decode_alpha_v_static > 0.0f) return d_tcq_decode_alpha_v_static;
    if (n_kv < 1) n_kv = 1;
    const float ln_ctx = logf((float)n_kv);
    if (v_type == GGML_TYPE_RQ3_ISO) {
        // v3: 3-point fit from fine-grained KLD sweeps at 8K/16K/32K on Qwen3.5-27B.
        // Per-token: n_kv=512→1.044, 2K→1.024, 8K→1.004, 16K→0.994, 32K→0.984
        // Clamped [0.98, 1.06] — early tokens (small n_kv) use higher alpha.
        return fmaxf(0.98f, fminf(1.06f, 1.1484f - 0.01443f * ln_ctx));
    } else if (v_type == GGML_TYPE_RQ4_ISO) {
        // v2: 4-point fit from fine-grained 0.005-step KLD sweeps at 2K/7K/16K/32K.
        // Per-token: n_kv=512→1.008, 2K→1.035, 7K→1.060, 16K→1.076, 32K→1.089
        return fmaxf(1.00f, fminf(1.12f, 0.8865f + 0.0195f * ln_ctx));
    }
    return 1.0f;
}

static void load_tcq_decode_alpha() {
    if (d_tcq_decode_alpha_loaded) return;
    d_tcq_decode_alpha_loaded = true;
    const char * s = getenv("RQ_ISO_DECODE_ALPHA_V");
    if (s) {
        char * end;
        errno = 0;
        float a = strtof(s, &end);
        if (end == s || errno != 0 || a <= 0.0f || a >= 10.0f) {
            fprintf(stderr, "TCQ: invalid RQ_ISO_DECODE_ALPHA_V='%s'\n", s);
        } else {
            d_tcq_decode_alpha_v_static = a;
            cudaMemcpyToSymbol(d_tcq_decode_alpha_v_fattn, &a, sizeof(float));
            fprintf(stderr, "TCQ decode: V alpha=%.4f (static override)\n", a);
        }
    } else {
        fprintf(stderr, "TCQ decode: context-adaptive V alpha enabled\n");
    }
    const char * sk = getenv("RQ_ISO_DECODE_ALPHA_K");
    if (sk) {
        char * end;
        errno = 0;
        float a = strtof(sk, &end);
        if (end == sk || errno != 0 || a <= 0.0f || a >= 10.0f) {
            fprintf(stderr, "TCQ: invalid RQ_ISO_DECODE_ALPHA_K='%s'\n", sk);
        } else {
            d_tcq_decode_alpha_k = a;
            fprintf(stderr, "TCQ decode: K alpha=%.4f\n", a);
        }
    }
}

// === RQ prefill: bulk dequant to fp16 + MMA attention ===
// During prefill (Q->ne[1] > 1), dequantize rq K/V to fp16 temp buffers
// and use the fast MMA kernel instead of the slower vec kernel.

static __global__ void k_rq2_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_RQ2;
    const int j_in_blk = j % QK_RQ2;
    const block_rq2_0 * blk = (const block_rq2_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const uint8_t idx = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
    const float val = d_rq_centroids_2bit_fattn[idx] * norm;

    dst[strm * (ne2 * ne1 * ne0) + head * (ne1 * ne0) + row * ne0 + j] = __float2half(val);
}

static __global__ void k_rq3_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_RQ3;
    const int j_in_blk = j % QK_RQ3;
    const block_rq3_0 * blk = (const block_rq3_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const uint8_t low2 = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
    const uint8_t hi1  = (blk->signs[j_in_blk / 8] >> (j_in_blk % 8)) & 0x1;
    const float val = d_rq_centroids_3bit_fattn[low2 | (hi1 << 2)] * norm;

    dst[strm * (ne2 * ne1 * ne0) + head * (ne1 * ne0) + row * ne0 + j] = __float2half(val);
}

static __global__ void k_rq3_iso_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x; // element index within row (0..ne0-1)
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_RQ3_ISO;
    const int t = j % QK_RQ3_ISO; // element index within 128-element block
    const block_rq3_iso * blk = (const block_rq3_iso *)src_row + blk_idx;

    const float norm = __half2float(blk->norm) * alpha;

    // Sliding window decode: read 9-bit state from bitstream at bit offset t*3
    const int bit_pos = t * 3;
    const int byte_idx = bit_pos / 8;
    const int bit_off = bit_pos % 8;
    const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
    const int state = (raw >> bit_off) & 0x1FF;
    const float val = d_rq3_iso_codebook_fattn[state] * norm;

    dst[strm * (ne2 * ne1 * ne0) + head * (ne1 * ne0) + row * ne0 + j] = __float2half(val);
}

static __global__ void k_rq4_iso_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_RQ4_ISO;
    const int t = j % QK_RQ4_ISO;
    const block_rq4_iso * blk = (const block_rq4_iso *)src_row + blk_idx;

    const float norm = __half2float(blk->norm) * alpha;

    // Sliding window decode: read 8-bit state from bitstream at bit offset t*2
    const int bit_pos = t * 2;
    const int byte_idx = bit_pos / 8;
    const int bit_off = bit_pos % 8;
    const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
    const int state = (raw >> bit_off) & 0xFF;
    const float val = d_rq4_iso_codebook_fattn[state] * norm;

    dst[strm * (ne2 * ne1 * ne0) + head * (ne1 * ne0) + row * ne0 + j] = __float2half(val);
}

static __global__ void k_rq4_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_RQ4;
    const int j_in_blk = j % QK_RQ4;
    const block_rq4_0 * blk = (const block_rq4_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const uint8_t idx = (j_in_blk & 1) ? (blk->qs[j_in_blk / 2] >> 4) : (blk->qs[j_in_blk / 2] & 0xF);
    const float val = d_rq_centroids_4bit_fattn[idx] * norm;

    dst[strm * (ne2 * ne1 * ne0) + head * (ne1 * ne0) + row * ne0 + j] = __float2half(val);
}

// rq4 K dequant with inverse Givens rotation: produces K in original (unrotated) domain
// so Q does NOT need pre-rotation. 128 threads per block, loops over 128-element rq4 blocks.
static __global__ void k_rq4_dequant_f16_inv_givens(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne2 * ne1 * ne0) + head * (ne1 * ne0) + row * ne0;

    __shared__ float smem[128];

    const int n_blocks = (int)(ne0 / QK_RQ4);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_rq4_0 * blk = (const block_rq4_0 *)src_row + blk_idx;
        const float norm = __half2float(blk->norm);

        // Extract 4-bit index, lookup centroid
        const uint8_t idx = (tid & 1) ? (blk->qs[tid / 2] >> 4) : (blk->qs[tid / 2] & 0xF);
        smem[tid] = d_rq_centroids_4bit_fattn[idx];
        __syncthreads();

        // Inverse Givens rotation (64 pairs, each thread handles one pair)
        if (tid < 64) {
            const int p = tid;
            float v0 = smem[2*p], v1 = smem[2*p + 1];
            smem[2*p]     =  RQ_COS[p] * v0 + RQ_SIN[p] * v1;
            smem[2*p + 1] = -RQ_SIN[p] * v0 + RQ_COS[p] * v1;
        }
        __syncthreads();

        // Undo InnerQ scaling, apply norm, cast to fp16
        float val = smem[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        dst[dst_base + blk_idx * 128 + tid] = __float2half(val);
    }
}

// Persistent Q rotation buffer per device (shared between prefill and decode paths)
static float * q_rot_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t  q_rot_buf_size[GGML_CUDA_MAX_DEVICES] = {};

// Persistent K/V fp16 dequant buffers per device (shared between prefill and decode paths)
static half * kv_dequant_k_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t  kv_dequant_k_buf_size[GGML_CUDA_MAX_DEVICES] = {};
static half * kv_dequant_v_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t  kv_dequant_v_buf_size[GGML_CUDA_MAX_DEVICES] = {};

// === Givens rotation kernel for pre-rotate-queries approach ===
// Forward Givens rotation on Q before attention (both prefill and decode paths).
// One block per 128-element group, 128 threads per block (64 active for rotation).
static __global__ void k_rq_givens_forward(
        const float * __restrict__ src, float * __restrict__ dst,
        const int64_t n_elements) {
    const int64_t offset = blockIdx.x * 128;
    if (offset >= n_elements) return;

    __shared__ float buf[128];

    if (threadIdx.x < 128) {
        // InnerQ: apply inverse channel scale to Q before rotation
        // This compensates for the channel scaling applied to K in SET_ROWS
        buf[threadIdx.x] = src[offset + threadIdx.x] * d_innerq_channel_scale_inv_fattn[threadIdx.x];
    }
    __syncthreads();

    // Forward Givens rotation: 64 pairs, each thread handles one pair
    if (threadIdx.x < 64) {
        const int p = threadIdx.x;
        float v0 = buf[2*p], v1 = buf[2*p + 1];
        buf[2*p]     = RQ_COS[p] * v0 - RQ_SIN[p] * v1;
        buf[2*p + 1] = RQ_SIN[p] * v0 + RQ_COS[p] * v1;
    }
    __syncthreads();

    if (threadIdx.x < 128) {
        float val = buf[threadIdx.x];
        dst[offset + threadIdx.x] = val;

        // Q² calibration: accumulate per-position squared values
        if (d_q_calibrate_fattn) {
            atomicAdd(&d_q_channel_sq_fattn[threadIdx.x], (double)(val * val));
            if (threadIdx.x == 0) atomicAdd(&d_q_channel_count_fattn, 1);
        }
    }
}

static void ggml_cuda_rq_prefill_attend(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    load_tcq_decode_alpha();
    cudaStream_t stream = ctx.stream();
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool rq_k = K->type == GGML_TYPE_RQ2_0 || K->type == GGML_TYPE_RQ3_0 || K->type == GGML_TYPE_RQ4_0 || K->type == GGML_TYPE_RQ3_ISO || K->type == GGML_TYPE_RQ4_ISO;
    const bool rq_v = V->type == GGML_TYPE_RQ2_0 || V->type == GGML_TYPE_RQ3_0 || V->type == GGML_TYPE_RQ4_0 || V->type == GGML_TYPE_RQ3_ISO || V->type == GGML_TYPE_RQ4_ISO;

    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    half * k_fp16 = nullptr;
    half * v_fp16 = nullptr;

    // Allocate and dequant K to fp16 (rq2, rq3, or rq4)
    if (rq_k) {
        const size_t k_size = K->ne[0] * K->ne[1] * K->ne[2] * K->ne[3] * sizeof(half);
        if (k_size > kv_dequant_k_buf_size[device]) {
            if (kv_dequant_k_buf[device]) CUDA_CHECK(cudaFree(kv_dequant_k_buf[device]));
            CUDA_CHECK(cudaMalloc(&kv_dequant_k_buf[device], k_size));
            kv_dequant_k_buf_size[device] = k_size;
        }
        k_fp16 = kv_dequant_k_buf[device];
        dim3 grid_k(K->ne[1], K->ne[2], K->ne[3]);
        if (K->type == GGML_TYPE_RQ2_0) {
            k_rq2_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else if (K->type == GGML_TYPE_RQ3_0) {
            k_rq3_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else if (K->type == GGML_TYPE_RQ3_ISO) {
            {
                static bool tcq_fattn_k_cb_loaded = false;
                if (!tcq_fattn_k_cb_loaded) {
                    tcq_fattn_k_cb_loaded = true;
                    const char *cb_path = getenv("RQ_ISO_CB");
                    if (cb_path) {
                        float cb[512];
                        FILE *f = fopen(cb_path, "rb");
                        if (f && fread(cb, sizeof(float), 512, f) == 512) {
                            fclose(f);
                            cudaMemcpyToSymbol(d_rq3_iso_codebook_fattn, cb, 512*sizeof(float));
                            fprintf(stderr, "TCQ K prefill: loaded codebook from %s\n", cb_path);
                        } else {
                            if (f) fclose(f);
                            fprintf(stderr, "TCQ K prefill: FAILED to load codebook from %s\n", cb_path);
                        }
                    }
                }
            }
            k_rq3_iso_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
        } else if (K->type == GGML_TYPE_RQ4_ISO) {
            {
                static bool tcq2_fattn_k_cb_loaded = false;
                if (!tcq2_fattn_k_cb_loaded) {
                    tcq2_fattn_k_cb_loaded = true;
                    const char *cb_path = getenv("RQ_ISO_CB2");
                    if (cb_path) {
                        float cb[256];
                        FILE *f = fopen(cb_path, "rb");
                        if (f && fread(cb, sizeof(float), 256, f) == 256) {
                            fclose(f);
                            cudaMemcpyToSymbol(d_rq4_iso_codebook_fattn, cb, 256*sizeof(float));
                            fprintf(stderr, "TCQ2 K prefill: loaded 2-bit codebook from %s\n", cb_path);
                        } else {
                            if (f) fclose(f);
                            fprintf(stderr, "TCQ2 K prefill: FAILED to load codebook from %s\n", cb_path);
                        }
                    }
                }
            }
            k_rq4_iso_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
        } else {
            // rq4 K: inverse Givens dequant → produces K in original domain (no Q rotation needed)
            k_rq4_dequant_f16_inv_givens<<<grid_k, 128, 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        }
    }

    // Allocate and dequant V to fp16 (rq2, rq3, or rq4)
    if (rq_v) {
        const size_t v_size = V->ne[0] * V->ne[1] * V->ne[2] * V->ne[3] * sizeof(half);
        if (v_size > kv_dequant_v_buf_size[device]) {
            if (kv_dequant_v_buf[device]) CUDA_CHECK(cudaFree(kv_dequant_v_buf[device]));
            CUDA_CHECK(cudaMalloc(&kv_dequant_v_buf[device], v_size));
            kv_dequant_v_buf_size[device] = v_size;
        }
        v_fp16 = kv_dequant_v_buf[device];
        dim3 grid_v(V->ne[1], V->ne[2], V->ne[3]);
        if (V->type == GGML_TYPE_RQ2_0) {
            k_rq2_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else if (V->type == GGML_TYPE_RQ3_0) {
            k_rq3_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else if (V->type == GGML_TYPE_RQ3_ISO) {
            // Runtime codebook loading for 3-bit V decode (in case K is a different type)
            {
                static bool tcq_fattn_v_cb_loaded = false;
                if (!tcq_fattn_v_cb_loaded) {
                    tcq_fattn_v_cb_loaded = true;
                    const char *cb_path = getenv("RQ_ISO_CB");
                    if (cb_path) {
                        float cb[512];
                        FILE *f = fopen(cb_path, "rb");
                        if (f && fread(cb, sizeof(float), 512, f) == 512) {
                            fclose(f);
                            cudaMemcpyToSymbol(d_rq3_iso_codebook_fattn, cb, 512*sizeof(float));
                            fprintf(stderr, "TCQ V decode: loaded 3-bit codebook from %s\n", cb_path);
                        } else {
                            if (f) fclose(f);
                        }
                    }
                }
            }
            k_rq3_iso_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]));
        } else if (V->type == GGML_TYPE_RQ4_ISO) {
            // Runtime codebook loading for 2-bit V decode (in case K is a different type)
            {
                static bool tcq2_fattn_v_cb_loaded = false;
                if (!tcq2_fattn_v_cb_loaded) {
                    tcq2_fattn_v_cb_loaded = true;
                    const char *cb_path = getenv("RQ_ISO_CB2");
                    if (cb_path) {
                        float cb[256];
                        FILE *f = fopen(cb_path, "rb");
                        if (f && fread(cb, sizeof(float), 256, f) == 256) {
                            fclose(f);
                            cudaMemcpyToSymbol(d_rq4_iso_codebook_fattn, cb, 256*sizeof(float));
                            fprintf(stderr, "TCQ2 V decode: loaded 2-bit codebook from %s\n", cb_path);
                        } else {
                            if (f) fclose(f);
                        }
                    }
                }
            }
            k_rq4_iso_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]));
        } else {
            k_rq4_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        }
    }

    // Create fp16 tensor copies on stack
    ggml_tensor K_f16 = *K;
    ggml_tensor V_f16 = *V;

    if (k_fp16) {
        K_f16.type = GGML_TYPE_F16;
        K_f16.data = k_fp16;
        K_f16.nb[0] = sizeof(half);
        K_f16.nb[1] = K->ne[0] * sizeof(half);
        K_f16.nb[2] = K->ne[0] * K->ne[1] * sizeof(half);
        K_f16.nb[3] = K->ne[0] * K->ne[1] * K->ne[2] * sizeof(half);
    }

    if (v_fp16) {
        V_f16.type = GGML_TYPE_F16;
        V_f16.data = v_fp16;
        V_f16.nb[0] = sizeof(half);
        V_f16.nb[1] = V->ne[0] * sizeof(half);
        V_f16.nb[2] = V->ne[0] * V->ne[1] * sizeof(half);
        V_f16.nb[3] = V->ne[0] * V->ne[1] * V->ne[2] * sizeof(half);
    }

    // Rotate Q for rq pre-rotate-queries (only when K is in rotated space)
    // rq4 K is dequanted via inverse Givens → original domain, so Q stays unrotated
    const ggml_tensor * Q = dst->src[0];
    float * q_rotated = nullptr;
    if (rq_k && K->type != GGML_TYPE_RQ4_0 && Q->ne[0] % 128 == 0) {
        const size_t q_size = ggml_nelements(Q) * sizeof(float);
        if (q_size > q_rot_buf_size[device]) {
            if (q_rot_buf[device]) CUDA_CHECK(cudaFree(q_rot_buf[device]));
            CUDA_CHECK(cudaMalloc(&q_rot_buf[device], q_size));
            q_rot_buf_size[device] = q_size;
        }
        q_rotated = q_rot_buf[device];
        const int64_t n_q_groups = ggml_nelements(Q) / 128;
        k_rq_givens_forward<<<(int)n_q_groups, 128, 0, stream>>>(
            (const float *)Q->data, q_rotated, ggml_nelements(Q));
    }

    // Temporarily swap src pointers to fp16 K/V and rotated Q
    ggml_tensor * orig_q = dst->src[0];
    ggml_tensor * orig_k = dst->src[1];
    ggml_tensor * orig_v = dst->src[2];

    ggml_tensor Q_rot;
    if (q_rotated) {
        Q_rot = *Q;
        Q_rot.data = q_rotated;
        dst->src[0] = &Q_rot;
    }
    dst->src[1] = k_fp16 ? &K_f16 : orig_k;
    dst->src[2] = v_fp16 ? &V_f16 : orig_v;

    // Dispatch to MMA kernel (sees rotated Q, fp16 K/V, uses tensor cores)
    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);

    // Restore original tensor pointers
    dst->src[0] = orig_q;
    dst->src[1] = orig_k;
    dst->src[2] = orig_v;

    // K/V fp16 buffers are persistent (grow-only), no free needed
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

#define FATTN_VEC_CASES_ALL_D_512(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \
    FATTN_VEC_CASE(512, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)

    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ2_0, GGML_TYPE_RQ2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ3_0, GGML_TYPE_RQ3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ4_0, GGML_TYPE_RQ4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ2_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_RQ2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_RQ3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_RQ4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ4_0, GGML_TYPE_RQ3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ3_0, GGML_TYPE_RQ4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ2_0, GGML_TYPE_RQ3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ3_0, GGML_TYPE_RQ2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ3_ISO, GGML_TYPE_RQ3_ISO)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ4_ISO, GGML_TYPE_RQ4_ISO)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ3_ISO, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ4_ISO, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,       GGML_TYPE_RQ3_ISO)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,       GGML_TYPE_RQ4_ISO)
#else
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ2_0, GGML_TYPE_RQ2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ3_0, GGML_TYPE_RQ3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ4_0, GGML_TYPE_RQ4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ2_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_RQ2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_RQ3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_RQ4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ4_0, GGML_TYPE_RQ3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ3_0, GGML_TYPE_RQ4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ2_0, GGML_TYPE_RQ3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ3_0, GGML_TYPE_RQ2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ3_ISO, GGML_TYPE_RQ3_ISO)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ4_ISO, GGML_TYPE_RQ4_ISO)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ3_ISO, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_RQ4_ISO, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,       GGML_TYPE_RQ3_ISO)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,       GGML_TYPE_RQ4_ISO)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_WMMA_F16 = 300,
    BEST_FATTN_KERNEL_MMA_F16  = 400,
};

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

    switch (K->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return BEST_FATTN_KERNEL_NONE;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
        case GGML_TYPE_RQ2_0:
        case GGML_TYPE_RQ3_0:
        case GGML_TYPE_RQ4_0:
        case GGML_TYPE_RQ3_ISO:
        case GGML_TYPE_RQ4_ISO:
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:

    // RotorQuant: only the vec kernel has native rq dequant support.
    // No FATTN_KQ_STRIDE alignment needed — vec kernel handles arbitrary lengths.
    if (K->type == GGML_TYPE_RQ2_0 || V->type == GGML_TYPE_RQ2_0 ||
        K->type == GGML_TYPE_RQ3_0 || V->type == GGML_TYPE_RQ3_0 ||
        K->type == GGML_TYPE_RQ4_0 || V->type == GGML_TYPE_RQ4_0 ||
        K->type == GGML_TYPE_RQ3_ISO || V->type == GGML_TYPE_RQ3_ISO ||
        K->type == GGML_TYPE_RQ4_ISO || V->type == GGML_TYPE_RQ4_ISO) {
        if (Q->ne[0] <= 512 && Q->ne[0] % 64 == 0)
            return BEST_FATTN_KERNEL_VEC;
        return BEST_FATTN_KERNEL_NONE;
    }

    // D=512: MMA/TILE templates don't support this head_dim, use VEC unconditionally
    if (Q->ne[0] == 512) {
        return BEST_FATTN_KERNEL_VEC;
    }

    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        int gqa_ratio_eff = 1;
        const int ncols2_max = Q->ne[0] == 576 ? 16 : 8;
        while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
            gqa_ratio_eff *= 2;
        }
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use the WMMA kernel if possible:
    if (ggml_cuda_should_use_wmma_fattn(cc) && K->ne[1] % FATTN_KQ_STRIDE == 0 && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 512 && Q->ne[0] != 576) {
        if (can_use_vector_kernel && Q->ne[1] <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        return BEST_FATTN_KERNEL_WMMA_F16;
    }

    if (amd_wmma_available(cc) && GGML_CUDA_CC_IS_RDNA4(cc) && gqa_opt_applies && Q->ne[0] <= 128 && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (Q->ne[1] == 1) {
                    if (!gqa_opt_applies) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            } else {
                if (Q->ne[1] <= 2) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        }
        int gqa_ratio_eff = 1;
        const int ncols2_max = Q->ne[0] == 576 ? 16 : 8;
        while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
            gqa_ratio_eff *= 2;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 8) {
            return BEST_FATTN_KERNEL_TILE; // AMD WMMA is only faster if the full tile width of 16 can be utilized.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use MFMA flash attention for CDNA (MI100+):
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 256 && Q->ne[0] != 512 && Q->ne[0] != 576) {
        const int64_t eff_nq = Q->ne[1] * (gqa_opt_applies ? gqa_ratio : 1);
        // MMA vs tile crossover benchmarked on MI300X @ d32768:
        //   hsk=64  (gqa=4): MMA wins at eff >= 128 (+11%)
        //   hsk=128 (gqa=4): MMA wins at eff >= 128 (+4%)
        if (eff_nq >= (GGML_CUDA_CC_IS_CDNA1(cc) && Q->ne[0] == 64 ? 64 : 128)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        // Fall through to tile kernel for small effective batch sizes.
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // RQ prefill: dequant to fp16 and use tensor core MMA for batched attention.
    // rq4 K uses inverse Givens during dequant — mixes centroids in float32 shmem before
    // fp16 cast, so precision is fine. rq2/rq3 use simple centroid×norm dequant.
    // Set RQ_PREFILL_VEC=1 to force vec kernel for all rq types (debug override).
    static const bool rq_prefill_vec = [] {
        const char * e = getenv("RQ_PREFILL_VEC");
        if (e) fprintf(stderr, "RQ_PREFILL_VEC=%s: forcing vec prefill for rq types\n", e);
        return e != nullptr;
    }();
    const bool rq_kv = K->type == GGML_TYPE_RQ2_0 || K->type == GGML_TYPE_RQ3_0 || K->type == GGML_TYPE_RQ4_0 || K->type == GGML_TYPE_RQ3_ISO || K->type == GGML_TYPE_RQ4_ISO ||
                          V->type == GGML_TYPE_RQ2_0 || V->type == GGML_TYPE_RQ3_0 || V->type == GGML_TYPE_RQ4_0 || V->type == GGML_TYPE_RQ3_ISO || V->type == GGML_TYPE_RQ4_ISO;
    if (rq_kv && !rq_prefill_vec && Q->ne[1] > 1 && Q->ne[0] <= 256 && turing_mma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        // Prefill path: rq4 K uses inverse Givens dequant (original domain, no Q rotation),
        // rq2/3 K uses simple dequant (rotated domain, Q pre-rotated). V un-rotation at graph level.
        ggml_cuda_rq_prefill_attend(ctx, dst);
    } else {
        load_tcq_decode_alpha();

        // Update VEC __constant__ alpha for context-adaptive mode
        if (d_tcq_decode_alpha_v_static == 0.0f &&
            (V->type == GGML_TYPE_RQ3_ISO || V->type == GGML_TYPE_RQ4_ISO)) {
            float alpha = tcq_compute_alpha_v(V->type, V->ne[1]);
            cudaMemcpyToSymbol(d_tcq_decode_alpha_v_fattn, &alpha, sizeof(float));
        }

        // Load runtime codebooks for TCQ types (needed by both dequant and native VEC paths)
        if (K->type == GGML_TYPE_RQ3_ISO || V->type == GGML_TYPE_RQ3_ISO) {
            static bool tcq3_cb_loaded = false;
            if (!tcq3_cb_loaded) {
                tcq3_cb_loaded = true;
                const char *cb_path = getenv("RQ_ISO_CB");
                if (cb_path) {
                    float cb[512];
                    FILE *f = fopen(cb_path, "rb");
                    if (f && fread(cb, sizeof(float), 512, f) == 512) {
                        fclose(f);
                        cudaMemcpyToSymbol(d_rq3_iso_codebook_fattn, cb, 512*sizeof(float));
                        fprintf(stderr, "TCQ decode: loaded 3-bit codebook from %s\n", cb_path);
                    } else {
                        if (f) fclose(f);
                        fprintf(stderr, "TCQ decode: FAILED to load 3-bit codebook from %s\n", cb_path);
                    }
                }
            }
        }
        if (K->type == GGML_TYPE_RQ4_ISO || V->type == GGML_TYPE_RQ4_ISO) {
            static bool tcq2_cb_loaded = false;
            if (!tcq2_cb_loaded) {
                tcq2_cb_loaded = true;
                const char *cb_path = getenv("RQ_ISO_CB2");
                if (cb_path) {
                    float cb[256];
                    FILE *f = fopen(cb_path, "rb");
                    if (f && fread(cb, sizeof(float), 256, f) == 256) {
                        fclose(f);
                        cudaMemcpyToSymbol(d_rq4_iso_codebook_fattn, cb, 256*sizeof(float));
                        fprintf(stderr, "TCQ decode: loaded 2-bit codebook from %s\n", cb_path);
                    } else {
                        if (f) fclose(f);
                        fprintf(stderr, "TCQ decode: FAILED to load 2-bit codebook from %s\n", cb_path);
                    }
                }
            }
        }

        cudaStream_t stream = ctx.stream();

        // Dequant rq K/V to fp16 for decode: MMA tensor cores on fp16 beat VEC scalar
        // on rq bits for dense models. Bandwidth savings from rq's 3/16 footprint are
        // negligible relative to FFN compute (~1% slower native on Qwen3.5-27B, 3090).
        // Set GGML_RQ_DECODE_NATIVE=1 to force native VEC path (may help bandwidth-limited configs).
        static const bool rq_decode_native = (getenv("GGML_RQ_DECODE_NATIVE") != nullptr);
        // Skip dequant for D>256: MMA templates only go up to 256; use native VEC path instead
        const bool do_decode_dequant = !rq_decode_native && rq_kv && Q->ne[0] <= 256;

        half * k_fp16_dec = nullptr;
        half * v_fp16_dec = nullptr;
        ggml_tensor K_f16_dec, V_f16_dec;
        ggml_tensor * orig_k_decode = nullptr;
        ggml_tensor * orig_v_decode = nullptr;

        int device_dec;
        CUDA_CHECK(cudaGetDevice(&device_dec));

        if (do_decode_dequant) {
            if (K->type == GGML_TYPE_RQ2_0 || K->type == GGML_TYPE_RQ3_0 || K->type == GGML_TYPE_RQ3_ISO || K->type == GGML_TYPE_RQ4_ISO) {
                const size_t k_size = K->ne[0] * K->ne[1] * K->ne[2] * K->ne[3] * sizeof(half);
                if (k_size > kv_dequant_k_buf_size[device_dec]) {
                    if (kv_dequant_k_buf[device_dec]) CUDA_CHECK(cudaFree(kv_dequant_k_buf[device_dec]));
                    CUDA_CHECK(cudaMalloc(&kv_dequant_k_buf[device_dec], k_size));
                    kv_dequant_k_buf_size[device_dec] = k_size;
                }
                k_fp16_dec = kv_dequant_k_buf[device_dec];
                // K dequant to fp16 in rotated domain (Q rotation handles domain matching)
                dim3 grid_k(K->ne[1], K->ne[2], K->ne[3]);
                if (K->type == GGML_TYPE_RQ2_0) {
                    k_rq2_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_RQ3_ISO) {
                    k_rq3_iso_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
                } else if (K->type == GGML_TYPE_RQ4_ISO) {
                    k_rq4_iso_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
                } else {
                    k_rq3_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                }
                K_f16_dec = *K;
                K_f16_dec.type = GGML_TYPE_F16;
                K_f16_dec.data = k_fp16_dec;
                K_f16_dec.nb[0] = sizeof(half);
                K_f16_dec.nb[1] = K->ne[0] * sizeof(half);
                K_f16_dec.nb[2] = K->ne[0] * K->ne[1] * sizeof(half);
                K_f16_dec.nb[3] = K->ne[0] * K->ne[1] * K->ne[2] * sizeof(half);
                orig_k_decode = dst->src[1];
                dst->src[1] = &K_f16_dec;
            }
            if (V->type == GGML_TYPE_RQ2_0 || V->type == GGML_TYPE_RQ3_0 || V->type == GGML_TYPE_RQ3_ISO || V->type == GGML_TYPE_RQ4_ISO) {
                const size_t v_size = V->ne[0] * V->ne[1] * V->ne[2] * V->ne[3] * sizeof(half);
                if (v_size > kv_dequant_v_buf_size[device_dec]) {
                    if (kv_dequant_v_buf[device_dec]) CUDA_CHECK(cudaFree(kv_dequant_v_buf[device_dec]));
                    CUDA_CHECK(cudaMalloc(&kv_dequant_v_buf[device_dec], v_size));
                    kv_dequant_v_buf_size[device_dec] = v_size;
                }
                v_fp16_dec = kv_dequant_v_buf[device_dec];
                dim3 grid_v(V->ne[1], V->ne[2], V->ne[3]);
                if (V->type == GGML_TYPE_RQ2_0) {
                    k_rq2_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                } else if (V->type == GGML_TYPE_RQ3_ISO) {
                    k_rq3_iso_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]));
                } else if (V->type == GGML_TYPE_RQ4_ISO) {
                    k_rq4_iso_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]));
                } else {
                    k_rq3_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                }
                V_f16_dec = *V;
                V_f16_dec.type = GGML_TYPE_F16;
                V_f16_dec.data = v_fp16_dec;
                V_f16_dec.nb[0] = sizeof(half);
                V_f16_dec.nb[1] = V->ne[0] * sizeof(half);
                V_f16_dec.nb[2] = V->ne[0] * V->ne[1] * sizeof(half);
                V_f16_dec.nb[3] = V->ne[0] * V->ne[1] * V->ne[2] * sizeof(half);
                orig_v_decode = dst->src[2];
                dst->src[2] = &V_f16_dec;
            }
        }

        // Pre-rotate Q for rq (K/V stored in rotated space, whether rq3 or dequanted fp16)
        ggml_tensor Q_rot_decode;
        ggml_tensor * orig_q_decode = nullptr;
        const bool rq_k_any = (K->type == GGML_TYPE_RQ2_0 || K->type == GGML_TYPE_RQ3_0 || K->type == GGML_TYPE_RQ4_0 || K->type == GGML_TYPE_RQ3_ISO || K->type == GGML_TYPE_RQ4_ISO);
        if (rq_k_any && Q->ne[0] % 128 == 0) {
            const size_t q_size = ggml_nelements(Q) * sizeof(float);
            if (q_size > q_rot_buf_size[device_dec]) {
                if (q_rot_buf[device_dec]) CUDA_CHECK(cudaFree(q_rot_buf[device_dec]));
                CUDA_CHECK(cudaMalloc(&q_rot_buf[device_dec], q_size));
                q_rot_buf_size[device_dec] = q_size;
            }
            const int64_t n_q_groups = ggml_nelements(Q) / 128;
            k_rq_givens_forward<<<(int)n_q_groups, 128, 0, stream>>>(
                (const float *)Q->data, q_rot_buf[device_dec], ggml_nelements(Q));
            Q_rot_decode = *Q;
            Q_rot_decode.data = q_rot_buf[device_dec];
            orig_q_decode = dst->src[0];
            dst->src[0] = &Q_rot_decode;
        }

        switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
            case BEST_FATTN_KERNEL_NONE:
                GGML_ABORT("fatal error");
            case BEST_FATTN_KERNEL_TILE:
                ggml_cuda_flash_attn_ext_tile(ctx, dst);
                break;
            case BEST_FATTN_KERNEL_VEC:
                ggml_cuda_flash_attn_ext_vec(ctx, dst);
                break;
            case BEST_FATTN_KERNEL_WMMA_F16:
                ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
                break;
            case BEST_FATTN_KERNEL_MMA_F16:
                ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
                break;
        }

        if (orig_q_decode) dst->src[0] = orig_q_decode;
        if (orig_k_decode) dst->src[1] = orig_k_decode;
        if (orig_v_decode) dst->src[2] = orig_v_decode;
        // K/V fp16 buffers are persistent (grow-only), no free needed
    }

    // Output inverse rotation for rq V types is handled at graph level
    // (ggml_rq_rotate op in llama-graph.cpp) to maintain CUDA graph compatibility.
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
