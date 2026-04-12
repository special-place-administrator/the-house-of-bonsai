/*
 * RotorQuant: KV cache compression via Givens/Quaternion rotation
 *
 * PlanarQuant (Givens 2D rotation) for rq2_0, rq3_0, rq4_0
 * IsoQuant   (Quaternion 4D rotation) for rq3_iso, rq4_iso
 *
 * Precomputed rotation constants are hardcoded for d=128.
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

/* ---------- constants ---------- */

#define RQ_D 128  /* rotation group size = head_dim */

/* Lloyd-Max centroids for N(0, 1/sqrt(128)) */

static const float CENTROIDS_2BIT[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };

static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

static const float CENTROIDS_4BIT[16] = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};

static const float MIDPOINTS_4BIT[15] = {
    -0.212232f, -0.162977f, -0.127056f, -0.097191f, -0.070693f,
    -0.046190f, -0.022832f,  0.000000f,  0.022832f,  0.046190f,
     0.070693f,  0.097191f,  0.127056f,  0.162977f,  0.212232f,
};

/* ---------- PlanarQuant: precomputed Givens cos/sin (64 pairs for d=128) ---------- */

static const float RQ_COS[64]={0.7386546135f, 0.8607548475f, -0.7411674857f, 0.9674890637f, -0.7723053098f, -0.8056974411f, -0.0412844308f, 0.2707833052f, 0.9315500855f, 0.6698185802f, 0.9167487621f, -0.8320636749f, 0.6818146110f, -0.9108457565f, -0.0559285842f, -0.9032276273f, 0.7519487143f, -0.8941103816f, -0.1039871648f, -0.6961420774f, -0.1230370328f, -0.9328963161f, -0.2905603051f, 0.4910068214f, 0.7889407277f, -0.1221836656f, -0.6316579580f, 0.3128163815f, -0.9563610554f, 0.9992509484f, 0.9540294409f, 0.8902468085f, 0.7543080449f, -0.8664138913f, -0.5232898593f, 0.3621287644f, -0.8825117350f, 0.8234673142f, -0.9416025877f, -0.5480425358f, -0.6644080281f, -0.6585279703f, -0.2460795939f, 0.9438471198f, 0.2427810431f, -0.1960992366f, 0.2403578013f, -0.8461306095f, 0.0246123374f, 0.3372744620f, 0.9994974732f, -0.3494733870f, 0.7438930869f, 0.8452339768f, -0.6177822948f, -0.2662552595f, -0.5457068086f, -0.9985070229f, 0.7757105827f, 0.6141811609f, -0.9805000424f, 0.5425475240f, -0.5663578510f, -0.4696439803f};
static const float RQ_SIN[64]={-0.6740840673f, -0.5090196729f, 0.6713201404f, -0.2529129684f, 0.6352515221f, -0.5923272967f, 0.9991474152f, -0.9626403451f, -0.3636130989f, 0.7425247431f, -0.3994642496f, -0.5546801090f, -0.7315250039f, -0.4127469361f, -0.9984347820f, 0.4291617870f, -0.6592215896f, -0.4478466809f, 0.9945786595f, -0.7179040313f, 0.9924020767f, 0.3601450622f, 0.9568566680f, -0.8711557388f, 0.6144692898f, 0.9925075173f, 0.7752471566f, 0.9498136044f, -0.2921875417f, 0.0386975110f, -0.2997128963f, 0.4554784000f, -0.6565206647f, -0.4993265271f, 0.8521547318f, -0.9321280718f, -0.4702904224f, -0.5673637390f, -0.3367263079f, 0.8364504576f, -0.7473700047f, 0.7525562644f, -0.9692496061f, -0.3303825557f, -0.9700810909f, 0.9805840850f, -0.9706843495f, -0.5329755545f, -0.9996970892f, 0.9414063692f, 0.0316982083f, 0.9369462729f, 0.6682986617f, -0.5343964100f, -0.7863491774f, -0.9639025331f, -0.8379761577f, 0.0546237342f, -0.6310887933f, 0.7891650796f, -0.1965190321f, 0.8400250673f, -0.8241594434f, 0.8828558922f};

/* ---------- IsoQuant: precomputed unit quaternions (32 groups for d=128) ---------- */

static const float RQ_ISO_QW[32]={0.5765609741f, 0.3176580369f, -0.3234235942f, -0.5127438903f, 0.9233905673f, -0.3323571086f, 0.5468608141f, -0.2500519454f, -0.5812215805f, 0.3228830695f, -0.7299832702f, -0.4535493255f, -0.7338157296f, -0.2884652913f, -0.9000198841f, -0.0377033800f, 0.5104404092f, 0.2033989877f, -0.2462528497f, 0.2314069420f, 0.0072374810f, 0.3923372924f, 0.4958070219f, -0.7235037088f, -0.9383618832f, 0.4430379272f, -0.2075705230f, 0.1983736306f, -0.8834578991f, 0.7389573455f, -0.0156172011f, 0.7738668919f};
static const float RQ_ISO_QX[32]={0.4450169504f, -0.5780548453f, 0.7089627385f, -0.3940812945f, -0.0897334740f, 0.4727236331f, 0.5542563796f, 0.0450818054f, -0.3657043576f, -0.4298477769f, 0.4666220546f, 0.7556306720f, -0.5284956098f, 0.7042509317f, 0.0230921544f, 0.7110687494f, 0.3024962246f, -0.1157865301f, 0.7490812540f, -0.2582575679f, -0.2255804837f, 0.3838746250f, -0.3209520578f, -0.3477301002f, 0.1824720055f, 0.4032751918f, 0.8433781862f, 0.9533935785f, -0.0620501526f, 0.0927560627f, 0.2964956462f, 0.2402082384f};
static const float RQ_ISO_QY[32]={0.2695076466f, -0.0201656222f, -0.1687686443f, -0.5415957570f, -0.2796611190f, 0.3510629535f, 0.2609911859f, -0.2715902030f, -0.0937586129f, 0.3095585108f, -0.4123268127f, -0.4394895136f, 0.0626545250f, -0.4811822474f, -0.0407132693f, -0.4566248953f, 0.7834537029f, -0.6187923551f, 0.0809760988f, -0.8879503012f, -0.8928058147f, 0.8350352049f, -0.6994170547f, 0.5606835485f, 0.2933705449f, 0.7377059460f, 0.4534837306f, -0.0009816211f, -0.3632916510f, -0.3959124386f, 0.1631654203f, 0.5088164806f};
static const float RQ_ISO_QZ[32]={-0.6300023794f, -0.7513582706f, -0.6035611629f, 0.5370919704f, 0.2471584976f, 0.7367672324f, 0.5706370473f, 0.9282674193f, 0.7208684087f, -0.7843156457f, -0.2817355990f, -0.1736787707f, 0.4222335219f, -0.4350655377f, 0.4333281815f, 0.5333415866f, 0.1847889870f, 0.7498788238f, 0.6096553802f, -0.3021556735f, -0.3898189068f, 0.0377884321f, 0.4024685621f, 0.2031257302f, 0.0107116764f, -0.3112498820f, 0.1999502629f, -0.2273492515f, 0.2892593443f, 0.5372074246f, 0.9408631325f, 0.2907505929f};

/* ---------- nearest centroid ---------- */

static int nearest_centroid_2bit(float val) {
    if (val < -0.086728f) return 0;
    if (val <  0.000000f) return 1;
    if (val <  0.086728f) return 2;
    return 3;
}

static int nearest_centroid_3bit(float val) {
    if (val < -0.154259f) return 0;
    if (val < -0.091775f) return 1;
    if (val < -0.043589f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.043589f) return 4;
    if (val <  0.091775f) return 5;
    if (val <  0.154259f) return 6;
    return 7;
}

static int nearest_centroid_4bit(float val) {
    if (val < MIDPOINTS_4BIT[7]) {
        if (val < MIDPOINTS_4BIT[3]) {
            if (val < MIDPOINTS_4BIT[1]) return val < MIDPOINTS_4BIT[0] ? 0 : 1;
            else                         return val < MIDPOINTS_4BIT[2] ? 2 : 3;
        } else {
            if (val < MIDPOINTS_4BIT[5]) return val < MIDPOINTS_4BIT[4] ? 4 : 5;
            else                         return val < MIDPOINTS_4BIT[6] ? 6 : 7;
        }
    } else {
        if (val < MIDPOINTS_4BIT[11]) {
            if (val < MIDPOINTS_4BIT[9])  return val < MIDPOINTS_4BIT[8] ? 8 : 9;
            else                          return val < MIDPOINTS_4BIT[10] ? 10 : 11;
        } else {
            if (val < MIDPOINTS_4BIT[13]) return val < MIDPOINTS_4BIT[12] ? 12 : 13;
            else                          return val < MIDPOINTS_4BIT[14] ? 14 : 15;
        }
    }
}

/* ---------- quaternion multiply helper for IsoQuant ---------- */

static inline void quat_mul(float aw, float ax, float ay, float az,
                             float bw, float bx, float by, float bz,
                             float *rw, float *rx, float *ry, float *rz) {
    *rw = aw*bw - ax*bx - ay*by - az*bz;
    *rx = aw*bx + ax*bw + ay*bz - az*by;
    *ry = aw*by - ax*bz + ay*bw + az*bx;
    *rz = aw*bz + ax*by - ay*bx + az*bw;
}

/* ========================================================================
 * PlanarQuant (Givens 2D rotation) -- rq2_0, rq3_0, rq4_0
 *
 * Forward rotation:  r0 = cos*v0 - sin*v1,  r1 = sin*v0 + cos*v1
 * Inverse rotation:  f0 = cos*q0 + sin*q1,  f1 = -sin*q0 + cos*q1
 *
 * The rotation operates on 128-element groups (64 pairs).
 * For rq2_0/rq3_0 (QK=32), 4 consecutive blocks form one rotation group.
 * For rq4_0 (QK=128), each block is one rotation group.
 * ======================================================================== */

/* ---------- RQ2_0: PlanarQuant 2-bit ---------- */

void quantize_row_rq2_0_ref(const float * GGML_RESTRICT x, block_rq2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % RQ_D == 0);
    const int ngroups = k / RQ_D;
    const int blocks_per_group = RQ_D / QK_RQ2;  /* 128/32 = 4 */

    for (int g = 0; g < ngroups; g++) {
        const float * src = x + g * RQ_D;

        /* Compute group norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < RQ_D; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[RQ_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < RQ_D; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, sizeof(normalized));
        }

        /* Forward Givens rotation (64 pairs) */
        float rotated[RQ_D];
        for (int p = 0; p < 64; p++) {
            float v0 = normalized[2*p];
            float v1 = normalized[2*p + 1];
            rotated[2*p]     = RQ_COS[p] * v0 - RQ_SIN[p] * v1;
            rotated[2*p + 1] = RQ_SIN[p] * v0 + RQ_COS[p] * v1;
        }

        /* Quantize to 2-bit centroids */
        uint8_t indices[RQ_D];
        for (int i = 0; i < RQ_D; i++) {
            indices[i] = (uint8_t)nearest_centroid_2bit(rotated[i]);
        }

        /* Norm correction */
        float recon_sq = 0.0f;
        for (int i = 0; i < RQ_D; i++) {
            float r = CENTROIDS_2BIT[indices[i]];
            recon_sq += r * r;
        }
        float recon_norm = sqrtf(recon_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? norm / recon_norm : norm;

        /* Pack into 4 consecutive blocks */
        for (int b = 0; b < blocks_per_group; b++) {
            int bi = g * blocks_per_group + b;
            y[bi].norm = GGML_FP32_TO_FP16(corrected_norm);
            memset(y[bi].qs, 0, QK_RQ2 / 4);
            for (int j = 0; j < QK_RQ2; j++) {
                int elem = b * QK_RQ2 + j;
                y[bi].qs[j / 4] |= (indices[elem] & 0x3) << ((j % 4) * 2);
            }
        }
    }
}

void dequantize_row_rq2_0(const block_rq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % RQ_D == 0);
    const int ngroups = k / RQ_D;
    const int blocks_per_group = RQ_D / QK_RQ2;

    for (int g = 0; g < ngroups; g++) {
        float norm = GGML_FP16_TO_FP32(x[g * blocks_per_group].norm);

        /* Unpack 2-bit indices from 4 consecutive blocks */
        float rotated_recon[RQ_D];
        for (int b = 0; b < blocks_per_group; b++) {
            int bi = g * blocks_per_group + b;
            for (int j = 0; j < QK_RQ2; j++) {
                int elem = b * QK_RQ2 + j;
                uint8_t idx = (x[bi].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
                rotated_recon[elem] = CENTROIDS_2BIT[idx];
            }
        }

        /* Inverse Givens rotation */
        float * dst = y + g * RQ_D;
        for (int p = 0; p < 64; p++) {
            float q0 = rotated_recon[2*p];
            float q1 = rotated_recon[2*p + 1];
            dst[2*p]     =  RQ_COS[p] * q0 + RQ_SIN[p] * q1;
            dst[2*p + 1] = -RQ_SIN[p] * q0 + RQ_COS[p] * q1;
        }

        /* Scale by norm */
        for (int i = 0; i < RQ_D; i++) {
            dst[i] *= norm;
        }
    }
}

size_t quantize_rq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                       int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_RQ2 == 0);

    size_t row_size = (n_per_row / QK_RQ2) * sizeof(block_rq2_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_rq2_0_ref(
            src + row * n_per_row,
            (block_rq2_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- RQ3_0: PlanarQuant 3-bit ---------- */

void quantize_row_rq3_0_ref(const float * GGML_RESTRICT x, block_rq3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % RQ_D == 0);
    const int ngroups = k / RQ_D;
    const int blocks_per_group = RQ_D / QK_RQ3;  /* 128/32 = 4 */

    for (int g = 0; g < ngroups; g++) {
        const float * src = x + g * RQ_D;

        /* Compute group norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < RQ_D; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[RQ_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < RQ_D; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, sizeof(normalized));
        }

        /* Forward Givens rotation (64 pairs) */
        float rotated[RQ_D];
        for (int p = 0; p < 64; p++) {
            float v0 = normalized[2*p];
            float v1 = normalized[2*p + 1];
            rotated[2*p]     = RQ_COS[p] * v0 - RQ_SIN[p] * v1;
            rotated[2*p + 1] = RQ_SIN[p] * v0 + RQ_COS[p] * v1;
        }

        /* Quantize to 3-bit centroids */
        uint8_t indices[RQ_D];
        for (int i = 0; i < RQ_D; i++) {
            indices[i] = (uint8_t)nearest_centroid_3bit(rotated[i]);
        }

        /* Norm correction */
        float recon_sq = 0.0f;
        for (int i = 0; i < RQ_D; i++) {
            float r = CENTROIDS_3BIT[indices[i]];
            recon_sq += r * r;
        }
        float recon_norm = sqrtf(recon_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? norm / recon_norm : norm;

        /* Pack into 4 consecutive blocks: lower 2 bits in qs[], upper 1 bit in signs[] */
        for (int b = 0; b < blocks_per_group; b++) {
            int bi = g * blocks_per_group + b;
            y[bi].norm = GGML_FP32_TO_FP16(corrected_norm);
            memset(y[bi].qs, 0, QK_RQ3 / 4);
            memset(y[bi].signs, 0, QK_RQ3 / 8);
            for (int j = 0; j < QK_RQ3; j++) {
                int elem = b * QK_RQ3 + j;
                uint8_t idx = indices[elem];
                /* lower 2 bits */
                y[bi].qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                /* upper 1 bit */
                y[bi].signs[j / 8] |= ((idx >> 2) & 0x1) << (j % 8);
            }
        }
    }
}

void dequantize_row_rq3_0(const block_rq3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % RQ_D == 0);
    const int ngroups = k / RQ_D;
    const int blocks_per_group = RQ_D / QK_RQ3;

    for (int g = 0; g < ngroups; g++) {
        float norm = GGML_FP16_TO_FP32(x[g * blocks_per_group].norm);

        /* Unpack 3-bit indices from 4 consecutive blocks */
        float rotated_recon[RQ_D];
        for (int b = 0; b < blocks_per_group; b++) {
            int bi = g * blocks_per_group + b;
            for (int j = 0; j < QK_RQ3; j++) {
                int elem = b * QK_RQ3 + j;
                uint8_t low2 = (x[bi].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
                uint8_t hi1  = (x[bi].signs[j / 8] >> (j % 8)) & 0x1;
                uint8_t idx  = low2 | (hi1 << 2);
                rotated_recon[elem] = CENTROIDS_3BIT[idx];
            }
        }

        /* Inverse Givens rotation */
        float * dst = y + g * RQ_D;
        for (int p = 0; p < 64; p++) {
            float q0 = rotated_recon[2*p];
            float q1 = rotated_recon[2*p + 1];
            dst[2*p]     =  RQ_COS[p] * q0 + RQ_SIN[p] * q1;
            dst[2*p + 1] = -RQ_SIN[p] * q0 + RQ_COS[p] * q1;
        }

        /* Scale by norm */
        for (int i = 0; i < RQ_D; i++) {
            dst[i] *= norm;
        }
    }
}

size_t quantize_rq3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                       int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_RQ3 == 0);

    size_t row_size = (n_per_row / QK_RQ3) * sizeof(block_rq3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_rq3_0_ref(
            src + row * n_per_row,
            (block_rq3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- RQ4_0: PlanarQuant 4-bit ---------- */

void quantize_row_rq4_0_ref(const float * GGML_RESTRICT x, block_rq4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_RQ4 == 0);
    const int nb = k / QK_RQ4;
    const int d  = QK_RQ4;  /* 128 */

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        /* Compute norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[RQ_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, sizeof(normalized));
        }

        /* Forward Givens rotation (64 pairs) */
        float rotated[RQ_D];
        for (int p = 0; p < 64; p++) {
            float v0 = normalized[2*p];
            float v1 = normalized[2*p + 1];
            rotated[2*p]     = RQ_COS[p] * v0 - RQ_SIN[p] * v1;
            rotated[2*p + 1] = RQ_SIN[p] * v0 + RQ_COS[p] * v1;
        }

        /* Quantize to 4-bit centroids */
        uint8_t indices[RQ_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_4bit(rotated[i]);
        }

        /* Norm correction */
        float recon_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            float r = CENTROIDS_4BIT[indices[i]];
            recon_sq += r * r;
        }
        float recon_norm = sqrtf(recon_sq);
        y[block].norm = GGML_FP32_TO_FP16((recon_norm > 1e-10f) ? norm / recon_norm : norm);

        /* Pack 4-bit indices: 2 per byte, low nibble first */
        for (int i = 0; i < d; i += 2) {
            y[block].qs[i / 2] = (uint8_t)((indices[i + 1] << 4) | (indices[i] & 0xF));
        }
    }
}

void dequantize_row_rq4_0(const block_rq4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_RQ4 == 0);
    const int nb = k / QK_RQ4;
    const int d  = QK_RQ4;

    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);

        /* Unpack 4-bit indices and look up centroids */
        float rotated_recon[RQ_D];
        for (int i = 0; i < d; i++) {
            uint8_t idx = (i & 1) ? (x[block].qs[i / 2] >> 4) : (x[block].qs[i / 2] & 0xF);
            rotated_recon[i] = CENTROIDS_4BIT[idx];
        }

        /* Inverse Givens rotation */
        float * dst = y + block * d;
        for (int p = 0; p < 64; p++) {
            float q0 = rotated_recon[2*p];
            float q1 = rotated_recon[2*p + 1];
            dst[2*p]     =  RQ_COS[p] * q0 + RQ_SIN[p] * q1;
            dst[2*p + 1] = -RQ_SIN[p] * q0 + RQ_COS[p] * q1;
        }

        /* Scale by norm */
        for (int i = 0; i < d; i++) {
            dst[i] *= norm;
        }
    }
}

size_t quantize_rq4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                       int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_RQ4 == 0);

    size_t row_size = (n_per_row / QK_RQ4) * sizeof(block_rq4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_rq4_0_ref(
            src + row * n_per_row,
            (block_rq4_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ========================================================================
 * IsoQuant (Quaternion 4D rotation) -- rq3_iso, rq4_iso
 *
 * Forward:  q_L * v  (quaternion multiply)
 * Inverse:  conj(q_L) * v  (conjugate: negate x,y,z)
 *
 * 32 quaternion groups of 4 elements each for d=128.
 * Each block = one 128-element rotation group.
 * ======================================================================== */

/* ---------- RQ3_ISO: IsoQuant 3-bit ---------- */

void quantize_row_rq3_iso_ref(const float * GGML_RESTRICT x, block_rq3_iso * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_RQ3_ISO == 0);
    const int nb = k / QK_RQ3_ISO;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * QK_RQ3_ISO;

        /* Compute norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < QK_RQ3_ISO; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[RQ_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < QK_RQ3_ISO; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, sizeof(normalized));
        }

        /* Forward quaternion rotation (32 groups of 4) */
        float rotated[RQ_D];
        for (int g = 0; g < 32; g++) {
            float v0 = normalized[4*g];
            float v1 = normalized[4*g + 1];
            float v2 = normalized[4*g + 2];
            float v3 = normalized[4*g + 3];
            quat_mul(RQ_ISO_QW[g], RQ_ISO_QX[g], RQ_ISO_QY[g], RQ_ISO_QZ[g],
                     v0, v1, v2, v3,
                     &rotated[4*g], &rotated[4*g+1], &rotated[4*g+2], &rotated[4*g+3]);
        }

        /* Quantize to 3-bit centroids */
        uint8_t indices[RQ_D];
        for (int i = 0; i < QK_RQ3_ISO; i++) {
            indices[i] = (uint8_t)nearest_centroid_3bit(rotated[i]);
        }

        /* Norm correction */
        float recon_sq = 0.0f;
        for (int i = 0; i < QK_RQ3_ISO; i++) {
            float r = CENTROIDS_3BIT[indices[i]];
            recon_sq += r * r;
        }
        float recon_norm = sqrtf(recon_sq);
        y[block].norm = GGML_FP32_TO_FP16((recon_norm > 1e-10f) ? norm / recon_norm : norm);

        /* Pack 3-bit indices into qs[49] bitstream */
        memset(y[block].qs, 0, 49);
        y[block].pad = 0;
        for (int i = 0; i < QK_RQ3_ISO; i++) {
            uint32_t val = indices[i] & 0x7;
            int bit_pos = i * 3;
            int byte_pos = bit_pos / 8;
            int bit_off  = bit_pos % 8;
            y[block].qs[byte_pos] |= (uint8_t)(val << bit_off);
            if (bit_off > 5) {
                /* Bits spill into next byte */
                y[block].qs[byte_pos + 1] |= (uint8_t)(val >> (8 - bit_off));
            }
        }
    }
}

void dequantize_row_rq3_iso(const block_rq3_iso * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_RQ3_ISO == 0);
    const int nb = k / QK_RQ3_ISO;

    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);

        /* Unpack 3-bit indices from bitstream */
        float rotated_recon[RQ_D];
        for (int i = 0; i < QK_RQ3_ISO; i++) {
            int bit_pos = i * 3;
            int byte_pos = bit_pos / 8;
            int bit_off  = bit_pos % 8;
            uint16_t raw = (uint16_t)x[block].qs[byte_pos];
            if (byte_pos + 1 < 49) {
                raw |= (uint16_t)x[block].qs[byte_pos + 1] << 8;
            }
            uint8_t idx = (raw >> bit_off) & 0x7;
            rotated_recon[i] = CENTROIDS_3BIT[idx];
        }

        /* Inverse quaternion rotation: conj(q) * v */
        float * dst = y + block * QK_RQ3_ISO;
        for (int g = 0; g < 32; g++) {
            float q0 = rotated_recon[4*g];
            float q1 = rotated_recon[4*g + 1];
            float q2 = rotated_recon[4*g + 2];
            float q3 = rotated_recon[4*g + 3];
            quat_mul(RQ_ISO_QW[g], -RQ_ISO_QX[g], -RQ_ISO_QY[g], -RQ_ISO_QZ[g],
                     q0, q1, q2, q3,
                     &dst[4*g], &dst[4*g+1], &dst[4*g+2], &dst[4*g+3]);
        }

        /* Scale by norm */
        for (int i = 0; i < QK_RQ3_ISO; i++) {
            dst[i] *= norm;
        }
    }
}

size_t quantize_rq3_iso(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_RQ3_ISO == 0);

    size_t row_size = (n_per_row / QK_RQ3_ISO) * sizeof(block_rq3_iso);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_rq3_iso_ref(
            src + row * n_per_row,
            (block_rq3_iso *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- RQ4_ISO: IsoQuant 2-bit ---------- */
/* Note: despite the "rq4" name, the block struct qs[33] holds 264 bits   */
/* for 128 values = 2 bits/value, so we use 2-bit centroids here.         */

void quantize_row_rq4_iso_ref(const float * GGML_RESTRICT x, block_rq4_iso * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_RQ4_ISO == 0);
    const int nb = k / QK_RQ4_ISO;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * QK_RQ4_ISO;

        /* Compute norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < QK_RQ4_ISO; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[RQ_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < QK_RQ4_ISO; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, sizeof(normalized));
        }

        /* Forward quaternion rotation (32 groups of 4) */
        float rotated[RQ_D];
        for (int g = 0; g < 32; g++) {
            float v0 = normalized[4*g];
            float v1 = normalized[4*g + 1];
            float v2 = normalized[4*g + 2];
            float v3 = normalized[4*g + 3];
            quat_mul(RQ_ISO_QW[g], RQ_ISO_QX[g], RQ_ISO_QY[g], RQ_ISO_QZ[g],
                     v0, v1, v2, v3,
                     &rotated[4*g], &rotated[4*g+1], &rotated[4*g+2], &rotated[4*g+3]);
        }

        /* Quantize to 2-bit centroids */
        uint8_t indices[RQ_D];
        for (int i = 0; i < QK_RQ4_ISO; i++) {
            indices[i] = (uint8_t)nearest_centroid_2bit(rotated[i]);
        }

        /* Norm correction */
        float recon_sq = 0.0f;
        for (int i = 0; i < QK_RQ4_ISO; i++) {
            float r = CENTROIDS_2BIT[indices[i]];
            recon_sq += r * r;
        }
        float recon_norm = sqrtf(recon_sq);
        y[block].norm = GGML_FP32_TO_FP16((recon_norm > 1e-10f) ? norm / recon_norm : norm);

        /* Pack 2-bit indices into qs[33] bitstream */
        memset(y[block].qs, 0, 33);
        y[block].pad = 0;
        for (int i = 0; i < QK_RQ4_ISO; i++) {
            uint8_t val = indices[i] & 0x3;
            int bit_pos = i * 2;
            int byte_pos = bit_pos / 8;
            int bit_off  = bit_pos % 8;
            y[block].qs[byte_pos] |= (uint8_t)(val << bit_off);
        }
    }
}

void dequantize_row_rq4_iso(const block_rq4_iso * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_RQ4_ISO == 0);
    const int nb = k / QK_RQ4_ISO;

    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);

        /* Unpack 2-bit indices */
        float rotated_recon[RQ_D];
        for (int i = 0; i < QK_RQ4_ISO; i++) {
            int bit_pos = i * 2;
            int byte_pos = bit_pos / 8;
            int bit_off  = bit_pos % 8;
            uint8_t idx = (x[block].qs[byte_pos] >> bit_off) & 0x3;
            rotated_recon[i] = CENTROIDS_2BIT[idx];
        }

        /* Inverse quaternion rotation: conj(q) * v */
        float * dst = y + block * QK_RQ4_ISO;
        for (int g = 0; g < 32; g++) {
            float q0 = rotated_recon[4*g];
            float q1 = rotated_recon[4*g + 1];
            float q2 = rotated_recon[4*g + 2];
            float q3 = rotated_recon[4*g + 3];
            quat_mul(RQ_ISO_QW[g], -RQ_ISO_QX[g], -RQ_ISO_QY[g], -RQ_ISO_QZ[g],
                     q0, q1, q2, q3,
                     &dst[4*g], &dst[4*g+1], &dst[4*g+2], &dst[4*g+3]);
        }

        /* Scale by norm */
        for (int i = 0; i < QK_RQ4_ISO; i++) {
            dst[i] *= norm;
        }
    }
}

size_t quantize_rq4_iso(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_RQ4_ISO == 0);

    size_t row_size = (n_per_row / QK_RQ4_ISO) * sizeof(block_rq4_iso);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_rq4_iso_ref(
            src + row * n_per_row,
            (block_rq4_iso *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}
