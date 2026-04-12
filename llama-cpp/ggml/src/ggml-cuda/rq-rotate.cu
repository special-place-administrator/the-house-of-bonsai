#include "rq-rotate.cuh"
#include "rq-constants.cuh"

// Quaternion multiply helper for IsoQuant (device-side)
static __device__ __forceinline__ void rq_quat_mul(
        float aw, float ax, float ay, float az,
        float bw, float bx, float by, float bz,
        float *rw, float *rx, float *ry, float *rz) {
    *rw = aw*bw - ax*bx - ay*by - az*bz;
    *rx = aw*bx + ax*bw + ay*bz - az*by;
    *ry = aw*by - ax*bz + ay*bw + az*bx;
    *rz = aw*bz + ax*by - ay*bx + az*bw;
}

// PlanarQuant: Givens 2D rotation kernel (64 pairs for d=128)
// direction=0: forward (encode), direction=1: inverse (decode)
static __global__ void k_rq_rotate(
        const float * __restrict__ src, float * __restrict__ dst,
        const int64_t n_elements, const int direction) {

    const int64_t group = blockIdx.x;
    const int64_t offset = group * 128;
    if (offset >= n_elements) return;

    // Each thread handles one pair (p = threadIdx.x, 0..63)
    const int p = threadIdx.x;
    if (p >= 64) return;

    const float c = RQ_COS[p];
    const float s = RQ_SIN[p];
    const float v0 = src[offset + 2*p];
    const float v1 = src[offset + 2*p + 1];

    if (direction == 0) {
        // Forward: r0 = cos*v0 - sin*v1, r1 = sin*v0 + cos*v1
        dst[offset + 2*p]     = c * v0 - s * v1;
        dst[offset + 2*p + 1] = s * v0 + c * v1;
    } else {
        // Inverse: r0 = cos*v0 + sin*v1, r1 = -sin*v0 + cos*v1
        dst[offset + 2*p]     =  c * v0 + s * v1;
        dst[offset + 2*p + 1] = -s * v0 + c * v1;
    }
}

void ggml_cuda_op_rq_rotate(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const float * src_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    int direction;
    memcpy(&direction, dst->op_params, sizeof(int));

    const int64_t n_elements = ggml_nelements(src0);
    const int64_t n_groups = n_elements / 128;

    // 64 threads per block (one per Givens pair), one block per 128-element group
    k_rq_rotate<<<(int)n_groups, 64, 0, stream>>>(src_d, dst_d, n_elements, direction);
}
