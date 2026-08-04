#include <metal_stdlib>
using namespace metal;

// Router and normalisation: the small resident kernels that run before we know
// which experts to fetch.
//
// The router is on the critical path in a way its size does not suggest. Its
// output decides which ~12.6 MiB blobs get read from disk, so nothing can be
// prefetched until it has finished. That is also why its weights stay BF16
// while everything comparable is quantised.

constant constexpr uint kSimdWidth = 32;

// y[row] = sum_c W[row][c] * x[c] + bias[row], BF16 weights.
//
// One SIMD group per row. With 128 rows this is a small dispatch; it exists to
// keep the value on the GPU next to everything else rather than for throughput.
kernel void bf16_gemv_bias(
    device const bfloat *weight [[buffer(0)]],
    device const bfloat *bias   [[buffer(1)]],
    device const half   *x      [[buffer(2)]],
    device float        *y      [[buffer(3)]],
    constant uint       &cols   [[buffer(4)]],
    uint                 row    [[threadgroup_position_in_grid]],
    uint                 lane   [[thread_index_in_threadgroup]])
{
    device const bfloat *w = weight + row * cols;
    float acc = 0.0f;
    for (uint c = lane; c < cols; c += kSimdWidth) {
        acc += float(w[c]) * float(x[c]);
    }
    const float total = simd_sum(acc);
    if (lane == 0) { y[row] = total + float(bias[row]); }
}

// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight.
//
// Plain multiplicative weight, not Gemma's (1 + weight). Accumulation is in
// FP32 regardless of input precision, matching the reference, which upcasts
// before taking the variance.
kernel void rmsnorm(
    device const half   *x       [[buffer(0)]],
    device const bfloat *weight  [[buffer(1)]],
    device half         *out     [[buffer(2)]],
    constant uint       &width   [[buffer(3)]],
    constant float      &epsilon [[buffer(4)]],
    uint                 lane    [[thread_index_in_threadgroup]])
{
    float sum = 0.0f;
    for (uint i = lane; i < width; i += kSimdWidth) {
        const float value = float(x[i]);
        sum += value * value;
    }
    const float variance = simd_sum(sum) / float(width);
    const float inverse = rsqrt(variance + epsilon);

    for (uint i = lane; i < width; i += kSimdWidth) {
        out[i] = half(float(x[i]) * inverse * float(weight[i]));
    }
}

// Dequantises an affine int8 row group and multiplies: the trunk's counterpart
// to mxfp4_gemv. Codes are [rows, cols]; meta holds a BF16 scale and zero per
// group of 64, laid out as pairs.
kernel void int8_affine_gemv(
    device const uchar  *codes [[buffer(0)]],
    device const bfloat *meta  [[buffer(1)]],
    device const half   *x     [[buffer(2)]],
    device float        *y     [[buffer(3)]],
    constant uint       &cols  [[buffer(4)]],
    uint                 row   [[threadgroup_position_in_grid]],
    uint                 lane  [[thread_index_in_threadgroup]])
{
    const uint groups = cols / 64u;
    device const uchar *w = codes + row * cols;
    device const bfloat *m = meta + row * groups * 2u;

    float acc = 0.0f;
    for (uint g = lane; g < groups; g += kSimdWidth) {
        const float scale = float(m[g * 2u]);
        const float zero = float(m[g * 2u + 1u]);
        const uint base = g * 64u;

        // sum((q*scale + zero) * x) == scale * sum(q*x) + zero * sum(x)
        float dot = 0.0f;
        float xsum = 0.0f;
        for (uint i = 0; i < 64u; ++i) {
            const float xi = float(x[base + i]);
            dot += float(w[base + i]) * xi;
            xsum += xi;
        }
        acc += scale * dot + zero * xsum;
    }

    const float total = simd_sum(acc);
    if (lane == 0) { y[row] = total; }
}


// int8 affine projection for a run of tokens, with bias, producing half.
//
// The earlier version handled one token and had no bias, so the caller looped
// over tokens submitting a command buffer each and then added the bias on the
// CPU. Folding both in lets a whole attention block be one submission — which
// matters because submissions, not arithmetic, dominate decode.
kernel void int8_affine_gemv_bias_batched(
    device const uchar  *codes [[buffer(0)]],
    device const bfloat *meta  [[buffer(1)]],
    device const bfloat *bias  [[buffer(2)]],
    device const half   *x     [[buffer(3)]],   // [T, cols]
    device half         *y     [[buffer(4)]],   // [T, rows]
    constant uint       &cols  [[buffer(5)]],
    constant uint       &rows  [[buffer(6)]],
    uint2                group [[threadgroup_position_in_grid]],
    uint                 lane  [[thread_index_in_threadgroup]])
{
    const uint row = group.x;
    const uint token = group.y;
    const uint groups = cols / 64u;
    device const uchar *w = codes + row * cols;
    device const bfloat *m = meta + row * groups * 2u;
    device const half *xt = x + token * cols;

    float acc = 0.0f;
    for (uint g = lane; g < groups; g += kSimdWidth) {
        const float scale = float(m[g * 2u]);
        const float zero = float(m[g * 2u + 1u]);
        const uint base = g * 64u;
        float dot = 0.0f;
        float xsum = 0.0f;
        for (uint i = 0; i < 64u; ++i) {
            const float xi = float(xt[base + i]);
            dot += float(w[base + i]) * xi;
            xsum += xi;
        }
        acc += scale * dot + zero * xsum;
    }

    const float total = simd_sum(acc);
    if (lane == 0) { y[token * rows + row] = half(total + float(bias[row])); }
}
