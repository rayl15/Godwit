#include <metal_stdlib>
using namespace metal;

// Fused dequantise-and-multiply, the operation an MoE expert actually performs.
//
// Benchmarking dequantisation on its own is misleading: writing FP16 output is
// 2 bytes per weight against roughly 0.53 bytes read, so a standalone kernel is
// dominated by stores that a fused kernel never performs. These kernels keep
// dequantised weights in registers and consume them immediately.
//
// The two formats are given deliberately identical structure -- one SIMD group
// per output row, lanes striding over blocks, one simd_sum at the end -- so the
// only difference measured is the unpack itself:
//
//   MXFP4        32-weight blocks, E8M0 shared exponent, codebook lookup
//   affine int4  64-weight groups, BF16 scale and bias, fused multiply-add
//
// MXFP4 has half the block size (twice the scale loads per weight) and needs an
// indexed table read where affine needs an FMA. Whether that costs anything
// real is the question.

constant constexpr uint kSimdWidth = 32;

inline float mxfp4_code_to_float(uint code) {
    const float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float value = magnitude[code & 0x7u];
    return (code & 0x8u) ? -value : value;
}

inline float mxfp4_decode_scale(uchar raw) {
    return exp2(float(int(raw) - 127));
}

// y[row] = sum_c dequant(W[row][c]) * x[c]
//
// One threadgroup of 32 threads per row. `cols` must be a multiple of 32.
kernel void mxfp4_gemv(
    device const uchar *packed  [[buffer(0)]],   // rows * cols/2 bytes
    device const uchar *scales  [[buffer(1)]],   // rows * cols/32 bytes
    device const half  *x       [[buffer(2)]],   // cols
    device float       *y       [[buffer(3)]],   // rows
    constant uint      &cols    [[buffer(4)]],
    uint                row     [[threadgroup_position_in_grid]],
    uint                lane    [[thread_index_in_threadgroup]])
{
    const uint blocksPerRow = cols / 32u;
    const uint packedRowBase = row * (cols / 2u);
    const uint scaleRowBase  = row * blocksPerRow;

    float acc = 0.0f;
    for (uint b = lane; b < blocksPerRow; b += kSimdWidth) {
        const float scale = mxfp4_decode_scale(scales[scaleRowBase + b]);
        const uint base = packedRowBase + b * 16u;
        const uint col0 = b * 32u;

        float partial = 0.0f;
        for (uint i = 0; i < 16u; ++i) {
            const uchar byte = packed[base + i];
            partial += mxfp4_code_to_float(byte & 0x0Fu) * float(x[col0 + i * 2u]);
            partial += mxfp4_code_to_float(byte >> 4)    * float(x[col0 + i * 2u + 1u]);
        }
        acc += partial * scale;   // scale is uniform across the block
    }

    const float total = simd_sum(acc);
    if (lane == 0) { y[row] = total; }
}

// Same shape, MLX-style affine int4: 64-weight groups, BF16 scale and bias.
// `cols` must be a multiple of 64.
kernel void affine_int4_gemv(
    device const uchar  *packed [[buffer(0)]],   // rows * cols/2 bytes
    device const bfloat *scales [[buffer(1)]],   // rows * cols/64
    device const bfloat *biases [[buffer(2)]],   // rows * cols/64
    device const half   *x      [[buffer(3)]],   // cols
    device float        *y      [[buffer(4)]],   // rows
    constant uint       &cols   [[buffer(5)]],
    uint                 row    [[threadgroup_position_in_grid]],
    uint                 lane   [[thread_index_in_threadgroup]])
{
    const uint groupsPerRow = cols / 64u;
    const uint packedRowBase = row * (cols / 2u);
    const uint metaRowBase = row * groupsPerRow;

    float acc = 0.0f;
    for (uint g = lane; g < groupsPerRow; g += kSimdWidth) {
        const float scale = float(scales[metaRowBase + g]);
        const float bias  = float(biases[metaRowBase + g]);
        const uint base = packedRowBase + g * 32u;
        const uint col0 = g * 64u;

        // sum((q * scale + bias) * x) == scale * sum(q * x) + bias * sum(x),
        // so the affine transform leaves the inner loop entirely.
        float dot = 0.0f;
        float xsum = 0.0f;
        for (uint i = 0; i < 32u; ++i) {
            const uchar byte = packed[base + i];
            const float x0 = float(x[col0 + i * 2u]);
            const float x1 = float(x[col0 + i * 2u + 1u]);
            dot  += float(byte & 0x0Fu) * x0 + float(byte >> 4) * x1;
            xsum += x0 + x1;
        }
        acc += scale * dot + bias * xsum;
    }

    const float total = simd_sum(acc);
    if (lane == 0) { y[row] = total; }
}
