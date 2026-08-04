#include <metal_stdlib>
using namespace metal;

// MXFP4 (OCP microscaling) dequantisation.
//
// Layout: 32 weights per block. Codes are 4-bit E2M1 packed two per byte, low
// nibble first, in one stream; block scales are E8M0 bytes in a parallel
// stream.
//
// The open question this kernel exists to answer: MXFP4's block is 32 weights
// against MLX affine-int4's 64, so there are twice as many scale loads per
// weight. If that per-block overhead dominates, streamed MXFP4 inference is not
// viable and the target model family changes. Benchmark against an affine-int4
// kernel of identical shape before building anything on top of this.

constant constexpr uint kBlockSize = 32;
constant constexpr uint kPackedBytesPerBlock = kBlockSize / 2;

// E2M1: 1 sign, 2 exponent (bias 1), 1 mantissa. No infinities, so the eight
// representable magnitudes fit in an immediate table.
inline float mxfp4_code_to_float(uint code) {
    const float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float value = magnitude[code & 0x7u];
    return (code & 0x8u) ? -value : value;
}

// E8M0: raw biased power of two. 0xFF is reserved for NaN; there is no zero
// encoding, so byte 0 means 2^-127 rather than 0.
inline float mxfp4_decode_scale(uchar raw) {
    return (raw == 0xFF) ? NAN : exp2(float(int(raw) - 127));
}

// One thread per block. Deliberately the simplest possible mapping: this is a
// correctness reference for the CPU decoder and a baseline to optimise against,
// not a production kernel.
kernel void mxfp4_dequant_reference(
    device const uchar *packed   [[buffer(0)]],
    device const uchar *scales   [[buffer(1)]],
    device half        *out      [[buffer(2)]],
    constant uint      &blocks   [[buffer(3)]],
    uint                gid      [[thread_position_in_grid]])
{
    if (gid >= blocks) { return; }

    const float scale = mxfp4_decode_scale(scales[gid]);
    device const uchar *src = packed + gid * kPackedBytesPerBlock;
    device half *dst = out + gid * kBlockSize;

    for (uint i = 0; i < kPackedBytesPerBlock; ++i) {
        const uchar byte = src[i];
        dst[i * 2 + 0] = half(mxfp4_code_to_float(byte & 0x0Fu) * scale);
        dst[i * 2 + 1] = half(mxfp4_code_to_float(byte >> 4) * scale);
    }
}

// One SIMD lane per byte-pair, 16 lanes per block. Loads the scale once per
// block via broadcast rather than once per lane, which is the specific overhead
// we are trying to measure.
kernel void mxfp4_dequant_wide(
    device const uchar *packed   [[buffer(0)]],
    device const uchar *scales   [[buffer(1)]],
    device half        *out      [[buffer(2)]],
    constant uint      &blocks   [[buffer(3)]],
    uint                gid      [[thread_position_in_grid]])
{
    const uint lanesPerBlock = kPackedBytesPerBlock;
    const uint block = gid / lanesPerBlock;
    const uint lane  = gid % lanesPerBlock;
    if (block >= blocks) { return; }

    const float scale = mxfp4_decode_scale(scales[block]);
    const uchar byte = packed[block * kPackedBytesPerBlock + lane];

    device half *dst = out + block * kBlockSize + lane * 2;
    dst[0] = half(mxfp4_code_to_float(byte & 0x0Fu) * scale);
    dst[1] = half(mxfp4_code_to_float(byte >> 4) * scale);
}
