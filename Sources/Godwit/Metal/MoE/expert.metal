#include <metal_stdlib>
using namespace metal;

// One GPT-OSS expert: gate_up projection, gated activation, down projection.
//
// Three details here are easy to get wrong and produce plausible-looking
// garbage rather than an error, so they are spelled out:
//
//   1. gate and up are INTERLEAVED in the gate_up output, not stored as two
//      halves. gate = out[0::2], up = out[1::2].
//   2. The activation is not SiLU despite `hidden_act: "silu"` in config.json.
//      It is (up + 1) * gate * sigmoid(alpha * gate) with alpha = 1.702 -- a
//      GELU-style sigmoid gate, with a +1 shift on the up branch.
//   3. Clamping is asymmetric: gate is clamped above only, up on both sides.
//
// Source of truth: transformers' GptOssExperts.forward.

constant constexpr uint kSimdWidth = 32;
constant constexpr float kSwigluAlpha = 1.702f;

inline float mxfp4_code_to_float(uint code) {
    const float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float value = magnitude[code & 0x7u];
    return (code & 0x8u) ? -value : value;
}

inline float mxfp4_decode_scale(uchar raw) {
    return exp2(float(int(raw) - 127));
}

// y[row] = sum_c dequant(W[row][c]) * x[c] + bias[row]
//
// One threadgroup of 32 threads per row; `cols` must be a multiple of 32.
kernel void mxfp4_gemv_bias(
    device const uchar  *packed [[buffer(0)]],
    device const uchar  *scales [[buffer(1)]],
    device const bfloat *bias   [[buffer(2)]],
    device const half   *x      [[buffer(3)]],
    device float        *y      [[buffer(4)]],
    constant uint       &cols   [[buffer(5)]],
    uint                 row    [[threadgroup_position_in_grid]],
    uint                 lane   [[thread_index_in_threadgroup]])
{
    const uint blocksPerRow = cols / 32u;
    const uint packedRowBase = row * (cols / 2u);
    const uint scaleRowBase = row * blocksPerRow;

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
        acc += partial * scale;
    }

    const float total = simd_sum(acc);
    if (lane == 0) { y[row] = total + float(bias[row]); }
}

// Consumes the 2F-wide gate_up result and produces the F-wide activation.
//
// One thread per output element.
kernel void gptoss_expert_activation(
    device const float *gate_up [[buffer(0)]],   // 2 * F, interleaved
    device half        *out     [[buffer(1)]],   // F
    constant uint      &width   [[buffer(2)]],   // F
    constant float     &limit   [[buffer(3)]],   // swiglu_limit, 7.0
    uint                index   [[thread_position_in_grid]])
{
    if (index >= width) { return; }

    // Interleaved, not split halves.
    float gate = gate_up[index * 2u];
    float up   = gate_up[index * 2u + 1u];

    // Asymmetric on purpose: gate has no lower bound.
    gate = min(gate, limit);
    up = clamp(up, -limit, limit);

    const float glu = gate * (1.0f / (1.0f + exp(-kSwigluAlpha * gate)));
    out[index] = half((up + 1.0f) * glu);
}

// Accumulates one expert's output into the layer result, scaled by its routing
// weight. Separate from the GEMV so several experts can accumulate into the
// same destination without a second pass.
kernel void expert_accumulate(
    device const float *expert_out [[buffer(0)]],
    device float       *destination [[buffer(1)]],
    constant float     &weight     [[buffer(2)]],
    constant uint      &width      [[buffer(3)]],
    uint                index      [[thread_position_in_grid]])
{
    if (index >= width) { return; }
    destination[index] += expert_out[index] * weight;
}
