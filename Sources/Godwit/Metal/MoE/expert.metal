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
//      It is (up + 1) * gate * sigmoid(alpha * gate) -- a GELU-style sigmoid
//      gate, with a +1 shift on the up branch. Alpha and the clamp arrive as
//      parameters rather than literals, since they are model properties.
//   3. Clamping is asymmetric: gate is clamped above only, up on both sides.
//
// Source of truth: transformers' GptOssExperts.forward.

constant constexpr uint kSimdWidth = 32;

// The E2M1 codebook in the constant address space, indexed by the full 4-bit
// code including its sign.
//
// Declaring this inside the function, as the original did, makes it a
// thread-local array; a dynamic index into one of those can spill to stack
// memory instead of staying in registers, and this is the innermost operation
// in the whole runtime — two lookups per packed byte, billions per token.
constant float kFP4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

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
    constant float     &alpha   [[buffer(4)]],   // sigmoid steepness, 1.702
    uint                index   [[thread_position_in_grid]])
{
    if (index >= width) { return; }

    // Interleaved, not split halves.
    float gate = gate_up[index * 2u];
    float up   = gate_up[index * 2u + 1u];

    // Asymmetric on purpose: gate has no lower bound.
    gate = min(gate, limit);
    up = clamp(up, -limit, limit);

    const float glu = gate * (1.0f / (1.0f + exp(-alpha * gate)));
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

// Persistent-workgroup gate_up/down projection.
//
// The naive kernel above launches one 32-thread threadgroup per output row —
// 5,760 of them for gate_up. Measured at 13-33 G weights/s while using roughly
// a tenth of the M4's memory bandwidth, which says it is bound by launch
// overhead and occupancy rather than by arithmetic or by memory.
//
// Three changes:
//
//   1. 256-thread threadgroups holding 8 SIMD groups, each SIMD group taking a
//      row. That cuts the threadgroup count eightfold and gives each one real
//      work to do.
//   2. A grid-stride loop, so a fixed pool of threadgroups drains the rows
//      instead of one being launched per row. No atomics needed — the stride
//      partitions the work with no contention.
//   3. 16-byte vector loads. An MXFP4 block is exactly 16 packed bytes, and
//      because cols/2 is a multiple of 16 for these shapes, every block starts
//      16-byte aligned.
constant constexpr uint kSimdsPerGroup = 8;

kernel void mxfp4_gemv_bias_persistent(
    device const uchar  *packed [[buffer(0)]],
    device const uchar  *scales [[buffer(1)]],
    device const bfloat *bias   [[buffer(2)]],
    device const half   *x      [[buffer(3)]],
    device float        *y      [[buffer(4)]],
    constant uint       &cols   [[buffer(5)]],
    constant uint       &rows   [[buffer(6)]],
    uint                 tgid   [[threadgroup_position_in_grid]],
    uint                 tgSize [[threadgroups_per_grid]],
    uint                 tid    [[thread_index_in_threadgroup]])
{
    const uint simd = tid / 32u;
    const uint lane = tid % 32u;
    const uint blocksPerRow = cols / 32u;
    const uint rowStride = tgSize * kSimdsPerGroup;

    for (uint row = tgid * kSimdsPerGroup + simd; row < rows; row += rowStride) {
        const uint packedRowBase = row * (cols / 2u);
        const uint scaleRowBase = row * blocksPerRow;

        float acc = 0.0f;
        for (uint b = lane; b < blocksPerRow; b += 32u) {
            const float scale = mxfp4_decode_scale(scales[scaleRowBase + b]);
            // One block in one load.
            const uint4 chunk = *reinterpret_cast<device const uint4 *>(
                packed + packedRowBase + b * 16u);
            const uint col0 = b * 32u;

            float partial = 0.0f;
            for (uint word = 0; word < 4u; ++word) {
                const uint bits = chunk[word];
                for (uint byte = 0; byte < 4u; ++byte) {
                    const uint value = (bits >> (byte * 8u)) & 0xFFu;
                    const uint index = col0 + (word * 4u + byte) * 2u;
                    partial += mxfp4_code_to_float(value & 0x0Fu) * float(x[index]);
                    partial += mxfp4_code_to_float(value >> 4)    * float(x[index + 1u]);
                }
            }
            acc += partial * scale;
        }

        const float total = simd_sum(acc);
        if (lane == 0) { y[row] = total + float(bias[row]); }
    }
}

// As above, but the activation vector is staged in threadgroup memory first.
//
// Every lane of every SIMD group reads the whole of x while walking its rows,
// so with 8 SIMD groups and a grid-stride loop the same 5,760 bytes are pulled
// from device memory hundreds of times. Staging it once per threadgroup turns
// that into one cooperative load followed by hits in fast memory.
//
// This should matter most for the narrow projection: down is 2,880 rows against
// gate_up's 5,760, so it has half the work to hide its memory traffic behind.
constant constexpr uint kMaxStagedCols = 4096;

kernel void mxfp4_gemv_bias_staged(
    device const uchar  *packed [[buffer(0)]],
    device const uchar  *scales [[buffer(1)]],
    device const bfloat *bias   [[buffer(2)]],
    device const half   *x      [[buffer(3)]],
    device float        *y      [[buffer(4)]],
    constant uint       &cols   [[buffer(5)]],
    constant uint       &rows   [[buffer(6)]],
    threadgroup half    *shared [[threadgroup(0)]],
    uint                 tgid   [[threadgroup_position_in_grid]],
    uint                 tgSize [[threadgroups_per_grid]],
    uint                 tid    [[thread_index_in_threadgroup]],
    uint                 tgDim  [[threads_per_threadgroup]])
{
    for (uint i = tid; i < cols; i += tgDim) {
        shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint simd = tid / 32u;
    const uint lane = tid % 32u;
    const uint blocksPerRow = cols / 32u;
    const uint rowStride = tgSize * kSimdsPerGroup;

    for (uint row = tgid * kSimdsPerGroup + simd; row < rows; row += rowStride) {
        const uint packedRowBase = row * (cols / 2u);
        const uint scaleRowBase = row * blocksPerRow;

        float acc = 0.0f;
        for (uint b = lane; b < blocksPerRow; b += 32u) {
            const float scale = mxfp4_decode_scale(scales[scaleRowBase + b]);
            const uint4 chunk = *reinterpret_cast<device const uint4 *>(
                packed + packedRowBase + b * 16u);
            const uint col0 = b * 32u;

            float partial = 0.0f;
            for (uint word = 0; word < 4u; ++word) {
                const uint bits = chunk[word];
                for (uint byte = 0; byte < 4u; ++byte) {
                    const uint value = (bits >> (byte * 8u)) & 0xFFu;
                    const uint index = col0 + (word * 4u + byte) * 2u;
                    partial += mxfp4_code_to_float(value & 0x0Fu) * float(shared[index]);
                    partial += mxfp4_code_to_float(value >> 4)    * float(shared[index + 1u]);
                }
            }
            acc += partial * scale;
        }

        const float total = simd_sum(acc);
        if (lane == 0) { y[row] = total + float(bias[row]); }
    }
}


// Identical to mxfp4_gemv_bias_persistent except that it indexes the constant
// address space codebook. Kept as a separate kernel so the A/B measures the
// lookup and nothing else.
kernel void mxfp4_gemv_bias_lut(
    device const uchar  *packed [[buffer(0)]],
    device const uchar  *scales [[buffer(1)]],
    device const bfloat *bias   [[buffer(2)]],
    device const half   *x      [[buffer(3)]],
    device float        *y      [[buffer(4)]],
    constant uint       &cols   [[buffer(5)]],
    constant uint       &rows   [[buffer(6)]],
    uint                 tgid   [[threadgroup_position_in_grid]],
    uint                 tgSize [[threadgroups_per_grid]],
    uint                 tid    [[thread_index_in_threadgroup]])
{
    const uint simd = tid / 32u;
    const uint lane = tid % 32u;
    const uint blocksPerRow = cols / 32u;
    const uint rowStride = tgSize * kSimdsPerGroup;

    for (uint row = tgid * kSimdsPerGroup + simd; row < rows; row += rowStride) {
        const uint packedRowBase = row * (cols / 2u);
        const uint scaleRowBase = row * blocksPerRow;

        float acc = 0.0f;
        for (uint b = lane; b < blocksPerRow; b += 32u) {
            const float scale = mxfp4_decode_scale(scales[scaleRowBase + b]);
            const uint4 chunk = *reinterpret_cast<device const uint4 *>(
                packed + packedRowBase + b * 16u);
            const uint col0 = b * 32u;

            float partial = 0.0f;
            for (uint word = 0; word < 4u; ++word) {
                const uint bits = chunk[word];
                for (uint byte = 0; byte < 4u; ++byte) {
                    const uint value = (bits >> (byte * 8u)) & 0xFFu;
                    const uint index = col0 + (word * 4u + byte) * 2u;
                    partial += kFP4[value & 0x0Fu] * float(x[index]);
                    partial += kFP4[value >> 4]    * float(x[index + 1u]);
                }
            }
            acc += partial * scale;
        }

        const float total = simd_sum(acc);
        if (lane == 0) { y[row] = total + float(bias[row]); }
    }
}

// Four rows per SIMD group.
//
// The previous kernels are neither bandwidth-bound (19% of the M4's) nor
// ALU-bound (2%) — they are latency-bound. Each lane walks three blocks for its
// row, one dependent load at a time, with nothing to overlap the memory latency
// against.
//
// Processing four rows together gives four independent load streams, and the x
// values loaded for one row serve all four, so the ratio of arithmetic to
// loads improves at the same time.
constant constexpr uint kRowsPerSimd = 4;

kernel void mxfp4_gemv_bias_multirow(
    device const uchar  *packed [[buffer(0)]],
    device const uchar  *scales [[buffer(1)]],
    device const bfloat *bias   [[buffer(2)]],
    device const half   *x      [[buffer(3)]],
    device float        *y      [[buffer(4)]],
    constant uint       &cols   [[buffer(5)]],
    constant uint       &rows   [[buffer(6)]],
    uint                 tgid   [[threadgroup_position_in_grid]],
    uint                 tgSize [[threadgroups_per_grid]],
    uint                 tid    [[thread_index_in_threadgroup]])
{
    const uint simd = tid / 32u;
    const uint lane = tid % 32u;
    const uint blocksPerRow = cols / 32u;
    const uint packedStride = cols / 2u;
    const uint groupStride = tgSize * kSimdsPerGroup * kRowsPerSimd;

    for (uint base = (tgid * kSimdsPerGroup + simd) * kRowsPerSimd;
         base < rows; base += groupStride) {

        float acc[kRowsPerSimd];
        for (uint r = 0; r < kRowsPerSimd; ++r) { acc[r] = 0.0f; }
        const uint active = min(kRowsPerSimd, rows - base);

        for (uint b = lane; b < blocksPerRow; b += 32u) {
            const uint col0 = b * 32u;

            // One load of x serves every row in the group.
            float xv[32];
            for (uint i = 0; i < 32u; ++i) { xv[i] = float(x[col0 + i]); }

            for (uint r = 0; r < active; ++r) {
                const uint row = base + r;
                const float scale = mxfp4_decode_scale(scales[row * blocksPerRow + b]);
                const uint4 chunk = *reinterpret_cast<device const uint4 *>(
                    packed + row * packedStride + b * 16u);

                float partial = 0.0f;
                for (uint word = 0; word < 4u; ++word) {
                    const uint bits = chunk[word];
                    for (uint byte = 0; byte < 4u; ++byte) {
                        const uint value = (bits >> (byte * 8u)) & 0xFFu;
                        const uint slot = (word * 4u + byte) * 2u;
                        partial += kFP4[value & 0x0Fu] * xv[slot];
                        partial += kFP4[value >> 4]    * xv[slot + 1u];
                    }
                }
                acc[r] += partial * scale;
            }
        }

        for (uint r = 0; r < active; ++r) {
            const float total = simd_sum(acc[r]);
            if (lane == 0) { y[base + r] = total + float(bias[base + r]); }
        }
    }
}

// Plain SwiGLU: silu(gate) * up. What Qwen3 actually computes, and what
// `hidden_act: "silu"` means everywhere except GPT-OSS.
//
// The differences from `gptoss_expert_activation` are all load-bearing: no
// clamping, no `+1` shift on `up`, and the sigmoid has no alpha. Reusing the
// GPT-OSS kernel with alpha=1 would still leave the shift and the clamp, and
// would run without complaint.
//
// The interleaved gate/up order is shared, because the installer writes Qwen3's
// separate tensors into that order rather than teaching this a second layout.
kernel void expert_activation_swiglu(
    device const float *gate_up [[buffer(0)]],   // 2 * F, interleaved
    device half        *out     [[buffer(1)]],   // F
    constant uint      &width   [[buffer(2)]],   // F
    uint                index   [[thread_position_in_grid]])
{
    if (index >= width) { return; }

    const float gate = gate_up[index * 2u];
    const float up   = gate_up[index * 2u + 1u];

    const float silu = gate * (1.0f / (1.0f + exp(-gate)));
    out[index] = half(silu * up);
}

// RMS-normalises each head of q or k before RoPE, as Qwen3 does.
//
// Applied per head over head_dim, not over the whole projected vector: the
// weight is head_dim wide and shared across heads. Normalising across the full
// row instead would run, produce plausible numbers, and be wrong.
kernel void qk_head_rmsnorm(
    device half        *values   [[buffer(0)]],   // tokens * heads * head_dim
    device const half  *weight   [[buffer(1)]],   // head_dim
    constant uint      &headDim  [[buffer(2)]],
    constant float     &epsilon  [[buffer(3)]],
    uint                head     [[threadgroup_position_in_grid]],
    uint                lane     [[thread_position_in_threadgroup]],
    uint                width    [[threads_per_threadgroup]])
{
    device half *row = values + head * headDim;

    float sum = 0.0f;
    for (uint i = lane; i < headDim; i += width) {
        const float v = float(row[i]);
        sum += v * v;
    }
    sum = simd_sum(sum);
    const float inverse = rsqrt(sum / float(headDim) + epsilon);

    for (uint i = lane; i < headDim; i += width) {
        row[i] = half(float(row[i]) * inverse * float(weight[i]));
    }
}
