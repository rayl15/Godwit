#include <metal_stdlib>
using namespace metal;

// Grouped-query attention with sinks and an optional sliding window.
//
// The sink is a learned per-head logit appended to the row before softmax and
// dropped afterwards. It contributes to the denominator but carries no value,
// so it is a way for a head to attend to nothing — probability mass that goes
// nowhere instead of being forced onto real tokens. Leaving it out does not
// fail, it just scales every output up by a variable amount.
//
// Softmax runs online (running max and denominator) rather than materialising
// the score row. With a 131,072-token context a stored row would dwarf the
// output it produces.

constant constexpr uint kSimdWidth = 32;
constant constexpr uint kHeadDim = 64;
constant constexpr uint kPerLane = kHeadDim / kSimdWidth;   // 2

// One threadgroup of 32 lanes per (query position, head).
//
// Each lane owns two of the 64 head dimensions, so a dot product is two
// multiplies plus one simd_sum, and the accumulator stays in registers.
kernel void gqa_attention_sinks(
    device const half  *queries    [[buffer(0)]],  // [T, H, D]
    device const half  *keys       [[buffer(1)]],  // [S, KV, D]
    device const half  *values     [[buffer(2)]],  // [S, KV, D]
    device const bfloat *sinks     [[buffer(3)]],  // [H]
    device half        *out        [[buffer(4)]],  // [T, H, D]
    constant uint      &keyCount   [[buffer(5)]],  // S
    constant uint      &headCount  [[buffer(6)]],  // H
    constant uint      &kvHeads    [[buffer(7)]],  // KV
    constant uint      &window     [[buffer(8)]],  // 0 means full attention
    constant uint      &queryBase  [[buffer(9)]],  // absolute position of query 0
    constant float     &scale      [[buffer(10)]],
    constant uint      &ring       [[buffer(11)]], // ring capacity, 0 = linear
    uint2               group      [[threadgroup_position_in_grid]],
    uint                lane       [[thread_index_in_threadgroup]])
{
    const uint queryIndex = group.x;
    const uint head = group.y;
    const uint kvHead = head / (headCount / kvHeads);      // grouped-query
    const uint position = queryBase + queryIndex;

    device const half *q = queries + (queryIndex * headCount + head) * kHeadDim;

    float accumulator[kPerLane];
    for (uint i = 0; i < kPerLane; ++i) { accumulator[i] = 0.0f; }
    float runningMax = -INFINITY;
    float denominator = 0.0f;

    for (uint key = 0; key <= position && key < keyCount; ++key) {
        // (keyCount is the number of positions written so far, not slots)
        // Sliding layers see only the most recent `window` keys, inclusive of
        // the query's own position.
        if (window != 0 && position - key >= window) { continue; }

        // Sliding layers store K/V in a ring: a window of 128 needs 128 slots,
        // not one per position, which is what keeps KV bounded at long context.
        const uint slot = (ring != 0) ? (key % ring) : key;
        device const half *k = keys + (slot * kvHeads + kvHead) * kHeadDim;
        float partial = 0.0f;
        for (uint i = 0; i < kPerLane; ++i) {
            const uint index = lane + i * kSimdWidth;
            partial += float(q[index]) * float(k[index]);
        }
        const float score = simd_sum(partial) * scale;

        const float updatedMax = max(runningMax, score);
        const float rescale = exp(runningMax - updatedMax);
        const float weight = exp(score - updatedMax);

        device const half *v = values + (slot * kvHeads + kvHead) * kHeadDim;
        for (uint i = 0; i < kPerLane; ++i) {
            const uint index = lane + i * kSimdWidth;
            accumulator[i] = accumulator[i] * rescale + weight * float(v[index]);
        }
        denominator = denominator * rescale + weight;
        runningMax = updatedMax;
    }

    // The sink joins the denominator only. No value is accumulated, which is
    // precisely what lets a head decline to attend.
    {
        const float sink = float(sinks[head]);
        const float updatedMax = max(runningMax, sink);
        const float rescale = exp(runningMax - updatedMax);
        for (uint i = 0; i < kPerLane; ++i) { accumulator[i] *= rescale; }
        denominator = denominator * rescale + exp(sink - updatedMax);
    }

    const float inverse = 1.0f / max(denominator, 1e-20f);
    device half *destination = out + (queryIndex * headCount + head) * kHeadDim;
    for (uint i = 0; i < kPerLane; ++i) {
        const uint index = lane + i * kSimdWidth;
        destination[index] = half(accumulator[i] * inverse);
    }
}

// NeoX rotary embedding: the head is split in half and the halves rotate
// against each other. Interleaved-pair layouts exist and are wrong here.
//
// cos and sin tables already carry YaRN's attention factor.
kernel void apply_rope(
    device half        *vectors  [[buffer(0)]],  // [T, H, D], in place
    device const float *cosines  [[buffer(1)]],  // [T, D/2]
    device const float *sines    [[buffer(2)]],  // [T, D/2]
    constant uint      &headCount [[buffer(3)]],
    uint2               group    [[threadgroup_position_in_grid]],
    uint                lane     [[thread_index_in_threadgroup]])
{
    const uint token = group.x;
    const uint head = group.y;
    const uint half_ = kHeadDim / 2;

    device half *vector = vectors + (token * headCount + head) * kHeadDim;
    device const float *cosRow = cosines + token * half_;
    device const float *sinRow = sines + token * half_;

    for (uint i = lane; i < half_; i += kSimdWidth) {
        const float first = float(vector[i]);
        const float second = float(vector[i + half_]);
        vector[i] = half(first * cosRow[i] - second * sinRow[i]);
        vector[i + half_] = half(second * cosRow[i] + first * sinRow[i]);
    }
}


// Writes one run of K/V into the cache at absolute positions, wrapping when the
// layer uses a ring. Separate from attention so a decode step can append before
// attending without re-uploading the whole cache.
kernel void kv_cache_write(
    device const half *source   [[buffer(0)]],  // [T, KV, D]
    device half       *cache    [[buffer(1)]],  // [capacity, KV, D]
    constant uint     &kvHeads  [[buffer(2)]],
    constant uint     &base     [[buffer(3)]],  // absolute position of token 0
    constant uint     &ring     [[buffer(4)]],  // 0 = linear
    uint2              group    [[threadgroup_position_in_grid]],
    uint               lane     [[thread_index_in_threadgroup]])
{
    const uint token = group.x;
    const uint head = group.y;
    const uint position = base + token;
    const uint slot = (ring != 0) ? (position % ring) : position;

    device const half *src = source + (token * kvHeads + head) * kHeadDim;
    device half *dst = cache + (slot * kvHeads + head) * kHeadDim;
    for (uint i = lane; i < kHeadDim; i += kSimdWidth) {
        dst[i] = src[i];
    }
}
