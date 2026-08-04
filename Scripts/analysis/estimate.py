"""GPT-OSS-120B on a 16 GB M4: throughput estimate from measured inputs."""

# --- Model, from openai/gpt-oss-120b config.json ---
LAYERS, EXPERTS, TOP_K = 36, 128, 4
HIDDEN, INTERMEDIATE = 2880, 2880
VOCAB, HEADS, KV_HEADS, HEAD_DIM = 201088, 64, 8, 64
SLIDING_WINDOW = 128            # 18 sliding layers alternating with 18 full

MXFP4_BITS = 4 + 8 / 32         # 4-bit code + one E8M0 scale byte per 32 weights

# --- Measured on this machine (nvmebench, F_NOCACHE, 12.6 MiB random reads) ---
BW_1T = 3158.84 / 1024          # GiB/s, single thread
BW_8T = 5202.91 / 1024          # GiB/s, 8 threads
RAM_BW = 120.0                  # GiB/s, M4 unified memory (spec)

GiB = 1024 ** 3

def bits_to_gib(params, bits):
    return params * bits / 8 / GiB

# --- Expert size ---
per_expert_params = 2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE
expert_gib = bits_to_gib(per_expert_params, MXFP4_BITS)
expert_mib = expert_gib * 1024
routed_total = expert_gib * EXPERTS * LAYERS

print("=== Model geometry ===")
print(f"Params per expert      : {per_expert_params/1e6:.1f}M")
print(f"Expert blob            : {expert_mib:.2f} MiB")
print(f"Routed pool (streamed) : {routed_total:.1f} GiB")

# --- Resident trunk. config.json keeps attn/router/embed/head out of MXFP4 (bf16);
# we requantise them ourselves, which is the only reason this fits at all. ---
embed = bits_to_gib(VOCAB * HIDDEN, MXFP4_BITS)
head = embed                                    # untied in gpt-oss
attn_params = LAYERS * (HIDDEN*HEADS*HEAD_DIM + 2*HIDDEN*KV_HEADS*HEAD_DIM + HEADS*HEAD_DIM*HIDDEN)
attn_8bit = bits_to_gib(attn_params, 8.25)
router = bits_to_gib(LAYERS * HIDDEN * EXPERTS, 16)
resident = embed + head + attn_8bit + router

print(f"\n=== Resident trunk ===")
print(f"Embedding (4-bit)      : {embed:.2f} GiB")
print(f"Output head (4-bit)    : {head:.2f} GiB")
print(f"Attention (8-bit)      : {attn_8bit:.2f} GiB")
print(f"Router (bf16)          : {router:.3f} GiB")
print(f"Total resident         : {resident:.2f} GiB")

def kv_gib(ctx):
    per_tok_per_layer = 2 * KV_HEADS * HEAD_DIM * 2      # K+V, FP16
    sliding = LAYERS//2 * min(ctx, SLIDING_WINDOW) * per_tok_per_layer
    full = LAYERS//2 * ctx * per_tok_per_layer
    return (sliding + full) / GiB

print(f"KV at 8K context       : {kv_gib(8192):.2f} GiB")
print(f"KV at 32K context      : {kv_gib(32768):.2f} GiB")

# --- Decode, using simulated hit rates at the calibrated exponent ---
HIT = {4: 0.136, 6: 0.277, 8: 0.355, 12: 0.453, 16: 0.516}

print("\n=== Decode throughput (8K context) ===")
print("slots  cacheRAM  totalRAM   hit%   IO/tok   IO time  compute  tok/s(ideal)  tok/s(real)")
for slots, hit in HIT.items():
    cache = slots * LAYERS * expert_gib
    total = resident + kv_gib(8192) + cache
    reads = TOP_K * LAYERS * (1 - hit)
    io_gib = reads * expert_gib
    io_s = io_gib / BW_8T
    # GPU must also stream every activated expert out of RAM, hit or miss.
    compute_s = (TOP_K * LAYERS * expert_gib + resident) / RAM_BW
    ideal = 1 / io_s
    # No shared-expert branch in gpt-oss, so there is less resident GPU work to
    # hide reads behind than TurboFieldfare had. Assume overlap recovers only
    # half the compute.
    real = 1 / (io_s + compute_s * 0.5)
    flag = "  << over budget" if total > 11 else ""
    print(f"{slots:>4}  {cache:6.2f}GiB {total:6.2f}GiB  {hit:5.1%}  {reads:6.1f}  "
          f"{io_s*1000:6.0f}ms  {compute_s*1000:5.0f}ms  {ideal:8.2f}      {real:6.2f}{flag}")

# --- Prefill. A chunk of C tokens routes 4C times per layer over 128 experts;
# past a few hundred tokens essentially every expert is touched, so the cost
# floor is one full pass over the routed pool per chunk. ---
print("\n=== Prefill (cost floor = one full model read per chunk) ===")
def unique_experts(chunk):
    draws = TOP_K * chunk
    return EXPERTS * (1 - (1 - 1/EXPERTS) ** draws)   # coupon collector

print("chunk  uniqueExperts/layer  read/chunk   TTFT(3K prompt)")
for chunk in [128, 256, 512, 1024, 2048]:
    uniq = unique_experts(chunk)
    per_chunk = uniq * LAYERS * expert_gib
    chunks = -(-3000 // chunk)
    ttft = chunks * per_chunk / BW_8T
    print(f"{chunk:>5}  {uniq:17.1f}  {per_chunk:8.1f}GiB   {ttft:8.1f}s ({chunks} chunks)")
