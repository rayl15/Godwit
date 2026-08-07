import Foundation
import Metal

/// Per-layer key/value storage, FP16.
///
/// Two layouts, because GPT-OSS alternates two kinds of layer:
///
/// **Sliding layers** attend to the last 128 tokens, so they need 128 slots, not
/// one per position. A ring gives bounded memory at any context length — 18 such
/// layers cost 4.7 MiB total regardless of whether the context is 1K or 128K.
///
/// **Full layers** keep everything and grow linearly. These are what make long
/// context expensive: 18 of them at 8K is 302 MiB, at 32K it is 1.2 GiB.
///
/// FP16 rather than a quantised cache is a deliberate inheritance. TurboFieldfare
/// measured a packed 4-bit KV cache failing quality evaluation *and* using more
/// memory at long context than the exact FP16 layout. Not worth revisiting
/// without a quality harness.
public final class KVCache {
    public struct LayerCache {
        public let keys: MTLBuffer
        public let values: MTLBuffer
        /// Ring capacity in positions, or 0 when storage is linear.
        public let ring: Int
        /// Slots allocated.
        public let capacity: Int
    }

    public let spec: ArchitectureSpec
    public let maxContext: Int
    public private(set) var layers: [LayerCache] = []
    /// Positions written so far.
    public private(set) var length = 0

    private let context: MetalContext

    public init(context: MetalContext, spec: ArchitectureSpec, maxContext: Int,
                layerCount: Int? = nil) throws {
        self.context = context
        self.spec = spec
        self.maxContext = maxContext

        let width = spec.keyValueHeads * spec.headDimension
        for index in 0..<(layerCount ?? spec.layerCount) {
            let layer = spec.layers[index]
            let isSliding = layer.attention == .sliding
            // Sliding layers need the window itself plus room for a prefill chunk
            // to be written before the oldest entries it still needs are read.
            let capacity = isSliding ? min(layer.window * 2, maxContext) : maxContext
            layers.append(LayerCache(
                keys: try context.emptyBuffer(of: Float16.self, count: capacity * width),
                values: try context.emptyBuffer(of: Float16.self, count: capacity * width),
                ring: isSliding ? capacity : 0,
                capacity: capacity))
        }
    }

    /// Bytes held across all layers.
    public var byteCount: Int {
        let width = spec.keyValueHeads * spec.headDimension
        return layers.reduce(0) { $0 + $1.capacity * width * 2 * 2 }
    }

    public func advance(by tokens: Int) {
        length += tokens
    }

    public func reset() {
        length = 0
    }

    /// Drops everything after `position`, keeping the prefix valid.
    ///
    /// Entries past the new length are left in place and simply overwritten by
    /// the next write; nothing reads beyond `length`. Sliding layers hold a
    /// ring, so shrinking is safe for the same reason — the slots a shortened
    /// prefix maps to are exactly the ones it wrote.
    public func truncate(to position: Int) {
        precondition(position >= 0 && position <= length,
                     "truncate(to: \(position)) outside 0...\(length)")
        length = position
    }

    /// Appends a run of K/V for one layer at absolute `position`.
    ///
    /// Encodes into a caller-supplied command buffer so the write joins the
    /// surrounding attention work in one submission rather than costing two of
    /// its own.
    public func encodeWrite(
        _ commands: MTLCommandBuffer,
        keys: MTLBuffer, values: MTLBuffer, layer: Int, position: Int, tokenCount: Int
    ) throws {
        let cache = layers[layer]
        let pipeline = try context.pipeline(shader: "attention", function: "kv_cache_write",
                                            constants: [0: spec.headDimension])

        for (source, destination) in [(keys, cache.keys), (values, cache.values)] {
            guard let encoder = commands.makeComputeCommandEncoder()
            else { throw MetalError.encoderCreationFailed }
            var kvHeads = UInt32(spec.keyValueHeads)
            var base = UInt32(position)
            var ring = UInt32(cache.ring)
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(source, offset: 0, index: 0)
            encoder.setBuffer(destination, offset: 0, index: 1)
            encoder.setBytes(&kvHeads, length: 4, index: 2)
            encoder.setBytes(&base, length: 4, index: 3)
            encoder.setBytes(&ring, length: 4, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: tokenCount, height: spec.keyValueHeads, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }
}
