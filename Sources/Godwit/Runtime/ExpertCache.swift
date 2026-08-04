import Darwin
import Foundation
import Metal

/// Resident expert slots, one bounded set per layer.
///
/// This is where the project's central bet is finally cashed. Without it every
/// decode step re-reads all 144 experts — about 1.8 GiB per token. Measured
/// routing says eight slots per layer capture 73.6% of requests, which should
/// leave roughly 0.5 GiB.
///
/// Two things make the read cheap:
///
/// **One `pread` per expert, not six.** The installer laid the six sub-tensors
/// out inside a single page-aligned stride, so the whole expert arrives in one
/// sequential read and the sections are addressed by buffer offset afterwards.
/// The 5% of the stride that is alignment padding is worth not making six
/// syscalls for.
///
/// **Slots are allocated once.** The buffers are page-aligned host memory
/// wrapped with `makeBuffer(bytesNoCopy:)`, so a hit costs nothing at all and a
/// miss is a read into memory the GPU already holds a handle to.
public final class ExpertCache {
    /// Set to record read time separately from everything else.
    public var profiler: Profiler?

    public struct Stats: Sendable {
        public var hits = 0
        public var misses = 0
        public var bytesRead = 0
        public var hitRate: Double {
            let total = hits + misses
            return total == 0 ? 0 : Double(hits) / Double(total)
        }
    }

    public let slotCount: Int
    public let layerCount: Int
    public let stride: Int
    public private(set) var stats = Stats()

    private let reader: ModelReader
    private var planners: [ExpertCachePlanner]
    private var buffers: [[MTLBuffer]]
    private var descriptors: [Int32]
    private let lock = NSLock()

    public init(context: MetalContext, reader: ModelReader, slotCount: Int = 8,
                bypassPageCache: Bool = true) throws {
        self.reader = reader
        self.slotCount = slotCount
        self.layerCount = reader.manifest.layerCount
        self.stride = reader.manifest.expertStride

        let expertCount = reader.manifest.expertCount
        precondition(slotCount >= reader.manifest.spec.maxExpertsPerToken,
                     "need at least top-k slots to plan a single token")

        self.planners = (0..<layerCount).map { _ in
            ExpertCachePlanner(slotCount: slotCount, expertCount: expertCount)
        }
        self.descriptors = []
        self.buffers = []

        let pageSize = Int(getpagesize())
        let allocation = (stride + pageSize - 1) / pageSize * pageSize
        for layer in 0..<layerCount {
            let path = reader.directory
                .appendingPathComponent("experts")
                .appendingPathComponent(String(format: "layer_%02d.bin", layer))
                .path
            let fd = open(path, O_RDONLY)
            guard fd >= 0 else { throw ExpertBlobError.openFailed(path: path, errno: errno) }

            // Bypass the unified buffer cache.
            //
            // We already hold the experts we want in our own slots, so the page
            // cache only buys a second copy of 59 GiB it cannot hold anyway on a
            // 16 GB machine — every read pays a copy into cache and out again,
            // and evicts something useful doing it. Reads here are page-aligned
            // in both offset and destination, which is what F_NOCACHE needs to
            // be fast rather than pathological.
            if bypassPageCache {
                if fcntl(fd, F_NOCACHE, 1) < 0 { perror("F_NOCACHE") }
                if fcntl(fd, F_RDAHEAD, 0) < 0 { perror("F_RDAHEAD") }
            }
            descriptors.append(fd)

            var layerBuffers: [MTLBuffer] = []
            for _ in 0..<slotCount {
                var raw: UnsafeMutableRawPointer?
                guard posix_memalign(&raw, pageSize, allocation) == 0, let pointer = raw else {
                    throw ExpertBlobError.allocationFailed(bytes: allocation)
                }
                nonisolated(unsafe) let owned = pointer
                guard let buffer = context.device.makeBuffer(
                    bytesNoCopy: pointer, length: allocation,
                    options: .storageModeShared, deallocator: { _, _ in free(owned) })
                else {
                    free(pointer)
                    throw ExpertBlobError.bufferWrapFailed
                }
                layerBuffers.append(buffer)
            }
            buffers.append(layerBuffers)
        }
    }

    deinit {
        for fd in descriptors { close(fd) }
    }

    /// Bytes held resident by the cache.
    public var byteCount: Int { slotCount * layerCount * stride }

    /// Makes `experts` resident in `layer` and returns their slots, in order.
    ///
    /// Misses are read here; hits cost nothing. The planner decides placement
    /// and is the only thing that decides it — this method never evicts on its
    /// own, which is what keeps the policy testable in isolation.
    public func acquire(layer: Int, experts: [Int]) throws -> [Int] {
        lock.lock()
        let plan = planners[layer].plan(experts: experts)
        lock.unlock()

        stats.hits += plan.hitCount
        stats.misses += plan.missCount

        guard !plan.missIndices.isEmpty else { return plan.slots }

        // Read the misses concurrently.
        //
        // Serially these ran at 1.97 GiB/s against the 5.08 GiB/s the same
        // 12.6 MiB reads reach at eight threads — one outstanding request never
        // saturates NVMe, and profiling put 71.6% of decode in this loop. The
        // misses are independent by construction: the planner has already
        // assigned each a distinct slot.
        let start = CFAbsoluteTimeGetCurrent()
        let misses = plan.missIndices
        let errorLock = NSLock()
        nonisolated(unsafe) var failure: Error?

        if misses.count == 1 {
            try read(layer: layer, expert: plan.experts[misses[0]], slot: plan.slots[misses[0]])
        } else {
            DispatchQueue.concurrentPerform(iterations: misses.count) { position in
                let index = misses[position]
                do {
                    try read(layer: layer, expert: plan.experts[index], slot: plan.slots[index])
                } catch {
                    errorLock.lock()
                    if failure == nil { failure = error }
                    errorLock.unlock()
                }
            }
        }
        if let failure { throw failure }

        profiler?.add("io:expert-read", wall: CFAbsoluteTimeGetCurrent() - start,
                      bytes: stride * misses.count)
        stats.bytesRead += stride * misses.count
        return plan.slots
    }

    public func buffer(layer: Int, slot: Int) -> MTLBuffer {
        buffers[layer][slot]
    }

    /// Concurrent chunks per expert read.
    ///
    /// Set to 1: splitting a read was tried and measured slightly *worse*
    /// (1.54 → 1.46 tok/s). The premise was that one `pread` is one outstanding
    /// request and the device wants several, but benchmarking the real model
    /// file showed one thread and eight threads both reach ~2 GiB/s. Whatever
    /// limits reads here, it is not the number of requests in flight.
    public static let readChunks = 1

    /// Reads one whole expert stride into a slot.
    private func read(layer: Int, expert: Int, slot: Int) throws {
        let fd = descriptors[layer]
        let destination = buffers[layer][slot].contents()
        let base = expert * stride

        let chunks = Self.readChunks
        guard chunks > 1 else {
            try readRange(fd: fd, into: destination, offset: base, length: stride, layer: layer)
            return
        }

        // Page-aligned chunk boundaries: F_NOCACHE requires alignment to be
        // fast, and the stride is a page multiple by construction.
        let pageSize = Int(getpagesize())
        let raw = (stride + chunks - 1) / chunks
        let chunkSize = (raw + pageSize - 1) / pageSize * pageSize

        let errorLock = NSLock()
        nonisolated(unsafe) var failure: Error?
        DispatchQueue.concurrentPerform(iterations: chunks) { index in
            let start = index * chunkSize
            guard start < stride else { return }
            let length = min(chunkSize, stride - start)
            do {
                try readRange(fd: fd, into: destination.advanced(by: start),
                              offset: base + start, length: length, layer: layer)
            } catch {
                errorLock.lock()
                if failure == nil { failure = error }
                errorLock.unlock()
            }
        }
        if let failure { throw failure }
    }

    private func readRange(
        fd: Int32, into destination: UnsafeMutableRawPointer,
        offset: Int, length: Int, layer: Int
    ) throws {
        var filled = 0
        while filled < length {
            let got = pread(fd, destination.advanced(by: filled), length - filled,
                            off_t(offset + filled))
            if got < 0 {
                throw ExpertBlobError.readFailed(section: "layer \(layer)", errno: errno)
            }
            if got == 0 {
                throw ExpertBlobError.shortRead(section: "layer \(layer)",
                                                got: filled, want: length)
            }
            filled += got
        }
    }

    public func resetStats() { stats = Stats() }
}
