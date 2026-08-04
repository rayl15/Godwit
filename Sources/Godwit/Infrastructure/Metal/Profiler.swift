import Foundation
import Metal

/// Accumulates where time actually goes, rather than where arithmetic says.
///
/// Two clocks matter and they answer different questions. Wall time is what the
/// user waits. `MTLCommandBuffer.gpuStartTime`/`gpuEndTime` report what the GPU
/// was busy with. The gap between the sum of GPU time and the wall time spent
/// submitting is idle — the CPU preparing the next submission while the GPU has
/// nothing to do — and no amount of kernel optimisation touches it.
public final class Profiler: @unchecked Sendable {
    public struct Entry: Sendable {
        public var wall: Double = 0
        public var gpu: Double = 0
        public var count: Int = 0
        public var bytes: Int = 0
    }

    private var entries: [String: Entry] = [:]
    private var order: [String] = []
    private let lock = NSLock()
    public private(set) var totalWall: Double = 0

    public init() {}

    public func reset() {
        lock.lock()
        entries.removeAll()
        order.removeAll()
        totalWall = 0
        lock.unlock()
    }

    /// Times a CPU-side block.
    public func measure<T>(_ name: String, _ body: () throws -> T) rethrows -> T {
        let start = CFAbsoluteTimeGetCurrent()
        defer { add(name, wall: CFAbsoluteTimeGetCurrent() - start) }
        return try body()
    }

    /// Records a completed command buffer: how long the caller waited, and how
    /// much of that the GPU was actually working.
    public func record(_ name: String, commandBuffer: MTLCommandBuffer, wall: Double) {
        let gpu = max(0, commandBuffer.gpuEndTime - commandBuffer.gpuStartTime)
        add(name, wall: wall, gpu: gpu)
    }

    public func add(_ name: String, wall: Double, gpu: Double = 0, bytes: Int = 0) {
        lock.lock()
        if entries[name] == nil { order.append(name) }
        var entry = entries[name] ?? Entry()
        entry.wall += wall
        entry.gpu += gpu
        entry.count += 1
        entry.bytes += bytes
        entries[name] = entry
        lock.unlock()
    }

    public func setTotal(_ seconds: Double) {
        lock.lock(); totalWall = seconds; lock.unlock()
    }

    public func report() -> String {
        lock.lock()
        defer { lock.unlock() }

        let measured = entries.values.reduce(0) { $0 + $1.wall }
        let gpuTotal = entries.values.reduce(0) { $0 + $1.gpu }
        var lines = [String(format: "%-22@ %8@ %9@ %9@ %7@",
                            "phase" as NSString, "calls" as NSString,
                            "wall s" as NSString, "gpu s" as NSString, "%" as NSString)]
        lines.append(String(repeating: "-", count: 60))

        let denominator = totalWall > 0 ? totalWall : measured
        for name in order.sorted(by: { (entries[$0]?.wall ?? 0) > (entries[$1]?.wall ?? 0) }) {
            guard let entry = entries[name] else { continue }
            lines.append(String(format: "%-22@ %8d %9.3f %9.3f %6.1f%%",
                                name as NSString, entry.count, entry.wall, entry.gpu,
                                entry.wall / denominator * 100))
        }
        lines.append(String(repeating: "-", count: 60))
        lines.append(String(format: "%-22@ %8@ %9.3f %9.3f",
                            "measured" as NSString, "" as NSString, measured, gpuTotal))
        if totalWall > 0 {
            lines.append(String(format: "%-22@ %8@ %9.3f %9@ %6.1f%%",
                                "unaccounted" as NSString, "" as NSString,
                                totalWall - measured, "" as NSString,
                                (totalWall - measured) / totalWall * 100))
            lines.append(String(format: "%-22@ %8@ %9.3f", "total" as NSString,
                                "" as NSString, totalWall))
            lines.append("")
            lines.append(String(format: "GPU busy %.1f%% of wall time — the rest is the CPU "
                                + "preparing work the GPU is waiting for.",
                                gpuTotal / totalWall * 100))
        }
        return lines.joined(separator: "\n")
    }
}
