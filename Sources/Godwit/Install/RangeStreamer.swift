import Darwin
import Foundation

/// Streams HTTP byte ranges straight into their final position on disk.
///
/// The installer never materialises a shard. It issues one ranged request per
/// (layer, tensor) — 216 requests for the whole model rather than the ~28,000 a
/// per-expert request pattern would need — and redistributes the arriving bytes
/// into the destination's expert-strided layout as they land.
///
/// Memory stays bounded at roughly one expert slice regardless of tensor size,
/// which is the same rule the runtime follows: no step should scale its
/// footprint with the size of the checkpoint.
/// Stateless and safe to share: `URLSession` is thread-safe and this type adds
/// no mutable state of its own.
public final class RangeStreamer: Sendable {
    private let session: URLSession

    public init(timeout: TimeInterval = 300) {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = timeout * 20
        configuration.httpMaximumConnectionsPerHost = 4
        self.session = URLSession(configuration: configuration)
    }

    /// Fetches a byte range into memory. For headers and small tensors.
    public func fetch(url: URL, offset: Int? = nil, length: Int? = nil) async throws -> Data {
        var request = URLRequest(url: url)
        if let offset, let length {
            request.setValue("bytes=\(offset)-\(offset + length - 1)", forHTTPHeaderField: "Range")
        }
        let (data, response) = try await session.data(for: request)
        try Self.check(response, url: url)
        if let length, data.count != length {
            throw InstallError.shortTransfer(expected: length, got: data.count)
        }
        return data
    }

    /// Slices delivered per HTTP request.
    ///
    /// `URLSession.bytes` yields one byte at a time, which is unusable across a
    /// gigabyte-scale tensor, so ranges are fetched in batches instead. Eight
    /// expert slices is roughly 66 MiB — large enough that per-request latency
    /// disappears into transfer time, small enough to keep the footprint
    /// bounded and to lose little work when a request has to be retried.
    public static let slicesPerRequest = 8

    /// Fetches a byte range in bounded batches, handing out fixed-size slices.
    ///
    /// `sliceLength` is the destination's natural unit — one expert's worth of
    /// one tensor — so `handle` is called once per expert with that expert's
    /// index. Any trailing partial slice is delivered last.
    public func stream(
        url: URL,
        offset: Int,
        length: Int,
        sliceLength: Int,
        handle: (Int, UnsafeRawBufferPointer) throws -> Void
    ) async throws {
        precondition(sliceLength > 0, "sliceLength must be positive")
        let batchBytes = sliceLength * Self.slicesPerRequest
        var consumed = 0
        var sliceIndex = 0

        while consumed < length {
            let take = min(batchBytes, length - consumed)
            let batch = try await fetch(url: url, offset: offset + consumed, length: take)

            try batch.withUnsafeBytes { raw in
                var cursor = 0
                while cursor < take {
                    let span = min(sliceLength, take - cursor)
                    try handle(sliceIndex, UnsafeRawBufferPointer(
                        rebasing: raw[cursor..<(cursor + span)]))
                    cursor += span
                    sliceIndex += 1
                }
            }
            consumed += take
        }
    }

    private static func check(_ response: URLResponse, url: URL) throws {
        guard let http = response as? HTTPURLResponse else { return }
        // 206 for a satisfied range, 200 when the server ignores it.
        guard http.statusCode == 200 || http.statusCode == 206 else {
            throw InstallError.transferFailed(url: url.absoluteString, status: http.statusCode)
        }
    }
}

/// A file opened for positional writes.
///
/// `pwrite` rather than seek-then-write so sections can be filled in whatever
/// order they arrive, which is what lets the streamer redistribute a
/// tensor-major download into an expert-major file.
/// `@unchecked Sendable` because every write goes to an explicit offset via
/// `pwrite`, which does not touch the shared file position. Concurrent writes
/// to disjoint ranges are safe, and the expert installer only ever issues
/// disjoint ones — each expert owns its own stride.
public final class PositionalWriter: @unchecked Sendable {
    private let fd: Int32
    public let path: String

    public init(path: String, size: Int) throws {
        self.path = path
        let descriptor = open(path, O_RDWR | O_CREAT | O_TRUNC, 0o644)
        guard descriptor >= 0 else {
            throw InstallError.writeFailed(path: path, errno: errno)
        }
        self.fd = descriptor
        // Reserve the full extent up front so the file is laid out contiguously
        // rather than grown piecemeal as slices arrive out of order.
        if ftruncate(descriptor, off_t(size)) != 0 {
            close(descriptor)
            throw InstallError.writeFailed(path: path, errno: errno)
        }
    }

    deinit { close(fd) }

    public func write(_ bytes: UnsafeRawBufferPointer, at offset: Int) throws {
        guard let base = bytes.baseAddress else { return }
        var written = 0
        while written < bytes.count {
            let n = pwrite(fd, base.advanced(by: written),
                           bytes.count - written, off_t(offset + written))
            if n <= 0 { throw InstallError.writeFailed(path: path, errno: errno) }
            written += n
        }
    }

    public func write(_ data: Data, at offset: Int) throws {
        try data.withUnsafeBytes { try write($0, at: offset) }
    }

    /// Flushes to stable storage. Called once per file rather than per write —
    /// an interrupted install is restarted, not resumed mid-file.
    public func sync() throws {
        if fsync(fd) != 0 { throw InstallError.writeFailed(path: path, errno: errno) }
    }
}
