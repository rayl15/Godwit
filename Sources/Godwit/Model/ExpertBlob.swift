import Darwin
import Foundation
import Metal

/// Description of one expert stored on disk, as written by
/// `Scripts/analysis/fetch_expert.py`.
public struct ExpertManifest: Codable, Sendable {
    public struct Section: Codable, Sendable {
        public let offset: Int
        public let length: Int
        public let shape: [Int]
        public let dtype: String
    }

    public struct Reference: Codable, Sendable {
        public let x: String
        public let y: String
        public let seed: Int
    }

    public let model: String
    public let layer: Int
    public let expert: Int
    public let rows: Int
    public let cols: Int
    public let blockSize: Int
    public let pageAlignment: Int
    public let sections: [String: Section]
    public let reference: Reference?
}

public enum ExpertBlobError: Error, CustomStringConvertible {
    case openFailed(path: String, errno: Int32)
    case missingSection(String)
    case readFailed(section: String, errno: Int32)
    case shortRead(section: String, got: Int, want: Int)
    case allocationFailed(bytes: Int)
    case bufferWrapFailed

    public var description: String {
        switch self {
        case .openFailed(let path, let code):
            return "could not open \(path): \(String(cString: strerror(code)))"
        case .missingSection(let name):
            return "manifest has no section '\(name)'"
        case .readFailed(let section, let code):
            return "reading \(section) failed: \(String(cString: strerror(code)))"
        case .shortRead(let section, let got, let want):
            return "reading \(section) returned \(got) of \(want) bytes"
        case .allocationFailed(let bytes):
            return "could not allocate \(bytes) bytes"
        case .bufferWrapFailed:
            return "could not wrap host memory as an MTLBuffer"
        }
    }
}

/// Reads expert sections off disk with `pread` into Metal-visible memory.
///
/// This is the streaming path in miniature. `pread` rather than `mmap` is a
/// settled question — demand paging hands read timing to the virtual-memory
/// system, which measured roughly 3.5x slower cold. Destination pages are
/// allocated aligned and wrapped with `makeBuffer(bytesNoCopy:)`, so the bytes
/// the CPU reads are the bytes the GPU executes against, with no copy between.
public final class ExpertBlobReader {
    public let manifest: ExpertManifest
    private let fd: Int32

    public init(directory: URL) throws {
        let manifestData = try Data(contentsOf: directory.appendingPathComponent("manifest.json"))
        self.manifest = try JSONDecoder().decode(ExpertManifest.self, from: manifestData)

        let binaryPath = directory.appendingPathComponent("expert.bin").path
        let descriptor = open(binaryPath, O_RDONLY)
        guard descriptor >= 0 else {
            throw ExpertBlobError.openFailed(path: binaryPath, errno: errno)
        }
        self.fd = descriptor
    }

    deinit { close(fd) }

    /// `pread`s one section into a fresh page-aligned, GPU-visible buffer.
    public func load(section name: String, device: MTLDevice) throws -> MTLBuffer {
        guard let section = manifest.sections[name] else {
            throw ExpertBlobError.missingSection(name)
        }

        let pageSize = Int(getpagesize())
        // Metal requires the allocation to be page-aligned in both address and
        // length before it will adopt it without copying.
        let allocation = ((section.length + pageSize - 1) / pageSize) * pageSize

        var raw: UnsafeMutableRawPointer?
        guard posix_memalign(&raw, pageSize, allocation) == 0, let pointer = raw else {
            throw ExpertBlobError.allocationFailed(bytes: allocation)
        }

        var filled = 0
        while filled < section.length {
            let got = pread(fd,
                            pointer.advanced(by: filled),
                            section.length - filled,
                            off_t(section.offset + filled))
            if got < 0 {
                free(pointer)
                throw ExpertBlobError.readFailed(section: name, errno: errno)
            }
            if got == 0 {
                free(pointer)
                throw ExpertBlobError.shortRead(section: name, got: filled, want: section.length)
            }
            filled += got
        }

        nonisolated(unsafe) let owned = pointer
        guard let buffer = device.makeBuffer(bytesNoCopy: pointer,
                                             length: allocation,
                                             options: .storageModeShared,
                                             deallocator: { _, _ in free(owned) })
        else {
            free(pointer)
            throw ExpertBlobError.bufferWrapFailed
        }
        return buffer
    }
}
