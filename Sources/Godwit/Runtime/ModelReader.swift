import Darwin
import Foundation
import Metal

/// Reads experts out of a `.gwt` installation.
///
/// One `pread` per expert sub-tensor, straight into page-aligned memory the GPU
/// can execute against without a copy. There is deliberately no caching here —
/// residency is `ExpertCachePlanner`'s job, and mixing the two would make the
/// eviction policy untestable.
public final class ModelReader {
    public let manifest: GodwitManifest
    public let directory: URL
    private var layerDescriptors: [Int: Int32] = [:]
    private let lock = NSLock()

    public init(directory: URL) throws {
        self.directory = directory
        let manifestURL = directory.appendingPathComponent("manifest.json")
        // Checked before reading so a mistyped path produces a sentence rather
        // than an NSCocoaErrorDomain dump. This is most people's first command,
        // and the raw error buries the one fact that matters.
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            let exists = FileManager.default.fileExists(atPath: directory.path)
            throw ModelReaderError.notAnInstall(
                exists
                ? "\(directory.path) has no manifest.json — not a .gwt install, "
                  + "or an install that did not finish"
                : "no such directory: \(directory.path). Run `godwit install "
                  + "--output <dir>` first, or pass --model <dir>")
        }
        let data = try Data(contentsOf: manifestURL)
        self.manifest = try JSONDecoder().decode(GodwitManifest.self, from: data)
        guard manifest.formatVersion == GodwitManifest.currentFormatVersion else {
            throw InstallError.verificationFailed(
                "format v\(manifest.formatVersion), runtime expects v\(GodwitManifest.currentFormatVersion)")
        }
    }

    deinit {
        for descriptor in layerDescriptors.values { close(descriptor) }
    }

    private func descriptor(forLayer layer: Int) throws -> Int32 {
        lock.lock()
        defer { lock.unlock() }
        if let cached = layerDescriptors[layer] { return cached }

        let path = directory
            .appendingPathComponent("experts")
            .appendingPathComponent(String(format: "layer_%02d.bin", layer))
            .path
        let descriptor = open(path, O_RDONLY)
        guard descriptor >= 0 else {
            throw ExpertBlobError.openFailed(path: path, errno: errno)
        }
        layerDescriptors[layer] = descriptor
        return descriptor
    }

    /// Reads one sub-tensor of one expert into fresh GPU-visible memory.
    public func loadSection(
        layer: Int, expert: Int, section: ExpertLayout.Section, device: MTLDevice
    ) throws -> MTLBuffer {
        guard let span = manifest.expertSections[section.rawValue] else {
            throw ExpertBlobError.missingSection(section.rawValue)
        }
        let offset = expert * manifest.expertStride + span.offset
        return try read(layer: layer, offset: offset, length: span.length, device: device)
    }

    /// Reads an entire expert stride, all six sub-tensors in one call.
    public func loadExpertStride(layer: Int, expert: Int, device: MTLDevice) throws -> MTLBuffer {
        try read(layer: layer, offset: expert * manifest.expertStride,
                 length: manifest.expertStride, device: device)
    }

    private func read(layer: Int, offset: Int, length: Int, device: MTLDevice) throws -> MTLBuffer {
        let fd = try descriptor(forLayer: layer)
        let pageSize = Int(getpagesize())
        let allocation = (length + pageSize - 1) / pageSize * pageSize

        var raw: UnsafeMutableRawPointer?
        guard posix_memalign(&raw, pageSize, allocation) == 0, let pointer = raw else {
            throw ExpertBlobError.allocationFailed(bytes: allocation)
        }

        var filled = 0
        while filled < length {
            let got = pread(fd, pointer.advanced(by: filled), length - filled, off_t(offset + filled))
            if got < 0 {
                free(pointer)
                throw ExpertBlobError.readFailed(section: "layer \(layer)", errno: errno)
            }
            if got == 0 {
                free(pointer)
                throw ExpertBlobError.shortRead(section: "layer \(layer)", got: filled, want: length)
            }
            filled += got
        }

        nonisolated(unsafe) let owned = pointer
        guard let buffer = device.makeBuffer(bytesNoCopy: pointer, length: allocation,
                                             options: .storageModeShared,
                                             deallocator: { _, _ in free(owned) })
        else {
            free(pointer)
            throw ExpertBlobError.bufferWrapFailed
        }
        return buffer
    }
}

extension ModelReader {
    /// Reads a named trunk section into GPU-visible memory.
    ///
    /// The trunk is small and resident, so unlike experts these are read once at
    /// load time. Still `pread` rather than `mmap`, for the same reason: we want
    /// to decide when the read happens.
    public func loadTrunk(section name: String, device: MTLDevice) throws -> MTLBuffer {
        guard let section = manifest.trunkSections.first(where: { $0.name == name }) else {
            throw ExpertBlobError.missingSection(name)
        }
        let path = directory.appendingPathComponent("trunk.bin").path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { throw ExpertBlobError.openFailed(path: path, errno: errno) }
        defer { close(fd) }

        let pageSize = Int(getpagesize())
        let allocation = (section.length + pageSize - 1) / pageSize * pageSize
        var raw: UnsafeMutableRawPointer?
        guard posix_memalign(&raw, pageSize, allocation) == 0, let pointer = raw else {
            throw ExpertBlobError.allocationFailed(bytes: allocation)
        }

        var filled = 0
        while filled < section.length {
            let got = pread(fd, pointer.advanced(by: filled),
                            section.length - filled, off_t(section.offset + filled))
            if got <= 0 {
                free(pointer)
                throw ExpertBlobError.readFailed(section: name, errno: errno)
            }
            filled += got
        }

        nonisolated(unsafe) let owned = pointer
        guard let buffer = device.makeBuffer(bytesNoCopy: pointer, length: allocation,
                                             options: .storageModeShared,
                                             deallocator: { _, _ in free(owned) })
        else {
            free(pointer)
            throw ExpertBlobError.bufferWrapFailed
        }
        return buffer
    }

    /// The tokeniser that shipped with this installation.
    public func loadTokenizer() throws -> Tokenizer {
        let url = directory.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw TokenizerError.notFound(url.path)
        }
        return try Tokenizer(contentsOf: url)
    }

    public func trunkShape(_ name: String) throws -> [Int] {
        guard let section = manifest.trunkSections.first(where: { $0.name == name }) else {
            throw ExpertBlobError.missingSection(name)
        }
        return section.shape
    }
}

/// Problems with the install directory itself, kept separate from install-time
/// verification so the message a first-time user sees is one sentence.
public enum ModelReaderError: Error, CustomStringConvertible {
    case notAnInstall(String)
    public var description: String {
        switch self { case .notAnInstall(let detail): return detail }
    }
}
