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
        let data = try Data(contentsOf: directory.appendingPathComponent("manifest.json"))
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
