import Foundation

/// Describes a completed `.gwt` installation.
///
/// The runtime accepts an installation only if this file is present and
/// consistent, so writing it is the last step: a directory without a manifest is
/// an interrupted install, not a usable model.
public struct GodwitManifest: Codable, Sendable {
    public struct SectionSpan: Codable, Sendable, Equatable {
        public let offset: Int
        public let length: Int

        public init(offset: Int, length: Int) {
            self.offset = offset
            self.length = length
        }
    }

    public struct TrunkSection: Codable, Sendable {
        public let name: String
        public let offset: Int
        public let length: Int
        public let dtype: String
        public let shape: [Int]

        public init(name: String, offset: Int, length: Int, dtype: String, shape: [Int]) {
            self.name = name
            self.offset = offset
            self.length = length
            self.dtype = dtype
            self.shape = shape
        }
    }

    /// Bumped whenever the on-disk layout changes incompatibly.
    public static let currentFormatVersion = 1

    public let formatVersion: Int
    public let model: String
    public let revision: String
    public let spec: ArchitectureSpec
    public let layerCount: Int
    public let expertCount: Int
    /// Page-aligned bytes per expert within a layer file.
    public let expertStride: Int
    /// Offsets of each sub-tensor within one expert's stride.
    public let expertSections: [String: SectionSpan]
    public let trunkSections: [TrunkSection]
    public let bytesWritten: Int

    public init(model: String, revision: String, spec: ArchitectureSpec,
                layerCount: Int, expertCount: Int, expertStride: Int,
                expertSections: [String: SectionSpan],
                trunkSections: [TrunkSection], bytesWritten: Int) {
        self.formatVersion = Self.currentFormatVersion
        self.model = model
        self.revision = revision
        self.spec = spec
        self.layerCount = layerCount
        self.expertCount = expertCount
        self.expertStride = expertStride
        self.expertSections = expertSections
        self.trunkSections = trunkSections
        self.bytesWritten = bytesWritten
    }

    /// Whether every layer was installed, as opposed to a partial test install.
    public var isComplete: Bool { layerCount == spec.layerCount }

    public func layerFileSize() -> Int { expertStride * expertCount }

    /// Checks that files exist and are the size the manifest claims.
    ///
    /// Cheap structural validation, not a hash check — enough to catch an
    /// interrupted or truncated install before the runtime reads garbage.
    public func validate(at directory: URL) throws {
        let manager = FileManager.default
        let expected = layerFileSize()

        for layer in 0..<layerCount {
            let path = directory
                .appendingPathComponent("experts")
                .appendingPathComponent(String(format: "layer_%02d.bin", layer))
            guard let size = try manager.attributesOfItem(atPath: path.path)[.size] as? Int else {
                throw InstallError.verificationFailed("layer \(layer) file is unreadable")
            }
            guard size == expected else {
                throw InstallError.verificationFailed(
                    "layer \(layer) is \(size) bytes, expected \(expected)")
            }
        }

        let trunkPath = directory.appendingPathComponent("trunk.bin")
        guard let trunkSize = try manager.attributesOfItem(atPath: trunkPath.path)[.size] as? Int
        else { throw InstallError.verificationFailed("trunk.bin is unreadable") }
        if let last = trunkSections.max(by: { $0.offset < $1.offset }),
           trunkSize < last.offset + last.length {
            throw InstallError.verificationFailed(
                "trunk.bin is \(trunkSize) bytes, shorter than its last section")
        }
    }
}
