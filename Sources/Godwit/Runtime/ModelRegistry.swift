import Foundation

/// Finds installed models, and knows which ones can be installed.
///
/// A `.gwt` directory is self-describing — its manifest carries the whole
/// `ArchitectureSpec` — so discovery is a directory scan rather than a
/// configuration file the user has to keep in step with reality.
public struct ModelRegistry {
    /// A model already on disk and ready to load.
    public struct Installed: Sendable, Codable {
        public let name: String
        public let path: String
        public let model: String
        public let layers: Int
        public let experts: Int
        public let bytes: Int
        public let complete: Bool
        public let hasRangeMap: Bool
    }

    /// A model Godwit knows how to install.
    public struct Available: Sendable, Codable {
        public let id: String
        public let title: String
        public let layers: Int
        public let experts: Int
        /// Approximate installed size in bytes.
        public let bytes: Int
        public let note: String
    }

    /// What `install` accepts today.
    ///
    /// Deliberately a short, honest list rather than every MoE checkpoint on
    /// Hugging Face: these are the two that have actually been run end to end.
    public static let catalogue: [Available] = [
        Available(id: "openai/gpt-oss-20b", title: "GPT-OSS-20B",
                  layers: 24, experts: 32, bytes: 12_079_595_520,
                  note: "Faster, and fits comfortably. MLX runs it quicker still."),
        Available(id: "openai/gpt-oss-120b", title: "GPT-OSS-120B",
                  layers: 36, experts: 128, bytes: 63_278_346_240,
                  note: "The reason this project exists — it fits nowhere else."),
        Available(id: "Qwen/Qwen3-30B-A3B", title: "Qwen3-30B-A3B",
                  layers: 48, experts: 128, bytes: 16_900_000_000,
                  note: "A second family. Quantised here rather than shipped quantised."),
    ]

    public let root: URL

    public init(root: URL) {
        self.root = root
    }

    /// Scans for `.gwt` directories one level down.
    public func installed() -> [Installed] {
        let manager = FileManager.default
        guard let entries = try? manager.contentsOfDirectory(
            at: root, includingPropertiesForKeys: nil) else { return [] }

        return entries.compactMap { directory -> Installed? in
            let manifestURL = directory.appendingPathComponent("manifest.json")
            guard let data = try? Data(contentsOf: manifestURL),
                  let manifest = try? JSONDecoder().decode(GodwitManifest.self, from: data)
            else { return nil }

            // Size on disk, not the manifest's own claim — a partial install
            // should look partial.
            var bytes = 0
            if let walker = manager.enumerator(at: directory, includingPropertiesForKeys: [.fileSizeKey]) {
                for case let url as URL in walker {
                    bytes += (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
                }
            }

            return Installed(
                name: directory.lastPathComponent,
                path: directory.path,
                model: manifest.model,
                layers: manifest.layerCount,
                experts: manifest.expertCount,
                bytes: bytes,
                complete: manifest.isComplete,
                hasRangeMap: manager.fileExists(
                    atPath: directory.appendingPathComponent("range.json").path))
        }
        .sorted { $0.bytes < $1.bytes }
    }

    /// The spec matching a repository id, or nil if it is not one we support.
    public static func spec(for repository: String) -> ArchitectureSpec? {
        let id = repository.lowercased()
        if id.contains("gpt-oss-20b") { return .gptOSS20B }
        if id.contains("gpt-oss-120b") { return .gptOSS120B }
        if id.contains("qwen3-30b-a3b") { return .qwen3MoE30B }
        return nil
    }
}
