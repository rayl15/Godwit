import Foundation

/// Builds a `.gwt` installation by streaming the checkpoint and repacking it.
///
/// Two rules shape the whole thing:
///
/// **Never materialise a shard.** Peak memory is roughly one expert slice
/// regardless of how large the checkpoint is, matching the rule the runtime
/// follows at decode time.
///
/// **Never re-quantise the experts.** MXFP4 bytes are copied through untouched;
/// only their arrangement changes. Round-tripping 56.7 GiB through a float
/// representation would be slow and would lose precision for no reason. The
/// trunk is different — it arrives as BF16 and must be quantised, which is a
/// deliberate lossy step measured in `Scripts/analysis/requant_quality.py`.
public struct Installer {
    public struct Progress: Sendable {
        public let stage: String
        public let completed: Int
        public let total: Int
        public let bytesWritten: Int
    }

    public struct Options: Sendable {
        public let repository: String
        public let revision: String
        /// Install only the first N layers. For testing the pipeline without a
        /// 60 GiB transfer; a real install uses all of them.
        public let layerLimit: Int?

        public init(repository: String = "openai/gpt-oss-120b",
                    revision: String = "main",
                    layerLimit: Int? = nil) {
            self.repository = repository
            self.revision = revision
            self.layerLimit = layerLimit
        }

        var base: URL {
            URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)")!
        }
    }

    public let spec: ArchitectureSpec
    public let options: Options
    private let streamer = RangeStreamer()

    public init(spec: ArchitectureSpec = .gptOSS120B, options: Options = Options()) {
        self.spec = spec
        self.options = options
    }

    // MARK: - Checkpoint index

    /// Tensor name -> shard, plus lazily fetched per-shard headers.
    private actor Catalog {
        let base: URL
        let streamer: RangeStreamer
        let weightMap: [String: String]
        private var headers: [String: SafetensorsHeader] = [:]

        init(base: URL, streamer: RangeStreamer) async throws {
            self.base = base
            self.streamer = streamer
            let data = try await streamer.fetch(url: base.appendingPathComponent("model.safetensors.index.json"))
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let map = root["weight_map"] as? [String: String]
            else { throw InstallError.malformedHeader("index has no weight_map") }
            self.weightMap = map
        }

        func header(forTensor name: String) async throws -> (URL, SafetensorsHeader) {
            guard let shard = weightMap[name] else { throw InstallError.missingTensor(name) }
            let url = base.appendingPathComponent(shard)
            if let cached = headers[shard] { return (url, cached) }

            let prefix = try await streamer.fetch(url: url, offset: 0, length: 8)
            let headerLength = prefix.withUnsafeBytes {
                Int($0.loadUnaligned(as: UInt64.self).littleEndian)
            }
            let json = try await streamer.fetch(url: url, offset: 8, length: headerLength)
            let header = try SafetensorsHeader(headerJSON: json, dataStart: 8 + headerLength)
            headers[shard] = header
            return (url, header)
        }
    }

    // MARK: - Install

    public func install(
        to directory: URL,
        progress: @Sendable (Progress) -> Void = { _ in }
    ) async throws -> GodwitManifest {
        let layerCount = min(options.layerLimit ?? spec.layerCount, spec.layerCount)
        let expertLayout = ExpertLayout(hiddenSize: spec.hiddenSize,
                                        intermediateSize: spec.intermediateSize,
                                        expertCount: spec.layers[0].routedExpertCount)

        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("experts"),
            withIntermediateDirectories: true)

        let catalog = try await Catalog(base: options.base, streamer: streamer)
        var bytesWritten = 0

        for layer in 0..<layerCount {
            let path = directory
                .appendingPathComponent("experts")
                .appendingPathComponent(String(format: "layer_%02d.bin", layer))
                .path
            let writer = try PositionalWriter(path: path, size: expertLayout.layerFileSize)

            for section in ExpertLayout.Section.allCases {
                let tensor = section.tensorName(layer: layer)
                let (url, header) = try await catalog.header(forTensor: tensor)
                let entry = try header.entry(tensor)

                guard entry.shape.first == expertLayout.expertCount else {
                    throw InstallError.unexpectedShape(tensor: tensor, shape: entry.shape)
                }
                let itemSize = try SafetensorsHeader.itemSize(of: entry.dtype)
                let sliceLength = entry.elementsPerLeadingIndex * itemSize
                guard sliceLength == expertLayout.size(of: section) else {
                    throw InstallError.verificationFailed(
                        "\(tensor): source slice \(sliceLength) bytes, layout expects "
                        + "\(expertLayout.size(of: section))")
                }

                let range = try header.range(of: tensor)
                try await streamer.stream(
                    url: url, offset: range.offset, length: range.length,
                    sliceLength: sliceLength
                ) { expert, bytes in
                    try writer.write(bytes,
                                     at: expertLayout.offset(expert: expert, section: section))
                }
                bytesWritten += range.length
            }

            try writer.sync()
            progress(Progress(stage: "experts", completed: layer + 1,
                              total: layerCount, bytesWritten: bytesWritten))
        }

        let trunk = try await installTrunk(to: directory, catalog: catalog,
                                           layerCount: layerCount, progress: progress)
        bytesWritten += trunk.totalSize

        let manifest = GodwitManifest(
            model: options.repository,
            revision: options.revision,
            spec: spec,
            layerCount: layerCount,
            expertCount: expertLayout.expertCount,
            expertStride: expertLayout.stride,
            expertSections: expertLayout.offsets.reduce(into: [:]) { result, pair in
                result[pair.key.rawValue] = GodwitManifest.SectionSpan(
                    offset: pair.value, length: expertLayout.size(of: pair.key))
            },
            trunkSections: trunk.sections.map {
                GodwitManifest.TrunkSection(name: $0.name, offset: $0.offset,
                                            length: $0.length, dtype: $0.dtype, shape: $0.shape)
            },
            bytesWritten: bytesWritten)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest)
            .write(to: directory.appendingPathComponent("manifest.json"))
        return manifest
    }

    // MARK: - Trunk

    /// Downloads the BF16 trunk and quantises it to int8 on the way to disk.
    ///
    /// Rows are processed in bounded chunks rather than whole tensors: the
    /// embedding alone is 1.16 GiB in BF16, and holding it would break the same
    /// rule the expert path is careful to keep.
    private func installTrunk(
        to directory: URL,
        catalog: Catalog,
        layerCount: Int,
        progress: @Sendable (Progress) -> Void
    ) async throws -> TrunkLayout {
        var plan = TrunkLayout()
        var quantized: [(name: String, tensor: String, rows: Int, cols: Int)] = []
        var copied: [(name: String, tensor: String)] = []

        for (name, tensor) in [("embed", "model.embed_tokens.weight"),
                               ("head", "lm_head.weight")] {
            let (_, header) = try await catalog.header(forTensor: tensor)
            let shape = try header.entry(tensor).shape
            plan.append(name: name,
                        length: TrunkLayout.quantizedSize(rows: shape[0], cols: shape[1]),
                        dtype: "int8-affine64", shape: shape)
            quantized.append((name, tensor, shape[0], shape[1]))
        }

        for layer in 0..<layerCount {
            for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                let tensor = "model.layers.\(layer).self_attn.\(projection).weight"
                let (_, header) = try await catalog.header(forTensor: tensor)
                let shape = try header.entry(tensor).shape
                let name = "layer\(layer).\(projection)"
                plan.append(name: name,
                            length: TrunkLayout.quantizedSize(rows: shape[0], cols: shape[1]),
                            dtype: "int8-affine64", shape: shape)
                quantized.append((name, tensor, shape[0], shape[1]))
            }
            // Small, and each feeds something precision-sensitive: norms scale
            // everything downstream, the router picks experts, and sinks sit
            // inside a softmax. Copied as BF16 rather than quantised.
            for (suffix, tensor) in [
                ("input_norm", "model.layers.\(layer).input_layernorm.weight"),
                ("post_norm", "model.layers.\(layer).post_attention_layernorm.weight"),
                ("router_w", "model.layers.\(layer).mlp.router.weight"),
                ("router_b", "model.layers.\(layer).mlp.router.bias"),
                ("sinks", "model.layers.\(layer).self_attn.sinks"),
            ] {
                let (_, header) = try await catalog.header(forTensor: tensor)
                let entry = try header.entry(tensor)
                plan.append(name: "layer\(layer).\(suffix)", length: entry.byteCount,
                            dtype: "bf16", shape: entry.shape)
                copied.append(("layer\(layer).\(suffix)", tensor))
            }
        }
        let (_, normHeader) = try await catalog.header(forTensor: "model.norm.weight")
        let normEntry = try normHeader.entry("model.norm.weight")
        plan.append(name: "final_norm", length: normEntry.byteCount,
                    dtype: "bf16", shape: normEntry.shape)
        copied.append(("final_norm", "model.norm.weight"))

        let writer = try PositionalWriter(
            path: directory.appendingPathComponent("trunk.bin").path, size: plan.totalSize)

        var done = 0
        let total = quantized.count + copied.count

        for item in quantized {
            guard let section = plan.section(item.name) else { continue }
            let (url, header) = try await catalog.header(forTensor: item.tensor)
            try await quantizeTensor(
                url: url, header: header, tensor: item.tensor,
                rows: item.rows, cols: item.cols,
                writer: writer, base: section.offset)
            done += 1
            progress(Progress(stage: "trunk", completed: done, total: total,
                              bytesWritten: plan.totalSize))
        }

        for item in copied {
            guard let section = plan.section(item.name) else { continue }
            let (url, header) = try await catalog.header(forTensor: item.tensor)
            let range = try header.range(of: item.tensor)
            let data = try await streamer.fetch(url: url, offset: range.offset, length: range.length)
            try writer.write(data, at: section.offset)
            done += 1
            progress(Progress(stage: "trunk", completed: done, total: total,
                              bytesWritten: plan.totalSize))
        }

        try writer.sync()
        return plan
    }

    /// Rows to quantise per network request. At 2880-5760 columns this keeps a
    /// chunk in the low tens of MiB.
    private static let quantizeRowChunk = 4096

    private func quantizeTensor(
        url: URL, header: SafetensorsHeader, tensor: String,
        rows: Int, cols: Int, writer: PositionalWriter, base: Int
    ) async throws {
        let entry = try header.entry(tensor)
        guard entry.dtype == "BF16" else { throw InstallError.unsupportedDType(entry.dtype) }
        let range = try header.range(of: tensor)
        let rowBytes = cols * 2
        let groupsPerRow = cols / Int8Affine.groupSize

        // Codes first, then scale/zero pairs, so the GPU can bind each as a
        // contiguous region without an interleaved stride.
        let codesBase = base
        let metaBase = base + rows * cols

        var row = 0
        while row < rows {
            let take = min(Self.quantizeRowChunk, rows - row)
            let data = try await streamer.fetch(url: url,
                                                offset: range.offset + row * rowBytes,
                                                length: take * rowBytes)
            var codes = [UInt8](repeating: 0, count: take * cols)
            var meta = [UInt16](repeating: 0, count: take * groupsPerRow * 2)

            data.withUnsafeBytes { raw in
                for local in 0..<take {
                    let values = BFloat16.decode(
                        UnsafeRawBufferPointer(rebasing: raw[(local * rowBytes)..<((local + 1) * rowBytes)]),
                        count: cols)
                    let (rowCodes, groups) = Int8Affine.quantize(row: values)
                    codes.replaceSubrange((local * cols)..<((local + 1) * cols), with: rowCodes)
                    for (g, group) in groups.enumerated() {
                        let index = (local * groupsPerRow + g) * 2
                        meta[index] = BFloat16.fromFloat(group.scale)
                        meta[index + 1] = BFloat16.fromFloat(group.zero)
                    }
                }
            }

            try codes.withUnsafeBytes { try writer.write($0, at: codesBase + row * cols) }
            try meta.withUnsafeBytes {
                try writer.write($0, at: metaBase + row * groupsPerRow * 4)
            }
            row += take
        }
    }
}
