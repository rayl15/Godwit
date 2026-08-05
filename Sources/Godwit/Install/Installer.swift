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
    fileprivate actor Catalog {
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
                                        expertCount: spec.layers[0].routedExpertCount,
                                        expertBias: spec.expertBias)

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

            if spec.naming.expertLayout == .perExpertTensors {
                bytesWritten += try await installPerExpertLayer(
                    layer: layer, writer: writer, layout: expertLayout, catalog: catalog)
                try writer.sync()
                progress(Progress(stage: "experts", completed: layer + 1,
                                  total: layerCount, bytesWritten: bytesWritten))
                continue
            }

            for section in ExpertLayout.Section.allCases
            where expertLayout.size(of: section) > 0 {
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

        // The tokeniser travels with the weights. Reading the checkpoint's own
        // tokenizer.json means there is no second source of truth to drift.
        let tokenizerData = try await streamer.fetch(
            url: options.base.appendingPathComponent("tokenizer.json"))
        try tokenizerData.write(to: directory.appendingPathComponent("tokenizer.json"))

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

        let naming = spec.naming
        var headTensors: [(String, String)] = [("embed", naming.embedding)]
        if let head = naming.outputHead { headTensors.append(("head", head)) }

        for (name, tensor) in headTensors {
            let (_, header) = try await catalog.header(forTensor: tensor)
            let shape = try header.entry(tensor).shape
            plan.append(name: name,
                        length: TrunkLayout.quantizedSize(rows: shape[0], cols: shape[1]),
                        dtype: "int8-affine64", shape: shape)
            quantized.append((name, tensor, shape[0], shape[1]))
        }

        for layer in 0..<layerCount {
            for (projection, template) in [
                ("q_proj", naming.queryProjection), ("k_proj", naming.keyProjection),
                ("v_proj", naming.valueProjection), ("o_proj", naming.outputProjection),
            ] {
                let tensor = naming.resolve(template, layer: layer)
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
            // Small, and each feeds something precision-sensitive. Which of
            // these exist is a property of the family: GPT-OSS has biases and
            // sinks and no QK-norm, Qwen3 the exact reverse. A nil name means
            // the tensor is not in the checkpoint, so asking for it would 404.
            var small: [(String, String)] = [
                ("input_norm", naming.resolve(naming.inputNorm, layer: layer)),
                ("post_norm", naming.resolve(naming.postAttentionNorm, layer: layer)),
                ("router_w", naming.resolve(naming.router, layer: layer)),
            ]
            for (suffix, template) in [
                ("q_bias", naming.queryBias), ("k_bias", naming.keyBias),
                ("v_bias", naming.valueBias), ("o_bias", naming.outputBias),
                ("router_b", naming.routerBias), ("sinks", naming.attentionSinks),
                ("q_norm", naming.queryNorm), ("k_norm", naming.keyNorm),
            ] {
                guard let template else { continue }
                small.append((suffix, naming.resolve(template, layer: layer)))
            }

            for (suffix, tensor) in small {
                let (_, header) = try await catalog.header(forTensor: tensor)
                let entry = try header.entry(tensor)
                plan.append(name: "layer\(layer).\(suffix)", length: entry.byteCount,
                            dtype: "bf16", shape: entry.shape)
                copied.append(("layer\(layer).\(suffix)", tensor))
            }
        }
        let (_, normHeader) = try await catalog.header(forTensor: naming.finalNorm)
        let normEntry = try normHeader.entry(naming.finalNorm)
        plan.append(name: "final_norm", length: normEntry.byteCount,
                    dtype: "bf16", shape: normEntry.shape)
        copied.append(("final_norm", naming.finalNorm))

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

// MARK: - Per-expert checkpoints

extension Installer {
    /// Experts fetched concurrently. One expert is three round trips of a
    /// couple of megabytes, and at 18,432 tensors the run is dominated by
    /// latency rather than bandwidth: serially this measured 9.9 minutes per
    /// layer, which is eight hours for the model. Eight at a time holds peak
    /// memory near 80 MB, still honouring the rule that nothing here scales
    /// with checkpoint size.
    static let expertFetchWidth = 8

    /// Installs one layer's experts from a checkpoint that stores them
    /// separately, three tensors per expert.
    ///
    /// GPT-OSS hands over a layer's experts as two stacked tensors already in
    /// MXFP4, so the installer slices and copies. Qwen3 hands over
    /// `mlp.experts.<n>.{gate,up,down}_proj` in BF16, so contiguity has to be
    /// built rather than inherited and the bytes quantised on the way past.
    fileprivate func installPerExpertLayer(
        layer: Int,
        writer: PositionalWriter,
        layout: ExpertLayout,
        catalog: Catalog
    ) async throws -> Int {
        let naming = spec.naming
        let hidden = spec.hiddenSize
        let inner = spec.intermediateSize
        let streamer = self.streamer
        let counter = Counter()

        try await withThrowingTaskGroup(of: Void.self) { group in
            var inFlight = 0
            for expert in 0..<layout.expertCount {
                if inFlight >= Self.expertFetchWidth {
                    try await group.next()
                    inFlight -= 1
                }
                inFlight += 1
                group.addTask {
                    let read = try await installOneExpert(
                        layer: layer, expert: expert, naming: naming,
                        hidden: hidden, inner: inner, layout: layout,
                        writer: writer, catalog: catalog, streamer: streamer)
                    counter.add(read)
                }
            }
            try await group.waitForAll()
        }
        return counter.value
    }
}

/// Fetches, interleaves, quantises and writes one expert.
///
/// Free rather than a method so a task group can run several without sending
/// the whole `Installer`.
private func installOneExpert(
    layer: Int, expert: Int, naming: TensorNaming,
    hidden: Int, inner: Int, layout: ExpertLayout,
    writer: PositionalWriter, catalog: Installer.Catalog, streamer: RangeStreamer
) async throws -> Int {
    async let gateData = fetchNamedTensor(
        naming.resolve(naming.expertGate, layer: layer, expert: expert), catalog, streamer)
    async let upData = fetchNamedTensor(
        naming.resolve(naming.expertUp, layer: layer, expert: expert), catalog, streamer)
    async let downData = fetchNamedTensor(
        naming.resolve(naming.expertDown, layer: layer, expert: expert), catalog, streamer)

    let (gateRaw, gateShape) = try await gateData
    let (upRaw, upShape) = try await upData
    let (downRaw, downShape) = try await downData

    guard gateShape == [inner, hidden], upShape == [inner, hidden] else {
        throw InstallError.unexpectedShape(
            tensor: naming.resolve(naming.expertGate, layer: layer, expert: expert),
            shape: gateShape)
    }
    guard downShape == [hidden, inner] else {
        throw InstallError.unexpectedShape(
            tensor: naming.resolve(naming.expertDown, layer: layer, expert: expert),
            shape: downShape)
    }

    let gate = gateRaw.withUnsafeBytes { BFloat16.decode($0, count: inner * hidden) }
    let up = upRaw.withUnsafeBytes { BFloat16.decode($0, count: inner * hidden) }
    let down = downRaw.withUnsafeBytes { BFloat16.decode($0, count: hidden * inner) }

    // The kernel reads gate and up interleaved row-wise — gate 0, up 0, gate 1,
    // up 1 — because that is how GPT-OSS ships them and how the expert kernel
    // indexes. Qwen3 keeps them apart, so interleave here rather than teach the
    // kernel a second order. Getting this backwards is the failure that runs
    // fine and produces nonsense: measured, the swap costs 21 dB.
    var gateUp = [Float](repeating: 0, count: 2 * inner * hidden)
    for row in 0..<inner {
        let source = row * hidden
        gateUp.replaceSubrange((2 * row) * hidden..<(2 * row + 1) * hidden,
                               with: gate[source..<source + hidden])
        gateUp.replaceSubrange((2 * row + 1) * hidden..<(2 * row + 2) * hidden,
                               with: up[source..<source + hidden])
    }

    try writeQuantised(gateUp, rows: 2 * inner, cols: hidden, expert: expert,
                       blocks: .gateUpBlocks, scales: .gateUpScales,
                       layout: layout, writer: writer)
    try writeQuantised(down, rows: hidden, cols: inner, expert: expert,
                       blocks: .downBlocks, scales: .downScales,
                       layout: layout, writer: writer)
    return gateRaw.count + upRaw.count + downRaw.count
}

/// Quantises a row-major matrix to MXFP4 and writes both streams.
///
/// Per row rather than across the whole matrix: a block must not span two rows,
/// or a row's last partial block would share an exponent with the next row.
private func writeQuantised(
    _ values: [Float], rows: Int, cols: Int, expert: Int,
    blocks: ExpertLayout.Section, scales: ExpertLayout.Section,
    layout: ExpertLayout, writer: PositionalWriter
) throws {
    let blocksPerRow = cols / MXFP4.blockSize
    var packed = [UInt8](repeating: 0, count: rows * blocksPerRow * MXFP4.packedBytesPerBlock)
    var scaleBytes = [UInt8](repeating: 127, count: rows * blocksPerRow)

    for row in 0..<rows {
        let (rowPacked, rowScales) = MXFP4.encode(Array(values[row * cols..<(row + 1) * cols]))
        let packedStart = row * blocksPerRow * MXFP4.packedBytesPerBlock
        packed.replaceSubrange(packedStart..<packedStart + rowPacked.count, with: rowPacked)
        let scaleStart = row * blocksPerRow
        scaleBytes.replaceSubrange(scaleStart..<scaleStart + rowScales.count, with: rowScales)
    }

    try packed.withUnsafeBytes {
        try writer.write($0, at: layout.offset(expert: expert, section: blocks))
    }
    try scaleBytes.withUnsafeBytes {
        try writer.write($0, at: layout.offset(expert: expert, section: scales))
    }
}

/// Fetches one whole tensor and reports its shape.
private func fetchNamedTensor(
    _ name: String, _ catalog: Installer.Catalog, _ streamer: RangeStreamer
) async throws -> (Data, [Int]) {
    let (url, header) = try await catalog.header(forTensor: name)
    let entry = try header.entry(name)
    let range = try header.range(of: name)
    let data = try await streamer.fetch(url: url, offset: range.offset, length: range.length)
    return (data, entry.shape)
}

/// A counter that several fetches can add to.
private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var total = 0
    var value: Int { lock.lock(); defer { lock.unlock() }; return total }
    func add(_ amount: Int) { lock.lock(); total += amount; lock.unlock() }
}
