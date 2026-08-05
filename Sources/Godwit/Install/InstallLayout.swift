import Foundation

/// Where every byte lands in a `.gwt` installation.
///
/// The layout exists to serve one access pattern: at decode time the runtime
/// knows a layer and an expert id and must turn that into a single `pread` with
/// no indirection. So experts sit at a fixed, page-aligned stride within a
/// per-layer file, and each expert's six sub-tensors sit at fixed offsets within
/// that stride.
///
/// Page alignment costs a little padding — the six sections do not sum to a
/// page multiple — and buys reads that never straddle a page boundary
/// unnecessarily.
public struct ExpertLayout: Sendable, Equatable {
    public static let pageSize = 16384

    /// Sub-tensors of one expert, in the order they are stored.
    public enum Section: String, CaseIterable, Sendable {
        case gateUpBlocks = "gate_up_blocks"
        case gateUpScales = "gate_up_scales"
        case downBlocks = "down_blocks"
        case downScales = "down_scales"
        case gateUpBias = "gate_up_bias"
        case downBias = "down_bias"

        /// Name of the corresponding safetensors tensor for `layer`.
        public func tensorName(layer: Int) -> String {
            let field: String
            switch self {
            case .gateUpBlocks: field = "gate_up_proj_blocks"
            case .gateUpScales: field = "gate_up_proj_scales"
            case .downBlocks: field = "down_proj_blocks"
            case .downScales: field = "down_proj_scales"
            case .gateUpBias: field = "gate_up_proj_bias"
            case .downBias: field = "down_proj_bias"
            }
            return "model.layers.\(layer).mlp.experts.\(field)"
        }
    }

    public let hiddenSize: Int
    public let intermediateSize: Int
    public let expertCount: Int
    /// Byte offset of each section within one expert's stride.
    public let offsets: [Section: Int]
    public let sizes: [Section: Int]
    /// Page-aligned bytes per expert.
    public let stride: Int

    public init(hiddenSize: Int, intermediateSize: Int, expertCount: Int,
                expertBias: Bool = true) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.expertCount = expertCount

        let gateUpRows = 2 * intermediateSize
        let downRows = hiddenSize
        let blocksPerRow = hiddenSize / MXFP4.blockSize

        let sizes: [Section: Int] = [
            .gateUpBlocks: gateUpRows * blocksPerRow * MXFP4.packedBytesPerBlock,
            .gateUpScales: gateUpRows * blocksPerRow,
            .downBlocks: downRows * (intermediateSize / MXFP4.blockSize) * MXFP4.packedBytesPerBlock,
            .downScales: downRows * (intermediateSize / MXFP4.blockSize),
            // Qwen3 has no expert biases. Zero-length rather than absent keeps
            // the section list the same shape for every family, so the reader
            // needs no special case — it just never reads them.
            .gateUpBias: expertBias ? gateUpRows * 2 : 0,        // BF16
            .downBias: expertBias ? downRows * 2 : 0,            // BF16
        ]

        // Each section starts page-aligned so a read of one never shares a page
        // with a section the caller did not ask for.
        var offsets: [Section: Int] = [:]
        var cursor = 0
        for section in Section.allCases {
            guard sizes[section]! > 0 else { offsets[section] = cursor; continue }
            cursor = Self.alignUp(cursor)
            offsets[section] = cursor
            cursor += sizes[section]!
        }
        self.offsets = offsets
        self.sizes = sizes
        self.stride = Self.alignUp(cursor)
    }

    public static func alignUp(_ value: Int) -> Int {
        (value + pageSize - 1) / pageSize * pageSize
    }

    /// Byte offset of `expert` within its layer file.
    public func expertOffset(_ expert: Int) -> Int {
        precondition(expert >= 0 && expert < expertCount, "expert \(expert) out of range")
        return expert * stride
    }

    /// Byte offset of one section of one expert within its layer file.
    public func offset(expert: Int, section: Section) -> Int {
        expertOffset(expert) + offsets[section]!
    }

    public func size(of section: Section) -> Int { sizes[section]! }

    /// Total bytes in one layer file.
    public var layerFileSize: Int { stride * expertCount }

    /// Bytes of real data per expert, excluding alignment padding.
    public var payloadPerExpert: Int { sizes.values.reduce(0, +) }

    /// Fraction of the stride lost to alignment.
    public var paddingOverhead: Double {
        1 - Double(payloadPerExpert) / Double(stride)
    }
}

/// Trunk tensors, quantised to int8 or copied as BF16.
///
/// Unlike experts these are read once at load time and stay resident, so they
/// are laid out for simplicity rather than for a hot access pattern.
public struct TrunkLayout: Sendable {
    public struct Section: Sendable {
        public let name: String
        public let offset: Int
        public let length: Int
        public let dtype: String
        public let shape: [Int]
    }

    public private(set) var sections: [Section] = []
    public private(set) var totalSize = 0

    public init() {}

    public mutating func append(name: String, length: Int, dtype: String, shape: [Int]) {
        totalSize = ExpertLayout.alignUp(totalSize)
        sections.append(Section(name: name, offset: totalSize,
                                length: length, dtype: dtype, shape: shape))
        totalSize += length
    }

    public func section(_ name: String) -> Section? {
        sections.first { $0.name == name }
    }

    /// Bytes an int8-quantised `[rows, cols]` tensor occupies, including its
    /// BF16 scale and zero per group.
    public static func quantizedSize(rows: Int, cols: Int) -> Int {
        let groups = rows * (cols / Int8Affine.groupSize)
        return rows * cols + groups * 2 * 2
    }
}
