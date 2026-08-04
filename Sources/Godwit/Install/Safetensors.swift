import Foundation

/// Minimal safetensors reader: enough to locate byte ranges, not to load data.
///
/// The format is an 8-byte little-endian header length, a JSON header mapping
/// tensor names to dtype/shape/offsets, then the tensor data. Because every
/// tensor is contiguous and expert index is the leading dimension of GPT-OSS's
/// MoE weights, a single expert is one byte range — which is what lets the
/// installer work without ever materialising a shard.
public struct SafetensorsHeader: Sendable {
    public struct Entry: Sendable {
        public let dtype: String
        public let shape: [Int]
        public let begin: Int
        public let end: Int

        public var byteCount: Int { end - begin }
        /// Elements in one slice along the leading dimension.
        public var elementsPerLeadingIndex: Int {
            shape.dropFirst().reduce(1, *)
        }
    }

    /// Byte offset where tensor data begins, i.e. past the JSON header.
    public let dataStart: Int
    public let entries: [String: Entry]

    public init(headerJSON: Data, dataStart: Int) throws {
        self.dataStart = dataStart
        guard let root = try JSONSerialization.jsonObject(with: headerJSON) as? [String: Any] else {
            throw InstallError.malformedHeader("root is not an object")
        }
        var entries: [String: Entry] = [:]
        for (name, value) in root where name != "__metadata__" {
            guard let fields = value as? [String: Any],
                  let dtype = fields["dtype"] as? String,
                  let shape = fields["shape"] as? [Int],
                  let offsets = fields["data_offsets"] as? [Int], offsets.count == 2
            else {
                throw InstallError.malformedHeader("tensor '\(name)' is missing fields")
            }
            entries[name] = Entry(dtype: dtype, shape: shape,
                                  begin: offsets[0], end: offsets[1])
        }
        self.entries = entries
    }

    /// Absolute byte range of one slice along the leading dimension.
    ///
    /// For `gate_up_proj_blocks` with shape `[128, 5760, 90, 16]`, index 3 is
    /// expert 3's weights.
    public func range(of name: String, leadingIndex: Int) throws -> (offset: Int, length: Int) {
        guard let entry = entries[name] else { throw InstallError.missingTensor(name) }
        let itemSize = try Self.itemSize(of: entry.dtype)
        let span = entry.elementsPerLeadingIndex * itemSize
        return (dataStart + entry.begin + leadingIndex * span, span)
    }

    /// Absolute byte range of a whole tensor.
    public func range(of name: String) throws -> (offset: Int, length: Int) {
        guard let entry = entries[name] else { throw InstallError.missingTensor(name) }
        return (dataStart + entry.begin, entry.byteCount)
    }

    public func entry(_ name: String) throws -> Entry {
        guard let entry = entries[name] else { throw InstallError.missingTensor(name) }
        return entry
    }

    public static func itemSize(of dtype: String) throws -> Int {
        switch dtype {
        case "U8", "I8": return 1
        case "BF16", "F16", "I16": return 2
        case "F32", "I32": return 4
        case "F64", "I64": return 8
        default: throw InstallError.unsupportedDType(dtype)
        }
    }
}

public enum InstallError: Error, CustomStringConvertible {
    case malformedHeader(String)
    case missingTensor(String)
    case unsupportedDType(String)
    case unexpectedShape(tensor: String, shape: [Int])
    case transferFailed(url: String, status: Int)
    case shortTransfer(expected: Int, got: Int)
    case writeFailed(path: String, errno: Int32)
    case verificationFailed(String)

    public var description: String {
        switch self {
        case .malformedHeader(let why): return "malformed safetensors header: \(why)"
        case .missingTensor(let name): return "checkpoint has no tensor '\(name)'"
        case .unsupportedDType(let d): return "unsupported dtype '\(d)'"
        case .unexpectedShape(let t, let s): return "tensor '\(t)' has unexpected shape \(s)"
        case .transferFailed(let url, let status): return "HTTP \(status) for \(url)"
        case .shortTransfer(let want, let got): return "expected \(want) bytes, received \(got)"
        case .writeFailed(let path, let code):
            return "writing \(path) failed: \(String(cString: strerror(code)))"
        case .verificationFailed(let why): return "verification failed: \(why)"
        }
    }
}
