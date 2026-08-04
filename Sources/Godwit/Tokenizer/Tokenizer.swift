import Foundation

/// Byte-level BPE, loaded from the checkpoint's own `tokenizer.json`.
///
/// Reading the model's tokeniser rather than reimplementing a named one means
/// there is no second source of truth to drift: if the vocabulary in the file
/// disagrees with the weights, nothing else can save us anyway.
///
/// The scheme is the GPT-2 lineage. Text is split by a regex into pieces, each
/// piece's UTF-8 bytes are mapped into a printable-character alphabet so BPE
/// never has to reason about raw bytes, and merges are applied lowest-rank
/// first until none apply.
public final class Tokenizer {
    /// Special tokens, which bypass BPE entirely and are matched literally.
    public struct Special: Sendable {
        public let text: String
        public let id: Int
    }

    private let vocabulary: [String: Int]
    private let reverse: [Int: String]
    /// Merge pair -> rank. Lower merges first.
    private let ranks: [String: Int]
    private let specials: [Special]
    private let specialByText: [String: Int]
    private let pattern: NSRegularExpression

    /// Byte value -> the character standing in for it, and back.
    private let byteEncoder: [Character]
    private let byteDecoder: [Character: UInt8]

    public var count: Int { vocabulary.count + specials.count }

    public init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any],
              let vocab = model["vocab"] as? [String: Int]
        else { throw TokenizerError.malformed("model.vocab missing") }

        self.vocabulary = vocab
        var reverse: [Int: String] = [:]
        reverse.reserveCapacity(vocab.count)
        for (token, id) in vocab { reverse[id] = token }

        // Merges are either ["a", "b"] pairs or "a b" strings depending on the
        // file's vintage; accept both.
        var ranks: [String: Int] = [:]
        if let merges = model["merges"] as? [[String]] {
            ranks.reserveCapacity(merges.count)
            for (rank, pair) in merges.enumerated() where pair.count == 2 {
                ranks[pair[0] + "\u{0}" + pair[1]] = rank
            }
        } else if let merges = model["merges"] as? [String] {
            for (rank, line) in merges.enumerated() {
                let parts = line.split(separator: " ", maxSplits: 1)
                if parts.count == 2 {
                    ranks[String(parts[0]) + "\u{0}" + String(parts[1])] = rank
                }
            }
        }
        guard !ranks.isEmpty else { throw TokenizerError.malformed("model.merges missing") }
        self.ranks = ranks

        var specials: [Special] = []
        for entry in (root["added_tokens"] as? [[String: Any]]) ?? [] {
            if let content = entry["content"] as? String, let id = entry["id"] as? Int {
                specials.append(Special(text: content, id: id))
                reverse[id] = content
            }
        }
        // Longest first, so <|start|> never matches inside a longer token.
        self.specials = specials.sorted { $0.text.count > $1.text.count }
        self.specialByText = Dictionary(uniqueKeysWithValues: specials.map { ($0.text, $0.id) })
        self.reverse = reverse

        let source = Self.extractPattern(root) ?? Self.defaultPattern
        self.pattern = try NSRegularExpression(pattern: source, options: [])

        let (encoder, decoder) = Self.byteLevelAlphabet()
        self.byteEncoder = encoder
        self.byteDecoder = decoder
    }

    // MARK: - Encoding

    public func encode(_ text: String) -> [Int] {
        var output: [Int] = []
        var remaining = Substring(text)

        // Specials are matched literally before anything else sees the text.
        while !remaining.isEmpty {
            var matched: (range: Range<Substring.Index>, id: Int)?
            for special in specials {
                if let found = remaining.range(of: special.text) {
                    if matched == nil || found.lowerBound < matched!.range.lowerBound {
                        matched = (found, special.id)
                    }
                }
            }
            guard let matched else {
                output.append(contentsOf: encodeOrdinary(String(remaining)))
                break
            }
            if matched.range.lowerBound > remaining.startIndex {
                output.append(contentsOf:
                    encodeOrdinary(String(remaining[remaining.startIndex..<matched.range.lowerBound])))
            }
            output.append(matched.id)
            remaining = remaining[matched.range.upperBound...]
        }
        return output
    }

    private func encodeOrdinary(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }
        var output: [Int] = []
        let full = NSRange(text.startIndex..., in: text)

        pattern.enumerateMatches(in: text, options: [], range: full) { match, _, _ in
            guard let match, let range = Range(match.range, in: text) else { return }
            let piece = String(text[range])
            // Bytes into the stand-in alphabet, then merge.
            let mapped = String(Array(piece.utf8).map { byteEncoder[Int($0)] })
            for token in merge(mapped) {
                if let id = vocabulary[token] {
                    output.append(id)
                } else {
                    // Unknown only happens if the vocabulary is inconsistent with
                    // its own alphabet; fall back to single characters.
                    for character in token {
                        if let id = vocabulary[String(character)] { output.append(id) }
                    }
                }
            }
        }
        return output
    }

    /// Applies BPE merges to one pre-tokenised piece, lowest rank first.
    private func merge(_ piece: String) -> [String] {
        var parts = piece.map(String.init)
        guard parts.count > 1 else { return parts }

        while parts.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for index in 0..<(parts.count - 1) {
                if let rank = ranks[parts[index] + "\u{0}" + parts[index + 1]], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }
            guard bestIndex >= 0 else { break }
            parts[bestIndex] += parts[bestIndex + 1]
            parts.remove(at: bestIndex + 1)
        }
        return parts
    }

    // MARK: - Decoding

    public func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            guard let token = reverse[id] else { continue }
            if specialByText[token] != nil {
                bytes.append(contentsOf: Array(token.utf8))
            } else {
                for character in token {
                    if let byte = byteDecoder[character] { bytes.append(byte) }
                }
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    public func id(of special: String) -> Int? { specialByText[special] }

    // MARK: - Byte-level alphabet

    /// Maps all 256 byte values onto printable characters.
    ///
    /// Printable ASCII and Latin-1 stand for themselves; everything else moves
    /// into the private range at 256+. Without this, BPE would have to merge
    /// over raw bytes including whitespace and control codes.
    private static func byteLevelAlphabet() -> ([Character], [Character: UInt8]) {
        var byteValues: [Int] = []
        byteValues.append(contentsOf: Int(Character("!").asciiValue!)...Int(Character("~").asciiValue!))
        byteValues.append(contentsOf: 0xA1...0xAC)
        byteValues.append(contentsOf: 0xAE...0xFF)

        var codePoints = byteValues
        var extra = 0
        for byte in 0..<256 where !byteValues.contains(byte) {
            byteValues.append(byte)
            codePoints.append(256 + extra)
            extra += 1
        }

        var encoder = [Character](repeating: " ", count: 256)
        var decoder: [Character: UInt8] = [:]
        for (byte, code) in zip(byteValues, codePoints) {
            let character = Character(UnicodeScalar(code)!)
            encoder[byte] = character
            decoder[character] = UInt8(byte)
        }
        return (encoder, decoder)
    }

    private static func extractPattern(_ root: [String: Any]) -> String? {
        func search(_ node: Any) -> String? {
            if let dictionary = node as? [String: Any] {
                if let regex = (dictionary["pattern"] as? [String: Any])?["Regex"] as? String {
                    return regex
                }
                for value in dictionary.values {
                    if let found = search(value) { return found }
                }
            } else if let array = node as? [Any] {
                for value in array {
                    if let found = search(value) { return found }
                }
            }
            return nil
        }
        return search(root)
    }

    /// The o200k split pattern, if the file does not carry one.
    private static let defaultPattern =
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
        + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
        + "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
        + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
        + "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
}

public enum TokenizerError: Error, CustomStringConvertible {
    case malformed(String)
    case notFound(String)

    public var description: String {
        switch self {
        case .malformed(let why): return "tokenizer.json is malformed: \(why)"
        case .notFound(let path): return "no tokenizer.json at \(path)"
        }
    }
}
