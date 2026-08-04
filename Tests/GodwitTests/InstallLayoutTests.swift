import Foundation
import Testing

@testable import Godwit

@Suite("Install layout")
struct InstallLayoutTests {
    let layout = ExpertLayout(hiddenSize: 2880, intermediateSize: 2880, expertCount: 128)

    @Test("Section sizes match the checkpoint's tensor shapes")
    func sectionSizes() {
        // Read off the real safetensors header: gate_up is [128, 5760, 90, 16],
        // down is [128, 2880, 90, 16], scales drop the trailing 16.
        #expect(layout.size(of: .gateUpBlocks) == 5760 * 90 * 16)
        #expect(layout.size(of: .gateUpScales) == 5760 * 90)
        #expect(layout.size(of: .downBlocks) == 2880 * 90 * 16)
        #expect(layout.size(of: .downScales) == 2880 * 90)
        #expect(layout.size(of: .gateUpBias) == 5760 * 2)
        #expect(layout.size(of: .downBias) == 2880 * 2)
    }

    @Test("Payload reproduces the 12.62 MiB expert")
    func expertSize() {
        let mib = Double(layout.payloadPerExpert) / 1_048_576
        #expect(mib > 12.6 && mib < 12.7, "got \(mib) MiB")
    }

    @Test("Every section is page-aligned")
    func alignment() {
        for section in ExpertLayout.Section.allCases {
            let offset = layout.offset(expert: 0, section: section)
            #expect(offset % ExpertLayout.pageSize == 0, "\(section) at \(offset)")
        }
        #expect(layout.stride % ExpertLayout.pageSize == 0)
    }

    @Test("Padding overhead stays small")
    func paddingIsCheap() {
        // Six page-aligned sections can waste at most 6 pages out of a ~12.6 MiB
        // stride. Anything more means the layout has drifted.
        #expect(layout.paddingOverhead < 0.01, "got \(layout.paddingOverhead)")
    }

    @Test("Experts never overlap")
    func expertsDisjoint() {
        for expert in 0..<layout.expertCount {
            let start = layout.expertOffset(expert)
            #expect(start + layout.payloadPerExpert <= start + layout.stride)
            if expert > 0 {
                #expect(start >= layout.expertOffset(expert - 1) + layout.stride)
            }
        }
        #expect(layout.layerFileSize == layout.stride * 128)
    }

    @Test("Whole-model size lands on the estimate")
    func totalSize() {
        let gib = Double(layout.layerFileSize) * 36 / 1_073_741_824
        // ESTIMATE.md says 56.7 GiB of payload; alignment adds a little.
        #expect(gib > 56 && gib < 58, "got \(gib) GiB")
    }

    @Test("Trunk sections are aligned and ordered")
    func trunkLayout() {
        var plan = TrunkLayout()
        plan.append(name: "embed", length: 100, dtype: "int8-affine64", shape: [10, 10])
        plan.append(name: "head", length: 200, dtype: "int8-affine64", shape: [20, 10])

        #expect(plan.section("embed")?.offset == 0)
        #expect(plan.section("head")?.offset == ExpertLayout.pageSize)
        #expect(plan.totalSize == ExpertLayout.pageSize + 200)
    }

    @Test("Quantised size accounts for scale and zero")
    func quantizedSize() {
        // 64 columns is one group: 64 code bytes plus a BF16 scale and zero.
        #expect(TrunkLayout.quantizedSize(rows: 1, cols: 64) == 64 + 4)
        #expect(TrunkLayout.quantizedSize(rows: 10, cols: 128) == 10 * 128 + 10 * 2 * 4)
    }
}

@Suite("Int8 affine quantisation")
struct Int8AffineTests {
    @Test("Round-trip is accurate to a fraction of a step")
    func roundTrip() {
        var generator = SystemRandomNumberGenerator()
        let row = (0..<256).map { _ in Float.random(in: -2...2, using: &generator) }
        let (codes, groups) = Int8Affine.quantize(row: row)
        let back = Int8Affine.dequantize(codes: codes, groups: groups)

        for index in row.indices {
            // 255 levels across a group's range; half a step is the bound.
            let step = groups[index / Int8Affine.groupSize].scale
            #expect(abs(back[index] - row[index]) <= step * 0.51,
                    "index \(index): \(row[index]) -> \(back[index])")
        }
    }

    @Test("Extremes are represented exactly")
    func endpointsExact() {
        var row = [Float](repeating: 0, count: 64)
        row[0] = -3
        row[63] = 5
        let (codes, groups) = Int8Affine.quantize(row: row)
        let back = Int8Affine.dequantize(codes: codes, groups: groups)

        #expect(abs(back[0] - (-3)) < 1e-4, "group minimum should be exact")
        #expect(abs(back[63] - 5) < 1e-3, "group maximum should be near exact")
        #expect(codes[0] == 0)
        #expect(codes[63] == 255)
    }

    @Test("A constant group does not divide by zero")
    func constantGroup() {
        let row = [Float](repeating: 1.25, count: 64)
        let (_, groups) = Int8Affine.quantize(row: row)
        let back = Int8Affine.dequantize(codes: Array(repeating: 0, count: 64), groups: groups)
        #expect(groups[0].scale > 0)
        #expect(back.allSatisfy { abs($0 - 1.25) < 1e-5 })
    }

    @Test("BF16 round-trips values it can represent")
    func bfloatRoundTrip() {
        for value in [0.0, 1.0, -1.0, 0.5, 2.0, -0.25, 1024.0] as [Float] {
            #expect(BFloat16.toFloat(BFloat16.fromFloat(value)) == value, "\(value)")
        }
    }

    @Test("BF16 rounds to nearest rather than truncating")
    func bfloatRounding() {
        // Truncation biases toward zero, and that bias compounds across a
        // 2880-wide dot product.
        var errors: [Float] = []
        var state: UInt32 = 12345
        for _ in 0..<2000 {
            state = state &* 1_664_525 &+ 1_013_904_223
            let value = Float(Int32(bitPattern: state) % 10_000) / 3_000.0
            errors.append(BFloat16.toFloat(BFloat16.fromFloat(value)) - value)
        }
        let mean = errors.reduce(0, +) / Float(errors.count)
        let scale = errors.map { abs($0) }.max() ?? 1
        #expect(abs(mean) < scale * 0.1, "mean error \(mean) suggests a directional bias")
    }
}
