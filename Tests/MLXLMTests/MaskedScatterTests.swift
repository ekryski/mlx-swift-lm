// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

/// Regression tests for the VLM masked-scatter pattern after the PR 0 follow-up
/// swap from `mask.asArray(Bool.self) + Swift loop` to `argWhere(mask, count:)`.
///
/// These tests pin the semantic identity between the old Swift-side-loop
/// formulation and the new GPU-side `argWhere` formulation, independent of any
/// specific VLM model. Both paths must produce the same output embeddings for
/// any (mask, features) pair.
struct MaskedScatterTests {

    /// The old CPU-loop pattern that callers used to write inline.
    private func cpuReference(
        flatEmbeds: MLXArray,
        flatFeatures: MLXArray,
        flatMask: MLXArray
    ) -> MLXArray {
        let maskValues = flatMask.asType(.bool).asArray(Bool.self)
        let indices = maskValues.enumerated().compactMap { i, v -> UInt32? in
            v ? UInt32(i) : nil
        }
        guard !indices.isEmpty && indices.count == flatFeatures.shape[0] else {
            return flatEmbeds
        }
        var result = flatEmbeds
        result[MLXArray(indices)] = flatFeatures
        return result
    }

    /// The new GPU-only pattern (matches the `maskedScatter` bodies in
    /// `Gemma3.swift` and `Gemma4.swift`).
    private func gpuScatter(
        flatEmbeds: MLXArray,
        flatFeatures: MLXArray,
        flatMask: MLXArray
    ) -> MLXArray {
        let expected = flatFeatures.shape[0]
        let actual = flatMask.asType(.int32).sum().item(Int.self)
        guard actual == expected, expected > 0 else {
            return flatEmbeds
        }
        let rawIndices = argWhere(flatMask.asType(.bool), count: expected)
        let positions = rawIndices.asType(DType.uint32)
        var result = flatEmbeds
        result[positions] = flatFeatures
        return result
    }

    @Test
    func `GPU scatter matches CPU-loop reference — single image block`() {
        // 12 token positions, 4 image tokens clustered in one block.
        let n = 12
        let d = 8
        let maskRaw: [Bool] = [false, false, true, true, true, true, false, false, false, false, false, false]
        let mask = MLXArray(maskRaw)

        let embeds = MLXArray.zeros([n, d], dtype: .float32)
        let featureValues: [Float] = (0 ..< (4 * d)).map { Float($0) }
        let features = MLXArray(featureValues, [4, d])
        let flatEmbeds = embeds.flattened()
        let flatFeatures = features.flattened()
        let flatMask = mask  // 1-D already; shape [n]

        // To match the VLM callers, the real mask is expanded to [n, d] and flattened
        // so it selects full rows. Build that form:
        let expandedMask = broadcast(
            mask.reshaped([n, 1]), to: [n, d])
        let flatMaskExpanded = expandedMask.flattened()

        let cpu = cpuReference(
            flatEmbeds: flatEmbeds, flatFeatures: flatFeatures, flatMask: flatMaskExpanded)
        let gpu = gpuScatter(
            flatEmbeds: flatEmbeds, flatFeatures: flatFeatures, flatMask: flatMaskExpanded)

        eval(cpu, gpu)
        #expect(cpu.asArray(Float.self) == gpu.asArray(Float.self))
    }

    @Test
    func `GPU scatter matches CPU-loop reference — interleaved mask`() {
        // 3 image tokens scattered across 8 positions.
        let n = 8
        let d = 4
        let maskRaw: [Bool] = [false, true, false, true, false, false, true, false]
        let mask = MLXArray(maskRaw)
        let expanded = broadcast(mask.reshaped([n, 1]), to: [n, d]).flattened()

        let embedValues: [Float] = (0 ..< (n * d)).map { Float($0) * 0.1 }
        let embeds = MLXArray(embedValues, [n, d])
        let featureValues: [Float] = (0 ..< (3 * d)).map { Float($0) + 100 }
        let features = MLXArray(featureValues, [3, d])

        let cpu = cpuReference(
            flatEmbeds: embeds.flattened(),
            flatFeatures: features.flattened(),
            flatMask: expanded)
        let gpu = gpuScatter(
            flatEmbeds: embeds.flattened(),
            flatFeatures: features.flattened(),
            flatMask: expanded)

        eval(cpu, gpu)
        #expect(cpu.asArray(Float.self) == gpu.asArray(Float.self))
    }

    @Test
    func `GPU scatter returns input unchanged when no image tokens`() {
        let n = 6
        let d = 4
        let mask = MLXArray.zeros([n * d], dtype: .bool)
        let embedValues: [Float] = (0 ..< (n * d)).map { Float($0) }
        let embeds = MLXArray(embedValues, [n, d])
        let emptyFeatures = MLXArray.zeros([0, d], dtype: .float32)

        let result = gpuScatter(
            flatEmbeds: embeds.flattened(),
            flatFeatures: emptyFeatures.flattened(),
            flatMask: mask)

        eval(result)
        #expect(result.asArray(Float.self) == embeds.flattened().asArray(Float.self))
    }

    @Test
    func `GPU scatter returns input unchanged on mask-count mismatch`() {
        // Mask says 3 image tokens, features provide only 2 → mismatch branch.
        let n = 6
        let d = 4
        let maskRaw: [Bool] = [true, false, true, true, false, false]
        let expanded = broadcast(MLXArray(maskRaw).reshaped([n, 1]), to: [n, d]).flattened()
        let embedValues: [Float] = (0 ..< (n * d)).map { Float($0) }
        let embeds = MLXArray(embedValues, [n, d])
        // Count mismatch: we pass only 2·d features for 3·d mask positions.
        let features = MLXArray.zeros([2, d], dtype: .float32)

        let result = gpuScatter(
            flatEmbeds: embeds.flattened(),
            flatFeatures: features.flattened(),
            flatMask: expanded)

        eval(result)
        // Falls through the guard, returns flatEmbeds unchanged.
        #expect(result.asArray(Float.self) == embeds.flattened().asArray(Float.self))
    }
}
