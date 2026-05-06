// Copyright © 2026 Apple Inc.

import MLXLMCommon
import XCTest

/// Coverage for the model-free helpers in `WiredMemoryUtils` (env parsing,
/// KV-bytes formula, batch scaling).
///
/// Functions that depend on a loaded `LanguageModel` (`estimateBudget`,
/// `estimatedTicket`, `resolveTicket`) are exercised by the bench harness and
/// the smoke test on real models; replicating them here would require model
/// downloads that the MLXLMTests target intentionally avoids.
final class WiredMemoryUtilsTests: XCTestCase {

    // MARK: - parseMemoryLimit

    func testParseMemoryLimitNilOrEmpty() {
        XCTAssertNil(WiredMemoryUtils.parseMemoryLimit(nil))
        XCTAssertNil(WiredMemoryUtils.parseMemoryLimit(""))
        XCTAssertNil(WiredMemoryUtils.parseMemoryLimit("   "))
    }

    func testParseMemoryLimitPlainBytes() {
        XCTAssertEqual(WiredMemoryUtils.parseMemoryLimit("1024"), 1024)
        XCTAssertEqual(WiredMemoryUtils.parseMemoryLimit("1073741824"), 1_073_741_824)
    }

    func testParseMemoryLimitGigabytes() {
        XCTAssertEqual(WiredMemoryUtils.parseMemoryLimit("32g"), 32 * 1_073_741_824)
        XCTAssertEqual(WiredMemoryUtils.parseMemoryLimit("32G"), 32 * 1_073_741_824)
        XCTAssertEqual(WiredMemoryUtils.parseMemoryLimit("32gb"), 32 * 1_073_741_824)
        XCTAssertEqual(WiredMemoryUtils.parseMemoryLimit("32GB"), 32 * 1_073_741_824)
    }

    func testParseMemoryLimitMegabytes() {
        XCTAssertEqual(WiredMemoryUtils.parseMemoryLimit("512m"), 512 * 1_048_576)
        XCTAssertEqual(WiredMemoryUtils.parseMemoryLimit("512MB"), 512 * 1_048_576)
    }

    func testParseMemoryLimitKilobytes() {
        XCTAssertEqual(WiredMemoryUtils.parseMemoryLimit("4k"), 4 * 1024)
        XCTAssertEqual(WiredMemoryUtils.parseMemoryLimit("4KB"), 4 * 1024)
    }

    func testParseMemoryLimitFractional() {
        XCTAssertEqual(WiredMemoryUtils.parseMemoryLimit("1.5g"), Int(1.5 * 1_073_741_824))
        XCTAssertEqual(WiredMemoryUtils.parseMemoryLimit("0.5gb"), Int(0.5 * 1_073_741_824))
    }

    func testParseMemoryLimitWhitespaceTolerated() {
        XCTAssertEqual(WiredMemoryUtils.parseMemoryLimit("  32g  "), 32 * 1_073_741_824)
        XCTAssertEqual(WiredMemoryUtils.parseMemoryLimit("\t32GB\n"), 32 * 1_073_741_824)
    }

    func testParseMemoryLimitRejectsGarbage() {
        XCTAssertNil(WiredMemoryUtils.parseMemoryLimit("abc"))
        XCTAssertNil(WiredMemoryUtils.parseMemoryLimit("g32"))
        XCTAssertNil(WiredMemoryUtils.parseMemoryLimit("--32g"))
    }

    func testParseMemoryLimitRejectsZeroAndNegative() {
        XCTAssertNil(WiredMemoryUtils.parseMemoryLimit("0"))
        XCTAssertNil(WiredMemoryUtils.parseMemoryLimit("0g"))
        XCTAssertNil(WiredMemoryUtils.parseMemoryLimit("-32g"))
    }

    func testParseMemoryLimitRejectsOverflow() {
        XCTAssertNil(WiredMemoryUtils.parseMemoryLimit("99999999999999999999g"))
        XCTAssertNil(WiredMemoryUtils.parseMemoryLimit("inf"))
    }

    // MARK: - kvBytesPerTokenPerHead

    func testKVBytesFP16Default() {
        XCTAssertEqual(WiredMemoryUtils.kvBytesPerTokenPerHead(headDim: 128), 128 * 2 * 2)
        XCTAssertEqual(WiredMemoryUtils.kvBytesPerTokenPerHead(headDim: 64), 64 * 2 * 2)
    }

    func testKVBytesTurboFallsBackToFP16() {
        let fp16 = WiredMemoryUtils.kvBytesPerTokenPerHead(headDim: 128)
        XCTAssertEqual(
            WiredMemoryUtils.kvBytesPerTokenPerHead(
                headDim: 128, algorithm: .turbo(keyBits: 4, valueBits: 4)),
            fp16,
            "TurboQuant decode keeps FP16 in the cache; bytes/token must equal FP16.")
        XCTAssertEqual(
            WiredMemoryUtils.kvBytesPerTokenPerHead(
                headDim: 128, algorithm: .turbo(keyBits: 4, valueBits: 2)),
            fp16)
    }

    func testKVBytesAffineQuantizationIsSmaller() {
        let fp16 = WiredMemoryUtils.kvBytesPerTokenPerHead(headDim: 128)
        let q8 = WiredMemoryUtils.kvBytesPerTokenPerHead(
            headDim: 128, algorithm: .affine(bits: 8, groupSize: 64))
        let q4 = WiredMemoryUtils.kvBytesPerTokenPerHead(
            headDim: 128, algorithm: .affine(bits: 4, groupSize: 64))
        XCTAssertLessThan(q8, fp16, "8-bit affine should be smaller than FP16")
        XCTAssertLessThan(q4, q8, "4-bit affine should be smaller than 8-bit")
    }

    func testKVBytesIgnoresZeroAndSixteenBitOverride() {
        let fp16 = WiredMemoryUtils.kvBytesPerTokenPerHead(headDim: 128)
        // bits=0 / bits>=16 fall back to FP16 baseline.
        XCTAssertEqual(
            WiredMemoryUtils.kvBytesPerTokenPerHead(
                headDim: 128, algorithm: .affine(bits: 0, groupSize: 64)),
            fp16)
        XCTAssertEqual(
            WiredMemoryUtils.kvBytesPerTokenPerHead(
                headDim: 128, algorithm: .affine(bits: 16, groupSize: 64)),
            fp16)
    }

    // MARK: - estimateKVBytes

    func testEstimateKVBytesMatchesHandFormula() {
        let kvHeads = Array(repeating: 8, count: 32)
        let bytes = WiredMemoryUtils.estimateKVBytes(
            tokens: 32_768, kvHeads: kvHeads, headDim: 128, batchSize: 1)
        XCTAssertEqual(bytes, 32_768 * (8 * 32) * (128 * 2 * 2))
        XCTAssertEqual(bytes, 4_294_967_296)  // 4.29 GB — Qwen3.5-9B FP16 KV @ 32k
    }

    func testEstimateKVBytesScalesLinearlyWithBatch() {
        let kvHeads = Array(repeating: 8, count: 32)
        let b1 = WiredMemoryUtils.estimateKVBytes(
            tokens: 32_768, kvHeads: kvHeads, headDim: 128, batchSize: 1)
        let b4 = WiredMemoryUtils.estimateKVBytes(
            tokens: 32_768, kvHeads: kvHeads, headDim: 128, batchSize: 4)
        XCTAssertEqual(b4, b1 * 4)
    }

    func testEstimateKVBytesDegenerateInputsReturnZero() {
        let kvHeads = Array(repeating: 8, count: 32)
        XCTAssertEqual(
            WiredMemoryUtils.estimateKVBytes(
                tokens: 0, kvHeads: kvHeads, headDim: 128), 0)
        XCTAssertEqual(
            WiredMemoryUtils.estimateKVBytes(
                tokens: 1024, kvHeads: [], headDim: 128), 0)
        XCTAssertEqual(
            WiredMemoryUtils.estimateKVBytes(
                tokens: 1024, kvHeads: kvHeads, headDim: 0), 0)
        XCTAssertEqual(
            WiredMemoryUtils.estimateKVBytes(
                tokens: 1024, kvHeads: kvHeads, headDim: 128, batchSize: 0), 0)
    }

    func testEstimateKVBytesAffineQuantizationShrinksTotal() {
        let kvHeads = Array(repeating: 8, count: 32)
        let fp16 = WiredMemoryUtils.estimateKVBytes(
            tokens: 32_768, kvHeads: kvHeads, headDim: 128)
        let q4 = WiredMemoryUtils.estimateKVBytes(
            tokens: 32_768, kvHeads: kvHeads, headDim: 128,
            algorithm: .affine(bits: 4, groupSize: 64))
        XCTAssertLessThan(q4, fp16)
        XCTAssertLessThan(Double(q4) / Double(fp16), 0.5,
            "4-bit affine should be at least 2× smaller than FP16.")
    }

    func testEstimateKVBytesHandlesPerLayerVariation() {
        // Some hybrid models have varying kvHeads per layer (e.g. Gemma sliding
        // window mixed with full attention). The estimate sums per-layer heads.
        let mixed = [8, 8, 8, 4, 4, 4]
        let bytes = WiredMemoryUtils.estimateKVBytes(
            tokens: 4096, kvHeads: mixed, headDim: 128)
        let expected = 4096 * (8 + 8 + 8 + 4 + 4 + 4) * (128 * 2 * 2)
        XCTAssertEqual(bytes, expected)
    }

    // MARK: - envSmartMemoryEnabled

    func testSmartMemoryDefaultsToOn() {
        let prior = ProcessInfo.processInfo.environment["MLX_SMART_MEMORY"]
        unsetenv("MLX_SMART_MEMORY")
        defer {
            if let prior { setenv("MLX_SMART_MEMORY", prior, 1) }
        }
        XCTAssertTrue(WiredMemoryUtils.envSmartMemoryEnabled())
    }

    func testSmartMemoryDisabledOnlyWhenZero() {
        let prior = ProcessInfo.processInfo.environment["MLX_SMART_MEMORY"]
        defer {
            if let prior { setenv("MLX_SMART_MEMORY", prior, 1) }
            else { unsetenv("MLX_SMART_MEMORY") }
        }

        setenv("MLX_SMART_MEMORY", "0", 1)
        XCTAssertFalse(WiredMemoryUtils.envSmartMemoryEnabled())

        setenv("MLX_SMART_MEMORY", "1", 1)
        XCTAssertTrue(WiredMemoryUtils.envSmartMemoryEnabled())

        setenv("MLX_SMART_MEMORY", "", 1)
        XCTAssertTrue(WiredMemoryUtils.envSmartMemoryEnabled())
    }

    // MARK: - envMemoryLimit

    func testEnvMemoryLimitParsesEnv() {
        let prior = ProcessInfo.processInfo.environment["MLX_MEMORY_LIMIT"]
        defer {
            if let prior { setenv("MLX_MEMORY_LIMIT", prior, 1) }
            else { unsetenv("MLX_MEMORY_LIMIT") }
        }

        unsetenv("MLX_MEMORY_LIMIT")
        XCTAssertNil(WiredMemoryUtils.envMemoryLimit())

        setenv("MLX_MEMORY_LIMIT", "16g", 1)
        XCTAssertEqual(WiredMemoryUtils.envMemoryLimit(), 16 * 1_073_741_824)

        setenv("MLX_MEMORY_LIMIT", "garbage", 1)
        XCTAssertNil(WiredMemoryUtils.envMemoryLimit())
    }
}
