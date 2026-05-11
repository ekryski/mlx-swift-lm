// Copyright © 2026 ekryski.
//
// Multi-token prediction (MTP) — unit tests for the protocol surface,
// the variant-A route decision, and the variant-B assistant draft
// registry.
//
// Spec: specs/030-multi-token-prediction.md.
//
// These tests don't engage a live model. The full iterator round-trip
// against a real `MTPInjector` conformance lands in a follow-up once a
// model carries the head implementation (e.g. DeepseekV4MTP).

import Foundation
@testable import MLXLMCommon
import Testing

@Suite
struct MTPTests {

    // MARK: - mtpRouteDecision

    @Test
    func `route declines on default parameters`() {
        let p = GenerateParameters(temperature: 0.0)
        let d = mtpRouteDecision(parameters: p)
        #expect(d.shouldEngage == false)
    }

    @Test
    func `route declines on non-greedy temperature`() {
        var p = GenerateParameters(temperature: 0.7)
        p.mtpEnabled = true
        let d = mtpRouteDecision(parameters: p)
        #expect(d.shouldEngage == false)
    }

    @Test
    func `route engages on Swift opt-in (greedy)`() {
        var p = GenerateParameters(temperature: 0.0)
        p.mtpEnabled = true
        let d = mtpRouteDecision(parameters: p)
        #expect(d.shouldEngage == true)
        // Default draft count fills in when caller didn't set one.
        #expect(d.parameters.mtpDraftCount >= 1)
    }

    @Test
    func `route preserves explicit draft count`() {
        var p = GenerateParameters(temperature: 0.0)
        p.mtpEnabled = true
        p.mtpDraftCount = 3
        let d = mtpRouteDecision(parameters: p)
        #expect(d.shouldEngage == true)
        #expect(d.parameters.mtpDraftCount == 3)
    }

    // MARK: - MTPLoader probing

    @Test
    func `MTPLoader detects mtp prefix`() {
        let keys: Set<String> = [
            "model.layers.0.self_attn.q_proj.weight",
            "mtp.eh_proj.weight",
        ]
        #expect(MTPLoader.hasInTrunkMTPHead(keys: keys))
    }

    @Test
    func `MTPLoader detects DeepSeek-style layer-N convention`() {
        let keys: Set<String> = [
            "model.layers.42.self_attn.q_proj.weight",
            "model.layers.43.eh_proj.weight",  // layer-N MTP
        ]
        #expect(MTPLoader.hasInTrunkMTPHead(
            keys: keys, deepseekStyleLayerN: 43))
    }

    @Test
    func `MTPLoader returns false on plain weights`() {
        let keys: Set<String> = [
            "model.layers.0.self_attn.q_proj.weight",
            "lm_head.weight",
        ]
        #expect(MTPLoader.hasInTrunkMTPHead(keys: keys) == false)
        #expect(MTPLoader.hasInTrunkMTPHead(
            keys: keys, deepseekStyleLayerN: 99) == false)
    }

    @Test
    func `MTPLoader.mtpLoadEnabled false by default`() {
        let p = GenerateParameters()
        #expect(MTPLoader.mtpLoadEnabled(parameters: p) == false)
    }

    @Test
    func `MTPLoader.mtpLoadEnabled true when parameters opt in`() {
        var p = GenerateParameters()
        p.mtpEnabled = true
        #expect(MTPLoader.mtpLoadEnabled(parameters: p) == true)
    }

    // MARK: - AssistantDraftRegistry

    @Test
    func `registry resolves Gemma-4 26B-A4B target`() {
        let entry = AssistantDraftRegistry.resolve(
            targetId: "mlx-community/gemma-4-26B-A4B-it-bf16")
        #expect(entry?.draftId
            == "mlx-community/gemma-4-26B-A4B-it-assistant-bf16")
        #expect(entry?.recommendedNumDraftTokens == 6)
    }

    @Test
    func `registry resolves Gemma-4 31B target`() {
        let entry = AssistantDraftRegistry.resolve(
            targetId: "mlx-community/gemma-4-31B-it-4bit")
        #expect(entry?.draftId
            == "mlx-community/gemma-4-31B-it-assistant-bf16")
    }

    @Test
    func `registry resolves Gemma-4 E2B target`() {
        let entry = AssistantDraftRegistry.resolve(
            targetId: "mlx-community/gemma-4-E2B-it-bf16")
        #expect(entry?.draftId
            == "mlx-community/gemma-4-E2B-it-assistant-bf16")
    }

    @Test
    func `registry resolves Gemma-4 E4B target`() {
        let entry = AssistantDraftRegistry.resolve(
            targetId: "mlx-community/gemma-4-E4B-it-bf16")
        #expect(entry?.draftId
            == "mlx-community/gemma-4-E4B-it-assistant-bf16")
    }

    @Test
    func `registry returns nil on unsupported target`() {
        let entry = AssistantDraftRegistry.resolve(
            targetId: "mlx-community/Llama-3.2-3B-Instruct-4bit")
        #expect(entry == nil)
    }

    @Test
    func `registry org-strip fallback matches local bundles`() {
        // A local-only bundle path that ends with the same suffix as a
        // registered prefix should match via the org-strip fallback.
        let entry = AssistantDraftRegistry.resolve(
            targetId: "LocalDir/gemma-4-26B-A4B-it-bf16")
        #expect(entry?.draftId
            == "mlx-community/gemma-4-26B-A4B-it-assistant-bf16")
    }

    // MARK: - GenerateParameters threading

    @Test
    func `GenerateParameters defaults disable MTP`() {
        let p = GenerateParameters()
        #expect(p.mtpEnabled == false)
        #expect(p.mtpDraftCount == 0)
        #expect(p.mtpAssistantDraftId == nil)
    }

    @Test
    func `GenerateParameters initializer threads MTP fields`() {
        let p = GenerateParameters(
            temperature: 0.0,
            mtpEnabled: true, mtpDraftCount: 2,
            mtpAssistantDraftId: "custom/draft-id")
        #expect(p.mtpEnabled == true)
        #expect(p.mtpDraftCount == 2)
        #expect(p.mtpAssistantDraftId == "custom/draft-id")
    }

    // MARK: - MTPContract

    @Test
    func `MTPContract stores per-family knobs`() {
        let c = MTPContract(
            hiddenVariant: .preNorm,
            concatOrder: .embeddingHidden,
            maxHeads: 1, family: "deepseek-v4")
        #expect(c.maxHeads == 1)
        #expect(c.family == "deepseek-v4")
        if case .preNorm = c.hiddenVariant { } else { Issue.record() }
        if case .embeddingHidden = c.concatOrder { } else { Issue.record() }
    }
}
