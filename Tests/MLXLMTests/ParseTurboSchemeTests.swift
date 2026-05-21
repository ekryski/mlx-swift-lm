// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Pin the user-facing scheme-string contract that drives BOTH single-batch
// `TurboQuantKVCache.toTurboQuantized(...)` AND the batched-decode turbo
// cache factory in vllm-swift's Bridge.swift. The parser is the single
// source of truth for "turboNvM" → (keyBits, valueBits) — every CLI / config
// flag funnels through here, so an off-by-one in the parse rule would
// silently flip K and V across the whole stack.

import Foundation
import MLXLMCommon
import Testing

@Suite("parseTurboScheme")
struct ParseTurboSchemeTests {

    @Test
    func `symmetric turbo4 yields nil keyBits and valueBits`() throws {
        let r = parseTurboScheme("turbo4")
        #expect(r.bits == 4)
        #expect(r.keyBits == nil)
        #expect(r.valueBits == nil)
    }

    @Test
    func `symmetric turbo3 parses bit-width`() throws {
        let r = parseTurboScheme("turbo3")
        #expect(r.bits == 3)
        #expect(r.keyBits == nil)
        #expect(r.valueBits == nil)
    }

    @Test
    func `symmetric turbo8 parses bit-width`() throws {
        let r = parseTurboScheme("turbo8")
        #expect(r.bits == 8)
    }

    @Test
    func `asymmetric turbo4v2 splits K and V`() throws {
        // The scheme tag that tripped the v0.5.1 alpha tester. K=4 (16
        // codebook centroids), V=2 (only 4 centroids — aggressive). Buddy's
        // ".2.2.2.2..." drift was BatchedKVCache silently dropping this
        // scheme; this test pins the parse so the silent-drop can't come
        // back through a parser regression.
        let r = parseTurboScheme("turbo4v2")
        #expect(r.bits == 4)         // max(4, 2) — used as legacy "bits"
        #expect(r.keyBits == 4)
        #expect(r.valueBits == 2)
    }

    @Test
    func `asymmetric turbo8v4 K8 V4`() throws {
        // The buddy-recommended workaround in the v0.5.1 alpha report
        // ("bump kv_bits to 8") — K=8 keeps high-precision keys (where
        // softmax amplifies error), V=4 stays compressed.
        let r = parseTurboScheme("turbo8v4")
        #expect(r.bits == 8)
        #expect(r.keyBits == 8)
        #expect(r.valueBits == 4)
    }

    @Test
    func `asymmetric turbo0v4 raw-key mode`() throws {
        // K=0 = raw FP16 keys, V=4 compressed. The single biggest TurboQuant+
        // quality finding — softmax amplification makes V compression nearly
        // free. BatchedKVCache's `rawKeyMode` flag flips on when keyBits == 0.
        let r = parseTurboScheme("turbo0v4")
        #expect(r.bits == 4)         // max(0, 4)
        #expect(r.keyBits == 0)
        #expect(r.valueBits == 4)
    }

    @Test
    func `asymmetric turbo3v2 GPT-OSS family`() throws {
        let r = parseTurboScheme("turbo3v2")
        #expect(r.keyBits == 3)
        #expect(r.valueBits == 2)
    }

    @Test
    func `asymmetric turbo4v3`() throws {
        let r = parseTurboScheme("turbo4v3")
        #expect(r.keyBits == 4)
        #expect(r.valueBits == 3)
    }

    @Test
    func `asymmetric turbo4v4 same as symmetric turbo4 for keyBits and valueBits`()
        throws
    {
        // Both spellings should produce the same effective configuration
        // when consumed by the cache factory (after defaulting nil → bits).
        let asym = parseTurboScheme("turbo4v4")
        #expect(asym.keyBits == 4)
        #expect(asym.valueBits == 4)
        // Symmetric keeps keyBits/valueBits nil so the caller defaults from
        // .bits. Both end up at (4, 4) downstream.
        let sym = parseTurboScheme("turbo4")
        #expect(sym.bits == 4)
        #expect((sym.keyBits ?? sym.bits) == 4)
        #expect((sym.valueBits ?? sym.bits) == 4)
    }
}
