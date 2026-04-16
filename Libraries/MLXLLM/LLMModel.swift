// Copyright © 2024 Apple Inc.

import MLX
import MLXLMCommon

/// Marker protocol for LLMModels
public protocol LLMModel: LanguageModel, LoRAModel {

    /// Models can implement this is they need a custom `MessageGenerator`.
    ///
    /// The default implementation returns `DefaultMessageGenerator`.
    func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator
}

extension LLMModel {

    /// Default prepare step for ``LLMModel``.
    ///
    /// Matches Python mlx-lm's `generate_step` prefill loop: process every
    /// prompt token except the LAST one through chunked prefill (with
    /// `eval(cache)` + `clearCache()` between chunks). The trailing token is
    /// returned to the iterator's "primes the pump" call.
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let prefillStepSize = windowSize ?? 512
        var y = input.text

        // Prepare the prompt in chunks, leaving exactly 1 token for the iterator.
        // `asyncEval` lets the CPU build chunk N+1's graph while the GPU evaluates
        // chunk N — a pipeline win Python mlx-lm gets automatically because its
        // bindings defer eval until a value is read. Swift's previous pattern
        // called `eval(cache)` (sync) between chunks, draining the GPU pipeline
        // and leaving the CPU idle while the next chunk's graph was still being
        // built. For GDN-heavy arches (Qwen3.5/3.6) this stall compounds across
        // 30 linear-attention layers per chunk.
        while y.tokens.size > 1 {
            let chunkSize = min(prefillStepSize, y.tokens.size - 1)
            let input = y[.newAxis, ..<chunkSize]
            _ = self(input, cache: cache.isEmpty ? nil : cache, state: nil)

            // async barrier — lets CPU keep building graph while GPU works.
            var cacheArrays: [MLXArray] = []
            for c in cache { cacheArrays.append(contentsOf: c.innerState()) }
            asyncEval(cacheArrays)

            y = y[chunkSize...]
        }

        // Single sync + clearCache AFTER the prefill loop. Avoids per-chunk
        // pipeline stalls without blowing out memory (one final flush is enough
        // for multi-chunk prefill; the buffer pool only grows temporarily).
        eval(cache)
        MLX.Memory.clearCache()

        return .tokens(y)
    }

    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        DefaultMessageGenerator()
    }
}
