// Copyright © 2024 Apple Inc.

import MLX
import MLXLMCommon
import Tokenizers

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
    ///
    /// Previously the loop only ran when the prompt exceeded `prefillStepSize`,
    /// so a 1024-token prompt with chunk size 512 chunked correctly but a
    /// 256-token prompt would skip the loop entirely and process the whole
    /// thing inside the iterator with no cache cleanup, leaking ~1 GB of
    /// activation buffers per model into the MLX recycle pool.
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let prefillStepSize = windowSize ?? 512
        var y = input.text

        // Prepare the prompt in chunks, leaving exactly 1 token for the iterator.
        while y.tokens.size > 1 {
            let chunkSize = min(prefillStepSize, y.tokens.size - 1)
            let input = y[.newAxis, ..<chunkSize]
            _ = self(input, cache: cache.isEmpty ? nil : cache, state: nil)
            eval(cache)
            y = y[chunkSize...]
            // Free intermediate activation buffers between chunks to reduce memory pressure,
            // matching Python mlx-lm's mx.clear_cache() after each prefill chunk.
            MLX.Memory.clearCache()
        }

        return .tokens(y)
    }

    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        DefaultMessageGenerator()
    }
}
