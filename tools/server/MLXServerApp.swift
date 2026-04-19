import Foundation
import MLX
import MLXLLM
import MLXVLM
import MLXLMCommon
import MLXHuggingFace
import MLXNN
import HuggingFace
import Tokenizers

// MARK: - Entry Point

@main
struct MLXServerApp {
    @MainActor
    static func main() async throws {
        // Ignore SIGPIPE — client disconnects during streaming should not crash
        signal(SIGPIPE, SIG_IGN)

        let args = CommandLine.arguments
        let model = args.firstIndex(of: "--model").flatMap { i in
            i + 1 < args.count ? args[i + 1] : nil
        } ?? args.firstIndex(of: "-m").flatMap { i in
            i + 1 < args.count ? args[i + 1] : nil
        } ?? "mlx-community/gemma-4-e2b-it-4bit"

        let port = args.firstIndex(of: "--port").flatMap { i in
            i + 1 < args.count ? UInt16(args[i + 1]) : nil
        } ?? 8080

        let slots = args.firstIndex(of: "--slots").flatMap { i in
            i + 1 < args.count ? Int(args[i + 1]) : nil
        } ?? 4

        let kvScheme = args.firstIndex(of: "--kv").flatMap { i in
            i + 1 < args.count ? args[i + 1] : nil
        }

        // KV cache quantization bits (alternative to --kv scheme)
        let kvBits = (args.firstIndex(of: "--kv-bits") ?? args.firstIndex(of: "--cache-type-k")).flatMap { i in
            i + 1 < args.count ? Int(args[i + 1]) : nil
        }

        let kvStart = args.firstIndex(of: "--kv-start").flatMap { i in
            i + 1 < args.count ? Int(args[i + 1]) : nil
        }

        // Context size (like llama.cpp --ctx-size / -c)
        let ctxSize = (args.firstIndex(of: "--ctx-size") ?? args.firstIndex(of: "-c")).flatMap { i in
            i + 1 < args.count ? Int(args[i + 1]) : nil
        }

        // Max prediction tokens default (like llama.cpp --n-predict / -n)
        let nPredict = (args.firstIndex(of: "--n-predict") ?? args.firstIndex(of: "-n")).flatMap { i in
            i + 1 < args.count ? Int(args[i + 1]) : nil
        }

        // Reasoning mode (like llama-server --reasoning)
        // "on" = always enable thinking (default), "off" = disable, strip from output either way
        let reasoning = args.firstIndex(of: "--reasoning").flatMap { i in
            i + 1 < args.count ? args[i + 1] : nil
        } ?? "on"
        let enableThinking = (reasoning != "off")

        // Help
        if args.contains("--help") || args.contains("-h") {
            printUsage()
            return
        }

        log("Loading model: \(model)")
        let config: ModelConfiguration
        if model.hasPrefix("/") || model.hasPrefix("~") || model.hasPrefix(".") {
            let expandedPath = NSString(string: model).expandingTildeInPath
            config = ModelConfiguration(directory: URL(fileURLWithPath: expandedPath))
        } else {
            config = ModelConfiguration(id: model)
        }

        // Load model
        let downloader: any Downloader = #hubDownloader()
        let tokenizerLoader: any TokenizerLoader = #huggingFaceTokenizerLoader()
        let container: ModelContainer
        do {
            container = try await LLMModelFactory.shared.loadContainer(
                from: downloader, using: tokenizerLoader,
                configuration: config) { p in
                if p.fractionCompleted > 0.99 { log("Model loaded (LLM)") }
            }
        } catch {
            log("LLM load failed, trying VLM...")
            container = try await VLMModelFactory.shared.loadContainer(
                from: downloader, using: tokenizerLoader,
                configuration: config) { p in
                if p.fractionCompleted > 0.99 { log("Model loaded (VLM)") }
            }
        }

        let server = SimpleHTTPServer(port: port, container: container, modelId: model, slotCount: slots,
                                       kvScheme: kvScheme, kvBits: kvBits, kvStart: kvStart,
                                       ctxSize: ctxSize, defaultMaxTokens: nPredict,
                                       enableThinking: enableThinking)
        log("Reasoning: \(reasoning) (enable_thinking=\(enableThinking))")
        if let kv = kvScheme { log("KV scheme: \(kv)") }
        if let bits = kvBits { log("KV bits: \(bits)") }
        if let ctx = ctxSize { log("Context size: \(ctx)") }
        if let np = nPredict { log("Default max tokens: \(np)") }
        try server.start()
    }

    static func printUsage() {
        let usage = """
        MLXServer — OpenAI-compatible inference server for MLX Swift

        USAGE:
          MLXServer --model <model> [options]

        OPTIONS:
          -m, --model <path|id>    Model path or HuggingFace ID (required)
          --port <port>            Listen port (default: 8080)
          --slots <n>              Parallel inference slots (default: 4)
          --kv <scheme>            KV cache scheme (e.g. turbo4v2)
          --kv-bits <n>            KV cache quantization bits
          --kv-start <n>           Layer to start KV quantization
          -c, --ctx-size <n>       Context size limit
          -n, --n-predict <n>      Default max generation tokens
          --reasoning <on|off>     Enable/disable thinking (default: on)
          -h, --help               Show this help
        """
        print(usage)
    }
}
