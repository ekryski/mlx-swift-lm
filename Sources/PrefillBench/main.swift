import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon

typealias PB2Init = @convention(c) (Int32, Int32, Int32, Int32, Int32, Int32) -> Int32
typealias PB2SetWeight = @convention(c) (UnsafePointer<CChar>, UnsafeMutableRawPointer) -> Int32
typealias PB2Finalize = @convention(c) () -> Int32
typealias PB2Run = @convention(c) (UnsafePointer<Int32>, Int32, UnsafeMutablePointer<Double>, UnsafeMutablePointer<Float>) -> Int32
typealias PB2TestWeight = @convention(c) (UnsafePointer<CChar>, UnsafeMutableRawPointer) -> Int32
typealias PB2Cleanup = @convention(c) () -> Void

func log(_ msg: String) {
    FileHandle.standardError.write(Data("[Swift] \(msg)\n".utf8))
}

@main
struct PrefillBenchmark {
    static func main() async {
        setlinebuf(stdout)
        let mode = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "swift"
        log("START mode=\(mode)")

        do {
            let config = ModelConfiguration(id: "mlx-community/gemma-4-e2b-it-4bit")
            let container = try await LLMModelFactory.shared.loadContainer(
                configuration: config) { p in
                if p.fractionCompleted > 0.99 { log("Loaded") }
            }
            log("MODEL_LOADED")

            let data = try Data(contentsOf: URL(fileURLWithPath: "/tmp/bench_tokens_1024.json"))
            let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
            let allTokens = (json["tokens"] as! [Int]).map { Int32($0) }

            if mode == "inject" {
                guard let lib = dlopen("/tmp/libprefill_bridge_v2.dylib", RTLD_NOW) else {
                    log("ERROR: \(String(cString: dlerror()))"); return
                }
                let pb2Init = unsafeBitCast(dlsym(lib, "pb2_init")!, to: PB2Init.self)
                let pb2Set = unsafeBitCast(dlsym(lib, "pb2_set_weight")!, to: PB2SetWeight.self)
                let pb2Fin = unsafeBitCast(dlsym(lib, "pb2_finalize")!, to: PB2Finalize.self)
                let pb2Run = unsafeBitCast(dlsym(lib, "pb2_run")!, to: PB2Run.self)
                let pb2Test = unsafeBitCast(dlsym(lib, "pb2_test_weight")!, to: PB2TestWeight.self)
                log("Bridge symbols loaded")

                // Init + pass weights + finalize inside perform
                try await container.perform { ctx in
                    log("Inside perform")
                    let model = ctx.model

                    let _ = pb2Init(15, 1536, 8, 1, 512, 5)
                    log("pb2_init done")

                    // Test one weight first
                    guard let module = model as? Module else { log("Not a Module!"); return }
                    let params = module.parameters().flattened()
                    log("Got \(params.count) params")

                    // Find a known key and test the borrow
                    if let (key, arr) = params.first(where: { $0.0.contains("q_proj.scales") }) {
                        let k = key.hasPrefix("model.") ? String(key.dropFirst(6)) : key
                        log("Testing weight: \(k)")
                        let rawPtr = arr.ctx.ctx!
                        let testRC = k.withCString { pb2Test($0, rawPtr) }
                        log("Test result: \(testRC)")
                    }

                    // Pass all weights
                    var count = 0
                    for (key, arr) in params {
                        let k = key.hasPrefix("model.") ? String(key.dropFirst(6)) : key
                        let _ = k.withCString { pb2Set($0, arr.ctx.ctx!) }
                        count += 1
                    }
                    log("Set \(count) weights")

                    let finRC = pb2Fin()
                    log("Finalize: \(finRC)")
                    if finRC != 0 { return }

                    // Isolated bridge timing: 16 and 1024 tokens
                    var ms: Double = 0; var ck: Float = 0
                    for n in [16, 1024] {
                        let toks = Array(allTokens.prefix(n))
                        // warmup
                        for _ in 0..<3 { let _ = pb2Run(toks, Int32(n), &ms, &ck) }
                        // timed
                        var times: [Double] = []
                        for _ in 0..<5 { let _ = pb2Run(toks, Int32(n), &ms, &ck); times.append(ms) }
                        let avg = times.reduce(0, +) / 5.0
                        log(String(format: "Bridge %4d tok: %.1fms (%.0f tok/s) cksum=%.4f",
                            n, avg, Double(n)/(avg/1000.0), ck))
                    }

                    // --- Task 2: K/V injection ---
                    typealias PB2KVNbytes = @convention(c) (Int32) -> Int
                    typealias PB2KVShape = @convention(c) (Int32, UnsafeMutablePointer<Int32>, UnsafeMutablePointer<Int32>, UnsafeMutablePointer<Int32>) -> Int32
                    typealias PB2ExportKV = @convention(c) (Int32, UnsafeMutableRawPointer, UnsafeMutableRawPointer) -> Int32

                    let kvNbytes = unsafeBitCast(dlsym(lib, "pb2_kv_nbytes")!, to: PB2KVNbytes.self)
                    let kvShape = unsafeBitCast(dlsym(lib, "pb2_kv_shape")!, to: PB2KVShape.self)
                    let exportKV = unsafeBitCast(dlsym(lib, "pb2_export_kv")!, to: PB2ExportKV.self)

                    let cache = model.newCache(parameters: nil)
                    log("Cache created: \(cache.count) entries")
                    for i in 0..<15 {
                        log("  export layer \(i)...")
                        let nb = kvNbytes(Int32(i))
                        log("  nb=\(nb)")
                        guard nb > 0 else { log("Layer \(i): no data"); continue }
                        var kvH: Int32 = 0, seqL: Int32 = 0, hd: Int32 = 0
                        let _ = kvShape(Int32(i), &kvH, &seqL, &hd)
                        log("  shape: [\(kvH),\(seqL),\(hd)]")

                        let kBuf = UnsafeMutableRawPointer.allocate(byteCount: nb, alignment: 16)
                        let vBuf = UnsafeMutableRawPointer.allocate(byteCount: nb, alignment: 16)
                        log("  allocated bufs")
                        let rc = exportKV(Int32(i), kBuf, vBuf)
                        log("  exported rc=\(rc)")
                        guard rc == 0 else { kBuf.deallocate(); vBuf.deallocate(); log("Export \(i) failed"); continue }

                        let shape = [1, Int(kvH), Int(seqL), Int(hd)]
                        let numElements = shape.reduce(1, *)
                        let bpe = nb / numElements
                        let kData = Data(bytes: kBuf, count: nb)
                        let vData = Data(bytes: vBuf, count: nb)
                        kBuf.deallocate(); vBuf.deallocate()

                        let kArr: MLXArray
                        let vArr: MLXArray
                        if bpe == 2 {
                            // bfloat16 — view as bf16
                            kArr = MLXArray(kData, shape, type: UInt16.self).view(dtype: .bfloat16)
                            vArr = MLXArray(vData, shape, type: UInt16.self).view(dtype: .bfloat16)
                        } else {
                            // float32 — convert to bf16
                            kArr = MLXArray(kData, shape, type: Float.self).asType(.bfloat16)
                            vArr = MLXArray(vData, shape, type: Float.self).asType(.bfloat16)
                        }

                        log("  Layer \(i): K=\(kArr.shape) nb=\(nb)")
                        let _ = cache[i].update(keys: kArr, values: vArr)
                    }
                    log("All layers injected, evaluating cache...")
                    eval(cache)
                    let c0sum = MLX.sum(cache[0].state[0]).item(Float.self)
                    log(String(format: "Cache[0] K cksum: %.4f (bridge: %.4f) %@",
                        c0sum, ck, abs(c0sum - ck) < 0.01 ? "✓" : "✗"))

                    // --- Task 3: Decode correctness ---
                    log("--- DECODE (native prefill → Swift decode) ---")
                    let lastTok = MLXArray([Array(allTokens.prefix(16)).last!]).reshaped(1, 1)
                    var nativeDecode: [Int32] = []
                    var inp = lastTok
                    for _ in 0..<8 {
                        let logits = model(inp, cache: cache)
                        let next = MLX.argMax(logits[0..., -1, 0...], axis: -1)
                        eval(next)
                        let tok = next.item(Int32.self)
                        nativeDecode.append(tok)
                        inp = MLXArray([tok]).reshaped(1, 1)
                    }
                    log("Native decode: \(nativeDecode)")

                    log("--- DECODE (Swift prefill → Swift decode) ---")
                    let swiftCache = model.newCache(parameters: nil)
                    let swiftArr = MLXArray(Array(allTokens.prefix(16))).reshaped(1, 16)
                    let _ = model(swiftArr, cache: swiftCache); eval(swiftCache)

                    var swiftDecode: [Int32] = []
                    inp = lastTok
                    for _ in 0..<8 {
                        let logits = model(inp, cache: swiftCache)
                        let next = MLX.argMax(logits[0..., -1, 0...], axis: -1)
                        eval(next)
                        let tok = next.item(Int32.self)
                        swiftDecode.append(tok)
                        inp = MLXArray([tok]).reshaped(1, 1)
                    }
                    log("Swift decode:  \(swiftDecode)")
                    log("MATCH: \(nativeDecode == swiftDecode ? "YES ✓" : "NO ✗")")
                }
                log("After perform")
                dlclose(lib)

            } else {
                // Swift-only path
                for n in [16, 1024] {
                    let tokens = Array(allTokens.prefix(n))
                    try await container.perform { ctx in
                        let model = ctx.model
                        let arr = MLXArray(tokens).reshaped(1, n)
                        for _ in 0..<3 {
                            let c = model.newCache(parameters: nil)
                            let _ = model(arr, cache: c); eval(c)
                            Stream.gpu.synchronize(); MLX.Memory.clearCache()
                        }
                        var times: [Double] = []
                        for _ in 0..<5 {
                            let c = model.newCache(parameters: nil)
                            let t0 = CFAbsoluteTimeGetCurrent()
                            let _ = model(arr, cache: c); eval(c)
                            Stream.gpu.synchronize()
                            times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                            MLX.Memory.clearCache()
                        }
                        let avg = times.reduce(0, +) / 5.0
                        log(String(format: "Swift %4d tok: %.1fms (%.0f tok/s)",
                            n, avg, Double(n)/(avg/1000.0)))
                    }
                }
            }
            log("Done.")
        } catch {
            log("ERROR: \(error)")
        }
    }
}
