// Minimal repro: MLX GPU compute path reads wrong buffer for live arrays
//
// PROVEN: CPU memory is stable. MLX sum(abs()) returns wrong values.
// This test splits the observer to isolate where the GPU path diverges.

#include <cstdio>
#include <cstring>
#include <cmath>
#include "mlx/mlx.h"

using namespace mlx::core;

// CPU-side float16 abssum for ground truth
static double cpu_f16_abssum(const array& a) {
    auto* raw = static_cast<_Float16*>(const_cast<allocator::Buffer&>(a.buffer()).raw_ptr());
    double s = 0;
    for (int i = 0; i < (int)a.size(); i++) {
        s += std::abs(static_cast<float>(raw[i]));
    }
    return s;
}

// CPU-side: read first N float16 values and return a checksum
static uint64_t cpu_f16_hash(const array& a, int n = 256) {
    auto* raw = static_cast<uint16_t*>(const_cast<allocator::Buffer&>(a.buffer()).raw_ptr());
    uint64_t h = 0;
    for (int i = 0; i < std::min(n, (int)a.size()); i++) {
        h = h * 31 + raw[i];
    }
    return h;
}

extern "C" {

int gp_repro_allocator_bug(void) {
    fprintf(stderr, "\n=== MLX GPU BINDING BUG REPRO ===\n");

    // 1. Create and eval a persistent array
    auto long_lived = random::normal({32768, 64}, float16);
    eval({long_lived});

    double cpu_baseline = cpu_f16_abssum(long_lived);
    uint64_t hash_baseline = cpu_f16_hash(long_lived);
    void* buf_ptr = long_lived.buffer().raw_ptr();
    fprintf(stderr, "baseline: cpu_abssum=%.2f buf=%p hash=%llx\n",
        cpu_baseline, buf_ptr, hash_baseline);

    // 2. Test MLX ops on long_lived BEFORE any pressure
    {
        auto a = abs(long_lived);
        eval({a});
        double cpu_abs = cpu_f16_abssum(a);
        float mlx_sum_a = sum(a).item<float>();
        fprintf(stderr, "pre-pressure: abs cpu_abssum=%.2f  sum(abs) via mlx=%.6f\n",
            cpu_abs, mlx_sum_a);
    }

    // 3. One round of pressure
    {
        int D = 4096;
        auto x = random::normal({4, D}, float16);
        auto q = matmul(x, transpose(random::normal({D, D}, float16)));
        auto k = matmul(x, transpose(random::normal({D, D}, float16)));
        auto v = matmul(x, transpose(random::normal({D, D}, float16)));
        auto out = matmul(x, transpose(random::normal({D, D}, float16)));
        auto gate = matmul(x, transpose(random::normal({14336, D}, float16)));
        auto up = matmul(x, transpose(random::normal({14336, D}, float16)));
        auto down = matmul(gate * sigmoid(gate) * up,
                           transpose(random::normal({D, 14336}, float16)));
        eval({out, down});
    }

    // 4. Test each op individually AFTER pressure
    fprintf(stderr, "\n--- after 1 round of pressure ---\n");

    // 4a. CPU ground truth
    double cpu_after = cpu_f16_abssum(long_lived);
    uint64_t hash_after = cpu_f16_hash(long_lived);
    void* ptr_after = long_lived.buffer().raw_ptr();
    fprintf(stderr, "cpu: abssum=%.2f hash=%llx ptr=%p %s\n",
        cpu_after, hash_after, ptr_after,
        (hash_after == hash_baseline) ? "STABLE" : "CHANGED");

    // 4b. abs() alone — eval it, then check its output on CPU
    {
        auto a = abs(long_lived);
        eval({a});
        void* abs_ptr = const_cast<allocator::Buffer&>(a.buffer()).raw_ptr();
        double cpu_abs_out = cpu_f16_abssum(a);
        fprintf(stderr, "abs(long_lived): output cpu_abssum=%.2f  output buf=%p  same_as_input=%s\n",
            cpu_abs_out, abs_ptr, (abs_ptr == buf_ptr) ? "YES(donated!)" : "no");
    }

    // 4c. sum() on a tiny known-good array
    {
        auto tiny = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
        eval({tiny});
        float s = sum(tiny).item<float>();
        fprintf(stderr, "sum([1..5]): mlx=%.6f expected=15.0 %s\n",
            s, (std::abs(s - 15.0f) < 0.01f) ? "OK" : "WRONG");
    }

    // 4d. Full sum(abs(long_lived)) via MLX
    {
        float mlx_val = sum(abs(long_lived)).item<float>();
        fprintf(stderr, "sum(abs(long_lived)): mlx=%.6f cpu=%.2f ratio=%.4f %s\n",
            mlx_val, cpu_after, mlx_val / cpu_after,
            (std::abs(mlx_val - cpu_after) / cpu_after > 0.01) ? "DIVERGED" : "OK");
    }

    // 4e. Force CPU eval path if possible: convert to float32, eval, read back
    {
        auto f32 = astype(long_lived, float32);
        eval({f32});
        auto* f32_raw = static_cast<float*>(f32.buffer().raw_ptr());
        double cpu_f32_sum = 0;
        for (int i = 0; i < std::min(1000, (int)f32.size()); i++) {
            cpu_f32_sum += std::abs(f32_raw[i]);
        }
        float mlx_f32_sum = sum(abs(f32)).item<float>();
        fprintf(stderr, "astype(f32): cpu_first1000=%.2f  mlx_full=%.6f\n",
            cpu_f32_sum, mlx_f32_sum);
    }

    fprintf(stderr, "=== END REPRO ===\n\n");
    return 0;
}

} // extern "C"
