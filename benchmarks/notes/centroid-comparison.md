# Centroid A/B Test: Beta vs N(0,1) Lloyd-Max

Empirical comparison on M5 Max, Qwen3.5-2B 8bit, turbo3.

## Results

| Context | Beta PPL | N(0,1) PPL | Beta Decode | N(0,1) Decode |
|---------|----------|------------|-------------|---------------|
| 128 | **1.72** | 2.53 | 156.5 | 158.4 |
| 1024 | **2.12** | 2.20 | 156.3 | 157.0 |
| 4096 | 2.33 | **2.14** | 153.9 | 151.4 |

## Verdict: Beta wins

Beta centroids (unit-sphere normalization) dramatically better at short context
(47% lower PPL at 128 tokens). N(0,1) slightly better at 4K but the short-context
advantage makes Beta the right default.

Speed is identical between both — the centroid values don't affect kernel performance.

Toggle available via TURBO_USE_N01_CENTROIDS=1 env var for further testing.
