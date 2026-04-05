# Centroid Comparison: Beta vs N(0,1) Lloyd-Max

Compared Eric's Beta distribution centroids against our N(0,1) Lloyd-Max centroids
(from llama.cpp ggml-turbo-quant.c) at dim=128, 4-bit.

## Result: ~2% difference, not worth changing

| Metric | Value |
|--------|-------|
| Mean absolute diff | 0.002209 |
| Max relative diff | 2.34% |

Eric's approach (Beta distribution after unit-sphere normalization) is more theoretically
correct. Our N(0,1) approach works because raw WHT-rotated values approximate N(0, sigma^2/d).
Both are valid. The 2% centroid difference has negligible impact on reconstruction MSE.

**Decision: keep Beta distribution centroids.** They match the actual distribution of
unit-sphere-rotated coordinates more precisely.
