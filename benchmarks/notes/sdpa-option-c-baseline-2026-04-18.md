# SDPA Option C — Phase 0 Baseline

Captured on legacy vector-path SDPA (`sdpa_vector` + `sdpa_vector_2pass`). Phase 1 unified-kernel output must match the `checksum` column within tolerance; Phase 2 AB migration must match byte-for-byte. The `median_us` column is the 1.25× perf floor the Phase 1 kernel must respect per-case (warning, not hard-fail).

Rows: 63

| Case key | Stratum | Det? | Checksum | median µs |
|---|---|---|---:|---:|
| `B1_Hq4_Hk4_Lq1_Lk1023_D64_nocausal_f32` | threshold | ✓ | 1.015504e+01 | 288.500000 |
| `B1_Hq4_Hk4_Lq1_Lk1023_D64_nocausal_f16` | threshold | ✓ | 1.071201e+01 | 407.292000 |
| `B1_Hq4_Hk4_Lq1_Lk1023_D64_nocausal_bf16` | threshold | ✓ | 1.083062e+01 | 305.833000 |
| `B1_Hq4_Hk4_Lq1_Lk1023_D64_causal_f32` | threshold | ✓ | 1.199107e+01 | 299.125000 |
| `B1_Hq4_Hk4_Lq1_Lk1023_D64_causal_f16` | threshold | ✓ | 9.013254e+00 | 282.125000 |
| `B1_Hq4_Hk4_Lq1_Lk1023_D64_causal_bf16` | threshold | ✓ | 1.102898e+01 | 274.292000 |
| `B1_Hq4_Hk4_Lq1_Lk1024_D64_nocausal_f32` | threshold | ✓ | 1.145824e+01 | 322.167000 |
| `B1_Hq4_Hk4_Lq1_Lk1024_D64_nocausal_f16` | threshold | ✓ | 1.160983e+01 | 263.000000 |
| `B1_Hq4_Hk4_Lq1_Lk1024_D64_nocausal_bf16` | threshold | ✓ | 1.025053e+01 | 798.583000 |
| `B1_Hq4_Hk4_Lq1_Lk1024_D64_causal_f32` | threshold | ✓ | 1.058025e+01 | 309.250000 |
| `B1_Hq4_Hk4_Lq1_Lk1024_D64_causal_f16` | threshold | ✓ | 1.009115e+01 | 289.416000 |
| `B1_Hq4_Hk4_Lq1_Lk1024_D64_causal_bf16` | threshold | ✓ | 1.112143e+01 | 397.875000 |
| `B1_Hq4_Hk4_Lq1_Lk1025_D64_nocausal_f32` | threshold | ✓ | 1.055516e+01 | 291.542000 |
| `B1_Hq4_Hk4_Lq1_Lk1025_D64_nocausal_f16` | threshold | ✓ | 1.066033e+01 | 277.334000 |
| `B1_Hq4_Hk4_Lq1_Lk1025_D64_nocausal_bf16` | threshold | ✓ | 1.074605e+01 | 285.500000 |
| `B1_Hq4_Hk4_Lq1_Lk1025_D64_causal_f32` | threshold | ✓ | 1.114183e+01 | 281.708000 |
| `B1_Hq4_Hk4_Lq1_Lk1025_D64_causal_f16` | threshold | ✓ | 1.105931e+01 | 303.792000 |
| `B1_Hq4_Hk4_Lq1_Lk1025_D64_causal_bf16` | threshold | ✓ | 9.561234e+00 | 553.667000 |
| `B1_Hq4_Hk4_Lq1_Lk4095_D64_nocausal_f32` | threshold | ✓ | 5.194726e+00 | 654.166000 |
| `B1_Hq4_Hk4_Lq1_Lk4095_D64_nocausal_f16` | threshold | ✓ | 5.620897e+00 | 304.000000 |
| `B1_Hq4_Hk4_Lq1_Lk4095_D64_nocausal_bf16` | threshold | ✓ | 5.173139e+00 | 351.333000 |
| `B1_Hq4_Hk4_Lq1_Lk4095_D64_causal_f32` | threshold | ✓ | 5.110060e+00 | 316.291000 |
| `B1_Hq4_Hk4_Lq1_Lk4095_D64_causal_f16` | threshold | ✓ | 4.866494e+00 | 366.000000 |
| `B1_Hq4_Hk4_Lq1_Lk4095_D64_causal_bf16` | threshold | ✓ | 4.996672e+00 | 305.792000 |
| `B1_Hq4_Hk4_Lq1_Lk4096_D64_nocausal_f32` | threshold | ✓ | 5.558503e+00 | 347.917000 |
| `B1_Hq4_Hk4_Lq1_Lk4096_D64_nocausal_f16` | threshold | ✓ | 5.151876e+00 | 307.083000 |
| `B1_Hq4_Hk4_Lq1_Lk4096_D64_nocausal_bf16` | threshold | ✓ | 5.836443e+00 | 322.167000 |
| `B1_Hq4_Hk4_Lq1_Lk4096_D64_causal_f32` | threshold | ✓ | 5.312666e+00 | 346.708000 |
| `B1_Hq4_Hk4_Lq1_Lk4096_D64_causal_f16` | threshold | ✓ | 4.837635e+00 | 332.833000 |
| `B1_Hq4_Hk4_Lq1_Lk4096_D64_causal_bf16` | threshold | ✓ | 4.774035e+00 | 352.917000 |
| `B1_Hq4_Hk4_Lq1_Lk4097_D64_nocausal_f32` | threshold | ✓ | 5.579906e+00 | 343.375000 |
| `B1_Hq4_Hk4_Lq1_Lk4097_D64_nocausal_f16` | threshold | ✓ | 5.509085e+00 | 292.500000 |
| `B1_Hq4_Hk4_Lq1_Lk4097_D64_nocausal_bf16` | threshold | ✓ | 5.299890e+00 | 654.458000 |
| `B1_Hq4_Hk4_Lq1_Lk4097_D64_causal_f32` | threshold | ✓ | 4.731006e+00 | 337.500000 |
| `B1_Hq4_Hk4_Lq1_Lk4097_D64_causal_f16` | threshold | ✓ | 5.060500e+00 | 301.833000 |
| `B1_Hq4_Hk4_Lq1_Lk4097_D64_causal_bf16` | threshold | ✓ | 5.737053e+00 | 363.208000 |
| `B1_Hq4_Hk4_Lq1_Lk1_D64_causal_f16` | diag_Lk | ✓ | 1.913527e+02 | 230.666000 |
| `B1_Hq4_Hk4_Lq1_Lk32_D64_causal_f16` | diag_Lk | ✓ | 5.181374e+01 | 231.333000 |
| `B1_Hq4_Hk4_Lq1_Lk64_D64_causal_f16` | diag_Lk | ✓ | 4.112006e+01 | 245.583000 |
| `B1_Hq4_Hk4_Lq1_Lk96_D64_causal_f16` | diag_Lk | ✓ | 3.407981e+01 | 230.375000 |
| `B1_Hq4_Hk4_Lq1_Lk128_D64_causal_f16` | diag_Lk | ✓ | 2.700492e+01 | 219.583000 |
| `B1_Hq4_Hk4_Lq1_Lk256_D64_causal_f16` | diag_Lk | ✓ | 1.855499e+01 | 699.958000 |
| `B1_Hq4_Hk4_Lq1_Lk768_D64_causal_f16` | diag_Lk | ✓ | 1.177109e+01 | 271.834000 |
| `B1_Hq4_Hk4_Lq1_Lk2048_D64_causal_f16` | diag_Lk | ✓ | 7.631152e+00 | 289.042000 |
| `B1_Hq4_Hk4_Lq1_Lk8192_D64_causal_f16` | diag_Lk | ✓ | 3.977320e+00 | 364.791000 |
| `B1_Hq4_Hk4_Lq1_Lk128_D64_causal_f16` | diag_D | ✓ | 2.720076e+01 | 255.959000 |
| `B1_Hq4_Hk4_Lq1_Lk128_D96_causal_f16` | diag_D | ✓ | 3.837166e+01 | 221.875000 |
| `B1_Hq4_Hk4_Lq1_Lk128_D128_causal_f16` | diag_D | ✓ | 6.877504e+01 | 210.292000 |
| `B1_Hq4_Hk4_Lq1_Lk128_D256_causal_f16` | diag_D | ✓ | 1.110345e+02 | 355.625000 |
| `B1_Hq1_Hk1_Lq1_Lk128_D64_causal_f16` | diag_Hq | ✓ | 7.443863e+00 | 213.500000 |
| `B1_Hq4_Hk4_Lq1_Lk128_D64_causal_f16` | diag_Hq | ✓ | 2.602850e+01 | 251.833000 |
| `B1_Hq8_Hk4_Lq1_Lk128_D64_causal_f16` | diag_Hq | ✓ | 5.161447e+01 | 236.000000 |
| `B1_Hq32_Hk4_Lq1_Lk128_D64_causal_f16` | diag_Hq | ✓ | 2.467281e+02 | 239.750000 |
| `B1_Hq8_Hk8_Lq1_Lk128_D64_causal_f16` | diag_gqa | ✓ | 6.194412e+01 | 399.709000 |
| `B1_Hq8_Hk4_Lq1_Lk128_D64_causal_f16` | diag_gqa | ✓ | 5.629164e+01 | 232.583000 |
| `B1_Hq8_Hk2_Lq1_Lk128_D64_causal_f16` | diag_gqa | ✓ | 5.336073e+01 | 250.583000 |
| `B1_Hq8_Hk1_Lq1_Lk128_D64_causal_f16` | diag_gqa | ✓ | 6.061090e+01 | 479.792000 |
| `B1_Hq4_Hk4_Lq1_Lk128_D64_causal_f16` | diag_B | ✓ | 2.709401e+01 | 219.417000 |
| `B2_Hq4_Hk4_Lq1_Lk128_D64_causal_f16` | diag_B | ✓ | 5.703309e+01 | 283.542000 |
| `B4_Hq4_Hk4_Lq1_Lk128_D64_causal_f16` | diag_B | ✓ | 1.113448e+02 | 239.000000 |
| `B1_Hq4_Hk4_Lq1_Lk128_D64_causal_f16` | diag_Lq | ✓ | 2.513481e+01 | 228.833000 |
| `B1_Hq4_Hk4_Lq4_Lk128_D64_causal_f16` | diag_Lq | ✓ | 1.204628e+02 | 275.625000 |
| `B1_Hq4_Hk4_Lq8_Lk128_D64_causal_f16` | diag_Lq | ✓ | 2.356975e+02 | 236.083000 |
