# NS-CPU-001 performance report

Measured on 2026-10-04 at commit `feat/cpu-f32-kernels` (after the vector25 backend). Raw JMH output:
[`jmh-NS-CPU-001.json`](jmh-NS-CPU-001.json); environment dump: [`env.txt`](env.txt). Tables were
generated with `python3 project/perf/jmh_report.py docs/perf/jmh-NS-CPU-001.json`.

## Environment

| Item | Value |
|---|---|
| CPU | AMD Ryzen 9 5900X (12 cores / 24 threads; WSL2 reports L2 6 MiB, L3 "32 MiB, 1 instance") |
| OS | Windows + WSL2, kernel 6.18.33.2-microsoft-standard-WSL2; 31 GB visible to the VM |
| JDK | Temurin 25.0.4+7-LTS, `--add-modules=jdk.incubator.vector`, default GC (G1), JMH default heap |
| Vector species | `FloatVector.SPECIES_PREFERRED` = 256-bit (8 lanes, AVX2) |
| Scala / sbt | 3.3.7 / 1.11.3, sbt-jmh 0.4.7 (JMH 1.37) |
| JMH | `@Fork(3)`, warmup 5 x 1 s, measurement 8 x 1 s, 1 thread, `-prof gc`, mode avgt |

**Not recorded:** process RSS, thermal throttling, warm/cold variation beyond JMH warmup. The desktop
was otherwise idle, but this is WSL2 on a desktop, not a dedicated benchmark host. Treat the absolute
numbers as indicative. Only single-thread kernels were measured; the 1/6/12/24-worker comparison of
§7.3 belongs with a real model workload and was not run.

## Benchmarks

* `legacyNpMatmul`: a copy of the numscala 0.2.0 float32 `np.matmul` path (`contiguous` + `toArray`
  copies, then boxed generic loops). This is the N0 baseline.
* `npMatmul`: `np.matmul` after this change (adapter, default `scalar` backend, fresh output array).
* `scalarGemmInto` / `vectorGemmInto`: the low-level kernels writing into a reused output.
* `KernelBench`: 512 random-row gather from a 32768 x 384 table; scatter-add of 512 rows onto 16 distinct
  ids; coalesce then scatter of the same; affine scan `[T=64, B=8, D=384]`; row log-sum-exp of 32 x 8192;
  axpy 512 x 384.


### GEMM (median us/op, p95 in parentheses; GFLOP/s from the median, FMA = 2 FLOPs)

| shape `MxNxK-op` | legacy np.matmul | np.matmul (new, scalar) | scalar gemmInto | vector25 gemmInto | vector GFLOP/s | B/op legacy | B/op np.matmul | B/op scalar | B/op vector |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x384x384-NN | 384.9 (398.2) | 40.1 (44.1) | 84.7 (120.4) | 14.7 (16.7) | 20.1 | 593,155 | 2,011 | 1 | 0 |
| 1x768x384-NN | 789.0 (801.8) | 88.7 (99.4) | 87.3 (223.1) | 32.4 (34.8) | 18.2 | 1,184,518 | 3,529 | 1 | 0 |
| 1x128x384-NN | 124.3 (125.2) | 15.7 (19.1) | 13.2 (13.8) | 3.8 (3.9) | 26.2 | 198,913 | 909 | 0 | 0 |
| 1x192x384-NN | 182.9 (186.5) | 20.3 (23.7) | 19.7 (20.0) | 5.4 (6.0) | 27.4 | 297,473 | 1,248 | 0 | 0 |
| 8x384x384-NN | 2,486.7 (2,518.8) | 332.3 (371.5) | 311.3 (314.1) | 35.3 (37.5) | 66.8 | 614,680 | 12,674 | 2 | 64 |
| 32x768x384-NN | 19,594.8 (20,423.9) | 2,601.7 (3,111.6) | 2,429.3 (2,459.4) | 292.3 (310.6) | 64.6 | 1,327,529 | 99,180 | 41 | 66 |
| 128x768x384-NN | 78,240.3 (84,009.4) | 10,393.6 (11,663.6) | 9,702.6 (9,832.2) | 1,120.8 (1,135.9) | 67.4 | 1,770,302 | 394,178 | 91 | 75 |
| 512x384x384-NN | 149,528.2 (162,123.5) | 20,617.4 (25,739.5) | 18,298.7 (18,464.9) | 2,110.2 (2,190.4) | 71.6 | 2,163,976 | 787,472 | 151 | 97 |
| 32x384x768-NT | 20,891.0 (22,719.7) | 3,507.4 (4,517.1) | 3,727.5 (4,102.2) | 395.4 (400.5) | 47.7 | 7,225,907 | 50,034 | 49 | 3 |
| 384x768x32-TN | 16,018.4 (17,211.3) | 3,165.4 (3,490.9) | 3,481.1 (3,810.7) | 330.4 (337.9) | 57.1 | 1,573,378 | 1,180,528 | 49 | 66 |
| 8x256x128-NT | 619.6 (656.5) | 102.2 (118.7) | 109.5 (124.9) | 11.9 (12.1) | 44.1 | 799,140 | 8,650 | 1 | 0 |
| 7x385x383-NN | 2,164.2 (2,200.5) | 325.6 (367.2) | 374.4 (437.2) | 55.3 (56.1) | 37.4 | 611,610 | 11,280 | 3 | 224 |

### Speedups (ratio of medians)

| shape | vector25 vs scalar gemmInto | new np.matmul vs legacy |
|---|---:|---:|
| 1x384x384-NN | 5.77x | 9.60x |
| 1x768x384-NN | 2.70x | 8.89x |
| 1x128x384-NN | 3.51x | 7.90x |
| 1x192x384-NN | 3.66x | 9.00x |
| 8x384x384-NN | 8.82x | 7.48x |
| 32x768x384-NN | 8.31x | 7.53x |
| 128x768x384-NN | 8.66x | 7.53x |
| 512x384x384-NN | 8.67x | 7.25x |
| 32x384x768-NT | 9.43x | 5.96x |
| 384x768x32-TN | 10.54x | 5.06x |
| 8x256x128-NT | 9.21x | 6.06x |
| 7x385x383-NN | 6.77x | 6.65x |
| **geometric mean** | **6.60x** | **7.29x** |

### Row ops, scan and reductions (median us/op; B/op)

| kernel | scalar | vector25 | vector vs scalar | B/op scalar | B/op vector |
|---|---:|---:|---:|---:|---:|
| affineScan64x8x384 | 91.11 | 73.75 | 1.24x | 2 | 0 |
| axpy512x384 | 83.32 | 49.96 | 1.67x | 1 | 0 |
| coalesceThenScatter512 | 94.51 | 92.56 | 1.02x | 33 | 29 |
| gatherRandom512 | 46.98 | 47.36 | 0.99x | 0 | 0 |
| rowLogSumExp32x8192 | 1,170.93 | 1,165.91 | 1.00x | 33 | 20 |
| scatterAddDuplicates512 | 54.58 | 15.84 | 3.45x | 0 | 0 |

Gate input (NS-CPU-001 §9.3): geometric mean of vector25/scalar over all 18 workloads = **3.93x**; worst workload = **0.99x** (gatherRandom512).

GC: the legacy path ran 80+ collections per run on the 1-token shapes, from boxing. The new kernels
ran none (`gc.count = 0` for 38 of the 60 configurations; the rest are `np.matmul` / legacy, which
allocate results). The few B/op on the `gemmInto` rows are JMH's per-iteration bookkeeping amortised
over very few calls (at most about 220 B/op, against 0.6–7 MB/op for the legacy path).

## Reading the results

* **Copies and boxing were the main cost of the old path.** Even with the same scalar algorithm,
  float32 `np.matmul` is 5–10x faster (geometric mean 7.29x). Allocation drops by 1–3 orders of
  magnitude: the remaining bytes are the result array itself.
* **The Vector API backend** reaches 45–72 GFLOP/s single-threaded on the batched shapes and
  15–27 GFLOP/s on 1-token GEMV shapes. It is 2.7–10.5x faster than the scalar kernels (geometric mean 6.60x).
* **Memory-bound kernels gain little from SIMD**, as §11 of the design expected. Gather (row copies),
  coalesce (sort) and log-sum-exp (`exp` stays scalar) are within ±2%. scatter-add (3.45x), axpy (1.67x)
  and scan (1.24x) improve.
* One anomaly: `scalarGemmInto` 1x384x384 (85 us, p95 120 us) is slower than `npMatmul` on the same
  shape (40 us), which runs the same scalar kernel. The wide p95 suggests a JIT-profile effect specific
  to that fork set, not a property of the kernel. Re-measure before relying on scalar 1-token numbers.

## After the determinism fix

The tables above were measured before `VectorF32Kernels` stopped using `reduceLanes(ADD)`. That
reduction's addition order is unspecified and differed between the interpreter and C2, which a
model-side resume test caught. The NT kernel and `rowSumSquaresInto` now sum lanes in a fixed order
(`DeterminismSuite`). Re-measured with the same JMH settings, only the NT shapes are affected:

| shape | vector25 before | vector25 after | vector vs scalar after |
|---|---:|---:|---:|
| 32x384x768-NT | 395.4 us | 422.7 us | 8.82x |
| 8x256x128-NT | 11.9 us | 14.4 us | 7.59x |

With these values the GEMM geometric mean is 6.48x and the all-workload geometric mean is 3.87x.

## Gate (NS-CPU-001 §9.3)

| Criterion | Result | Verdict |
|---|---|---|
| Correctness and existing compatibility | `sbt test` 656 passed; `vector25/test` 51 passed (contract, backend diff, determinism, integration model) | pass |
| Geometric-mean speedup of the optimised backend >= 1.20 | 3.87x over all 18 workloads (GEMM alone: 6.48x), after the determinism fix | pass |
| No important workload regresses by more than 10% | worst: `gatherRandom512` 0.99x (-1%) | pass |

So `vector25` qualifies as the **recommended backend wherever JDK 25 is available**
(`-Dnumscala.cpu.backend=vector25`, or `auto` to fall back silently). It stays an optional module and
the root default stays `scalar`, because the published `numscala` artifact must keep working on JDK 17
without incubator modules (§4.2). These are development-time measurements on one machine, not a
guarantee. The model project should confirm the gain in a whole training step (§9.3, last paragraph),
which this repository cannot measure without the model.
