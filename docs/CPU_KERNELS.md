# Float32 CPU kernels (`com.github.kmizu.numscala.cpu`)

Low-level Float32 kernels for code that runs the same CPU computation many times (for example a small
language model written in Scala). They implement the connection contract **NS-CPU-KERNEL-1** of the
design document NS-CPU-001. They sit next to the NumPy-style API and do not replace it: `NDArray`, dtype
promotion, views, `.toArray`/`.copy()` and the file formats behave as before. `np.matmul` on float32
now runs on these kernels.

```scala
import com.github.kmizu.numscala.cpu.*

val k  = F32Backend.default.kernels          // scalar unless -Dnumscala.cpu.backend=vector25|auto
val ws = new Workspace()                     // one per worker thread
val x  = MatrixF32.zeros(32, 384)            // [batch, in]
val w  = MatrixF32.zeros(768, 384)           // weights stored [out, in]
val y  = MatrixF32.zeros(32, 768)
k.gemmInto(x, Transpose.No, w, Transpose.Yes, y, ws)   // y := x w^T, y reused across steps
```

## What is provided (K1–K7)

| ID | API | Notes |
|---|---|---|
| K1 | `MatrixF32(data, offset, rows, cols, rowStride)`, `Transpose`, `gemmInto` | row-major, rows contiguous, padded rows allowed; `C := alpha op(A) op(B) + beta C` |
| K2 | `gatherRowsInto`, `scatterAddRowsInto`, `coalesceRowsInto` | `Int` row ids, out-of-range ids throw (no negative wrap-around); duplicates accumulate |
| K3 | `copyInto`, `fillInto`, `axpyInto`, `mulInto`, `sigmoidInto`, `siluInto`, `rowLogSumExpInto`, `rowSumSquaresInto` | overflow-free sigmoid/SiLU; stable log-sum-exp |
| K4 | `affineScanInto` | `s(t) = a(t) * s(t-1) + b(t)` elementwise over `[time, batch, feature]` stored as `(T*B) x D` |
| K5 | `Workspace`, `WorkspaceStats` | worker-owned scratch; `allocatedBytes`, `bytesPacked`, `bytesConverted`, `growCount` |
| K6 | `ScalarF32Kernels`, `numscala-vector25`, `F32Backend`, `KernelDiagnostics` | scalar reference always available; optional Vector API backend |
| K7 | `NDArrayF32Adapter`, `np.matmul` | existing NDArray/dtype/NumPy-compatible I/O |

Out of scope by design: tokenizers, layers, autograd, optimizers, losses as training objects,
checkpoints. For example `scatterAddRowsInto` exists, but whether a row then gets AdaGrad is up to the model.

## Contract

* **Validation first.** Every kernel checks shapes, buffer bounds (in `Long`, so no `Int` overflow),
  ids, aliasing and workspace capacity before it writes anything. A rejected call leaves outputs untouched.
* **GEMM special cases.** `M == 0` or `N == 0` touches nothing. `K == 0` or `alpha == 0` never reads A/B
  and only does `C := beta C`. `beta == 0` never reads C, so uninitialised or NaN-filled outputs are fine.
  Otherwise NaN/Infinity propagate: no `a == 0` shortcut, so `0 * inf` is NaN.
* **Aliasing.** Outputs must not overlap inputs. The exceptions are elementwise ops (`axpyInto`, `mulInto`,
  `sigmoidInto`, `siluInto`, `copyInto`, and `affineScanInto` for `a`/`b`), where the output may be
  *exactly* the same range as an input. A and B of a GEMM may overlap, since both are read-only. The
  check is conservative: same buffer plus intersecting address ranges.
* **Copies.** Row-major operands are used in place. The only GEMM layout that packs is `A^T B^T`, which
  packs `op(B)` into the workspace and counts it in `bytesPacked`. The NDArray adapter also packs views
  with negative or non-unit column strides, again counted. Broadcast batch axes are addressed per batch
  and never materialised.
* **Workspaces.** Not thread-safe: one per worker. Capacity grows only through `reserveFloats` /
  `reserveLongs`, never silently in a hot loop. Use `gemmWorkspaceFloats` / `coalesceWorkspaceLongs`
  to size it. `reset()` does not zero anything. Nothing a kernel borrows survives the call.
  `new Workspace(debug = true)` throws when a second thread uses it.
* **Threads.** Kernels are single-threaded and never create pools; callers split work across their
  own workers.
* **Numerics.** Computation is Float32. The scalar backend's row reductions accumulate in Double.
  Backends may differ in rounding (FMA, summation order); bit equality between backends is not promised.
  The tests use the componentwise bound `1e-6 + 4 γ(2K+4) (|alpha| Σ|a b| + |beta c|)`.
  `rowLogSumExpInto` returns NaN for a row with NaN, `+inf` for a row containing `+inf`, and `-inf` for
  all-`-inf` or empty rows. `silu(-inf) = -0`.

## Backends

| Backend | Where | Requirements |
|---|---|---|
| `scalar` | `numscala` (root artifact) | JDK 17+ |
| `vector25` | `vector25/` module (`numscala-vector25`, not yet published) | JDK 25 started with `--add-modules=jdk.incubator.vector` |

`F32Backend.select("scalar" | "vector25" | "auto")` returns a `BackendSelection` with the reason for the
choice. Asking for `vector25` when it cannot load throws with that reason. Only `auto` falls back to
`scalar`. The module is checked before any Vector API class is loaded. `F32Backend.default` is chosen
once from the system property `numscala.cpu.backend` (default `scalar`) and is what `np.matmul` uses
for float32. `KernelDiagnostics.report(selection, workspace)` gives one line for logs and benchmark records.

The vector backend uses `FloatVector.SPECIES_PREFERRED` (8 lanes on AVX2) with FMA. GEMM uses a 4x2
register-blocked micro-kernel for NN/TN/TT and a 1x4 dot-product block for NT. Every lane reduction sums in a fixed order (never `reduceLanes`), so a given backend instance and shape give bit-identical results run to run and across JIT tiers. GEMMs with
`m*n*k < 2048` stay on the scalar loops; the threshold is fixed per instance
(`new VectorF32Kernels(threshold)`).

## Building and running

```bash
sbt test                      # root only, JDK 17+ (the published artifact)
sbt vector25/test             # JDK 25; tests fork with --add-modules=jdk.incubator.vector
sbt "benchmarks/Jmh/run -prof gc .*GemmBench.*"   # JMH, see docs/perf/
```

The root project does not aggregate `vector25` or `benchmarks`, so `sbt test`, `publishLocal` and the
release still build only `numscala`.

## Measured performance

See [perf/NS-CPU-001-report.md](perf/NS-CPU-001-report.md).
