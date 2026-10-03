# numscala

**A nearly complete port of [NumPy](https://numpy.org) to Scala 3.**

n-dimensional strided arrays with views and broadcasting, NumPy's dtype system and type
promotion (statically typed), fancy and boolean indexing, ufuncs, reductions, sorting and
set routines, `np.linalg`, `np.fft`, `np.random` (bit-compatible with NumPy's generators),
`np.polynomial`, `np.ma`, `np.strings`, `.npy`/`.npz` I/O and `np.testing` — with output
formatted exactly like NumPy's `str()`/`repr()`.

```scala
libraryDependencies += "com.github.kmizu" %% "numscala" % "0.3.0"
```

Requires Scala 3.3+ and Java 17+. No dependencies.

> The library was first published as `"com.github.kmizu" %% "num-scala" % "0.1.0"` with
> package `numscala`. That artifact stays on Maven Central (published artifacts cannot be
> removed) but is superseded: use `numscala` and `import com.github.kmizu.numscala.*`.

```scala
import com.github.kmizu.numscala.*

val a = np.arange(12.0).reshape(3, 4)
println(a)
// [[ 0.  1.  2.  3.]
//  [ 4.  5.  6.  7.]
//  [ 8.  9. 10. 11.]]

a(1, ::)                 // row 1 (a view)                 a[1, :]
a(::, "1:3")             // columns 1..2                    a[:, 1:3]
a("::-1", 0)             // reversed first column           a[::-1, 0]
a(a > 5.0)               // boolean mask                    a[a > 5]
a(np.array(0, 2), ::)    // fancy indexing                  a[[0, 2], :]
a(---, None)             // ellipsis + new axis             a[..., np.newaxis]

(a * 2.0 + 1.0).sum()    // 144.0
a.mean(0).repr           // array([4., 5., 6., 7.])
a @@ a.T                 // matrix product                  a @ a.T
np.sqrt(np.arange(4))    // int -> float64 like NumPy

val b = a.copy()
b(0, ::) = 0.0           // assignment through a view       b[0, :] = 0
b(b > 9.0) = -1.0        // masked assignment               b[b > 9] = -1
```

## NumPy → numscala cheat sheet

| NumPy | numscala |
|---|---|
| `import numpy as np` | `import com.github.kmizu.numscala.*` |
| `np.array([[1, 2], [3, 4]])` | `np.array(Seq(Seq(1, 2), Seq(3, 4)))` or `np.array(Seq(1, 2), Seq(3, 4))` |
| `np.array([1, 2], dtype=np.float32)` | `np.arrayOf(Seq(1, 2), DType.Float32)` |
| `np.zeros((2, 3))`, `np.zeros(3, dtype=int)` | `np.zeros(2, 3)`, `np.zeros[Int](3)` |
| `a.astype(np.uint8)` | `a.astype[UInt8]` |
| `a[i, j]` (scalar) | `a(i, j)` |
| `a[i]` on a 2-D array | `a(i, ::)` or `a(i, ---)` |
| `a[1:3]`, `a[::-1]`, `a[-3:]` | `a("1:3")` or `a(1 until 3)`, `a("::-1")`, `a("-3:")` |
| `a[..., None]` | `a(---, None)` |
| `a == b`, `a != b` | `a === b`, `a =!= b` |
| `a @ b` | `a @@ b` |
| `a // b`, `a //= b` | ``a `//` b``, ``a `//=` b`` (or `np.floor_divide(a, b)`) |
| `a.sum()`, `a.sum(axis=0)` | `a.sum()`, `a.sum(0)` / `np.sum(a, axis = 0)` |
| `a.T`, `a.reshape(-1, 2)` | `a.T`, `a.reshape(-1, 2)` |
| `print(a)`, `repr(a)` | `println(a)`, `a.repr` |

## Typed dtypes

`NDArray[T]` is parameterised by its element type; the runtime `DType[T]` mirrors
`numpy.dtype`:

| Scala `T` | dtype | | Scala `T` | dtype |
|---|---|---|---|---|
| `Boolean` | `bool` | | `UInt8` … `UInt64` | `uint8` … `uint64` |
| `Byte`, `Short`, `Int`, `Long` | `int8` … `int64` | | `Float`, `Double` | `float32`, `float64` |
| `Complex` | `complex128` | | `String` | `str` |

Result dtypes follow NumPy's promotion rules and are known statically:

```scala
val i = np.array(1, 2, 3)          // NDArray[Int]    (int32)
val f = np.array(1.0f, 2.0f, 3.0f) // NDArray[Float]  (float32)
val r: NDArray[Double] = i + f     // int32 + float32 -> float64
val q: NDArray[Double] = i / i     // true division   -> float64
val s: Long = i.sum()              // small ints accumulate in int64
val m: Double = i.mean()
```

## Modules

Every sample below is checked by `src/test/scala/com/github/kmizu/numscala/ReadmeSuite.scala`.

```scala
// numpy.linalg — LU/QR/SVD/eig implemented in pure Scala, float64 and complex128, batched
val A = np.array(Seq(Seq(3.0, 1.0), Seq(1.0, 2.0)))
np.linalg.solve(A, np.array(9.0, 8.0))      // array([2., 3.])
val sv = np.linalg.svd(A)                     // sv.U, sv.S, sv.Vh
np.linalg.eigh(A).eigenvalues
np.einsum("ij,jk->ik", A, A)

// numpy.fft — any length (Bluestein for non powers of two)
np.fft.fft(np.array(0.0, 1.0, 0.0, -1.0))
np.fft.rfft(signal); np.fft.fftfreq(4)

// numpy.random — bit-for-bit identical streams to NumPy (PCG64, MT19937, SeedSequence, ...)
val rng = np.random.default_rng(42)
rng.random(3)                                 // 0.7739560485559633, 0.4388784397520523, ...
rng.normal(0.0, 1.0, 1000)
np.random.seed(0); np.random.rand(1)          // 0.5488135039273248, like NumPy

// ufuncs with reduce / accumulate / outer / at / reduceat
np.add.reduce(np.arange(5))                   // 10
np.multiply.outer(np.array(1, 2), np.array(1, 2, 3))

// statistics, sorting, sets
np.median(x); np.percentile(x, 50.0); np.histogram(x, 10); np.cov(m)
np.unique(x); np.argsort(x); np.searchsorted(x, v); np.where(c, x, y)

// numpy.polynomial
val p = np.polynomial.Polynomial.fit(xs, ys, 2)
np.polyval(np.array(1.0, 0.0, -1.0), np.array(2.0))

// numpy.ma
val m = np.ma.masked_less(np.array(1.0, -2.0, 3.0), 0.0)
println(m)                                    // [1.0 -- 3.0]
m.mean()                                      // 2.0

// numpy.strings / numpy.char
np.strings.upper(np.array("ab", "cd"))

// .npy / .npz files interoperate with NumPy byte-for-byte
np.save("a.npy", A); np.load("a.npy", DType.Float64)
np.savez("arrays.npz", "a" -> A); np.load_npz("arrays.npz")

// numpy.testing
np.testing.assert_allclose(actual, expected, rtol = 1e-7)
```

Also available: shape manipulation (`concatenate`, `stack`, `split`, `pad`, `tile`, `roll`, …),
`np.lib.stride_tricks.sliding_window_view`, `np.emath`, `np.printoptions`, `finfo`/`iinfo`,
window functions, `packbits`, `einsum`, and more — about 420 of the ~425 public NumPy
functions.

## Differences from NumPy

Scala is statically typed and has no `__getitem__` syntax, so a few things look different:

* **Indexing.** `a(i, j)` with integers returns an element; anything else (`::`, slice strings
  like `"1:-1"`, `Range`s, `None`, `---`, index arrays, masks) returns an array. A row of a
  2-D array is `a(i, ::)`, not `a(i)`.
* **Operators.** Elementwise equality is `===` / `=!=`; matrix product is `@@`;
  floor division is ``a `//` b`` — backquoted, because a bare `//` starts a comment in Scala.
  It has the same precedence as `*` and `/`, like NumPy's `//`.
* **Scalars are "weak" like Python scalars (NEP 50).** A Scala `Boolean`, `Int`, `Long`,
  `Double` or `Complex` scalar takes the array's dtype unless its kind is higher:
  `float32Array + 2.0` is float32, `int8Array + 1` is int8, `int32Array + 2.5` is float64.
  An int scalar outside the array's range throws `ArithmeticException` (NumPy's
  `OverflowError`); comparisons never overflow (`int8Array < 300` is all true). `Byte`,
  `Short`, `Float` and unsigned scalars act like NumPy scalars and promote normally. The
  same rules apply to ufuncs (`np.add(float32Array, 2.0)`). A bool array plus an `Int` is
  int32 (NumPy: int64), numscala's default integer.
* **Index results are `NDArray[Int]`** (NumPy: int64), since JVM arrays are Int-indexed.
* **Results that NumPy returns as scalars from per-matrix linalg functions** (`det`, `cond`,
  `matrix_rank`) are 0-d arrays; `eig`/`eigvals` always return complex arrays.
* **Not supported:** `datetime64`/`timedelta64`, `float16`/`complex64` dtypes (read from
  `.npy` as float32/complex128), object and structured arrays, `np.matrix`, `memmap`,
  pickled `.npy` content. `np.seterr` modes are recorded but floating-point errors are never
  raised (computation follows IEEE like NumPy's `"ignore"`).

## Performance

Arrays are backed by primitive JVM arrays; hot paths (arithmetic on float64/int32/int64,
reductions with NumPy's pairwise summation, sorting, matmul) use specialised loops. Generic
dtypes go through boxed element access and are several times slower. Run
`sbt "Test/runMain com.github.kmizu.numscala.bench.Bench"` to measure on your machine.

### Float32 CPU kernels

`com.github.kmizu.numscala.cpu` adds allocation-free Float32 kernels for repeated CPU work, such as a
small model's training loop: `gemmInto` into caller-owned buffers, row gather/scatter-add/coalesce,
sigmoid/SiLU, stable row log-sum-exp, an elementwise affine scan, and worker-owned `Workspace`s that
report how much was allocated, packed and converted. `np.matmul` on float32 uses them directly on the
arrays' buffers (transposed and gapped views included) instead of copying. An optional JDK 25 Vector API
backend lives in `vector25/`. See [docs/CPU_KERNELS.md](docs/CPU_KERNELS.md).

```scala
import com.github.kmizu.numscala.cpu.*
val ws = new Workspace()
val (x, w, y) = (MatrixF32.zeros(32, 384), MatrixF32.zeros(768, 384), MatrixF32.zeros(32, 768))
ScalarF32Kernels.gemmInto(x, Transpose.No, w, Transpose.Yes, y, ws)   // y := x w^T, y reused
```

## Building

```bash
sbt test                           # run the test suite
sbt "Test/runMain com.github.kmizu.numscala.bench.Bench"     # micro benchmarks
sbt publishLocal                   # install locally
sbt vector25/test                  # optional Vector API backend (JDK 25)
sbt "benchmarks/Jmh/run .*GemmBench.*"                       # JMH kernel benchmarks (JDK 25)
```

Publishing to Maven Central: see [docs/PUBLISHING.md](docs/PUBLISHING.md).
Contributing / architecture: see [docs/DEVELOPING.md](docs/DEVELOPING.md).

## License

BSD 3-Clause (like NumPy).
