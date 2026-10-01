# num-scala

**A nearly complete port of [NumPy](https://numpy.org) to Scala 3.**

n-dimensional strided arrays with views and broadcasting, NumPy's dtype system and type
promotion (statically typed), fancy and boolean indexing, ufuncs, reductions, sorting and
set routines, `np.linalg`, `np.fft`, `np.random` (bit-compatible with NumPy's generators),
`np.polynomial`, `np.ma`, `np.strings`, `.npy`/`.npz` I/O and `np.testing` — with output
formatted exactly like NumPy's `str()`/`repr()`.

```scala
libraryDependencies += "com.github.kmizu" %% "num-scala" % "<version>"
```

Requires Scala 3.3+ and Java 17+. No dependencies.

```scala
import numscala.*

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

## NumPy → num-scala cheat sheet

| NumPy | num-scala |
|---|---|
| `import numpy as np` | `import numscala.*` |
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
| `a // b` | `a.floorDiv(b)` / `np.floor_divide(a, b)` |
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

## Building

```bash
sbt test                           # run the test suite
sbt "Test/runMain bench.Bench"     # micro benchmarks
sbt publishLocal                   # install locally
```

Publishing to Maven Central: see [docs/PUBLISHING.md](docs/PUBLISHING.md).
Contributing / architecture: see [docs/DEVELOPING.md](docs/DEVELOPING.md).

## License

BSD 3-Clause (like NumPy).
