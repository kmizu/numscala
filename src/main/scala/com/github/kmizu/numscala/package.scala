package com.github.kmizu.numscala

/** Anything accepted as one component of an index expression `a(i0, i1, ...)`:
  *
  *  - `Int`              — integer index (`a[3]`)
  *  - `::`               — the full slice (`a[:]`)
  *  - `String`           — Python slice syntax: `"1:3"`, `"::-1"`, `"..."`, `"None"`
  *  - `Range`            — `1 until 4`, `0 until 10 by 2` (as a slice)
  *  - `None`             — new axis (`np.newaxis`)
  *  - `NDArray[Int]`     — integer-array (fancy) indexing
  *  - `NDArray[Boolean]` — boolean mask
  *  - `Seq[Int]`, `Array[Int]`, `Seq[Boolean]`, `Array[Boolean]`
  *  - [[Index]]          — an explicit index object (`Index.Slice`, `np.newaxis`, `---`)
  */
type IndexLike = Int | Index | String | Range | NDArray[?] | Array[?] | Seq[?] |
  scala.collection.immutable.::.type | None.type

/** An axis argument: a single axis or several axes. */
type Axis = Int | Seq[Int]

/** The ellipsis index (Python's `...`). */
val --- : Index = Index.Ellipsis

/** Builds a slice index, like Python's `slice(start, stop, step)`. */
def slice(start: Int, stop: Int, step: Int = 1): Index = Index.Slice(Some(start), Some(stop), step)
/** Slice from `start` to the end. */
def sliceFrom(start: Int, step: Int = 1): Index = Index.Slice(Some(start), None, step)
/** Slice from the beginning to `stop` (exclusive). */
def sliceTo(stop: Int, step: Int = 1): Index = Index.Slice(None, Some(stop), step)

private[numscala] def normAxes(axis: Axis, ndim: Int): Array[Int] =
  val raw: Seq[Int] = axis match
    case i: Int => Seq(i)
    case s: Seq[?] => s.asInstanceOf[Seq[Int]]
  val out = raw.map(Shape.normAxis(_, ndim))
  if out.distinct.length != out.length then throw new IllegalArgumentException("duplicate value in 'axis'")
  out.sorted.toArray

// Scalar-on-the-left arithmetic: `2.0 * a`, `1 - a`, `Complex(0, 1) * a`.
extension (s: Double)
  def +(c: Complex): Complex = Complex(s + c.re, c.im)
  def -(c: Complex): Complex = Complex(s - c.re, -c.im)
  def *(c: Complex): Complex = Complex(s * c.re, s * c.im)
  def /(c: Complex): Complex = Complex(s, 0.0) / c
  /** Imaginary literal: `2.0.j == Complex(0, 2)`. */
  def j: Complex = Complex(0.0, s)
  def +[U](a: NDArray[U])(using w: WeakPromote[U, Double]): NDArray[w.Out] = ScalarLeft.arith(s, a, w, Arith.Add)
  def -[U](a: NDArray[U])(using w: WeakPromote[U, Double]): NDArray[w.Out] = ScalarLeft.arith(s, a, w, Arith.Sub)
  def *[U](a: NDArray[U])(using w: WeakPromote[U, Double]): NDArray[w.Out] = ScalarLeft.arith(s, a, w, Arith.Mul)
  def /[U](a: NDArray[U])(using w: WeakPromote[U, Double])(using t: ToInexact[w.Out]): NDArray[t.Out] =
    ScalarLeft.div(s, a, w, t)
  def **[U](a: NDArray[U])(using w: WeakPromote[U, Double]): NDArray[w.Out] = ScalarLeft.arith(s, a, w, Arith.Pow)
  /** `` 7.0 `//` a `` (NumPy's `7.0 // a`). */
  def `//`[U](a: NDArray[U])(using w: WeakPromote[U, Double])(using r: RealDType[w.Out]): NDArray[w.Out] =
    ScalarLeft.floorDiv(s, a, w, r)

extension (s: Int)
  def j: Complex = Complex(0.0, s.toDouble)
  def +[U](a: NDArray[U])(using w: WeakPromote[U, Int]): NDArray[w.Out] = ScalarLeft.arith(s, a, w, Arith.Add)
  def -[U](a: NDArray[U])(using w: WeakPromote[U, Int]): NDArray[w.Out] = ScalarLeft.arith(s, a, w, Arith.Sub)
  def *[U](a: NDArray[U])(using w: WeakPromote[U, Int]): NDArray[w.Out] = ScalarLeft.arith(s, a, w, Arith.Mul)
  def /[U](a: NDArray[U])(using w: WeakPromote[U, Int])(using t: ToInexact[w.Out]): NDArray[t.Out] =
    ScalarLeft.div(s, a, w, t)
  def **[U](a: NDArray[U])(using w: WeakPromote[U, Int]): NDArray[w.Out] = ScalarLeft.arith(s, a, w, Arith.Pow)
  /** `` 7 `//` a `` (NumPy's `7 // a`). */
  def `//`[U](a: NDArray[U])(using w: WeakPromote[U, Int])(using r: RealDType[w.Out]): NDArray[w.Out] =
    ScalarLeft.floorDiv(s, a, w, r)

extension (s: Long)
  def +[U](a: NDArray[U])(using w: WeakPromote[U, Long]): NDArray[w.Out] = ScalarLeft.arith(s, a, w, Arith.Add)
  def -[U](a: NDArray[U])(using w: WeakPromote[U, Long]): NDArray[w.Out] = ScalarLeft.arith(s, a, w, Arith.Sub)
  def *[U](a: NDArray[U])(using w: WeakPromote[U, Long]): NDArray[w.Out] = ScalarLeft.arith(s, a, w, Arith.Mul)
  def /[U](a: NDArray[U])(using w: WeakPromote[U, Long])(using t: ToInexact[w.Out]): NDArray[t.Out] =
    ScalarLeft.div(s, a, w, t)
  def **[U](a: NDArray[U])(using w: WeakPromote[U, Long]): NDArray[w.Out] = ScalarLeft.arith(s, a, w, Arith.Pow)
  /** `` 7L `//` a `` (NumPy's `7 // a`). */
  def `//`[U](a: NDArray[U])(using w: WeakPromote[U, Long])(using r: RealDType[w.Out]): NDArray[w.Out] =
    ScalarLeft.floorDiv(s, a, w, r)

extension (s: Complex)
  def +[U](a: NDArray[U])(using w: WeakPromote[U, Complex]): NDArray[w.Out] = ScalarLeft.arith(s, a, w, Arith.Add)
  def -[U](a: NDArray[U])(using w: WeakPromote[U, Complex]): NDArray[w.Out] = ScalarLeft.arith(s, a, w, Arith.Sub)
  def *[U](a: NDArray[U])(using w: WeakPromote[U, Complex]): NDArray[w.Out] = ScalarLeft.arith(s, a, w, Arith.Mul)
  def /[U](a: NDArray[U])(using w: WeakPromote[U, Complex])(using t: ToInexact[w.Out]): NDArray[t.Out] =
    ScalarLeft.div(s, a, w, t)
  def **[U](a: NDArray[U])(using w: WeakPromote[U, Complex]): NDArray[w.Out] = ScalarLeft.arith(s, a, w, Arith.Pow)

/** Scalar-on-the-left operators, with the scalar converted per NEP 50 (`WeakPromote`). */
private[numscala] object ScalarLeft:
  def arith[S, U](s: S, a: NDArray[U], w: WeakPromote[U, S], op: Arith): NDArray[w.Out] =
    Ops.arith(NDArray.scalar(w.lift(s))(using w.dtype), a, w.dtype, op)
  def div[S, U](s: S, a: NDArray[U], w: WeakPromote[U, S], t: ToInexact[w.Out]): NDArray[t.Out] =
    Ops.arith(NDArray.scalar(w.lift(s))(using w.dtype), a, t.dtype, Arith.Div)
  def floorDiv[S, U](s: S, a: NDArray[U], w: WeakPromote[U, S], r: RealDType[w.Out]): NDArray[w.Out] =
    Ops.binary(NDArray.scalar(w.lift(s))(using w.dtype), a, r)(r.floorDiv)
