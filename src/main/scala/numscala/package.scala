package numscala

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
  def +[U](a: NDArray[U])(using d: InexactDType[U]): NDArray[U] = a.map(x => d.plus(d.fromDouble(s), x))
  def -[U](a: NDArray[U])(using d: InexactDType[U]): NDArray[U] = a.map(x => d.minus(d.fromDouble(s), x))
  def *[U](a: NDArray[U])(using d: InexactDType[U]): NDArray[U] = a.map(x => d.times(d.fromDouble(s), x))
  def /[U](a: NDArray[U])(using d: InexactDType[U]): NDArray[U] = a.map(x => d.div(d.fromDouble(s), x))
  def **[U](a: NDArray[U])(using d: InexactDType[U]): NDArray[U] = a.map(x => d.power(d.fromDouble(s), x))

extension (s: Int)
  def j: Complex = Complex(0.0, s.toDouble)
  def +[U](a: NDArray[U])(using d: NumDType[U]): NDArray[U] = a.map(x => d.plus(d.fromInt(s), x))
  def -[U](a: NDArray[U])(using d: NumDType[U]): NDArray[U] = a.map(x => d.minus(d.fromInt(s), x))
  def *[U](a: NDArray[U])(using d: NumDType[U]): NDArray[U] = a.map(x => d.times(d.fromInt(s), x))
  def /[U](a: NDArray[U])(using t: ToInexact[U]): NDArray[t.Out] =
    val od = t.dtype
    a.map(x => od.div(od.fromLong(s.toLong), od.castFrom(a.dtype, x)))(using od)
  def **[U](a: NDArray[U])(using d: NumDType[U]): NDArray[U] = a.map(x => d.power(d.fromInt(s), x))

extension (s: Complex)
  def +(a: NDArray[Complex]): NDArray[Complex] = a.map(x => s + x)
  def -(a: NDArray[Complex]): NDArray[Complex] = a.map(x => s - x)
  def *(a: NDArray[Complex]): NDArray[Complex] = a.map(x => s * x)
  def /(a: NDArray[Complex]): NDArray[Complex] = a.map(x => s / x)
