package com.github.kmizu.numscala

/** Phantom kinds selecting how a binary [[Ufunc]] resolves its result dtype. */
object UfuncKind:
  /** `add`, `multiply`: `np.result_type` for calls; reductions widen small ints like `sum`. */
  sealed trait Sum
  /** Arithmetic on numbers (`subtract`, `power`, `floor_divide`, `remainder`, `fmod`). */
  sealed trait Arith
  /** Always inexact (`divide`, `arctan2`, `hypot`, `logaddexp`, `copysign`, ...). */
  sealed trait Div
  /** Result in the common dtype of the inputs (`maximum`, `bitwise_and`, `gcd`, ...). */
  sealed trait Same
  /** Boolean result (`equal`, `less`, `logical_and`, ...). */
  sealed trait Cmp
  /** `float_power`: float64, or complex128 for complex inputs. */
  sealed trait FPow

/** Result dtype of `np.float_power`: `float64`, or `complex128` when an input is complex. */
trait FloatPowerOf[A, B]:
  type Out
  def dtype: InexactDType[Out]

object FloatPowerOf:
  type Aux[A, B, O] = FloatPowerOf[A, B] { type Out = O }
  private def make[A, B, O](d: InexactDType[O]): Aux[A, B, O] = new FloatPowerOf[A, B]:
    type Out = O
    val dtype: InexactDType[O] = d
  given cc: Aux[Complex, Complex, Complex] = make(DType.Complex128)
  given cr[B](using ev: RealDType[B]): Aux[Complex, B, Complex] = make(DType.Complex128)
  given rc[A](using ev: RealDType[A]): Aux[A, Complex, Complex] = make(DType.Complex128)
  given rr[A, B](using ev0: RealDType[A], ev1: RealDType[B]): Aux[A, B, Double] = make(DType.Float64)

/** Dtype resolution of a binary ufunc of kind `K` applied to inputs of types `A` and `B`:
  * the inputs are cast to `compute`, and the result has dtype `out`.
  */
trait UfuncTypes[K, A, B]:
  type Out
  def compute: DType[?]
  def out: DType[Out]

object UfuncTypes:
  type Aux[K, A, B, O] = UfuncTypes[K, A, B] { type Out = O }
  private def make[K, A, B, O](c: DType[?], o: DType[O]): Aux[K, A, B, O] = new UfuncTypes[K, A, B]:
    type Out = O
    val compute: DType[?] = c
    val out: DType[O] = o
  given sum[A, B](using p: Promote[A, B]): Aux[UfuncKind.Sum, A, B, p.Out] = make(p.dtype, p.dtype)
  given arith[A, B](using p: NumPromote[A, B]): Aux[UfuncKind.Arith, A, B, p.Out] = make(p.dtype, p.dtype)
  given div[A, B](using p: DivPromote[A, B]): Aux[UfuncKind.Div, A, B, p.Out] = make(p.dtype, p.dtype)
  given same[A, B](using p: Promote[A, B]): Aux[UfuncKind.Same, A, B, p.Out] = make(p.dtype, p.dtype)
  given cmp[A, B](using p: Promote[A, B]): Aux[UfuncKind.Cmp, A, B, Boolean] = make(p.dtype, DType.Bool)
  given fpow[A, B](using p: FloatPowerOf[A, B]): Aux[UfuncKind.FPow, A, B, p.Out] = make(p.dtype, p.dtype)

/** Dtype resolution of the reductions (`reduce`, `accumulate`, `reduceat`) of a ufunc of
  * kind `K` over an input of type `T`. `add`/`multiply` widen small integers and bool to
  * int64 like NumPy; logical/comparison ufuncs reduce in bool.
  */
trait UfuncReduce[K, T]:
  type Out
  def dtype: DType[Out]

object UfuncReduce:
  type Aux[K, T, O] = UfuncReduce[K, T] { type Out = O }
  private def make[K, T, O](d: DType[O]): Aux[K, T, O] = new UfuncReduce[K, T]:
    type Out = O
    val dtype: DType[O] = d
  given sum[T](using s: SumOf[T]): Aux[UfuncKind.Sum, T, s.Out] = make(s.dtype)
  given arith[T](using p: NumPromote[T, T]): Aux[UfuncKind.Arith, T, p.Out] = make(p.dtype)
  given div[T](using p: DivPromote[T, T]): Aux[UfuncKind.Div, T, p.Out] = make(p.dtype)
  given same[T](using p: Promote[T, T]): Aux[UfuncKind.Same, T, p.Out] = make(p.dtype)
  given cmp[T](using ev: DType[T]): Aux[UfuncKind.Cmp, T, Boolean] = make(DType.Bool)
  given fpow[T](using p: FloatPowerOf[T, T]): Aux[UfuncKind.FPow, T, p.Out] = make(p.dtype)

/** A binary universal function, the counterpart of a NumPy `ufunc` with two inputs.
  *
  * Calling it applies the operation elementwise with broadcasting; the NumPy ufunc
  * methods `reduce`, `accumulate`, `reduceat`, `outer` and `at` are available too:
  * {{{
  * np.add(a, b); np.add(a, 1.0); np.add.reduce(a); np.add.reduce(m, axis = 0)
  * np.multiply.outer(x, y); np.maximum.accumulate(a); np.add.at(a, Seq(0, 0, 2), 1.0)
  * }}}
  */
abstract class Ufunc[K] private[numscala] (val name: String, identityValue: Any, val reorderable: Boolean):
  /** Number of inputs. */
  def nin: Int = 2
  /** Number of outputs. */
  def nout: Int = 1
  def nargs: Int = 3
  /** The identity element used for empty reductions (`ufunc.identity`), if any. */
  def identity: Option[Any] = Option(identityValue)
  override def toString: String = s"<ufunc '$name'>"

  /** Element kernel for inputs cast to `c` producing elements of dtype `r`. Throws for unsupported dtypes. */
  protected[numscala] def kernel[C, R](c: DType[C], r: DType[R]): (C, C) => R
  /** Optional primitive kernel used when both computation and result dtypes are float64. */
  protected[numscala] def kernelD: (Double, Double) => Double = null
  /** Optional special lane reduction (e.g. pairwise summation for `add`). */
  protected[numscala] def laneReducer[R](r: DType[R]): (Array[R], Int) => R = null
  /** Optional whole-array fast path; returns null when not applicable. */
  protected[numscala] def fastExec[R](a: NDArray[Any], b: NDArray[Any], c: DType[Any], r: DType[R]): NDArray[R] = null

  // ------------------------------------------------------------------ calls

  /** Elementwise application with broadcasting (`ufunc(a, b)`). */
  def apply[A, B](a: NDArray[A], b: NDArray[B])(using t: UfuncTypes[K, A, B]): NDArray[t.Out] =
    exec(a, b, t.compute, t.out)
  /** Array and scalar. */
  def apply[A, B](a: NDArray[A], b: B)(using db: DType[B], t: UfuncTypes[K, A, B]): NDArray[t.Out] =
    exec(a, NDArray.scalar(b), t.compute, t.out)
  /** Scalar and array. */
  def apply[A, B](a: A, b: NDArray[B])(using da: DType[A], t: UfuncTypes[K, A, B]): NDArray[t.Out] =
    exec(NDArray.scalar(a), b, t.compute, t.out)
  /** Two scalars: returns a scalar. */
  def apply[A, B](a: A, b: B)(using da: DType[A], db: DType[B], t: UfuncTypes[K, A, B]): t.Out =
    exec(NDArray.scalar(a), NDArray.scalar(b), t.compute, t.out).item

  /** `ufunc.outer(a, b)`: result shape is `a.shape ++ b.shape`. */
  def outer[A, B](a: NDArray[A], b: NDArray[B])(using t: UfuncTypes[K, A, B]): NDArray[t.Out] =
    val ash = a.shapeArr ++ Array.fill(b.ndim)(1)
    exec(a.reshapeArr(ash), b, t.compute, t.out)

  // ------------------------------------------------------------------ reductions

  /** Reduces over all axes to a scalar (`ufunc.reduce(a, axis=None)`; for 1-D input this
    * is NumPy's default `axis=0`).
    */
  def reduce[T](a: NDArray[T])(using r: UfuncReduce[K, T]): r.Out =
    if !reorderable && a.ndim > 1 then throw notReorderable
    val ca = a.asType(using r.dtype.asInstanceOf[DType[Any]]).asInstanceOf[NDArray[r.Out]]
    Lanes.reduceAll(ca)(lane(r.dtype))

  /** Reduces along `axis` (one axis or several), like `ufunc.reduce(a, axis, keepdims=...)`. */
  def reduce[T](a: NDArray[T], axis: Axis, keepdims: Boolean = false)(using r: UfuncReduce[K, T]): NDArray[r.Out] =
    val ax = normAxes(axis, a.ndim)
    if !reorderable && ax.length > 1 then throw notReorderable
    val ca = a.asType(using r.dtype.asInstanceOf[DType[Any]]).asInstanceOf[NDArray[r.Out]]
    Lanes.reduceAxes(ca, ax, keepdims)(lane(r.dtype))(using r.dtype)

  /** Running reduction along `axis` (`ufunc.accumulate`). */
  def accumulate[T](a: NDArray[T], axis: Int = 0)(using r: UfuncReduce[K, T]): NDArray[r.Out] =
    if a.ndim == 0 then throw new IllegalArgumentException("cannot accumulate on a scalar")
    val d = r.dtype
    val f = kernel(d, d)
    Reduce.cumulate(a, axis, d)(f)

  /** Reductions over the slices `indices(i) until indices(i+1)` along `axis` (`ufunc.reduceat`). */
  def reduceat[T](a: NDArray[T], indices: Seq[Int] | NDArray[Int], axis: Int = 0)(using
      r: UfuncReduce[K, T]
  ): NDArray[r.Out] =
    if a.ndim == 0 then throw new IllegalArgumentException("reduceat does not allow 0-d input")
    val idx: Array[Int] = indices match
      case arr: NDArray[?] => arr.asInstanceOf[NDArray[Int]].toArray
      case s: Seq[?] => s.asInstanceOf[Seq[Int]].toArray
    val ax = Shape.normAxis(axis, a.ndim)
    val n = a.shapeArr(ax)
    idx.foreach(i =>
      if i < 0 || i >= n then throw new IndexOutOfBoundsException(s"index $i out-of-bounds in $name.reduceat [0, $n)")
    )
    val d = r.dtype
    val ca = a.asType(using d.asInstanceOf[DType[Any]]).asInstanceOf[NDArray[r.Out]]
    val f = kernel(d, d)
    val m = idx.length
    Lanes.transform(ca, ax, m) { (in: Array[r.Out], len: Int, out: Array[r.Out]) =>
      var k = 0
      while k < m do
        val s = idx(k)
        val e = if k + 1 < m then idx(k + 1) else len
        if s < e then
          var acc = in(s)
          var i = s + 1
          while i < e do
            acc = f(acc, in(i))
            i += 1
          out(k) = acc
        else out(k) = in(s)
        k += 1
    }(using d)

  // ------------------------------------------------------------------ at

  /** Unbuffered in-place `a[indices] = ufunc(a[indices], b)`; repeated indices accumulate (`ufunc.at`). */
  def at[T](a: NDArray[T], indices: IndexLike, b: NDArray[T])(using t: UfuncTypes[K, T, T]): Unit =
    execAt(a, Seq(Index.from(indices)), b, t.compute)
  def at[T](a: NDArray[T], indices: IndexLike, b: T)(using t: UfuncTypes[K, T, T]): Unit =
    execAt(a, Seq(Index.from(indices)), NDArray.scalar(b)(using a.dtype), t.compute)
  /** `ufunc.at(a, (i0, i1, ...), b)` with one index per axis. */
  def at[T](a: NDArray[T], indices: Tuple, b: NDArray[T])(using t: UfuncTypes[K, T, T]): Unit =
    execAt(a, tupleIndex(indices), b, t.compute)
  def at[T](a: NDArray[T], indices: Tuple, b: T)(using t: UfuncTypes[K, T, T]): Unit =
    execAt(a, tupleIndex(indices), NDArray.scalar(b)(using a.dtype), t.compute)

  // ------------------------------------------------------------------ machinery

  private def tupleIndex(t: Tuple): Seq[Index] = t.toList.map(v => Index.from(v.asInstanceOf[IndexLike]))

  private def notReorderable =
    new IllegalArgumentException(
      s"reduction operation '$name' is not reorderable, so at most one axis may be specified"
    )

  /** Fold of one lane, honouring the identity for empty lanes. */
  private def lane[R](d: DType[R]): (Array[R], Int) => R =
    val special = laneReducer(d)
    if special != null then
      (buf, n) =>
        if n == 0 then emptyValue(d) else special(buf, n)
    else
      val f = kernel(d, d)
      (buf, n) =>
        if n == 0 then emptyValue(d)
        else
          var acc = buf(0)
          var i = 1
          while i < n do
            acc = f(acc, buf(i))
            i += 1
          acc

  private def emptyValue[R](d: DType[R]): R =
    if identityValue == null then
      throw new IllegalArgumentException(s"zero-size array to reduction operation $name which has no identity")
    else d.coerce(identityValue)

  private[numscala] final def exec[R](a: NDArray[?], b: NDArray[?], c: DType[?], r: DType[R]): NDArray[R] =
    val cd = c.asInstanceOf[DType[Any]]
    val ca = a.asInstanceOf[NDArray[Any]].asType(using cd)
    val cb = b.asInstanceOf[NDArray[Any]].asType(using cd)
    val fast = fastExec(ca, cb, cd, r)
    if fast != null then fast
    else
      val kd = kernelD
      if (cd eq DType.Float64) && (r eq DType.Float64) && kd != null then
        MathK.zipD(ca.asInstanceOf[NDArray[Double]], cb.asInstanceOf[NDArray[Double]], kd).asInstanceOf[NDArray[R]]
      else NDArray.zipMap(ca, cb)(kernel(cd, r))(using r)

  private def execAt[T](a: NDArray[T], items: Seq[Index], b: NDArray[T], c: DType[?]): Unit =
    val (offs, shape) = NDArray.resolveIndex(a, items) match
      case Left(view) =>
        val out = new Array[Int](view.size)
        var k = 0
        view.foreachOffset { o =>
          out(k) = o
          k += 1
        }
        (out, view.shapeArr)
      case Right((sh, o)) => (o, sh)
    val vals = b.broadcastTo(shape*).toArray
    val ad = a.dtype
    val cd = c.asInstanceOf[DType[Any]]
    val f = kernel(cd, cd)
    val data = a.data
    var k = 0
    while k < offs.length do
      val o = offs(k)
      data(o) = ad.castFrom(cd, f(cd.castFrom(ad, data(o)), cd.castFrom(ad, vals(k))))
      k += 1

/** Per-kind element implementations of a same-dtype binary operation; `null` = unsupported. */
private[numscala] final case class K2(
    b: (Boolean, Boolean) => Boolean = null,
    i: (Long, Long) => Long = null,
    f: (Double, Double) => Double = null,
    c: (Complex, Complex) => Complex = null,
    s: (String, String) => String = null,
    f32: (Float, Float) => Float = null
):
  /** Specialises to a typed kernel for dtype `d`. */
  def realize[C](name: String, d: DType[C]): (C, C) => C =
    def fail = throw new IllegalArgumentException(s"ufunc '$name' not supported for the input types (${d.name})")
    ((d: DType[?]) match
      case DType.Bool => if b != null then b else fail
      case id: IntDType[?] =>
        if i == null then fail
        else
          val dd = id.asInstanceOf[IntDType[Any]]
          val g = i
          (x: Any, y: Any) => dd.fromLong(g(dd.toLong(x), dd.toLong(y)))
      case DType.Float64 => if f != null then f else fail
      case DType.Float32 =>
        if f32 != null then f32
        else if f == null then fail
        else
          val g = f
          (x: Float, y: Float) => g(x.toDouble, y.toDouble).toFloat
      case DType.Complex128 => if c != null then c else fail
      case DType.Str => if s != null then s else fail
      case _ => fail
    ).asInstanceOf[(C, C) => C]

/** A ufunc whose result dtype equals its computation dtype. */
private[numscala] final class SameUfunc[K](
    name: String,
    identityValue: Any,
    reorderable: Boolean,
    impl: DType[?] => K2,
    arith: Arith = null,
    reducer: String = null
) extends Ufunc[K](name, identityValue, reorderable):
  protected[numscala] def kernel[C, R](c: DType[C], r: DType[R]): (C, C) => R =
    val g = impl(c).realize(name, c)
    if c eq r then g.asInstanceOf[(C, C) => R]
    else (x, y) => r.castFrom(c, g(x, y))
  override protected[numscala] lazy val kernelD: (Double, Double) => Double = impl(DType.Float64).f
  override protected[numscala] def laneReducer[R](r: DType[R]): (Array[R], Int) => R =
    (reducer, r) match
      case ("sum", n: NumDType[?]) =>
        val nd = n.asInstanceOf[NumDType[R]]
        (buf, k) => Reduce.sumBuf(nd, buf, k, nd)
      case ("prod", n: NumDType[?]) =>
        val nd = n.asInstanceOf[NumDType[R]]
        (buf, k) => Reduce.prodBuf(nd, buf, k, nd)
      case _ => null
  override protected[numscala] def fastExec[R](a: NDArray[Any], b: NDArray[Any], c: DType[Any], r: DType[R]): NDArray[R] =
    if arith != null && (c eq r) then
      (c: DType[?]) match
        case n: NumDType[?] => Ops.arith(a, b, n.asInstanceOf[NumDType[R]], arith)
        case _ => null
    else null

/** A ufunc with a boolean result (comparisons and logical operations). */
private[numscala] final class BoolUfunc(
    name: String,
    identityValue: Any,
    reorderable: Boolean,
    impl: DType[Any] => (Any, Any) => Boolean
) extends Ufunc[UfuncKind.Cmp](name, identityValue, reorderable):
  protected[numscala] def kernel[C, R](c: DType[C], r: DType[R]): (C, C) => R =
    val g = impl(c.asInstanceOf[DType[Any]])
    if r eq DType.Bool then g.asInstanceOf[(C, C) => R]
    else (x, y) => r.castFrom(DType.Bool, g(x, y))
