package numscala

/** Clipping, complex-number helpers, multi-output float functions and `unwrap`. */
trait NpMathMisc:

  // ------------------------------------------------------------------ clip

  private def clipKernel[T](d: RealDType[T]): (T, T, T) => T = (d: DType[?]) match
    case f: FloatDType[?] =>
      val fd = f.asInstanceOf[FloatDType[T]]
      (x, lo, hi) =>
        fd.fromDouble(MathK.minimumD(MathK.maximumD(fd.toDouble(x), fd.toDouble(lo)), fd.toDouble(hi)))
    case _ =>
      (x, lo, hi) =>
        val v = d.toLong(x)
        val l = d.toLong(lo)
        val h = d.toLong(hi)
        if v < l then (if l > h then hi else lo) else if v > h then hi else x

  /** Limits the values to `[a_min, a_max]`; NaN propagates (`np.clip`). */
  def clip[T](a: NDArray[T], a_min: T, a_max: T)(using d: RealDType[T]): NDArray[T] =
    (d: DType[?]) match
      case DType.Float64 =>
        val lo = a_min.asInstanceOf[Double]
        val hi = a_max.asInstanceOf[Double]
        MathK.mapD(a.asInstanceOf[NDArray[Double]], x => MathK.minimumD(MathK.maximumD(x, lo), hi)).asInstanceOf[NDArray[T]]
      case _ =>
        val k = clipKernel(d)
        a.map(x => k(x, a_min, a_max))

  /** Clips with broadcast array bounds (`np.clip(a, a_min_array, a_max_array)`). */
  def clip[T](a: NDArray[T], a_min: NDArray[T], a_max: NDArray[T])(using d: RealDType[T]): NDArray[T] =
    NDArray.zipMap3(a, a_min, a_max)(clipKernel(d))

  /** Clips with optional bounds; `None` means unbounded on that side (`np.clip(a, None, hi)`). */
  def clip[T](a: NDArray[T], a_min: Option[T], a_max: Option[T])(using d: RealDType[T]): NDArray[T] =
    (a_min, a_max) match
      case (Some(lo), Some(hi)) => clip(a, lo, hi)
      case (Some(lo), None) => Ufuncs.maximum.exec(a, NDArray.scalar(lo), d, d)
      case (None, Some(hi)) => Ufuncs.minimum.exec(a, NDArray.scalar(hi), d, d)
      case (None, None) => a.copy()

  /** Scalar clip. */
  def clip(x: Double, a_min: Double, a_max: Double): Double = MathK.minimumD(MathK.maximumD(x, a_min), a_max)

  // ------------------------------------------------------------------ nan_to_num

  /** Replaces NaN with `nan`, +inf with `posinf` (default: largest finite value) and
    * -inf with `neginf` (default: most negative finite value); complex parts are
    * treated separately (`np.nan_to_num`).
    */
  def nan_to_num[T](x: NDArray[T], nan: Double = 0.0, posinf: Option[Double] = None, neginf: Option[Double] = None): NDArray[T] =
    def fix(v: Double, big: Double): Double =
      if v.isNaN then nan
      else if v == Double.PositiveInfinity then posinf.getOrElse(big)
      else if v == Double.NegativeInfinity then neginf.getOrElse(-big)
      else v
    (x.dtype: DType[?]) match
      case DType.Float64 => MathK.mapD(x.asInstanceOf[NDArray[Double]], v => fix(v, Double.MaxValue)).asInstanceOf[NDArray[T]]
      case DType.Float32 =>
        x.asInstanceOf[NDArray[Float]].map(v => fix(v.toDouble, Float.MaxValue.toDouble).toFloat).asInstanceOf[NDArray[T]]
      case DType.Complex128 =>
        x.asInstanceOf[NDArray[Complex]]
          .map(z => Complex(fix(z.re, Double.MaxValue), fix(z.im, Double.MaxValue)))
          .asInstanceOf[NDArray[T]]
      case _ => x.copy()

  /** Scalar `nan_to_num` with the default replacements. */
  def nan_to_num(x: Double): Double =
    if x.isNaN then 0.0 else if x == Double.PositiveInfinity then Double.MaxValue
    else if x == Double.NegativeInfinity then -Double.MaxValue else x

  // ------------------------------------------------------------------ complex helpers

  /** Real part; a view for real arrays, float64 for complex (`np.real`). */
  def real[T](x: NDArray[T])(using r: AbsOf[T]): NDArray[r.Out] =
    if x.dtype.isComplex then x.asInstanceOf[NDArray[Complex]].map(_.re).asInstanceOf[NDArray[r.Out]]
    else x.view(x.shapeArr.clone(), x.stridesArr.clone(), x.offset).asInstanceOf[NDArray[r.Out]]
  def real(x: Complex): Double = x.re
  def real(x: Double): Double = x

  /** Imaginary part; zeros of the same dtype for real arrays (`np.imag`). */
  def imag[T](x: NDArray[T])(using r: AbsOf[T]): NDArray[r.Out] =
    if x.dtype.isComplex then x.asInstanceOf[NDArray[Complex]].map(_.im).asInstanceOf[NDArray[r.Out]]
    else NDArray.zerosOf(x.dtype, x.shapeArr.clone()).asInstanceOf[NDArray[r.Out]]
  def imag(x: Complex): Double = x.im
  def imag(x: Double): Double = 0.0

  /** Complex conjugate; a copy for real arrays (`np.conj`). */
  def conj[T](x: NDArray[T])(using NumDType[T]): NDArray[T] =
    if x.dtype.isComplex then x.asInstanceOf[NDArray[Complex]].map(_.conj).asInstanceOf[NDArray[T]] else x.copy()
  def conj(x: Complex): Complex = x.conj
  /** Alias of [[conj]] (`np.conjugate`). */
  def conjugate[T](x: NDArray[T])(using NumDType[T]): NDArray[T] = conj(x)
  def conjugate(x: Complex): Complex = x.conj

  /** Counterclockwise angle from the positive real axis, in radians or degrees (`np.angle`). */
  def angle[T](z: NDArray[T], deg: Boolean = false)(using r: RealOf[T]): NDArray[r.Out] =
    val fac = if deg then 180.0 / math.Pi else 1.0
    if z.dtype.isComplex then
      z.asInstanceOf[NDArray[Complex]].map(c => math.atan2(c.im, c.re) * fac).asInstanceOf[NDArray[r.Out]]
    else MathK.mapFloat(z, r.dtype, "angle")(x => math.atan2(0.0, x) * fac)
  def angle(z: Complex): Double = math.atan2(z.im, z.re)
  def angle(z: Double): Double = math.atan2(0.0, z)

  /** The real part if all imaginary parts are within `tol` machine epsilons of zero
    * (when `tol > 1`) or below `tol`; otherwise the input (`np.real_if_close`).
    */
  def real_if_close[T](a: NDArray[T], tol: Double = 100.0): NDArray[?] =
    if !a.dtype.isComplex then a
    else
      val t = if tol > 1.0 then tol * DType.Float64.eps else tol
      val c = a.asInstanceOf[NDArray[Complex]]
      if c.toArray.forall(z => math.abs(z.im) < t) then c.map(_.re) else a

  /** True where the imaginary part is zero (`np.isreal`). */
  def isreal[T](x: NDArray[T]): NDArray[Boolean] =
    if x.dtype.isComplex then x.asInstanceOf[NDArray[Complex]].map(_.im == 0.0)
    else NDArray.fillOf(DType.Bool, x.shapeArr.clone(), true)
  /** True where the imaginary part is nonzero (`np.iscomplex`). */
  def iscomplex[T](x: NDArray[T]): NDArray[Boolean] =
    if x.dtype.isComplex then x.asInstanceOf[NDArray[Complex]].map(_.im != 0.0)
    else NDArray.fillOf(DType.Bool, x.shapeArr.clone(), false)
  /** True unless the array has a complex dtype (`np.isrealobj`). */
  def isrealobj(x: NDArray[?]): Boolean = !x.dtype.isComplex
  /** True if the array has a complex dtype (`np.iscomplexobj`). */
  def iscomplexobj(x: NDArray[?]): Boolean = x.dtype.isComplex

  // ------------------------------------------------------------------ multi-output functions

  /** Floor quotient and remainder at once (`np.divmod`). */
  def divmod[A, B](x1: NDArray[A], x2: NDArray[B])(using p: NumPromote[A, B]): (NDArray[p.Out], NDArray[p.Out]) =
    (Ufuncs.floor_divide.exec(x1, x2, p.dtype, p.dtype), Ufuncs.remainder.exec(x1, x2, p.dtype, p.dtype))
  def divmod[A, B](x1: NDArray[A], x2: B)(using db: DType[B], p: NumPromote[A, B]): (NDArray[p.Out], NDArray[p.Out]) =
    divmod(x1, NDArray.scalar(x2))
  def divmod(x1: Double, x2: Double): (Double, Double) = (MathK.floorDivD(x1, x2), MathK.modD(x1, x2))
  def divmod(x1: Long, x2: Long): (Long, Long) = (MathK.floorDivL(x1, x2), MathK.modL(x1, x2))

  /** Mantissa in `[0.5, 1)` and integer exponent with `x = m * 2**e` (`np.frexp`). */
  def frexp[T](x: NDArray[T])(using t: ToFloat[T]): (NDArray[t.Out], NDArray[Int]) =
    if x.dtype.isComplex then MathK.unsupported("frexp", x.dtype)
    val d = MathK.asDoubles(x).toArray
    val m = new Array[Double](d.length)
    val e = new Array[Int](d.length)
    var i = 0
    while i < d.length do
      val (mm, ee) = MathK.frexp(d(i))
      m(i) = mm
      e(i) = ee
      i += 1
    (NDArray.fromArray(m, x.shapeArr.clone()).asType(using t.dtype), NDArray.fromArray(e, x.shapeArr.clone()))
  def frexp(x: Double): (Double, Int) = MathK.frexp(x)

  /** Fractional and integral parts, both with the sign of the input (`np.modf`). */
  def modf[T](x: NDArray[T])(using t: ToFloat[T]): (NDArray[t.Out], NDArray[t.Out]) =
    if x.dtype.isComplex then MathK.unsupported("modf", x.dtype)
    val d = MathK.asDoubles(x).toArray
    val f = new Array[Double](d.length)
    val ip = new Array[Double](d.length)
    var i = 0
    while i < d.length do
      val (a, b) = MathK.modf(d(i))
      f(i) = a
      ip(i) = b
      i += 1
    (
      NDArray.fromArray(f, x.shapeArr.clone()).asType(using t.dtype),
      NDArray.fromArray(ip, x.shapeArr.clone()).asType(using t.dtype)
    )
  def modf(x: Double): (Double, Double) = MathK.modf(x)

  /** `x1 * 2**x2` with integer exponents (`np.ldexp`). */
  def ldexp[T, I](x1: NDArray[T], x2: NDArray[I])(using t: ToFloat[T], i: IntDType[I]): NDArray[t.Out] =
    if x1.dtype.isComplex then MathK.unsupported("ldexp", x1.dtype)
    val src = x1.dtype
    val od = t.dtype
    NDArray.zipMap(x1, x2)((a, n) => od.fromDouble(MathK.ldexp(src.toDouble(a), i.toLong(n))))(using od)
  def ldexp[T](x1: NDArray[T], x2: Int)(using t: ToFloat[T]): NDArray[t.Out] = ldexp(x1, NDArray.scalar(x2))
  def ldexp(x1: Double, x2: Int): Double = MathK.ldexp(x1, x2.toLong)

  // ------------------------------------------------------------------ unwrap

  /** Unwraps phase jumps larger than `discont` (default `period/2`) along `axis` by
    * adding multiples of `period` (`np.unwrap`).
    */
  def unwrap[T](p: NDArray[T], discont: Double = Double.NaN, axis: Int = -1, period: Double = 2.0 * math.Pi)(using
      t: ToFloat[T]
  ): NDArray[t.Out] =
    if p.dtype.isComplex then MathK.unsupported("unwrap", p.dtype)
    val disc = if discont.isNaN then period / 2.0 else discont
    val high = period / 2.0
    val low = -high
    val x = MathK.asDoubles(p)
    val res = Lanes.transform(x, axis) { (in: Array[Double], n: Int, out: Array[Double]) =>
      if n > 0 then out(0) = in(0)
      var cum = 0.0
      var i = 1
      while i < n do
        val dd = in(i) - in(i - 1)
        var ddmod = MathK.modD(dd - low, period) + low
        if ddmod == low && dd > 0 then ddmod = high
        val corr = if math.abs(dd) < disc then 0.0 else ddmod - dd
        cum += corr
        out(i) = in(i) + cum
        i += 1
    }
    val shaped = if p.ndim == 0 then res.reshapeArr(Array.emptyIntArray) else res
    shaped.asType(using t.dtype)
