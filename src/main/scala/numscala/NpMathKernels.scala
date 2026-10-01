package numscala

/** Scalar kernels and elementwise loops shared by the math module. */
private[numscala] object MathK:
  final val LN2 = 0.6931471805599453
  final val LOG2E = 1.4426950408889634
  final val LN10 = 2.302585092994046
  final val TWO28 = 268435456.0

  def unsupported(name: String, d: DType[?]): Nothing =
    throw new IllegalArgumentException(s"ufunc '$name' not supported for the input types (${d.name})")

  // ------------------------------------------------------------------ loops

  /** Broadcasting binary loop on doubles (specialised function, no boxing). */
  def zipD(a: NDArray[Double], b: NDArray[Double], f: (Double, Double) => Double): NDArray[Double] =
    val sh = Shape.broadcast(a.shapeArr, b.shapeArr)
    val sa = Strided.broadcastStrides(a.shapeArr, a.stridesArr, sh)
    val sb = Strided.broadcastStrides(b.shapeArr, b.stridesArr, sh)
    val res = new Array[Double](Shape.size(sh))
    val da = a.data
    val db = b.data
    var k = 0
    Strided.foreach2(sh, sa, a.offset, sb, b.offset) { (i, j) =>
      res(k) = f(da(i), db(j))
      k += 1
    }
    NDArray.fromArray(res, sh)

  /** Elementwise `f` on a float64 array (specialised function, no boxing). */
  def mapD(a: NDArray[Double], f: Double => Double): NDArray[Double] =
    val res = new Array[Double](a.size)
    val d = a.data
    var k = 0
    Strided.foreach1(a.shapeArr, a.stridesArr, a.offset) { o =>
      res(k) = f(d(o))
      k += 1
    }
    NDArray.fromArray(res, a.shapeArr.clone())

  /** The elements converted to doubles, as a float64 array (no copy if already float64). */
  def asDoubles[T](a: NDArray[T]): NDArray[Double] = a.asType(using DType.Float64)

  /** Applies a real function (`f`) or complex function (`fc`, null if unsupported) with
    * results in the inexact dtype `od`.
    */
  def mapInexact[T, U](a: NDArray[T], od: InexactDType[U], name: String)(
      f: Double => Double,
      fc: Complex => Complex
  ): NDArray[U] =
    (od: DType[?]) match
      case DType.Float64 => mapD(asDoubles(a), f).asInstanceOf[NDArray[U]]
      case DType.Float32 =>
        val src = a.dtype
        a.map(x => f(src.toDouble(x)).toFloat)(using DType.Float32).asInstanceOf[NDArray[U]]
      case _ =>
        if fc == null then unsupported(name, a.dtype)
        val src = a.dtype
        a.map(x => fc(src.toComplex(x)))(using DType.Complex128).asInstanceOf[NDArray[U]]

  /** Real-only function with results in the float dtype `od`. */
  def mapFloat[T, U](a: NDArray[T], od: FloatDType[U], name: String)(f: Double => Double): NDArray[U] =
    if a.dtype.isComplex || a.dtype.isString then unsupported(name, a.dtype)
    mapInexact(a, od, name)(f, null)

  /** Same-dtype unary operation chosen by kind; `null` entries are unsupported. */
  def mapSame[T](a: NDArray[T], name: String)(
      b: Boolean => Boolean = null,
      i: Long => Long = null,
      f: Double => Double = null,
      c: Complex => Complex = null
  ): NDArray[T] =
    val d = a.dtype
    ((d: DType[?]) match
      case DType.Bool if b != null => a.asInstanceOf[NDArray[Boolean]].map(b)
      case id: IntDType[?] if i != null =>
        val dd = id.asInstanceOf[IntDType[Any]]
        a.asInstanceOf[NDArray[Any]].map(x => dd.fromLong(i(dd.toLong(x))))(using dd)
      case DType.Float64 if f != null => mapD(a.asInstanceOf[NDArray[Double]], f)
      case DType.Float32 if f != null => a.asInstanceOf[NDArray[Float]].map(x => f(x.toDouble).toFloat)
      case DType.Complex128 if c != null => a.asInstanceOf[NDArray[Complex]].map(c)
      case _ => unsupported(name, d)
    ).asInstanceOf[NDArray[T]]

  /** Boolean-valued unary predicate chosen by kind. */
  def mapPred[T](a: NDArray[T], name: String)(
      b: Boolean => Boolean = null,
      i: Long => Boolean = null,
      f: Double => Boolean = null,
      c: Complex => Boolean = null
  ): NDArray[Boolean] =
    val d = a.dtype
    (d: DType[?]) match
      case DType.Bool if b != null => a.asInstanceOf[NDArray[Boolean]].map(b)
      case id: IntDType[?] if i != null =>
        val dd = id.asInstanceOf[IntDType[Any]]
        a.asInstanceOf[NDArray[Any]].map(x => i(dd.toLong(x)))
      case fd: FloatDType[?] if f != null =>
        val dd = fd.asInstanceOf[FloatDType[Any]]
        a.asInstanceOf[NDArray[Any]].map(x => f(dd.toDouble(x)))
      case DType.Complex128 if c != null => a.asInstanceOf[NDArray[Complex]].map(c)
      case _ => unsupported(name, d)

  // ------------------------------------------------------------------ real scalar functions

  /** C99 `pow` semantics (Java's `Math.pow` differs for `pow(1, nan)` and `pow(-1, ±inf)`). */
  def cpow(x: Double, y: Double): Double =
    if x == 1.0 then 1.0
    else if x == -1.0 && y.isInfinite then 1.0
    else math.pow(x, y)

  /** Integer power; negative exponents are an error like NumPy. */
  def ipow(x: Long, y: Long): Long =
    if y < 0 then throw new ArithmeticException("Integers to negative integer powers are not allowed.")
    var e = y
    var base = x
    var acc = 1L
    while e > 0 do
      if (e & 1L) == 1L then acc *= base
      base *= base
      e >>= 1
    acc

  /** NumPy's `npy_divmod`: Python floor division and modulo of doubles. */
  def floorDivD(a: Double, b: Double): Double =
    if b == 0.0 then a / b
    else
      val mod0 = a % b
      var div = (a - mod0) / b
      if mod0 != 0.0 then
        if (b < 0) != (mod0 < 0) then div -= 1.0
      if div != 0.0 then
        var fd = math.floor(div)
        if div - fd > 0.5 then fd += 1.0
        fd
      else math.copySign(0.0, a / b)

  def modD(a: Double, b: Double): Double =
    if b == 0.0 then a % b
    else
      var mod0 = a % b
      if mod0 != 0.0 then
        if (b < 0) != (mod0 < 0) then mod0 += b
        mod0
      else math.copySign(0.0, b)

  def floorDivL(a: Long, b: Long): Long = if b == 0L then 0L else Math.floorDiv(a, b)
  def modL(a: Long, b: Long): Long = if b == 0L then 0L else Math.floorMod(a, b)
  def fmodL(a: Long, b: Long): Long = if b == 0L then 0L else a % b

  def gcdL(a0: Long, b0: Long): Long =
    var a = math.abs(a0)
    var b = math.abs(b0)
    while b != 0L do
      val t = a % b
      a = b
      b = t
    a

  def lcmL(a: Long, b: Long): Long =
    val g = gcdL(a, b)
    if g == 0L then 0L else math.abs(a / g * b)

  def shiftLeft(bits: Int)(x: Long, s: Long): Long = if s < 0 || s >= bits then 0L else x << s
  def shiftRight(bits: Int)(x: Long, s: Long): Long =
    if s < 0 || s >= bits then (if x < 0 then -1L else 0L) else x >> s

  /** NumPy `maximum` (NaN-propagating; returns the second argument on ties, as NumPy 2 does). */
  def maximumD(a: Double, b: Double): Double = if a.isNaN then a else if b.isNaN then b else if a > b then a else b
  def minimumD(a: Double, b: Double): Double = if a.isNaN then a else if b.isNaN then b else if a < b then a else b
  def fmaxD(a: Double, b: Double): Double = if a >= b || b.isNaN then a else b
  def fminD(a: Double, b: Double): Double = if a <= b || b.isNaN then a else b

  private def cge(x: Complex, y: Complex): Boolean =
    (x.re > y.re && !x.im.isNaN && !y.im.isNaN) || (x.re == y.re && x.im >= y.im)
  private def cle(x: Complex, y: Complex): Boolean =
    (x.re < y.re && !x.im.isNaN && !y.im.isNaN) || (x.re == y.re && x.im <= y.im)
  def maximumC(a: Complex, b: Complex): Complex = if cge(a, b) || a.isNaN then a else b
  def minimumC(a: Complex, b: Complex): Complex = if cle(a, b) || a.isNaN then a else b
  def fmaxC(a: Complex, b: Complex): Complex = if cge(a, b) || b.isNaN then a else b
  def fminC(a: Complex, b: Complex): Complex = if cle(a, b) || b.isNaN then a else b

  def logaddexp(x: Double, y: Double): Double =
    if x == y then x + LN2
    else
      val t = x - y
      if t > 0 then x + math.log1p(math.exp(-t))
      else if t <= 0 then y + math.log1p(math.exp(t))
      else t

  def logaddexp2(x: Double, y: Double): Double =
    if x == y then x + 1.0
    else
      val t = x - y
      if t > 0 then x + LOG2E * math.log1p(exp2(-t))
      else if t <= 0 then y + LOG2E * math.log1p(exp2(t))
      else t

  def heaviside(x: Double, h0: Double): Double =
    if x.isNaN then x else if x == 0.0 then h0 else if x < 0 then 0.0 else 1.0

  def exp2(x: Double): Double = math.pow(2.0, x)

  /** Accurate base-2 logarithm (exact for powers of two). */
  def log2(x: Double): Double =
    if x.isNaN || x < 0 then Double.NaN
    else if x == 0.0 then Double.NegativeInfinity
    else if x.isInfinite then x
    else
      var e = math.getExponent(x)
      var m = x
      if e == java.lang.Double.MIN_EXPONENT - 1 then // subnormal
        m = x * TWO28 * TWO28
        e = math.getExponent(m) - 56
      m = math.scalb(m, -math.getExponent(m)) // in [1, 2)
      if m == 1.0 then e.toDouble
      else
        if m > math.sqrt(2.0) then
          m /= 2.0
          e += 1
        e + math.log(m) * LOG2E

  def asinh(x: Double): Double =
    val ax = math.abs(x)
    if x.isNaN || x.isInfinite then x
    else if ax < 3.7252902984e-09 then x
    else
      val w =
        if ax > TWO28 then math.log(ax) + LN2
        else if ax > 2.0 then math.log(2.0 * ax + 1.0 / (math.sqrt(x * x + 1.0) + ax))
        else
          val t = x * x
          math.log1p(ax + t / (1.0 + math.sqrt(1.0 + t)))
      math.copySign(w, x)

  def acosh(x: Double): Double =
    if x.isNaN || x < 1.0 then Double.NaN
    else if x >= TWO28 then (if x.isInfinite then x else math.log(x) + LN2)
    else if x == 1.0 then 0.0
    else if x > 2.0 then math.log(2.0 * x - 1.0 / (x + math.sqrt(x * x - 1.0)))
    else
      val t = x - 1.0
      math.log1p(t + math.sqrt(2.0 * t + t * t))

  def atanh(x: Double): Double =
    val ax = math.abs(x)
    if x.isNaN || ax > 1.0 then Double.NaN
    else if ax == 1.0 then math.copySign(Double.PositiveInfinity, x)
    else if ax < 3.7252902984e-09 then x
    else
      val t =
        if ax < 0.5 then
          val t2 = ax + ax
          0.5 * math.log1p(t2 + t2 * ax / (1.0 - ax))
        else 0.5 * math.log1p((ax + ax) / (1.0 - ax))
      math.copySign(t, x)

  /** `np.spacing` for float64: distance to the next float away from zero, negative for
    * negative `x` (`-0.0` counts as positive, like NumPy).
    */
  def spacing(x: Double): Double =
    if x.isNaN || x.isInfinite then Double.NaN
    else if x >= 0.0 then math.nextUp(x) - x
    else math.nextDown(x) - x
  def spacingF(x: Float): Float =
    if x.isNaN || x.isInfinite then Float.NaN
    else if x >= 0f then math.nextUp(x) - x
    else math.nextDown(x) - x

  def nextafterF(x: Float, y: Float): Float = math.nextAfter(x, y.toDouble)

  def sinc(x: Double): Double =
    val y = math.Pi * (if x == 0.0 then 1.0e-20 else x)
    math.sin(y) / y

  /** NumPy 2 complex `sign`: `z/|z|`, with infinite components handled like NumPy. */
  def signC(z: Complex): Complex =
    if z.isNaN then Complex(Double.NaN, Double.NaN)
    else if z.re == 0.0 && z.im == 0.0 then Complex.Zero
    else if z.re.isInfinite || z.im.isInfinite then
      if z.re.isInfinite && z.im.isInfinite then Complex(Double.NaN, Double.NaN)
      else if z.re.isInfinite then Complex(math.signum(z.re), 0.0)
      else Complex(0.0, math.signum(z.im))
    else
      val r = math.hypot(z.re, z.im)
      Complex(z.re / r, z.im / r)

  def sincC(z: Complex): Complex =
    val y = (if z.re == 0.0 && z.im == 0.0 then Complex(1.0e-20, 0.0) else z) * math.Pi
    CMath.sin(y) / y

  def fix(x: Double): Double = if x >= 0 then math.floor(x) else if x < 0 then math.ceil(x) else x
  def trunc(x: Double): Double = if x >= 0 then math.floor(x) else if x < 0 then math.ceil(x) else x

  /** NumPy `sign` for floats: -1, 0 (also for -0.0), 1, NaN. */
  def sign(x: Double): Double = if x > 0 then 1.0 else if x < 0 then -1.0 else if x == 0.0 then 0.0 else x

  def degrees(x: Double): Double = x * (180.0 / math.Pi)
  def radians(x: Double): Double = x * (math.Pi / 180.0)

  def signbit(x: Double): Boolean = (java.lang.Double.doubleToRawLongBits(x) >>> 63) != 0L

  /** `np.frexp`: mantissa in [0.5, 1) and exponent. */
  def frexp(x: Double): (Double, Int) =
    if x == 0.0 || x.isNaN || x.isInfinite then (x, 0)
    else
      var e = math.getExponent(x)
      var m = x
      var adj = 0
      if e == java.lang.Double.MIN_EXPONENT - 1 then
        m = x * TWO28 * TWO28
        adj = -56
        e = math.getExponent(m)
      (math.scalb(m, -(e + 1)), e + 1 + adj)

  /** `np.modf`: fractional and integral parts, both with the sign of `x`. */
  def modf(x: Double): (Double, Double) =
    if x.isNaN then (x, x)
    else if x.isInfinite then (math.copySign(0.0, x), x)
    else
      val ip = trunc(x)
      (math.copySign(x - ip, x), ip)

  def ldexp(x: Double, n: Long): Double =
    val k = if n > 100000L then 100000 else if n < -100000L then -100000 else n.toInt
    math.scalb(x, k)

  // ------------------------------------------------------------------ Bessel I0 (Cephes, as in NumPy)

  private val i0A: Array[Double] = Array(
    -4.41534164647933937950e-18, 3.33079451882223809783e-17, -2.43127984654795469359e-16,
    1.71539128555513303061e-15, -1.16853328779934516808e-14, 7.67618549860493561688e-14,
    -4.85644678311192946090e-13, 2.95505266312963983461e-12, -1.72682629144155570723e-11,
    9.67580903537323691224e-11, -5.18979560163526290666e-10, 2.65982372468238665035e-9,
    -1.30002500998624804212e-8, 6.04699502254191894932e-8, -2.67079385394061173391e-7,
    1.11738753912010371815e-6, -4.41673835845875056359e-6, 1.64484480707288970893e-5,
    -5.75419501008210370398e-5, 1.88502885095841655729e-4, -5.76375574538582365885e-4,
    1.63947561694133579842e-3, -4.32430999505057594430e-3, 1.05464603945949983183e-2,
    -2.37374148058994688156e-2, 4.93052842396707084878e-2, -9.49010970480476444210e-2,
    1.71620901522208775349e-1, -3.04682672343198398683e-1, 6.76795274409476084995e-1
  )

  private val i0B: Array[Double] = Array(
    -7.23318048787475395456e-18, -4.83050448594418207126e-18, 4.46562142029675999901e-17,
    3.46122286769746109310e-17, -2.82762398051658348494e-16, -3.42548561967721913462e-16,
    1.77256013305652638360e-15, 3.81168066935262242075e-15, -9.55484669882830764870e-15,
    -4.15056934728722208663e-14, 1.54008621752140982691e-14, 3.85277838274214270114e-13,
    7.18012445138366623367e-13, -1.79417853150680611778e-12, -1.32158118404477131188e-11,
    -3.14991652796324136454e-11, 1.18891471078464383424e-11, 4.94060238822496958910e-10,
    3.39623202570838634515e-9, 2.26666899049817806459e-8, 2.04891858946906374183e-7,
    2.89137052083475648297e-6, 6.88975834691682398426e-5, 3.36911647825569408990e-3,
    8.04490411014108831608e-1
  )

  private def chbevl(x: Double, vals: Array[Double]): Double =
    var b0 = vals(0)
    var b1 = 0.0
    var b2 = 0.0
    var i = 1
    while i < vals.length do
      b2 = b1
      b1 = b0
      b0 = x * b1 - b2 + vals(i)
      i += 1
    0.5 * (b0 - b2)

  /** Modified Bessel function of the first kind, order 0 (`np.i0`). */
  def i0(x0: Double): Double =
    val x = math.abs(x0)
    if x <= 8.0 then math.exp(x) * chbevl(x / 2.0 - 2.0, i0A)
    else math.exp(x) * chbevl(32.0 / x - 2.0, i0B) / math.sqrt(x)
