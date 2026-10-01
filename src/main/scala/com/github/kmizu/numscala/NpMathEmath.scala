package com.github.kmizu.numscala

/** `numpy.emath` (`numpy.lib.scimath`): math functions that return complex results for
  * inputs outside the real domain instead of NaN. Real-valued results are float64; an
  * array result is complex128 as soon as any element is out of the real domain, so the
  * array functions return `NDArray[Double] | NDArray[Complex]`.
  * {{{
  * np.emath.sqrt(-1.0)                    // Complex(0, 1)
  * np.emath.log(np.array(1.0, -1.0))     // complex128 array
  * }}}
  */
object Emath:
  private def scalar(x: Double, outOfDomain: Boolean, f: Double => Double, fc: Complex => Complex): Double | Complex =
    if outOfDomain then fc(Complex(x, 0.0)) else f(x)

  private def arr[T](x: NDArray[T], name: String, outOfDomain: Double => Boolean)(
      f: Double => Double,
      fc: Complex => Complex
  ): NDArray[Double] | NDArray[Complex] =
    if x.dtype.isComplex then x.asInstanceOf[NDArray[Complex]].map(fc)
    else if x.dtype.isString then MathK.unsupported(name, x.dtype)
    else
      val d = MathK.asDoubles(x)
      if d.toArray.exists(outOfDomain) then d.map(v => fc(Complex(v, 0.0)))
      else MathK.mapD(d, f)

  private val neg: Double => Boolean = _ < 0.0
  private val absGt1: Double => Boolean = v => math.abs(v) > 1.0

  /** Square root; negative inputs give imaginary results (`np.emath.sqrt`). */
  def sqrt[T](x: NDArray[T]): NDArray[Double] | NDArray[Complex] = arr(x, "sqrt", neg)(math.sqrt, CMath.sqrt)
  def sqrt(x: Double): Double | Complex = scalar(x, x < 0, math.sqrt, CMath.sqrt)
  def sqrt(x: Complex): Complex = CMath.sqrt(x)

  /** Natural logarithm; `log(-1) = pi*j` (`np.emath.log`). */
  def log[T](x: NDArray[T]): NDArray[Double] | NDArray[Complex] = arr(x, "log", neg)(math.log, CMath.log)
  def log(x: Double): Double | Complex = scalar(x, x < 0, math.log, CMath.log)
  def log(x: Complex): Complex = CMath.log(x)

  /** Base-2 logarithm (`np.emath.log2`). */
  def log2[T](x: NDArray[T]): NDArray[Double] | NDArray[Complex] = arr(x, "log2", neg)(MathK.log2, CMath.log2)
  def log2(x: Double): Double | Complex = scalar(x, x < 0, MathK.log2, CMath.log2)
  def log2(x: Complex): Complex = CMath.log2(x)

  /** Base-10 logarithm (`np.emath.log10`). */
  def log10[T](x: NDArray[T]): NDArray[Double] | NDArray[Complex] = arr(x, "log10", neg)(math.log10, CMath.log10)
  def log10(x: Double): Double | Complex = scalar(x, x < 0, math.log10, CMath.log10)
  def log10(x: Complex): Complex = CMath.log10(x)

  /** Logarithm of `x` to base `n` (`np.emath.logn`). */
  def logn[T](n: Double, x: NDArray[T]): NDArray[Double] | NDArray[Complex] =
    if n < 0 then
      val ln = CMath.log(Complex(n, 0.0))
      val lx = log(x) match
        case a: NDArray[?] if a.dtype.isComplex => a.asInstanceOf[NDArray[Complex]]
        case a => a.asInstanceOf[NDArray[Double]].map(v => Complex(v, 0.0))
      lx.map(_ / ln)
    else
      val ln = math.log(n)
      log(x) match
        case a: NDArray[?] if a.dtype.isComplex => a.asInstanceOf[NDArray[Complex]].map(_ / ln)
        case a => MathK.mapD(a.asInstanceOf[NDArray[Double]], _ / ln)
  def logn(n: Double, x: Double): Double | Complex =
    if n < 0 || x < 0 then CMath.log(Complex(x, 0.0)) / CMath.log(Complex(n, 0.0))
    else math.log(x) / math.log(n)

  /** `x ** p` with complex results for negative bases (`np.emath.power`). */
  def power[T](x: NDArray[T], p: Double): NDArray[Double] | NDArray[Complex] =
    val pc = Complex(p, 0.0)
    arr(x, "power", neg)(v => MathK.cpow(v, p), z => CMath.pow(z, pc))
  def power[T, U](x: NDArray[T], p: NDArray[U]): NDArray[Double] | NDArray[Complex] =
    if x.dtype.isComplex || p.dtype.isComplex || MathK.asDoubles(x).toArray.exists(neg) then
      val xs = x.asType(using DType.Complex128)
      val ps = p.asType(using DType.Complex128)
      NDArray.zipMap(xs, ps)(CMath.pow)
    else MathK.zipD(MathK.asDoubles(x), MathK.asDoubles(p), MathK.cpow)
  def power(x: Double, p: Double): Double | Complex =
    if x < 0 then CMath.pow(Complex(x, 0.0), Complex(p, 0.0)) else MathK.cpow(x, p)

  /** Inverse cosine; `|x| > 1` gives complex results (`np.emath.arccos`). */
  def arccos[T](x: NDArray[T]): NDArray[Double] | NDArray[Complex] = arr(x, "arccos", absGt1)(math.acos, CMath.acos)
  def arccos(x: Double): Double | Complex = scalar(x, math.abs(x) > 1, math.acos, CMath.acos)
  def arccos(x: Complex): Complex = CMath.acos(x)

  /** Inverse sine; `|x| > 1` gives complex results (`np.emath.arcsin`). */
  def arcsin[T](x: NDArray[T]): NDArray[Double] | NDArray[Complex] = arr(x, "arcsin", absGt1)(math.asin, CMath.asin)
  def arcsin(x: Double): Double | Complex = scalar(x, math.abs(x) > 1, math.asin, CMath.asin)
  def arcsin(x: Complex): Complex = CMath.asin(x)

  /** Inverse hyperbolic tangent; `|x| > 1` gives complex results (`np.emath.arctanh`). */
  def arctanh[T](x: NDArray[T]): NDArray[Double] | NDArray[Complex] = arr(x, "arctanh", absGt1)(MathK.atanh, CMath.atanh)
  def arctanh(x: Double): Double | Complex = scalar(x, math.abs(x) > 1, MathK.atanh, CMath.atanh)
  def arctanh(x: Complex): Complex = CMath.atanh(x)
