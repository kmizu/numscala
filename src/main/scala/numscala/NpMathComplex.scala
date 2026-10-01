package numscala

/** Complex elementary functions following C99 branch cuts and signed-zero conventions
  * (the behaviour of NumPy's `npy_c*` functions), using Kahan's formulas for the
  * inverse trigonometric and hyperbolic functions.
  */
private[numscala] object CMath:
  private final val LOG10E = 0.4342944819032518

  def sqrt(z: Complex): Complex = z.sqrt
  def exp(z: Complex): Complex = z.exp
  def log(z: Complex): Complex = Complex(math.log(math.hypot(z.re, z.im)), math.atan2(z.im, z.re))
  def log2(z: Complex): Complex =
    val l = log(z)
    Complex(l.re * MathK.LOG2E, l.im * MathK.LOG2E)
  def log10(z: Complex): Complex =
    val l = log(z)
    Complex(l.re * LOG10E, l.im * LOG10E)
  def log1p(z: Complex): Complex =
    val l = math.hypot(z.re + 1.0, z.im)
    Complex(math.log(l), math.atan2(z.im, z.re + 1.0))
  def expm1(z: Complex): Complex =
    val a = math.sin(z.im / 2.0)
    Complex(math.expm1(z.re) * math.cos(z.im) - 2.0 * a * a, math.exp(z.re) * math.sin(z.im))
  def exp2(z: Complex): Complex = exp(Complex(z.re * MathK.LN2, z.im * MathK.LN2))

  /** NumPy's complex power: exact repeated multiplication for small integral exponents. */
  def pow(a: Complex, b: Complex): Complex =
    if b.re == 0.0 && b.im == 0.0 then Complex.One
    else if a.re == 0.0 && a.im == 0.0 then
      // NumPy: 0 ** b is 0 when Re(b) > 0, otherwise nan+nanj
      if b.re > 0.0 then Complex.Zero else Complex(Double.NaN, Double.NaN)
    else if b.im == 0.0 && b.re == math.rint(b.re) && math.abs(b.re) < 100.0 then
      val n = b.re.toInt
      if n == 1 then a
      else if n == 2 then a * a
      else if n == 3 then a * a * a
      else
        var m = math.abs(n)
        var base = a
        var acc = Complex.One
        while m > 0 do
          if (m & 1) == 1 then acc = acc * base
          base = base * base
          m >>= 1
        if n < 0 then Complex.One / acc else acc
    else
      val l = log(a)
      exp(l * b)

  def sin(z: Complex): Complex =
    val x = z.re
    val y = z.im
    if x == 0.0 then Complex(x, math.sinh(y))
    else if y == 0.0 then Complex(math.sin(x), math.cos(x) * y)
    else Complex(math.sin(x) * math.cosh(y), math.cos(x) * math.sinh(y))

  def cos(z: Complex): Complex =
    val x = z.re
    val y = z.im
    if x == 0.0 then Complex(math.cosh(y), if y == 0.0 then -x * y else -x * math.signum(y))
    else if y == 0.0 then Complex(math.cos(x), -math.sin(x) * y)
    else Complex(math.cos(x) * math.cosh(y), -math.sin(x) * math.sinh(y))

  def sinh(z: Complex): Complex =
    val x = z.re
    val y = z.im
    if y == 0.0 then Complex(math.sinh(x), y)
    else if x == 0.0 then Complex(x * math.cos(y), math.sin(y))
    else Complex(math.sinh(x) * math.cos(y), math.cosh(x) * math.sin(y))

  def cosh(z: Complex): Complex =
    val x = z.re
    val y = z.im
    if y == 0.0 then Complex(math.cosh(x), if x == 0.0 then x * y else math.signum(x) * y)
    else if x == 0.0 then Complex(math.cos(y), x * math.sin(y))
    else Complex(math.cosh(x) * math.cos(y), math.sinh(x) * math.sin(y))

  /** Kahan's algorithm (as in FreeBSD's `ctanh`). */
  def tanh(z: Complex): Complex =
    val x = z.re
    val y = z.im
    if x.isNaN && y == 0.0 then Complex(x, y)
    else if math.abs(x) >= 22.0 then
      val e = math.exp(-2.0 * math.abs(x))
      Complex(math.copySign(1.0, x), 4.0 * math.sin(y) * math.cos(y) * e)
    else
      val t = math.tan(y)
      val beta = 1.0 + t * t
      val s = math.sinh(x)
      val rho = math.sqrt(1.0 + s * s)
      val denom = 1.0 + beta * s * s
      Complex(beta * rho * s / denom, t / denom)

  def tan(z: Complex): Complex =
    val w = tanh(Complex(-z.im, z.re))
    Complex(w.im, -w.re)

  // 1 - z, 1 + z, z - 1, z + 1 computed with a real 1 so that signed zeros survive (as in C99)
  private def oneMinus(z: Complex): Complex = Complex(1.0 - z.re, -z.im)
  private def onePlus(z: Complex): Complex = Complex(1.0 + z.re, z.im)
  private def minusOne(z: Complex): Complex = Complex(z.re - 1.0, z.im)

  def asin(z: Complex): Complex =
    val s1 = oneMinus(z).sqrt
    val s2 = onePlus(z).sqrt
    Complex(
      math.atan2(z.re, s1.re * s2.re - s1.im * s2.im),
      MathK.asinh(s1.re * s2.im - s1.im * s2.re)
    )

  def acos(z: Complex): Complex =
    val s1 = oneMinus(z).sqrt
    val s2 = onePlus(z).sqrt
    Complex(2.0 * math.atan2(s1.re, s2.re), MathK.asinh(s2.re * s1.im - s2.im * s1.re))

  def asinh(z: Complex): Complex =
    val w = asin(Complex(-z.im, z.re))
    Complex(w.im, -w.re)

  def acosh(z: Complex): Complex =
    val s1 = minusOne(z).sqrt
    val s2 = onePlus(z).sqrt
    Complex(MathK.asinh(s1.re * s2.re + s1.im * s2.im), 2.0 * math.atan2(s1.im, s2.re))

  def atanh(z: Complex): Complex =
    val x = z.re
    val y = z.im
    val omx = 1.0 - x
    val re = 0.25 * math.log1p(4.0 * x / (omx * omx + y * y))
    val im = 0.5 * math.atan2(2.0 * y, omx * (1.0 + x) - y * y)
    Complex(re, im)

  def atan(z: Complex): Complex =
    val w = atanh(Complex(-z.im, z.re))
    Complex(w.im, -w.re)

  def rint(z: Complex): Complex = Complex(math.rint(z.re), math.rint(z.im))
