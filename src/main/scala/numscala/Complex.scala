package numscala

/** A double-precision complex number (NumPy's `complex128`). */
final case class Complex(re: Double, im: Double):
  def real: Double = re
  def imag: Double = im

  def +(o: Complex): Complex = Complex(re + o.re, im + o.im)
  def -(o: Complex): Complex = Complex(re - o.re, im - o.im)
  def *(o: Complex): Complex = Complex(re * o.re - im * o.im, re * o.im + im * o.re)
  def /(o: Complex): Complex =
    // Smith's algorithm for numerical robustness.
    if o.im == 0.0 then Complex(re / o.re, im / o.re)
    else if math.abs(o.re) >= math.abs(o.im) then
      val r = o.im / o.re
      val d = o.re + o.im * r
      Complex((re + im * r) / d, (im - re * r) / d)
    else
      val r = o.re / o.im
      val d = o.re * r + o.im
      Complex((re * r + im) / d, (im * r - re) / d)
  def +(d: Double): Complex = Complex(re + d, im)
  def -(d: Double): Complex = Complex(re - d, im)
  def *(d: Double): Complex = Complex(re * d, im * d)
  def /(d: Double): Complex = Complex(re / d, im / d)
  def unary_- : Complex = Complex(-re, -im)
  def unary_+ : Complex = this

  def conj: Complex = Complex(re, -im)
  def abs: Double = math.hypot(re, im)
  def abs2: Double = re * re + im * im
  def arg: Double = math.atan2(im, re)
  def isNaN: Boolean = re.isNaN || im.isNaN
  def isInfinite: Boolean = re.isInfinite || im.isInfinite
  def isFinite: Boolean = !isNaN && !isInfinite

  def exp: Complex =
    val e = math.exp(re)
    if im == 0.0 then Complex(e, im) else Complex(e * math.cos(im), e * math.sin(im))
  def log: Complex = Complex(math.log(abs), arg)
  def sqrt: Complex =
    if re == 0.0 && im == 0.0 then Complex(0.0, im)
    else
      val t = math.sqrt((math.abs(re) + abs) / 2.0)
      if re >= 0.0 then Complex(t, im / (2.0 * t))
      else Complex(math.abs(im) / (2.0 * t), math.copySign(t, im))
  def pow(o: Complex): Complex =
    if o.re == 0.0 && o.im == 0.0 then Complex.One
    else if re == 0.0 && im == 0.0 then
      if o.im == 0.0 && o.re > 0.0 then Complex.Zero else Complex(Double.NaN, Double.NaN)
    else if o.im == 0.0 && o.re == math.rint(o.re) && math.abs(o.re) <= 64 then
      // exact repeated multiplication for small integral exponents
      var n = math.abs(o.re.toInt)
      var base = this
      var acc = Complex.One
      while n > 0 do
        if (n & 1) == 1 then acc = acc * base
        base = base * base
        n >>= 1
      if o.re < 0 then Complex.One / acc else acc
    else (log * o).exp
  def pow(d: Double): Complex = pow(Complex(d, 0.0))

  def sin: Complex = Complex(math.sin(re) * math.cosh(im), math.cos(re) * math.sinh(im))
  def cos: Complex = Complex(math.cos(re) * math.cosh(im), -math.sin(re) * math.sinh(im))
  def tan: Complex = sin / cos
  def sinh: Complex = Complex(math.sinh(re) * math.cos(im), math.cosh(re) * math.sin(im))
  def cosh: Complex = Complex(math.cosh(re) * math.cos(im), math.sinh(re) * math.sin(im))
  def tanh: Complex = sinh / cosh
  def asin: Complex =
    // -i * log(iz + sqrt(1 - z^2))
    val iz = Complex(-im, re)
    val w = (iz + (Complex.One - this * this).sqrt).log
    Complex(w.im, -w.re)
  def acos: Complex =
    // -i * log(z + i sqrt(1 - z^2))
    val s = (Complex.One - this * this).sqrt
    val w = (this + Complex(-s.im, s.re)).log
    Complex(w.im, -w.re)
  def atan: Complex =
    // i/2 * log((i + z)/(i - z))
    val i = Complex.I
    val w = ((i + this) / (i - this)).log
    Complex(-w.im / 2.0, w.re / 2.0)
  def asinh: Complex = (this + (this * this + Complex.One).sqrt).log
  def acosh: Complex = (this + (this + Complex.One).sqrt * (this - Complex.One).sqrt).log
  def atanh: Complex = ((Complex.One + this) / (Complex.One - this)).log * 0.5

  override def toString: String = Complex.format(this)

object Complex:
  val Zero: Complex = Complex(0.0, 0.0)
  val One: Complex = Complex(1.0, 0.0)
  val I: Complex = Complex(0.0, 1.0)
  def apply(re: Double): Complex = Complex(re, 0.0)
  def polar(r: Double, theta: Double): Complex = Complex(r * math.cos(theta), r * math.sin(theta))

  /** Lexicographic ordering on (re, im), as used by NumPy's sort; NaNs sort last. */
  given ordering: Ordering[Complex] with
    def compare(a: Complex, b: Complex): Int =
      val c = Format.compareDouble(a.re, b.re)
      if c != 0 then c else Format.compareDouble(a.im, b.im)

  private[numscala] def format(c: Complex): String =
    def short(d: Double): String =
      val t = Format.formatFloatShort(d)
      if t.endsWith(".0") then t.dropRight(2) else t
    val r = short(c.re)
    val i = short(math.abs(c.im))
    val sign = if c.im < 0 || (c.im == 0.0 && 1.0 / c.im < 0) then "-" else "+"
    val iStr = if c.im.isNaN then "nan" else i
    if c.re == 0.0 && 1.0 / c.re > 0 then s"${if c.im < 0 then "-" else ""}${iStr}j"
    else s"($r$sign${iStr}j)"
