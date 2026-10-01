package numscala

import PolyBasis.{asSeries, mapparms, toD, trimcoef}

/** Common base of the `numpy.polynomial` convenience classes ([[Polynomial]], [[Chebyshev]],
  * [[Legendre]], [[Hermite]], [[HermiteE]], [[Laguerre]]) — NumPy's `ABCPolyBase`.
  *
  * A series holds coefficients `coef` (lowest degree first) in its basis, a `domain` that is
  * mapped linearly onto the `window` before evaluation, and a `symbol` used for printing.
  */
abstract class PolySeries[S <: PolySeries[S]] protected (
    private[numscala] val coefArr: Array[Double],
    private[numscala] val domainArr: Array[Double],
    private[numscala] val windowArr: Array[Double],
    val symbol: String
):
  self: S =>

  /** The companion (class-level operations such as `fit`, `fromroots`, `basis`). */
  def companion: PolySeriesCompanion[S]
  private[numscala] def basis: PolyBasis = companion.impl
  private def make(c: Array[Double]): S = companion.create(c, domainArr, windowArr, symbol)

  /** Maximum power allowed by `**` (NumPy `maxpower`). */
  def maxpower: Int = 100

  /** Series coefficients, lowest degree first. */
  def coef: NDArray[Double] = NDArray.fromArray(coefArr.clone())
  /** The domain interval `[a, b]`. */
  def domain: NDArray[Double] = NDArray.fromArray(domainArr.clone())
  /** The window interval `[a, b]`. */
  def window: NDArray[Double] = NDArray.fromArray(windowArr.clone())
  /** Degree of the series (`len(coef) - 1`). */
  def degree: Int = coefArr.length - 1
  /** Number of coefficients (`len(p)`). */
  def length: Int = coefArr.length

  /** `(off, scl)` of the linear map from `domain` to `window`. */
  def mapparms: (Double, Double) = PolyBasis.mapparms(domainArr, windowArr)

  // ---------------------------------------------------------------- evaluation

  /** Evaluates the series at `x`. */
  def apply(x: Double): Double =
    val (off, scl) = mapparms
    basis.valScalar(off + scl * x, coefArr)

  /** Evaluates the series element-wise. */
  def apply(x: NDArray[?]): NDArray[Double] =
    val (off, scl) = mapparms
    toD(x).map(v => basis.valScalar(off + scl * v, coefArr))

  /** Evaluates this series at another series (`p(q)` in NumPy): the composition, in `arg`'s kind. */
  def compose[T <: PolySeries[T]](arg: T): T =
    val (off, scl) = mapparms
    val tb = arg.basis
    val mapped = tb.add(arg.coefArr.map(_ * scl), Array(off))
    arg.companion.create(basis.evalAtSeries(coefArr, tb, mapped), arg.domainArr, arg.windowArr, arg.symbol)

  // ---------------------------------------------------------------- arithmetic

  private def coefOf(o: S): Array[Double] =
    if o.getClass ne getClass then throw new IllegalArgumentException("Polynomial types differ")
    if !java.util.Arrays.equals(o.domainArr, domainArr) then throw new IllegalArgumentException("Domains differ")
    if !java.util.Arrays.equals(o.windowArr, windowArr) then throw new IllegalArgumentException("Windows differ")
    if o.symbol != symbol then throw new IllegalArgumentException("Polynomial symbols differ")
    o.coefArr

  def +(o: S): S = make(basis.add(coefArr, coefOf(o)))
  def -(o: S): S = make(basis.sub(coefArr, coefOf(o)))
  def *(o: S): S = make(basis.mul(coefArr, coefOf(o)))
  def +(s: Double): S = make(basis.add(coefArr, Array(s)))
  def -(s: Double): S = make(basis.sub(coefArr, Array(s)))
  def *(s: Double): S = make(basis.mul(coefArr, Array(s)))
  /** Division by a scalar (NumPy `/`; dividing by a series is not defined, use [[divmod]]). */
  def /(s: Double): S = make(basis.div(coefArr, Array(s))._1)
  def unary_- : S = make(coefArr.map(-_))
  def unary_+ : S = make(coefArr.clone())

  /** Floor division (NumPy `//`): the quotient of [[divmod]]. */
  def floorDiv(o: S): S = divmod(o)._1
  /** Floor division by a scalar. */
  def floorDiv(s: Double): S = make(basis.div(coefArr, Array(s))._1)
  /** Remainder (NumPy `%`). */
  def %(o: S): S = divmod(o)._2
  /** Remainder of division by a scalar. */
  def %(s: Double): S = make(basis.div(coefArr, Array(s))._2)
  /** `divmod(p, q)`: quotient and remainder. */
  def divmod(o: S): (S, S) =
    val (q, r) = basis.div(coefArr, coefOf(o))
    (make(q), make(r))
  /** Integer power. */
  def **(n: Int): S = make(basis.pow(coefArr, n, maxpower))

  // ---------------------------------------------------------------- calculus

  /** Derivative of order `m` (with respect to the domain variable). */
  def deriv(m: Int = 1): S =
    val (_, scl) = mapparms
    make(basis.der(coefArr.map(Array(_)), m, scl).map(_(0)))

  /** Integral of order `m` with constants `k`; the value at `lbnd` (default: the window
    * point mapped from 0, NumPy `lbnd=None`) is `k`. Pass a finite `lbnd` for NumPy's `lbnd=x`.
    */
  def integ(m: Int = 1, k: Seq[Double] = Nil, lbnd: Double = Double.NaN): S =
    val (off, scl) = mapparms
    val lb = if lbnd.isNaN then 0.0 else off + scl * lbnd
    make(basis.integ(coefArr.map(Array(_)), m, k, lb, 1.0 / scl).map(_(0)))

  /** Roots of the series in the domain (always complex, sorted). */
  def roots(): NDArray[Complex] =
    val r = basis.roots(coefArr)
    val (off, scl) = PolyBasis.mapparms(windowArr, domainArr)
    NDArray.fromArray(r.map(z => Complex(off + scl * z.re, scl * z.im)))

  // ---------------------------------------------------------------- misc

  /** `n` equally spaced points `x` in `domain` (default: this series' domain) and `self(x)`. */
  def linspace(n: Int = 100, domain: Seq[Double] | Null = null): (NDArray[Double], NDArray[Double]) =
    val d = if domain == null then domainArr else domain.nn.toArray
    val x = np.linspace(d(0), d(1), n)
    (x, apply(x))

  /** A copy with trailing coefficients of magnitude `<= tol` removed. */
  def trim(tol: Double = 0.0): S = make(trimcoef(coefArr, tol))

  /** A copy truncated to `size` coefficients. */
  def truncate(size: Int): S =
    if size < 1 then throw new IllegalArgumentException("size must be a positive integer")
    make(if size >= coefArr.length then coefArr.clone() else coefArr.take(size))

  /** A copy truncated to degree `deg`. */
  def cutdeg(deg: Int): S = truncate(deg + 1)

  def copy(): S = make(coefArr.clone())

  def has_samecoef(o: PolySeries[?]): Boolean = java.util.Arrays.equals(coefArr, o.coefArr)
  def has_samedomain(o: PolySeries[?]): Boolean = java.util.Arrays.equals(domainArr, o.domainArr)
  def has_samewindow(o: PolySeries[?]): Boolean = java.util.Arrays.equals(windowArr, o.windowArr)
  def has_sametype(o: PolySeries[?]): Boolean = o.getClass eq getClass

  /** Converts to the same kind with a new domain/window (defaults: the class defaults). */
  def convert(domain: Seq[Double] | Null = null, window: Seq[Double] | Null = null): S =
    convert(companion, domain, window)

  /** Converts to another kind of series (NumPy `convert(kind=...)`). */
  def convert[T <: PolySeries[T]](kind: PolySeriesCompanion[T]): T = convert(kind, null, null)

  /** Converts to another kind with the given domain and window (`null`: the kind's defaults). */
  def convert[T <: PolySeries[T]](kind: PolySeriesCompanion[T], domain: Seq[Double] | Null, window: Seq[Double] | Null): T =
    val d = if domain == null then kind.impl.defaultDomain.toSeq else domain.nn
    val w = if window == null then kind.impl.defaultWindow.toSeq else window.nn
    compose(kind.identity(d, w, symbol))

  // ---------------------------------------------------------------- printing

  override def equals(other: Any): Boolean = other match
    case o: PolySeries[?] =>
      (o.getClass eq getClass) && has_samedomain(o) && has_samewindow(o) && has_samecoef(o) && o.symbol == symbol
    case _ => false

  override def hashCode: Int =
    (getClass.getName, coefArr.toSeq, domainArr.toSeq, windowArr.toSeq, symbol).hashCode

  /** NumPy's `repr`, e.g. `Polynomial([1., 2.], domain=[-1.,  1.], window=[-1.,  1.], symbol='x')`. */
  def repr: String =
    def inner(a: NDArray[Double]): String =
      val r = Format.repr(a)
      r.substring(6, r.length - 1)
    s"${companion.name}(${inner(coef)}, domain=${inner(domain)}, window=${inner(window)}, symbol='$symbol')"

  /** NumPy's `str` (unicode or ascii per [[PolynomialModule.set_default_printstyle]]). */
  override def toString: String =
    val unicode = PolySeries.printStyle == "unicode"
    val linewidth = math.max(Format.printOptions.linewidth, 1)
    val fmt = PolySeries.formatFloat
    var out = fmt(coefArr(0), false)
    val (off, scl) = mapparms
    val (term, parens) =
      if off == 0 && scl == 1 then (symbol, false)
      else if scl == 1 then (s"${fmt(off, false)} + $symbol", true)
      else if off == 0 then (s"${fmt(scl, false)}$symbol", true)
      else (s"${fmt(off, false)} + ${fmt(scl, false)}$symbol", true)
    val scaled = if parens then s"($term)" else term
    val rest = coefArr.drop(1)
    for i <- rest.indices do
      out += " "
      val power = (i + 1).toString
      val c = rest(i)
      var next = if c >= 0 then "+ " + fmt(c, true) else "- " + fmt(-c, true)
      next += companion.termString(power, scaled, unicode)
      var lineLen = out.split("\n", -1).last.length + next.length
      if i < rest.length - 1 then lineLen += 2
      if lineLen >= linewidth then next = next.replaceFirst(" ", "\n")
      out += next
    out

object PolySeries:
  @volatile private[numscala] var printStyle: String = "unicode"

  /** NumPy `polyutils.format_float`. */
  private[numscala] def formatFloat(x: Double, parens: Boolean): String =
    val opts = Format.printOptions
    if x.isNaN then opts.nanstr
    else if x.isInfinite then opts.infstr
    else
      val a = math.abs(x)
      val expFormat = x != 0 && (a >= 1e8 || a < math.pow(10, math.min(0, Math.floorDiv(-(opts.precision - 1), 2)).toDouble))
      if expFormat then
        val s0 = Format.scientific(x, opts.precision, false)
        val s = s0.replace(".e", ".0e")
        if parens then "(" + s + ")" else s
      else
        val s = Format.positional(x, opts.precision, false)
        if s.endsWith(".") then s + "0" else s

  private val sub = "₀₁₂₃₄₅₆₇₈₉"
  private val sup = "⁰¹²³⁴⁵⁶⁷⁸⁹"
  private[numscala] def subscript(s: String): String = s.map(c => if c.isDigit then sub(c - '0') else c)
  private[numscala] def superscript(s: String): String = s.map(c => if c.isDigit then sup(c - '0') else c)

/** Class-level operations of a series kind (NumPy classmethods): construction, `fit`,
  * `fromroots`, `identity`, `basis`, `cast`, plus the default `domain` and `window`.
  */
abstract class PolySeriesCompanion[S <: PolySeries[S]] private[numscala] (private[numscala] val impl: PolyBasis):
  private[numscala] def create(c: Array[Double], d: Array[Double], w: Array[Double], symbol: String): S

  /** Class name (`"Chebyshev"`, ...). */
  def name: String = impl.className
  /** NumPy `basis_name` (`"T"` for Chebyshev, `None` for Polynomial). */
  def basis_name: String = impl.basisName

  /** The default domain. */
  def domain: NDArray[Double] = NDArray.fromArray(impl.defaultDomain.clone())
  /** The default window. */
  def window: NDArray[Double] = NDArray.fromArray(impl.defaultWindow.clone())

  private[numscala] def termString(power: String, arg: String, unicode: Boolean): String =
    if unicode then s"·${impl.basisName}${PolySeries.subscript(power)}($arg)"
    else s" ${impl.basisName}_$power($arg)"

  private def interval(v: Seq[Double] | Null, default: Array[Double], what: String): Array[Double] =
    if v == null then default.clone()
    else
      val a = v.nn.toArray
      if a.length != 2 then throw new IllegalArgumentException(s"$what has wrong number of elements.")
      a

  private def checkSymbol(symbol: String): Unit =
    if symbol.isEmpty || !(symbol.head.isLetter || symbol.head == '_') || !symbol.forall(c => c.isLetterOrDigit || c == '_') then
      throw new IllegalArgumentException("Symbol string must be a valid Python identifier")

  /** Creates a series from coefficients (lowest degree first) — NumPy's class constructor. */
  def apply(
      coef: Seq[Double] | NDArray[?],
      domain: Seq[Double] | Null = null,
      window: Seq[Double] | Null = null,
      symbol: String = "x"
  ): S =
    val c: Array[Double] = coef match
      case a: NDArray[?] => PolyBasis.seriesOf(a, trim = false)
      case s: Seq[?] => asSeries(s.asInstanceOf[Seq[Double]].toArray, trim = false)
    checkSymbol(symbol)
    create(c, interval(domain, impl.defaultDomain, "Domain"), interval(window, impl.defaultWindow, "Window"), symbol)

  /** The series representing the identity `x` over `domain` (NumPy `identity`). */
  def identity(domain: Seq[Double] | Null = null, window: Seq[Double] | Null = null, symbol: String = "x"): S =
    val d = interval(domain, impl.defaultDomain, "Domain")
    val w = interval(window, impl.defaultWindow, "Window")
    val (off, scl) = mapparms(w, d)
    checkSymbol(symbol)
    create(impl.line(off, scl), d, w, symbol)

  /** The basis polynomial of degree `deg` (NumPy `basis`). */
  def basis(deg: Int, domain: Seq[Double] | Null = null, window: Seq[Double] | Null = null, symbol: String = "x"): S =
    if deg < 0 then throw new IllegalArgumentException("deg must be non-negative integer")
    apply(Seq.fill(deg)(0.0) :+ 1.0, domain, window, symbol)

  /** The monic-in-this-basis series with the given roots (NumPy `fromroots`). The default
    * (empty) `domain` is the class domain; `null` uses the extent of the roots.
    */
  def fromroots(
      roots: Seq[Double] | NDArray[?],
      domain: Seq[Double] | Null = Nil,
      window: Seq[Double] | Null = null,
      symbol: String = "x"
  ): S =
    val r: Array[Double] = roots match
      case a: NDArray[?] => toD(a).toArray
      case s: Seq[?] => s.asInstanceOf[Seq[Double]].toArray
    val d =
      if domain == null then Array(r.min, r.max)
      else if domain.nn.isEmpty then impl.defaultDomain.clone()
      else interval(domain, impl.defaultDomain, "Domain")
    val w = interval(window, impl.defaultWindow, "Window")
    val (off, scl) = mapparms(d, w)
    val rnew = r.map(off + scl * _)
    val sc = math.pow(scl, r.length.toDouble)
    checkSymbol(symbol)
    create(impl.fromroots(rnew).map(_ / sc), d, w, symbol)

  /** Converts any series to this kind (NumPy `cast`). */
  def cast[T <: PolySeries[T]](series: T, domain: Seq[Double] | Null = null, window: Seq[Double] | Null = null): S =
    series.convert(this, domain, window)

  /** Least-squares fit of degree `deg` (or the listed degrees). `domain = null` uses the
    * extent of `x` (NumPy default), an empty `domain` the class default; `rcond < 0` means
    * NumPy's default. With `full = true` returns `(series, PolyFitInfo)`.
    */
  transparent inline def fit(
      x: NDArray[?],
      y: NDArray[?],
      deg: Int | Seq[Int],
      domain: Seq[Double] | Null = null,
      rcond: Double = -1.0,
      inline full: Boolean = false,
      w: NDArray[?] | Null = null,
      window: Seq[Double] | Null = null,
      symbol: String = "x"
  ) =
    inline if full then fitFull(x, y, deg, domain, rcond, w, window, symbol)
    else fitFull(x, y, deg, domain, rcond, w, window, symbol)._1

  /** Implementation of [[fit]] returning the series and the fit diagnostics. */
  def fitFull(
      x: NDArray[?],
      y: NDArray[?],
      deg: Int | Seq[Int],
      domain: Seq[Double] | Null,
      rcond: Double,
      w: NDArray[?] | Null,
      window: Seq[Double] | Null,
      symbol: String
  ): (S, PolyFitInfo) =
    val xd = toD(x)
    val d =
      if domain == null then
        val xs = xd.toArray
        if xs.isEmpty then throw new IllegalArgumentException("expected non-empty vector for x")
        Array(xs.min, xs.max)
      else if domain.nn.isEmpty then impl.defaultDomain.clone()
      else interval(domain, impl.defaultDomain, "Domain")
    val wd = interval(window, impl.defaultWindow, "Window")
    val (off, scl) = mapparms(d, wd)
    val xnew = xd.map(off + scl * _)
    val degs = deg match
      case i: Int => Seq(i)
      case s: Seq[?] => s.asInstanceOf[Seq[Int]]
    val (coef, resid, rank, sv, rc) = impl.fit(xnew, y, degs, rcond, w)
    if coef.ndim != 1 then throw new IllegalArgumentException("y must be 1-D to fit a series object")
    checkSymbol(symbol)
    (create(coef.toArray, d, wd, symbol), PolyFitInfo(resid, rank, sv, rc))

/** Diagnostics of a least-squares series fit (`[resid, rank, sv, rcond]` in NumPy). */
final case class PolyFitInfo(residuals: NDArray[Double], rank: Int, singularValues: NDArray[Double], rcond: Double)

/** A power series — `numpy.polynomial.Polynomial`. */
final class Polynomial private[numscala] (c: Array[Double], d: Array[Double], w: Array[Double], symbol: String)
    extends PolySeries[Polynomial](c, d, w, symbol):
  def companion: PolySeriesCompanion[Polynomial] = Polynomial

/** `numpy.polynomial.Polynomial` class methods. */
object Polynomial extends PolySeriesCompanion[Polynomial](PolyBasis.Power):
  private[numscala] def create(c: Array[Double], d: Array[Double], w: Array[Double], symbol: String): Polynomial =
    new Polynomial(c, d, w, symbol)
  override def basis_name: String = "None"
  override private[numscala] def termString(power: String, arg: String, unicode: Boolean): String =
    if unicode then (if power == "1" then s"·$arg" else s"·$arg${PolySeries.superscript(power)}")
    else if power == "1" then s" $arg"
    else s" $arg**$power"

/** A Chebyshev series (first kind) — `numpy.polynomial.Chebyshev`. */
final class Chebyshev private[numscala] (c: Array[Double], d: Array[Double], w: Array[Double], symbol: String)
    extends PolySeries[Chebyshev](c, d, w, symbol):
  def companion: PolySeriesCompanion[Chebyshev] = Chebyshev

/** `numpy.polynomial.Chebyshev` class methods. */
object Chebyshev extends PolySeriesCompanion[Chebyshev](PolyBasis.Cheb):
  private[numscala] def create(c: Array[Double], d: Array[Double], w: Array[Double], symbol: String): Chebyshev =
    new Chebyshev(c, d, w, symbol)

  /** Interpolates `func` at the Chebyshev points of the first kind (NumPy `Chebyshev.interpolate`). */
  def interpolate(func: Double => Double, deg: Int, domain: Seq[Double] | Null = null): Chebyshev =
    val d = if domain == null then impl.defaultDomain.clone() else domain.nn.toArray
    val (off, scl) = mapparms(impl.defaultWindow, d)
    val coef = PolyChebModule.chebinterpolate(x => func(off + scl * x), deg)
    apply(coef, d.toSeq)

/** A Legendre series — `numpy.polynomial.Legendre`. */
final class Legendre private[numscala] (c: Array[Double], d: Array[Double], w: Array[Double], symbol: String)
    extends PolySeries[Legendre](c, d, w, symbol):
  def companion: PolySeriesCompanion[Legendre] = Legendre

/** `numpy.polynomial.Legendre` class methods. */
object Legendre extends PolySeriesCompanion[Legendre](PolyBasis.Leg):
  private[numscala] def create(c: Array[Double], d: Array[Double], w: Array[Double], symbol: String): Legendre =
    new Legendre(c, d, w, symbol)

/** A physicists' Hermite series — `numpy.polynomial.Hermite`. */
final class Hermite private[numscala] (c: Array[Double], d: Array[Double], w: Array[Double], symbol: String)
    extends PolySeries[Hermite](c, d, w, symbol):
  def companion: PolySeriesCompanion[Hermite] = Hermite

/** `numpy.polynomial.Hermite` class methods. */
object Hermite extends PolySeriesCompanion[Hermite](PolyBasis.Herm):
  private[numscala] def create(c: Array[Double], d: Array[Double], w: Array[Double], symbol: String): Hermite =
    new Hermite(c, d, w, symbol)

/** A probabilists' Hermite series — `numpy.polynomial.HermiteE`. */
final class HermiteE private[numscala] (c: Array[Double], d: Array[Double], w: Array[Double], symbol: String)
    extends PolySeries[HermiteE](c, d, w, symbol):
  def companion: PolySeriesCompanion[HermiteE] = HermiteE

/** `numpy.polynomial.HermiteE` class methods. */
object HermiteE extends PolySeriesCompanion[HermiteE](PolyBasis.HermE):
  private[numscala] def create(c: Array[Double], d: Array[Double], w: Array[Double], symbol: String): HermiteE =
    new HermiteE(c, d, w, symbol)

/** A Laguerre series — `numpy.polynomial.Laguerre`. */
final class Laguerre private[numscala] (c: Array[Double], d: Array[Double], w: Array[Double], symbol: String)
    extends PolySeries[Laguerre](c, d, w, symbol):
  def companion: PolySeriesCompanion[Laguerre] = Laguerre

/** `numpy.polynomial.Laguerre` class methods. */
object Laguerre extends PolySeriesCompanion[Laguerre](PolyBasis.Lag):
  private[numscala] def create(c: Array[Double], d: Array[Double], w: Array[Double], symbol: String): Laguerre =
    new Laguerre(c, d, w, symbol)
