package com.github.kmizu.numscala

/** `numpy.polynomial`: the series classes and the functional sub-modules.
  *
  * {{{
  * val p = np.polynomial.Polynomial(Seq(1.0, 2.0, 3.0))
  * p(2.0)                                     // 17.0
  * val c = np.polynomial.Chebyshev.fit(x, y, 3)
  * np.polynomial.chebyshev.chebval(0.5, np.array(1.0, 2.0, 3.0))
  * np.polynomial.legendre.leggauss(5)
  * }}}
  */
object PolynomialModule:
  /** `numpy.polynomial.Polynomial`. */
  val Polynomial: com.github.kmizu.numscala.Polynomial.type = com.github.kmizu.numscala.Polynomial
  type Polynomial = com.github.kmizu.numscala.Polynomial
  /** `numpy.polynomial.Chebyshev`. */
  val Chebyshev: com.github.kmizu.numscala.Chebyshev.type = com.github.kmizu.numscala.Chebyshev
  type Chebyshev = com.github.kmizu.numscala.Chebyshev
  /** `numpy.polynomial.Legendre`. */
  val Legendre: com.github.kmizu.numscala.Legendre.type = com.github.kmizu.numscala.Legendre
  type Legendre = com.github.kmizu.numscala.Legendre
  /** `numpy.polynomial.Hermite`. */
  val Hermite: com.github.kmizu.numscala.Hermite.type = com.github.kmizu.numscala.Hermite
  type Hermite = com.github.kmizu.numscala.Hermite
  /** `numpy.polynomial.HermiteE`. */
  val HermiteE: com.github.kmizu.numscala.HermiteE.type = com.github.kmizu.numscala.HermiteE
  type HermiteE = com.github.kmizu.numscala.HermiteE
  /** `numpy.polynomial.Laguerre`. */
  val Laguerre: com.github.kmizu.numscala.Laguerre.type = com.github.kmizu.numscala.Laguerre
  type Laguerre = com.github.kmizu.numscala.Laguerre

  /** `numpy.polynomial.polynomial`. */
  val polynomial: PolyPowerModule.type = PolyPowerModule
  /** `numpy.polynomial.chebyshev`. */
  val chebyshev: PolyChebModule.type = PolyChebModule
  /** `numpy.polynomial.legendre`. */
  val legendre: PolyLegModule.type = PolyLegModule
  /** `numpy.polynomial.hermite`. */
  val hermite: PolyHermModule.type = PolyHermModule
  /** `numpy.polynomial.hermite_e`. */
  val hermite_e: PolyHermEModule.type = PolyHermEModule
  /** `numpy.polynomial.laguerre`. */
  val laguerre: PolyLagModule.type = PolyLagModule
  /** `numpy.polynomial.polyutils`. */
  val polyutils: PolyUtils.type = PolyUtils

  /** `numpy.polynomial.set_default_printstyle`: `"unicode"` (default) or `"ascii"`. */
  def set_default_printstyle(style: String): Unit = style match
    case "unicode" | "ascii" => PolySeries.printStyle = style
    case other => throw new IllegalArgumentException(s"Unsupported format string '$other'. Valid options are 'ascii' and 'unicode'")

/** `numpy.polynomial.polyutils`: helpers shared by the polynomial modules. */
object PolyUtils:
  /** `trimseq`: removes trailing zeros (keeping at least one coefficient). */
  def trimseq(seq: NDArray[?]): NDArray[Double] =
    val r = PolyBasis.trimseq(PolyBasis.seriesOf(seq, trim = false))
    NDArray.fromArray(r)

  /** `trimcoef`: removes trailing coefficients with magnitude `<= tol`. */
  def trimcoef(c: NDArray[?], tol: Double = 0.0): NDArray[Double] = PolyFunctional.trimcoef(c, tol)

  /** `as_series`: converts arrays to 1-D double series (trailing zeros trimmed when `trim`). */
  def as_series(alist: Seq[NDArray[?]], trim: Boolean = true): Seq[NDArray[Double]] =
    alist.map(a => NDArray.fromArray(PolyBasis.seriesOf(a, trim)))

  /** `getdomain`: the smallest interval `[min, max]` containing `x`. */
  def getdomain(x: NDArray[?]): NDArray[Double] =
    val xs = PolyBasis.toD(x).toArray
    if xs.isEmpty then throw new IllegalArgumentException("zero-size array to reduction operation minimum which has no identity")
    NDArray.fromArray(Array(xs.min, xs.max))

  /** `mapparms`: `(off, scl)` of the linear map from `old` to `new`. */
  def mapparms(old: Seq[Double], `new`: Seq[Double]): (Double, Double) =
    PolyBasis.mapparms(old.toArray, `new`.toArray)

  /** `mapdomain`: maps `x` linearly from the interval `old` to `new`. */
  def mapdomain(x: NDArray[?], old: Seq[Double], `new`: Seq[Double]): NDArray[Double] =
    val (off, scl) = mapparms(old, `new`)
    PolyBasis.toD(x).map(off + scl * _)

  /** `mapdomain` for a scalar. */
  def mapdomain(x: Double, old: Seq[Double], `new`: Seq[Double]): Double =
    val (off, scl) = mapparms(old, `new`)
    off + scl * x
