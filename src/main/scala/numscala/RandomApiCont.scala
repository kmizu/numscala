package numscala.random

import numscala.*
import Cons.*

/** Shared plumbing of [[Generator]] and [[RandomState]]. */
abstract class RandomCore:
  /** The underlying bit generator. */
  def bit_generator: BitGenerator
  protected[numscala] final def bg: BitGenerator = bit_generator

  // ---- primitive samplers (differ between Generator and the legacy RandomState) ----
  protected def pStdNormal(): Double
  protected def pStdExp(): Double
  protected def pStdGamma(shape: Double): Double
  protected def pNormal(loc: Double, scale: Double): Double
  protected def pExponential(scale: Double): Double
  protected def pGamma(shape: Double, scale: Double): Double
  protected def pBeta(a: Double, b: Double): Double
  protected def pChisquare(df: Double): Double
  protected def pNoncentralChisquare(df: Double, nonc: Double): Double
  protected def pF(dfnum: Double, dfden: Double): Double
  protected def pNoncentralF(dfnum: Double, dfden: Double, nonc: Double): Double
  protected def pStdCauchy(): Double
  protected def pStdT(df: Double): Double
  protected def pVonmises(mu: Double, kappa: Double): Double
  protected def pPareto(a: Double): Double
  protected def pWeibull(a: Double): Double
  protected def pPower(a: Double): Double
  protected def pLognormal(mean: Double, sigma: Double): Double
  protected def pRayleigh(scale: Double): Double
  protected def pWald(mean: Double, scale: Double): Double
  protected def uniformRangeCons: Cons

  protected inline def rd(s: SizeArg)(inline f: => Double): NDArray[Double] =
    val shp = RCommon.sizeShape(s)
    if shp == null then NDArray.scalar(f) else RCommon.fillD(RCommon.checkShape(shp))(f)

  protected def toSize(size: Seq[Int]): SizeArg = size

/** The continuous distributions shared by [[Generator]] and [[RandomState]]. */
abstract class RandomContinuous extends RandomCore:
  private def c1(size: SizeArg, a: Param, an: String, ac: Cons)(f: Double => Double): NDArray[Double] =
    RCommon.cont(size, Seq((a, an, ac)))(v => f(v(0)))
  private def c2(size: SizeArg, a: Param, an: String, ac: Cons, b: Param, bn: String, bc: Cons)(
      f: (Double, Double) => Double): NDArray[Double] =
    RCommon.cont(size, Seq((a, an, ac), (b, bn, bc)))(v => f(v(0), v(1)))
  private def c3(size: SizeArg, a: Param, an: String, ac: Cons, b: Param, bn: String, bc: Cons, c: Param, cn: String,
      cc: Cons)(f: (Double, Double, Double) => Double): NDArray[Double] =
    RCommon.cont(size, Seq((a, an, ac), (b, bn, bc), (c, cn, cc)))(v => f(v(0), v(1), v(2)))

  // ---------------- standard_normal / standard_exponential / standard_cauchy ----------------
  /** `standard_normal()`: one N(0, 1) sample. */
  def standard_normal(): Double = pStdNormal()
  /** `standard_normal(size)`: N(0, 1) samples of the given shape. */
  def standard_normal(size: Int*): NDArray[Double] = rd(toSize(size))(pStdNormal())
  /** `standard_normal(size)` with the shape given as a `Seq`. */
  @annotation.targetName("standard_normalSeq") def standard_normal(size: Seq[Int]): NDArray[Double] = rd(size)(pStdNormal())

  /** `standard_exponential()`: one Exp(1) sample. */
  def standard_exponential(): Double = pStdExp()
  /** `standard_exponential(size)`. */
  def standard_exponential(size: Int*): NDArray[Double] = rd(toSize(size))(pStdExp())
  /** `standard_exponential(size)` with the shape given as a `Seq`. */
  @annotation.targetName("standard_exponentialSeq") def standard_exponential(size: Seq[Int]): NDArray[Double] = rd(size)(pStdExp())

  /** `standard_cauchy()`: one standard Cauchy sample. */
  def standard_cauchy(): Double = pStdCauchy()
  /** `standard_cauchy(size)`. */
  def standard_cauchy(size: Int*): NDArray[Double] = rd(toSize(size))(pStdCauchy())
  /** `standard_cauchy(size)` with the shape given as a `Seq`. */
  @annotation.targetName("standard_cauchySeq") def standard_cauchy(size: Seq[Int]): NDArray[Double] = rd(size)(pStdCauchy())

  // ---------------- normal ----------------
  /** `normal(loc, scale)`: one sample. */
  def normal(loc: Double, scale: Double): Double = { RCommon.check(scale, "scale", NonNegative); pNormal(loc, scale) }
  /** `normal()`: one N(0, 1) sample. */
  def normal(): Double = normal(0.0, 1.0)
  /** `normal(loc, scale, size*)`. */
  def normal(loc: Double, scale: Double, size: Int*): NDArray[Double] = normalA(loc, scale, size)
  /** `normal(loc, scale, size)` with broadcasting array parameters. */
  def normal(loc: Param = 0.0, scale: Param = 1.0, size: SizeArg = null): NDArray[Double] = normalA(loc, scale, size)
  private def normalA(loc: Param, scale: Param, size: SizeArg) =
    c2(size, loc, "loc", NoCons, scale, "scale", NonNegative)(pNormal)

  // ---------------- uniform ----------------
  /** `uniform(low, high)`: one sample from `[low, high)`. */
  def uniform(low: Double, high: Double): Double =
    val r = high - low
    if r.isNaN || r.isInfinite then throw new ArithmeticException("high - low range exceeds valid bounds")
    RCommon.check(r, "high - low", uniformRangeCons)
    Dist.uniform(bg, low, r)
  /** `uniform()`: one sample from `[0, 1)`. */
  def uniform(): Double = uniform(0.0, 1.0)
  /** `uniform(low, high, size*)`. */
  def uniform(low: Double, high: Double, size: Int*): NDArray[Double] = uniformA(low, high, size)
  /** `uniform(low, high, size)` with broadcasting array parameters. */
  def uniform(low: Param = 0.0, high: Param = 1.0, size: SizeArg = null): NDArray[Double] = uniformA(low, high, size)
  private def uniformA(low: Param, high: Param, size: SizeArg): NDArray[Double] =
    if RCommon.isScalar(low) && RCommon.isScalar(high) then
      val l = RCommon.scalarD(low)
      val r = RCommon.scalarD(high) - l
      if r.isNaN || r.isInfinite then throw new ArithmeticException("high - low range exceeds valid bounds")
      c2(size, l, "", NoCons, r, "high - low", uniformRangeCons)((a, b) => Dist.uniform(bg, a, b))
    else
      val lo = RCommon.toD(low)
      val hi = RCommon.toD(high)
      val (shp, fl) = RCommon.broadcastAll(null, Seq(lo, hi))
      val rng = NDArray.fromArray(Array.tabulate(fl(0).length)(i => fl(1)(i) - fl(0)(i)), shp)
      if rng.toArray.exists(x => x.isNaN || x.isInfinite) then throw new ArithmeticException("Range exceeds valid bounds")
      c2(size, lo, "", NoCons, rng, "high - low", uniformRangeCons)((a, b) => Dist.uniform(bg, a, b))

  // ---------------- exponential ----------------
  /** `exponential(scale)`: one sample. */
  def exponential(scale: Double): Double = { RCommon.check(scale, "scale", NonNegative); pExponential(scale) }
  /** `exponential()`: one Exp(1) sample. */
  def exponential(): Double = exponential(1.0)
  /** `exponential(scale, size*)`. */
  def exponential(scale: Double, size: Int*): NDArray[Double] = exponentialA(scale, size)
  /** `exponential(scale, size)` with broadcasting. */
  def exponential(scale: Param = 1.0, size: SizeArg = null): NDArray[Double] = exponentialA(scale, size)
  private def exponentialA(scale: Param, size: SizeArg) = c1(size, scale, "scale", NonNegative)(pExponential)

  // ---------------- standard_gamma / gamma ----------------
  /** `standard_gamma(shape)`: one sample. */
  def standard_gamma(shape: Double): Double = { RCommon.check(shape, "shape", NonNegative); pStdGamma(shape) }
  /** `standard_gamma(shape, size*)`. */
  def standard_gamma(shape: Double, size: Int*): NDArray[Double] = standardGammaA(shape, size)
  /** `standard_gamma(shape, size)` with broadcasting. */
  def standard_gamma(shape: Param, size: SizeArg = null): NDArray[Double] = standardGammaA(shape, size)
  private def standardGammaA(shape: Param, size: SizeArg) = c1(size, shape, "shape", NonNegative)(pStdGamma)

  /** `gamma(shape, scale)`: one sample. */
  def gamma(shape: Double, scale: Double): Double =
    RCommon.check(shape, "shape", NonNegative); RCommon.check(scale, "scale", NonNegative)
    pGamma(shape, scale)
  /** `gamma(shape)`: one sample with unit scale. */
  def gamma(shape: Double): Double = gamma(shape, 1.0)
  /** `gamma(shape, scale, size*)`. */
  def gamma(shape: Double, scale: Double, size: Int*): NDArray[Double] = gammaA(shape, scale, size)
  /** `gamma(shape, scale, size)` with broadcasting. */
  def gamma(shape: Param, scale: Param = 1.0, size: SizeArg = null): NDArray[Double] = gammaA(shape, scale, size)
  private def gammaA(shape: Param, scale: Param, size: SizeArg) =
    c2(size, shape, "shape", NonNegative, scale, "scale", NonNegative)(pGamma)

  // ---------------- beta ----------------
  /** `beta(a, b)`: one sample. */
  def beta(a: Double, b: Double): Double =
    RCommon.check(a, "a", Positive); RCommon.check(b, "b", Positive)
    pBeta(a, b)
  /** `beta(a, b, size*)`. */
  def beta(a: Double, b: Double, size: Int*): NDArray[Double] = betaA(a, b, size)
  /** `beta(a, b, size)` with broadcasting. */
  def beta(a: Param, b: Param, size: SizeArg = null): NDArray[Double] = betaA(a, b, size)
  private def betaA(a: Param, b: Param, size: SizeArg) = c2(size, a, "a", Positive, b, "b", Positive)(pBeta)

  // ---------------- chisquare / noncentral ----------------
  /** `chisquare(df)`: one sample. */
  def chisquare(df: Double): Double = { RCommon.check(df, "df", Positive); pChisquare(df) }
  /** `chisquare(df, size*)`. */
  def chisquare(df: Double, size: Int*): NDArray[Double] = chisquareA(df, size)
  /** `chisquare(df, size)` with broadcasting. */
  def chisquare(df: Param, size: SizeArg = null): NDArray[Double] = chisquareA(df, size)
  private def chisquareA(df: Param, size: SizeArg) = c1(size, df, "df", Positive)(pChisquare)

  /** `noncentral_chisquare(df, nonc)`: one sample. */
  def noncentral_chisquare(df: Double, nonc: Double): Double =
    RCommon.check(df, "df", Positive); RCommon.check(nonc, "nonc", NonNegative)
    pNoncentralChisquare(df, nonc)
  /** `noncentral_chisquare(df, nonc, size*)`. */
  def noncentral_chisquare(df: Double, nonc: Double, size: Int*): NDArray[Double] = nccA(df, nonc, size)
  /** `noncentral_chisquare(df, nonc, size)` with broadcasting. */
  def noncentral_chisquare(df: Param, nonc: Param, size: SizeArg = null): NDArray[Double] = nccA(df, nonc, size)
  private def nccA(df: Param, nonc: Param, size: SizeArg) =
    c2(size, df, "df", Positive, nonc, "nonc", NonNegative)(pNoncentralChisquare)

  // ---------------- f / noncentral_f ----------------
  /** `f(dfnum, dfden)`: one sample. */
  def f(dfnum: Double, dfden: Double): Double =
    RCommon.check(dfnum, "dfnum", Positive); RCommon.check(dfden, "dfden", Positive)
    pF(dfnum, dfden)
  /** `f(dfnum, dfden, size*)`. */
  def f(dfnum: Double, dfden: Double, size: Int*): NDArray[Double] = fA(dfnum, dfden, size)
  /** `f(dfnum, dfden, size)` with broadcasting. */
  def f(dfnum: Param, dfden: Param, size: SizeArg = null): NDArray[Double] = fA(dfnum, dfden, size)
  private def fA(a: Param, b: Param, size: SizeArg) = c2(size, a, "dfnum", Positive, b, "dfden", Positive)(pF)

  /** `noncentral_f(dfnum, dfden, nonc)`: one sample. */
  def noncentral_f(dfnum: Double, dfden: Double, nonc: Double): Double =
    RCommon.check(dfnum, "dfnum", Positive); RCommon.check(dfden, "dfden", Positive)
    RCommon.check(nonc, "nonc", NonNegative)
    pNoncentralF(dfnum, dfden, nonc)
  /** `noncentral_f(dfnum, dfden, nonc, size*)`. */
  def noncentral_f(dfnum: Double, dfden: Double, nonc: Double, size: Int*): NDArray[Double] = ncfA(dfnum, dfden, nonc, size)
  /** `noncentral_f(dfnum, dfden, nonc, size)` with broadcasting. */
  def noncentral_f(dfnum: Param, dfden: Param, nonc: Param, size: SizeArg = null): NDArray[Double] =
    ncfA(dfnum, dfden, nonc, size)
  private def ncfA(a: Param, b: Param, c: Param, size: SizeArg) =
    c3(size, a, "dfnum", Positive, b, "dfden", Positive, c, "nonc", NonNegative)(pNoncentralF)

  // ---------------- standard_t ----------------
  /** `standard_t(df)`: one sample. */
  def standard_t(df: Double): Double = { RCommon.check(df, "df", Positive); pStdT(df) }
  /** `standard_t(df, size*)`. */
  def standard_t(df: Double, size: Int*): NDArray[Double] = stA(df, size)
  /** `standard_t(df, size)` with broadcasting. */
  def standard_t(df: Param, size: SizeArg = null): NDArray[Double] = stA(df, size)
  private def stA(df: Param, size: SizeArg) = c1(size, df, "df", Positive)(pStdT)

  // ---------------- vonmises ----------------
  /** `vonmises(mu, kappa)`: one sample. */
  def vonmises(mu: Double, kappa: Double): Double = { RCommon.check(kappa, "kappa", NonNegative); pVonmises(mu, kappa) }
  /** `vonmises(mu, kappa, size*)`. */
  def vonmises(mu: Double, kappa: Double, size: Int*): NDArray[Double] = vmA(mu, kappa, size)
  /** `vonmises(mu, kappa, size)` with broadcasting. */
  def vonmises(mu: Param, kappa: Param, size: SizeArg = null): NDArray[Double] = vmA(mu, kappa, size)
  private def vmA(mu: Param, kappa: Param, size: SizeArg) =
    c2(size, mu, "mu", NoCons, kappa, "kappa", NonNegative)(pVonmises)

  // ---------------- pareto / weibull / power ----------------
  /** `pareto(a)`: one Lomax sample. */
  def pareto(a: Double): Double = { RCommon.check(a, "a", Positive); pPareto(a) }
  /** `pareto(a, size*)`. */
  def pareto(a: Double, size: Int*): NDArray[Double] = paretoA(a, size)
  /** `pareto(a, size)` with broadcasting. */
  def pareto(a: Param, size: SizeArg = null): NDArray[Double] = paretoA(a, size)
  private def paretoA(a: Param, size: SizeArg) = c1(size, a, "a", Positive)(pPareto)

  /** `weibull(a)`: one sample. */
  def weibull(a: Double): Double = { RCommon.check(a, "a", NonNegative); pWeibull(a) }
  /** `weibull(a, size*)`. */
  def weibull(a: Double, size: Int*): NDArray[Double] = weibullA(a, size)
  /** `weibull(a, size)` with broadcasting. */
  def weibull(a: Param, size: SizeArg = null): NDArray[Double] = weibullA(a, size)
  private def weibullA(a: Param, size: SizeArg) = c1(size, a, "a", NonNegative)(pWeibull)

  /** `power(a)`: one sample. */
  def power(a: Double): Double = { RCommon.check(a, "a", Positive); pPower(a) }
  /** `power(a, size*)`. */
  def power(a: Double, size: Int*): NDArray[Double] = powerA(a, size)
  /** `power(a, size)` with broadcasting. */
  def power(a: Param, size: SizeArg = null): NDArray[Double] = powerA(a, size)
  private def powerA(a: Param, size: SizeArg) = c1(size, a, "a", Positive)(pPower)

  // ---------------- laplace / gumbel / logistic ----------------
  /** `laplace(loc, scale)`: one sample. */
  def laplace(loc: Double, scale: Double): Double =
    RCommon.check(scale, "scale", NonNegative); Dist.laplace(bg, loc, scale)
  /** `laplace()`: one standard sample. */
  def laplace(): Double = laplace(0.0, 1.0)
  /** `laplace(loc, scale, size*)`. */
  def laplace(loc: Double, scale: Double, size: Int*): NDArray[Double] = laplaceA(loc, scale, size)
  /** `laplace(loc, scale, size)` with broadcasting. */
  def laplace(loc: Param = 0.0, scale: Param = 1.0, size: SizeArg = null): NDArray[Double] = laplaceA(loc, scale, size)
  private def laplaceA(loc: Param, scale: Param, size: SizeArg) =
    c2(size, loc, "loc", NoCons, scale, "scale", NonNegative)((a, b) => Dist.laplace(bg, a, b))

  /** `gumbel(loc, scale)`: one sample. */
  def gumbel(loc: Double, scale: Double): Double =
    RCommon.check(scale, "scale", NonNegative); Dist.gumbel(bg, loc, scale)
  /** `gumbel()`: one standard sample. */
  def gumbel(): Double = gumbel(0.0, 1.0)
  /** `gumbel(loc, scale, size*)`. */
  def gumbel(loc: Double, scale: Double, size: Int*): NDArray[Double] = gumbelA(loc, scale, size)
  /** `gumbel(loc, scale, size)` with broadcasting. */
  def gumbel(loc: Param = 0.0, scale: Param = 1.0, size: SizeArg = null): NDArray[Double] = gumbelA(loc, scale, size)
  private def gumbelA(loc: Param, scale: Param, size: SizeArg) =
    c2(size, loc, "loc", NoCons, scale, "scale", NonNegative)((a, b) => Dist.gumbel(bg, a, b))

  /** `logistic(loc, scale)`: one sample. */
  def logistic(loc: Double, scale: Double): Double =
    RCommon.check(scale, "scale", NonNegative); Dist.logistic(bg, loc, scale)
  /** `logistic()`: one standard sample. */
  def logistic(): Double = logistic(0.0, 1.0)
  /** `logistic(loc, scale, size*)`. */
  def logistic(loc: Double, scale: Double, size: Int*): NDArray[Double] = logisticA(loc, scale, size)
  /** `logistic(loc, scale, size)` with broadcasting. */
  def logistic(loc: Param = 0.0, scale: Param = 1.0, size: SizeArg = null): NDArray[Double] = logisticA(loc, scale, size)
  private def logisticA(loc: Param, scale: Param, size: SizeArg) =
    c2(size, loc, "loc", NoCons, scale, "scale", NonNegative)((a, b) => Dist.logistic(bg, a, b))

  // ---------------- lognormal / rayleigh / wald ----------------
  /** `lognormal(mean, sigma)`: one sample. */
  def lognormal(mean: Double, sigma: Double): Double =
    RCommon.check(sigma, "sigma", NonNegative); pLognormal(mean, sigma)
  /** `lognormal()`: one standard sample. */
  def lognormal(): Double = lognormal(0.0, 1.0)
  /** `lognormal(mean, sigma, size*)`. */
  def lognormal(mean: Double, sigma: Double, size: Int*): NDArray[Double] = lognormalA(mean, sigma, size)
  /** `lognormal(mean, sigma, size)` with broadcasting. */
  def lognormal(mean: Param = 0.0, sigma: Param = 1.0, size: SizeArg = null): NDArray[Double] = lognormalA(mean, sigma, size)
  private def lognormalA(mean: Param, sigma: Param, size: SizeArg) =
    c2(size, mean, "mean", NoCons, sigma, "sigma", NonNegative)(pLognormal)

  /** `rayleigh(scale)`: one sample. */
  def rayleigh(scale: Double): Double = { RCommon.check(scale, "scale", NonNegative); pRayleigh(scale) }
  /** `rayleigh()`: one sample with unit scale. */
  def rayleigh(): Double = rayleigh(1.0)
  /** `rayleigh(scale, size*)`. */
  def rayleigh(scale: Double, size: Int*): NDArray[Double] = rayleighA(scale, size)
  /** `rayleigh(scale, size)` with broadcasting. */
  def rayleigh(scale: Param = 1.0, size: SizeArg = null): NDArray[Double] = rayleighA(scale, size)
  private def rayleighA(scale: Param, size: SizeArg) = c1(size, scale, "scale", NonNegative)(pRayleigh)

  /** `wald(mean, scale)`: one inverse-Gaussian sample. */
  def wald(mean: Double, scale: Double): Double =
    RCommon.check(mean, "mean", Positive); RCommon.check(scale, "scale", Positive)
    pWald(mean, scale)
  /** `wald(mean, scale, size*)`. */
  def wald(mean: Double, scale: Double, size: Int*): NDArray[Double] = waldA(mean, scale, size)
  /** `wald(mean, scale, size)` with broadcasting. */
  def wald(mean: Param, scale: Param, size: SizeArg = null): NDArray[Double] = waldA(mean, scale, size)
  private def waldA(mean: Param, scale: Param, size: SizeArg) =
    c2(size, mean, "mean", Positive, scale, "scale", Positive)(pWald)

  // ---------------- triangular ----------------
  /** `triangular(left, mode, right)`: one sample. */
  def triangular(left: Double, mode: Double, right: Double): Double =
    triCheck(left, mode, right)
    Dist.triangular(bg, left, mode, right)
  /** `triangular(left, mode, right, size*)`. */
  def triangular(left: Double, mode: Double, right: Double, size: Int*): NDArray[Double] = triA(left, mode, right, size)
  /** `triangular(left, mode, right, size)` with broadcasting. */
  def triangular(left: Param, mode: Param, right: Param, size: SizeArg = null): NDArray[Double] =
    triA(left, mode, right, size)
  private def triCheck(l: Double, m: Double, r: Double): Unit =
    if l > m then throw new IllegalArgumentException("left > mode")
    if m > r then throw new IllegalArgumentException("mode > right")
    if l == r then throw new IllegalArgumentException("left == right")
  private def triA(left: Param, mode: Param, right: Param, size: SizeArg): NDArray[Double] =
    val (_, fl) = RCommon.broadcastAll(null, Seq(RCommon.toD(left), RCommon.toD(mode), RCommon.toD(right)))
    val idx = fl(0).indices
    if idx.exists(i => fl(0)(i) > fl(1)(i)) then throw new IllegalArgumentException("left > mode")
    if idx.exists(i => fl(1)(i) > fl(2)(i)) then throw new IllegalArgumentException("mode > right")
    if idx.exists(i => fl(0)(i) == fl(2)(i)) then throw new IllegalArgumentException("left == right")
    c3(size, left, "", NoCons, mode, "", NoCons, right, "", NoCons)((a, b, c) => Dist.triangular(bg, a, b, c))
