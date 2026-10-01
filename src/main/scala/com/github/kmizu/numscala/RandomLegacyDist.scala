package com.github.kmizu.numscala.random

import scala.util.boundary
import scala.util.boundary.break

/** Bit generator plus the cached second Gaussian of the polar method (`aug_bitgen_t`). */
private[numscala] final class AugState(var bg: BitGenerator):
  var hasGauss: Boolean = false
  var gauss: Double = 0.0

/** Port of NumPy's `legacy-distributions.c` (the `RandomState` samplers). */
private[numscala] object LegacyDist:
  import java.lang.Math.*

  inline def nextDouble(s: AugState): Double = s.bg.nextDouble()

  def gauss(s: AugState): Double =
    if s.hasGauss then
      val t = s.gauss
      s.hasGauss = false
      s.gauss = 0.0
      t
    else
      var x1, x2, r2 = 0.0
      while
        x1 = 2.0 * nextDouble(s) - 1.0
        x2 = 2.0 * nextDouble(s) - 1.0
        r2 = x1 * x1 + x2 * x2
        r2 >= 1.0 || r2 == 0.0
      do ()
      val f = sqrt(-2.0 * log(r2) / r2)
      s.gauss = f * x1
      s.hasGauss = true
      f * x2

  def standardExponential(s: AugState): Double = -log(1.0 - nextDouble(s))

  def standardGamma(s: AugState, shape: Double): Double =
    if shape == 1.0 then standardExponential(s)
    else if shape == 0.0 then 0.0
    else if shape < 1.0 then
      while true do
        val u = nextDouble(s)
        val v = standardExponential(s)
        if u <= 1.0 - shape then
          val x = pow(u, 1.0 / shape)
          if x <= v then return x
        else
          val y = -log((1 - u) / shape)
          val x = pow(1.0 - shape + shape * y, 1.0 / shape)
          if x <= v + y then return x
      0.0
    else
      val b = shape - 1.0 / 3.0
      val c = 1.0 / sqrt(9 * b)
      while true do
        var x, v = 0.0
        while
          x = gauss(s)
          v = 1.0 + c * x
          v <= 0.0
        do ()
        v = v * v * v
        val u = nextDouble(s)
        if u < 1.0 - 0.0331 * (x * x) * (x * x) then return b * v
        if log(u) < 0.5 * x * x + b * (1.0 - v + log(v)) then return b * v
      0.0

  def gamma(s: AugState, shape: Double, scale: Double): Double = scale * standardGamma(s, shape)
  def pareto(s: AugState, a: Double): Double = exp(standardExponential(s) / a) - 1
  def weibull(s: AugState, a: Double): Double = if a == 0.0 then 0.0 else pow(standardExponential(s), 1.0 / a)
  def power(s: AugState, a: Double): Double = pow(1 - exp(-standardExponential(s)), 1.0 / a)
  def chisquare(s: AugState, df: Double): Double = 2.0 * standardGamma(s, df / 2.0)
  def rayleigh(s: AugState, mode: Double): Double = mode * sqrt(-2.0 * log1p(-nextDouble(s)))

  def noncentralChisquare(s: AugState, df: Double, nonc: Double): Double =
    if nonc == 0 then chisquare(s, df)
    else if 1 < df then
      val chi2 = chisquare(s, df - 1)
      val n = gauss(s) + sqrt(nonc)
      chi2 + n * n
    else
      val i = Dist.poisson(s.bg, nonc / 2.0)
      val out = chisquare(s, df + 2 * i)
      if nonc.isNaN then Double.NaN else out

  def noncentralF(s: AugState, dfnum: Double, dfden: Double, nonc: Double): Double =
    val t = noncentralChisquare(s, dfnum, nonc) * dfden
    t / (chisquare(s, dfden) * dfnum)

  def wald(s: AugState, mean: Double, scale: Double): Double =
    val mu2l = mean / (2 * scale)
    var y = gauss(s)
    y = mean * y * y
    val x = mean + mu2l * (y - sqrt(4 * scale * y + y * y))
    val u = nextDouble(s)
    if u <= mean / (mean + x) then x else mean * mean / x

  def normal(s: AugState, loc: Double, scale: Double): Double = loc + scale * gauss(s)
  def lognormal(s: AugState, mean: Double, sigma: Double): Double = exp(normal(s, mean, sigma))
  def standardT(s: AugState, df: Double): Double =
    val num = gauss(s)
    val denom = standardGamma(s, df / 2)
    sqrt(df / 2) * num / sqrt(denom)
  def negativeBinomial(s: AugState, n: Double, p: Double): Long =
    val y = gamma(s, n, (1 - p) / p)
    Dist.poisson(s.bg, y)
  def standardCauchy(s: AugState): Double =
    val a = gauss(s)
    val b = gauss(s)
    a / b

  def beta(s: AugState, a: Double, b: Double): Double =
    if a <= 1.0 && b <= 1.0 then
      while true do
        val u = nextDouble(s)
        val v = nextDouble(s)
        val x = pow(u, 1.0 / a)
        val y = pow(v, 1.0 / b)
        if x + y <= 1.0 then
          if x + y > 0 then return x / (x + y)
          else
            var logX = log(u) / a
            var logY = log(v) / b
            val logM = if logX > logY then logX else logY
            logX -= logM
            logY -= logM
            return exp(logX - log(exp(logX) + exp(logY)))
      0.0
    else
      val ga = standardGamma(s, a)
      val gb = standardGamma(s, b)
      ga / (ga + gb)

  def f(s: AugState, dfnum: Double, dfden: Double): Double =
    val a = chisquare(s, dfnum) * dfden
    val b = chisquare(s, dfden) * dfnum
    a / b
  def exponential(s: AugState, scale: Double): Double = scale * standardExponential(s)

  // ---- discrete ----
  private def hypergeometricHyp(bg: BitGenerator, good: Long, bad: Long, sample: Long): Long =
    val d1 = bad + good - sample
    val d2 = min(bad, good).toDouble
    var y = d2
    var k = sample
    boundary:
      while y > 0.0 do
        val u = bg.nextDouble()
        y -= floor(u + y / (d1 + k)).toLong
        k -= 1
        if k == 0 then break()
    var z = (d2 - y).toLong
    if good > bad then z = sample - z
    z

  private val D1 = 1.7155277699214135
  private val D2 = 0.8989161620588988

  private def hypergeometricHrua(bg: BitGenerator, good: Long, bad: Long, sample: Long): Long =
    val mingoodbad = min(good, bad)
    val popsize = good + bad
    val maxgoodbad = max(good, bad)
    val m = min(sample, popsize - sample)
    val d4 = mingoodbad.toDouble / popsize
    val d5 = 1.0 - d4
    val d6 = m * d4 + 0.5
    val d7 = sqrt((popsize - m).toDouble * sample * d4 * d5 / (popsize - 1) + 0.5)
    val d8 = D1 * d7 + D2
    val d9 = floor((m + 1).toDouble * (mingoodbad + 1) / (popsize + 2)).toLong
    val d10 = Dist.loggam((d9 + 1).toDouble) + Dist.loggam((mingoodbad - d9 + 1).toDouble) +
      Dist.loggam((m - d9 + 1).toDouble) + Dist.loggam((maxgoodbad - m + d9 + 1).toDouble)
    val d11 = min(min(m, mingoodbad) + 1.0, floor(d6 + 16 * d7))
    var z = 0L
    boundary:
      while true do
        val x = bg.nextDouble()
        val y = bg.nextDouble()
        val w = d6 + d8 * (y - 0.5) / x
        if !(w < 0.0 || w >= d11) then
          z = floor(w).toLong
          val t = d10 - (Dist.loggam((z + 1).toDouble) + Dist.loggam((mingoodbad - z + 1).toDouble) +
            Dist.loggam((m - z + 1).toDouble) + Dist.loggam((maxgoodbad - m + z + 1).toDouble))
          if (x * (4.0 - x) - 3.0) <= t then break()
          if !(x * (x - t) >= 1) then
            if 2.0 * log(x) <= t then break()
    if good > bad then z = m - z
    if m < sample then z = good - z
    z

  def hypergeometric(bg: BitGenerator, good: Long, bad: Long, sample: Long): Long =
    if sample > 10 then hypergeometricHrua(bg, good, bad, sample)
    else if sample > 0 then hypergeometricHyp(bg, good, bad, sample)
    else 0L

  def zipf(bg: BitGenerator, a: Double): Long =
    val am1 = a - 1.0
    val b = pow(2.0, am1)
    while true do
      val u = 1.0 - bg.nextDouble()
      val v = bg.nextDouble()
      val x = floor(pow(u, -1.0 / am1))
      if !(x > Long.MaxValue.toDouble || x < 1.0) then
        val t = pow(1.0 + 1.0 / x, am1)
        if v * x * (t - 1.0) / (b - 1.0) <= t / b then return x.toLong
    1L

  def geometric(bg: BitGenerator, p: Double): Long =
    if p >= 0.333333333333333333333333 then Dist.geometricSearch(bg, p)
    else ceil(log1p(-bg.nextDouble()) / log(1 - p)).toLong

  def logseries(bg: BitGenerator, p: Double): Long =
    val r = log(1.0 - p)
    while true do
      val v = bg.nextDouble()
      if v >= p then return 1L
      val u = bg.nextDouble()
      val q = 1.0 - exp(r * u)
      if v <= q * q then
        val result = floor(1 + log(v) / log(q)).toLong
        if !(result < 1 || v == 0.0) then return result
      else
        return if v >= q then 1L else 2L
    1L
