package com.github.kmizu.numscala.random

import scala.util.boundary
import scala.util.boundary.break

/** Cached binomial set-up (NumPy's `binomial_t`). */
private[numscala] final class BinomialState:
  var has: Boolean = false
  var psave: Double = 0.0
  var nsave: Long = 0L
  var r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4: Double = 0.0
  var m: Long = 0L

/** Port of NumPy's `numpy/random/src/distributions/distributions.c` (the `Generator` samplers). */
private[numscala] object Dist:
  import java.lang.Math.*

  inline def nextDouble(bg: BitGenerator): Double = bg.nextDouble()
  inline def nextFloat(bg: BitGenerator): Float = (bg.nextUInt32() >>> 8) * (1.0f / 16777216.0f)

  val ZigNorR = 3.6541528853610087963519472518
  val ZigNorInvR = 0.27366123732975827203338247596
  val ZigExpR = 7.6971174701310497140446280481
  val ZigNorRF = 3.6541528853610087963519472518f
  val ZigNorInvRF = 0.27366123732975827203338247596f
  val ZigExpRF = 7.6971174701310497140446280481f
  val Int64Max: Long = Long.MaxValue
  /** `POISSON_LAM_MAX`. */
  val PoissonLamMax: Double = Long.MaxValue.toDouble - sqrt(Long.MaxValue.toDouble) * 10

  // ---------------- exponential ----------------
  def standardExponential(bg: BitGenerator): Double =
    var ri = bg.nextUInt64() >>> 3
    val idx = (ri & 0xff).toInt
    ri >>>= 8
    val x = ri * ZigguratExpDouble.we(idx)
    if ri < ZigguratExpDouble.ke(idx) then x
    else expUnlikely(bg, idx, x)

  private def expUnlikely(bg: BitGenerator, idx: Int, x: Double): Double =
    if idx == 0 then ZigExpR - log1p(-nextDouble(bg))
    else if (ZigguratExpDouble.fe(idx - 1) - ZigguratExpDouble.fe(idx)) * nextDouble(bg) + ZigguratExpDouble.fe(idx) < exp(-x) then x
    else standardExponential(bg)

  def standardExponentialF(bg: BitGenerator): Float =
    var ri = bg.nextUInt32() >>> 1
    val idx = ri & 0xff
    ri >>>= 8
    val x = ri.toFloat * ZigguratExpFloat.we(idx)
    if ri < ZigguratExpFloat.ke(idx) then x
    else if idx == 0 then ZigExpRF - log1p(-nextFloat(bg).toDouble).toFloat
    else if (ZigguratExpFloat.fe(idx - 1) - ZigguratExpFloat.fe(idx)) * nextFloat(bg) + ZigguratExpFloat.fe(idx) < exp(-x.toDouble).toFloat then x
    else standardExponentialF(bg)

  // ---------------- normal ----------------
  def standardNormal(bg: BitGenerator): Double =
    while true do
      var r = bg.nextUInt64()
      val idx = (r & 0xff).toInt
      r >>>= 8
      val sign = r & 0x1
      val rabs = (r >>> 1) & 0x000fffffffffffffL
      var x = rabs * ZigguratDouble.wi(idx)
      if sign != 0 then x = -x
      if rabs < ZigguratDouble.ki(idx) then return x
      if idx == 0 then
        while true do
          val xx = -ZigNorInvR * log1p(-nextDouble(bg))
          val yy = -log1p(-nextDouble(bg))
          if yy + yy > xx * xx then
            return if ((rabs >>> 8) & 0x1) != 0 then -(ZigNorR + xx) else ZigNorR + xx
      else if ((ZigguratDouble.fi(idx - 1) - ZigguratDouble.fi(idx)) * nextDouble(bg) + ZigguratDouble.fi(idx)) < exp(-0.5 * x * x) then
        return x
    0.0

  def standardNormalF(bg: BitGenerator): Float =
    while true do
      val r = bg.nextUInt32()
      val idx = r & 0xff
      val sign = (r >>> 8) & 0x1
      val rabs = (r >>> 9) & 0x0007fffff
      var x = rabs.toFloat * ZigguratFloat.wi(idx)
      if sign != 0 then x = -x
      if rabs < ZigguratFloat.ki(idx) then return x
      if idx == 0 then
        while true do
          val xx = -ZigNorInvRF * log1p(-nextFloat(bg).toDouble).toFloat
          val yy = -log1p(-nextFloat(bg).toDouble).toFloat
          if yy + yy > xx * xx then
            return if ((rabs >>> 8) & 0x1) != 0 then -(ZigNorRF + xx) else ZigNorRF + xx
      else if (((ZigguratFloat.fi(idx - 1) - ZigguratFloat.fi(idx)) * nextFloat(bg) + ZigguratFloat.fi(idx)).toDouble) < exp(-0.5 * x * x) then
        return x
    0.0f

  // ---------------- gamma family ----------------
  def standardGamma(bg: BitGenerator, shape: Double): Double =
    if shape == 1.0 then standardExponential(bg)
    else if shape == 0.0 then 0.0
    else if shape < 1.0 then
      while true do
        val u = nextDouble(bg)
        val v = standardExponential(bg)
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
        var x = 0.0
        var v = 0.0
        while
          x = standardNormal(bg)
          v = 1.0 + c * x
          v <= 0.0
        do ()
        v = v * v * v
        val u = nextDouble(bg)
        if u < 1.0 - 0.0331 * (x * x) * (x * x) then return b * v
        if log(u) < 0.5 * x * x + b * (1.0 - v + log(v)) then return b * v
      0.0

  def standardGammaF(bg: BitGenerator, shape: Float): Float =
    if shape == 1.0f then standardExponentialF(bg)
    else if shape == 0.0f then 0.0f
    else if shape < 1.0f then
      while true do
        val u = nextFloat(bg)
        val v = standardExponentialF(bg)
        if u <= 1.0f - shape then
          val x = pow(u.toDouble, (1.0f / shape).toDouble).toFloat
          if x <= v then return x
        else
          val y = -log(((1.0f - u) / shape).toDouble).toFloat
          val x = pow((1.0f - shape + shape * y).toDouble, (1.0f / shape).toDouble).toFloat
          if x <= v + y then return x
      0.0f
    else
      val b = shape - 1.0f / 3.0f
      val c = 1.0f / sqrt((9.0f * b).toDouble).toFloat
      while true do
        var x = 0.0f
        var v = 0.0f
        while
          x = standardNormalF(bg)
          v = 1.0f + c * x
          v <= 0.0f
        do ()
        v = v * v * v
        val u = nextFloat(bg)
        if u < 1.0f - 0.0331f * (x * x) * (x * x) then return b * v
        if log(u.toDouble).toFloat < 0.5f * x * x + b * (1.0f - v + log(v.toDouble).toFloat) then return b * v
      0.0f

  def gamma(bg: BitGenerator, shape: Double, scale: Double): Double = scale * standardGamma(bg, shape)

  def beta(bg: BitGenerator, a: Double, b: Double): Double =
    if a <= 1.0 && b <= 1.0 then
      if a < 3e-103 && b < 3e-103 then
        val u = nextDouble(bg)
        return if (a + b) * u < a then 1.0 else 0.0
      while true do
        val u = nextDouble(bg)
        val v = nextDouble(bg)
        val x = pow(u, 1.0 / a)
        val y = pow(v, 1.0 / b)
        val xpy = x + y
        if xpy <= 1.0 && u + v > 0.0 then
          if x > 0 && y > 0 then return x / xpy
          else
            val logX = log(u) / a
            val logY = log(v) / b
            val delta = logX - logY
            return if delta > 0 then exp(-log1p(exp(-delta))) else exp(delta - log1p(exp(delta)))
      0.0
    else
      val ga = standardGamma(bg, a)
      val gb = standardGamma(bg, b)
      ga / (ga + gb)

  def chisquare(bg: BitGenerator, df: Double): Double = 2.0 * standardGamma(bg, df / 2.0)
  def f(bg: BitGenerator, dfnum: Double, dfden: Double): Double =
    val s1 = chisquare(bg, dfnum) * dfden
    val s2 = chisquare(bg, dfden) * dfnum
    s1 / s2
  def standardCauchy(bg: BitGenerator): Double =
    val a = standardNormal(bg)
    val b = standardNormal(bg)
    a / b
  def pareto(bg: BitGenerator, a: Double): Double = expm1(standardExponential(bg) / a)
  def weibull(bg: BitGenerator, a: Double): Double =
    if a == 0.0 then 0.0 else pow(standardExponential(bg), 1.0 / a)
  def power(bg: BitGenerator, a: Double): Double = pow(-expm1(-standardExponential(bg)), 1.0 / a)

  def laplace(bg: BitGenerator, loc: Double, scale: Double): Double =
    var u = nextDouble(bg)
    while u == 0.0 do u = nextDouble(bg)
    if u >= 0.5 then loc - scale * log(2.0 - u - u) else loc + scale * log(u + u)

  def gumbel(bg: BitGenerator, loc: Double, scale: Double): Double =
    var u = 1.0 - nextDouble(bg)
    while !(u < 1.0) do u = 1.0 - nextDouble(bg)
    loc - scale * log(-log(u))

  def logistic(bg: BitGenerator, loc: Double, scale: Double): Double =
    var u = nextDouble(bg)
    while !(u > 0.0) do u = nextDouble(bg)
    loc + scale * log(u / (1.0 - u))

  def normal(bg: BitGenerator, loc: Double, scale: Double): Double = loc + scale * standardNormal(bg)
  def lognormal(bg: BitGenerator, mean: Double, sigma: Double): Double = exp(normal(bg, mean, sigma))
  def rayleigh(bg: BitGenerator, mode: Double): Double = mode * sqrt(2.0 * standardExponential(bg))
  def exponential(bg: BitGenerator, scale: Double): Double = scale * standardExponential(bg)
  def uniform(bg: BitGenerator, lower: Double, range: Double): Double = lower + range * nextDouble(bg)

  def standardT(bg: BitGenerator, df: Double): Double =
    val num = standardNormal(bg)
    val denom = standardGamma(bg, df / 2)
    sqrt(df / 2) * num / sqrt(denom)

  // ---------------- log-gamma / log-factorial ----------------
  private val LoggamA = Array(8.333333333333333e-02, -2.777777777777778e-03, 7.936507936507937e-04,
    -5.952380952380952e-04, 8.417508417508418e-04, -1.917526917526918e-03, 6.410256410256410e-03,
    -2.955065359477124e-02, 1.796443723688307e-01, -1.39243221690590e+00)

  def loggam(x: Double): Double =
    if x == 1.0 || x == 2.0 then return 0.0
    val n: Long = if x < 7.0 then (7 - x).toLong else 0L
    var x0 = x + n
    val x2 = (1.0 / x0) * (1.0 / x0)
    val lg2pi = 1.8378770664093453e+00
    var gl0 = LoggamA(9)
    var k = 8
    while k >= 0 do
      gl0 *= x2
      gl0 += LoggamA(k)
      k -= 1
    var gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * log(x0) - x0
    if x < 7.0 then
      var kk = 1L
      while kk <= n do
        gl -= log(x0 - 1.0)
        x0 -= 1.0
        kk += 1
    gl

  def logfactorial(k: Long): Double =
    val t = LogFactTable.logfact
    if k < t.length then t(k.toInt)
    else
      val halfln2pi = 0.9189385332046728
      (k + 0.5) * log(k.toDouble) - k + (halfln2pi + (1.0 / k) * (1 / 12.0 - 1 / (360.0 * k * k)))

  // ---------------- poisson ----------------
  private def poissonMult(bg: BitGenerator, lam: Double): Long =
    val enlam = exp(-lam)
    var x = 0L
    var prod = 1.0
    while true do
      val u = nextDouble(bg)
      prod *= u
      if prod > enlam then x += 1 else return x
    x

  private val LS2PI = 0.91893853320467267
  private def poissonPtrs(bg: BitGenerator, lam: Double): Long =
    val slam = sqrt(lam)
    val loglam = log(lam)
    val b = 0.931 + 2.53 * slam
    val a = -0.059 + 0.02483 * b
    val invalpha = 1.1239 + 1.1328 / (b - 3.4)
    val vr = 0.9277 - 3.6224 / (b - 2)
    while true do
      val u = nextDouble(bg) - 0.5
      val v = nextDouble(bg)
      val us = 0.5 - abs(u)
      val k = floor((2 * a / us + b) * u + lam + 0.43).toLong
      if us >= 0.07 && v <= vr then return k
      if !(k < 0 || (us < 0.013 && v > us)) then
        if (log(v) + log(invalpha) - log(a / (us * us) + b)) <= (-lam + k.toDouble * loglam - loggam(k.toDouble + 1)) then
          return k
    0L

  def poisson(bg: BitGenerator, lam: Double): Long =
    if lam >= 10 then poissonPtrs(bg, lam)
    else if lam == 0 then 0L
    else poissonMult(bg, lam)

  def negativeBinomial(bg: BitGenerator, n: Double, p: Double): Long =
    val y = gamma(bg, n, (1 - p) / p)
    poisson(bg, y)

  // ---------------- binomial ----------------
  def binomialBtpe(bg: BitGenerator, n: Long, p: Double, bs: BinomialState): Long =
    if !bs.has || bs.nsave != n || bs.psave != p then
      bs.nsave = n; bs.psave = p; bs.has = true
      val r = min(p, 1.0 - p)
      bs.r = r
      val q = 1.0 - r
      bs.q = q
      bs.fm = n * r + r
      bs.m = floor(bs.fm).toLong
      bs.p1 = floor(2.195 * sqrt(n * r * q) - 4.6 * q) + 0.5
      bs.xm = bs.m + 0.5
      bs.xl = bs.xm - bs.p1
      bs.xr = bs.xm + bs.p1
      bs.c = 0.134 + 20.5 / (15.3 + bs.m)
      var a = (bs.fm - bs.xl) / (bs.fm - bs.xl * r)
      bs.laml = a * (1.0 + a / 2.0)
      a = (bs.xr - bs.fm) / (bs.xr * q)
      bs.lamr = a * (1.0 + a / 2.0)
      bs.p2 = bs.p1 * (1.0 + 2.0 * bs.c)
      bs.p3 = bs.p2 + bs.c / bs.laml
      bs.p4 = bs.p3 + bs.c / bs.lamr
    val r = bs.r; val q = bs.q; val fm = bs.fm; val m = bs.m; val p1 = bs.p1; val xm = bs.xm
    val xl = bs.xl; val xr = bs.xr; val c = bs.c; val laml = bs.laml; val lamr = bs.lamr
    val p2 = bs.p2; val p3 = bs.p3; val p4 = bs.p4
    var label = 10
    var y = 0L
    var u, v, x, nrq = 0.0
    while label != 60 do
      label match
        case 10 =>
          nrq = n * r * q
          u = nextDouble(bg) * p4
          v = nextDouble(bg)
          if u > p1 then label = 20
          else
            y = floor(xm - p1 * v + u).toLong
            label = 60
        case 20 =>
          if u > p2 then label = 30
          else
            x = xl + (u - p1) / c
            v = v * c + 1.0 - abs(m - x + 0.5) / p1
            if v > 1.0 then label = 10
            else
              y = floor(x).toLong
              label = 50
        case 30 =>
          if u > p3 then label = 40
          else
            y = floor(xl + log(v) / laml).toLong
            if y < 0 || v == 0.0 then label = 10
            else
              v = v * (u - p2) * laml
              label = 50
        case 40 =>
          y = floor(xr - log(v) / lamr).toLong
          if y > n || v == 0.0 then label = 10
          else
            v = v * (u - p3) * lamr
            label = 50
        case 50 =>
          val k = abs(y - m)
          if k > 20 && k < nrq / 2.0 - 1 then label = 52
          else
            val s = r / q
            val a = s * (n + 1)
            var ff = 1.0
            if m < y then
              var i = m + 1
              while i <= y do { ff *= (a / i - s); i += 1 }
            else if m > y then
              var i = y + 1
              while i <= m do { ff /= (a / i - s); i += 1 }
            label = if v > ff then 10 else 60
        case _ => // 52
          val k = abs(y - m)
          val rho = (k / nrq) * ((k * (k / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5)
          val t = (-k * k) / (2 * nrq)
          val aa = log(v)
          if aa < t - rho then label = 60
          else if aa > t + rho then label = 10
          else
            val x1 = y.toDouble + 1
            val f1 = m.toDouble + 1
            val z = n.toDouble + 1 - m.toDouble
            val w = n.toDouble - y.toDouble + 1
            val x2 = x1 * x1
            val f2 = f1 * f1
            val z2 = z * z
            val w2 = w * w
            val bound = xm * log(f1 / x1) + (n - m + 0.5) * log(z / w) + (y - m) * log(w * r / (x1 * q)) +
              (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / f2) / f2) / f2) / f2) / f1 / 166320.0 +
              (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / z2) / z2) / z2) / z2) / z / 166320.0 +
              (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2) / x1 / 166320.0 +
              (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / w2) / w2) / w2) / w2) / w / 166320.0
            label = if aa > bound then 10 else 60
    if p > 0.5 then n - y else y

  def binomialInversion(bg: BitGenerator, n: Long, p: Double, bs: BinomialState, legacy: Boolean): Long =
    if !bs.has || bs.nsave != n || bs.psave != p then
      bs.nsave = n; bs.psave = p; bs.has = true
      bs.q = 1.0 - p
      bs.r = if legacy then exp(n * log(bs.q)) else exp(n * log1p(-p))
      bs.c = n * p
      bs.m = min(n.toDouble, bs.c + 10.0 * sqrt(bs.c * bs.q + 1)).toLong
    val q = bs.q; val qn = bs.r; val bound = bs.m
    var x = 0L
    var px = qn
    var u = nextDouble(bg)
    while u > px do
      x += 1
      if x > bound then
        x = 0
        px = qn
        u = nextDouble(bg)
      else
        u -= px
        px = ((n - x + 1) * p * px) / (x * q)
    x

  def binomial(bg: BitGenerator, p: Double, n: Long, bs: BinomialState): Long =
    if n == 0L || p == 0.0 then 0L
    else if p <= 0.5 then
      if p * n <= 30.0 then binomialInversion(bg, n, p, bs, false) else binomialBtpe(bg, n, p, bs)
    else
      val q = 1.0 - p
      if q * n <= 30.0 then n - binomialInversion(bg, n, q, bs, false) else n - binomialBtpe(bg, n, q, bs)

  def legacyBinomial(bg: BitGenerator, p: Double, n: Long, bs: BinomialState): Long =
    if p <= 0.5 then
      if p * n <= 30.0 then binomialInversion(bg, n, p, bs, true) else binomialBtpe(bg, n, p, bs)
    else
      val q = 1.0 - p
      if q * n <= 30.0 then n - binomialInversion(bg, n, q, bs, true) else n - binomialBtpe(bg, n, q, bs)

  // ---------------- other continuous ----------------
  def noncentralChisquare(bg: BitGenerator, df: Double, nonc: Double): Double =
    if nonc.isNaN then Double.NaN
    else if nonc == 0 then chisquare(bg, df)
    else if 1 < df then
      val chi2 = chisquare(bg, df - 1)
      val n = standardNormal(bg) + sqrt(nonc)
      chi2 + n * n
    else
      val i = poisson(bg, nonc / 2.0)
      chisquare(bg, df + 2 * i)

  def noncentralF(bg: BitGenerator, dfnum: Double, dfden: Double, nonc: Double): Double =
    val t = noncentralChisquare(bg, dfnum, nonc) * dfden
    t / (chisquare(bg, dfden) * dfnum)

  def wald(bg: BitGenerator, mean: Double, scale: Double): Double =
    var y = standardNormal(bg)
    y = mean * y * y
    val d = 1 + sqrt(1 + 4 * scale / y)
    val x = mean * (1 - 2 / d)
    val u = nextDouble(bg)
    if u <= mean / (mean + x) then x else mean * mean / x

  def vonmises(bg: BitGenerator, mu: Double, kappa: Double, legacy: Boolean): Double =
    if kappa.isNaN then return Double.NaN
    if kappa < 1e-8 then return PI * (2 * nextDouble(bg) - 1)
    var s = 0.0
    if kappa < 1e-5 then s = 1.0 / kappa + kappa
    else if legacy || kappa <= 1e6 then
      val r = 1 + sqrt(1 + 4 * kappa * kappa)
      val rho = (r - sqrt(2 * r)) / (2 * kappa)
      s = (1 + rho * rho) / (2 * rho)
    else
      var result = mu + sqrt(1.0 / kappa) * standardNormal(bg)
      if result < -PI then result += 2 * PI
      if result > PI then result -= 2 * PI
      return result
    var w = 0.0
    boundary:
      while true do
        val u = nextDouble(bg)
        val z = cos(PI * u)
        w = (1 + s * z) / (s + z)
        val y = kappa * (s - w)
        val v = nextDouble(bg)
        if (y * (2 - y) - v >= 0) || (log(y / v) + 1 - y >= 0) then break()
    val u = nextDouble(bg)
    var result = acos(w)
    if u < 0.5 then result = -result
    result += mu
    val neg = result < 0
    var mod = abs(result)
    mod = ((mod + PI) % (2 * PI)) - PI
    if neg then mod *= -1
    mod

  def triangular(bg: BitGenerator, left: Double, mode: Double, right: Double): Double =
    val base = right - left
    val leftbase = mode - left
    val ratio = leftbase / base
    val leftprod = leftbase * base
    val rightprod = (right - mode) * base
    val u = nextDouble(bg)
    if u <= ratio then left + sqrt(u * leftprod) else right - sqrt((1.0 - u) * rightprod)

  // ---------------- discrete ----------------
  def logseries(bg: BitGenerator, p: Double): Long =
    val r = log1p(-p)
    while true do
      val v = nextDouble(bg)
      if v >= p then return 1L
      val u = nextDouble(bg)
      val q = -expm1(r * u)
      if v <= q * q then
        val result = floor(1 + log(v) / log(q)).toLong
        if !(result < 1 || v == 0.0) then return result
      else
        return if v >= q then 1L else 2L
    1L

  def geometricSearch(bg: BitGenerator, p: Double): Long =
    var x = 1L
    var sum = p
    var prod = p
    val q = 1.0 - p
    val u = nextDouble(bg)
    while u > sum do
      prod *= q
      sum += prod
      x += 1
    x

  def geometricInversion(bg: BitGenerator, p: Double): Long =
    val z = ceil(-standardExponential(bg) / log1p(-p))
    if z >= 9.223372036854776e+18 then Int64Max else z.toLong

  def geometric(bg: BitGenerator, p: Double): Long =
    if p >= 0.333333333333333333333333 then geometricSearch(bg, p) else geometricInversion(bg, p)

  def zipf(bg: BitGenerator, a: Double): Long =
    if a >= 1025 then return 1L
    val am1 = a - 1.0
    val b = pow(2.0, am1)
    val umin = pow(Int64Max.toDouble, -am1)
    while true do
      val u01 = nextDouble(bg)
      val u = u01 * umin + (1 - u01)
      val v = nextDouble(bg)
      val x = floor(pow(u, -1.0 / am1))
      if !(x > Int64Max.toDouble || x < 1.0) then
        val t = pow(1.0 + 1.0 / x, am1)
        if v * x * (t - 1.0) / (b - 1.0) <= t / b then return x.toLong
    1L

  /** `random_interval(max)`: uniform in `[0, max]` by masked rejection. */
  def interval(bg: BitGenerator, max: Long): Long =
    if max == 0 then return 0L
    var mask = max
    mask |= mask >>> 1; mask |= mask >>> 2; mask |= mask >>> 4
    mask |= mask >>> 8; mask |= mask >>> 16; mask |= mask >>> 32
    var value = 0L
    if !U64.ltU(0xffffffffL, max) then
      while
        value = (bg.nextUInt32() & 0xffffffffL) & mask
        U64.ltU(max, value)
      do ()
    else
      while
        value = bg.nextUInt64() & mask
        U64.ltU(max, value)
      do ()
    value

  // ---------------- hypergeometric ----------------
  private def hypergeometricSample(bg: BitGenerator, good: Long, bad: Long, sample: Long): Long =
    val total = good + bad
    val computedSample = if sample > total / 2 then total - sample else sample
    var cs = computedSample
    var remainingTotal = total
    var remainingGood = good
    while cs > 0 && remainingGood > 0 && remainingTotal > remainingGood do
      remainingTotal -= 1
      if interval(bg, remainingTotal) < remainingGood then remainingGood -= 1
      cs -= 1
    if remainingTotal == remainingGood then remainingGood -= cs
    if sample > total / 2 then remainingGood else good - remainingGood

  private val D1 = 1.7155277699214135
  private val D2 = 0.8989161620588988

  private def hypergeometricHrua(bg: BitGenerator, good: Long, bad: Long, sample: Long): Long =
    val popsize = good + bad
    val computedSample = min(sample, popsize - sample)
    val mingoodbad = min(good, bad)
    val maxgoodbad = max(good, bad)
    val p = mingoodbad.toDouble / popsize
    val q = maxgoodbad.toDouble / popsize
    val mu = computedSample * p
    val a = mu + 0.5
    val variance = (popsize - computedSample).toDouble * computedSample * p * q / (popsize - 1)
    val c = sqrt(variance + 0.5)
    val h = D1 * c + D2
    val m = floor((computedSample + 1).toDouble * (mingoodbad + 1) / (popsize + 2)).toLong
    val g = logfactorial(m) + logfactorial(mingoodbad - m) + logfactorial(computedSample - m) +
      logfactorial(maxgoodbad - computedSample + m)
    val b = min((min(computedSample, mingoodbad) + 1).toDouble, floor(a + 16 * c))
    var k = 0L
    boundary:
      while true do
        val u = nextDouble(bg)
        val v = nextDouble(bg)
        val x = a + h * (v - 0.5) / u
        if !(x < 0.0 || x >= b) then
          k = floor(x).toLong
          val gp = logfactorial(k) + logfactorial(mingoodbad - k) + logfactorial(computedSample - k) +
            logfactorial(maxgoodbad - computedSample + k)
          val t = g - gp
          if (u * (4.0 - u) - 3.0) <= t then break()
          if !(u * (u - t) >= 1) then
            if 2.0 * log(u) <= t then break()
    if good > bad then k = computedSample - k
    if computedSample < sample then k = good - k
    k

  def hypergeometric(bg: BitGenerator, good: Long, bad: Long, sample: Long): Long =
    if sample >= 10 && sample <= good + bad - 10 then hypergeometricHrua(bg, good, bad, sample)
    else hypergeometricSample(bg, good, bad, sample)

  // ---------------- multivariate ----------------
  def multinomial(bg: BitGenerator, n: Long, out: Array[Long], off: Int, pix: Array[Double], poff: Int, d: Int,
      bs: BinomialState): Unit =
    var remainingP = 1.0
    var dn = n
    var j = 0
    boundary:
      while j < d - 1 do
        out(off + j) = binomial(bg, pix(poff + j) / remainingP, dn, bs)
        dn -= out(off + j)
        if dn <= 0 then break()
        remainingP -= pix(poff + j)
        j += 1
    if dn > 0 then out(off + d - 1) = dn

  def mvhgCount(bg: BitGenerator, total: Long, colors: Array[Long], nsample0: Long, numVariates: Int,
      variates: Array[Long]): Unit =
    val numColors = colors.length
    if total == 0 || nsample0 == 0 || numVariates == 0 then return
    val choices = new Array[Int](total.toInt)
    var k = 0
    for i <- 0 until numColors; _ <- 0L until colors(i) do { choices(k) = i; k += 1 }
    val moreThanHalf = nsample0 > total / 2
    val nsample = if moreThanHalf then total - nsample0 else nsample0
    var i = 0
    while i < numVariates * numColors do
      var j = 0
      while j < nsample do
        val kk = j + interval(bg, total - j - 1).toInt
        val tmp = choices(kk); choices(kk) = choices(j); choices(j) = tmp
        j += 1
      j = 0
      while j < nsample do { variates(i + choices(j)) += 1; j += 1 }
      if moreThanHalf then
        for c <- 0 until numColors do variates(i + c) = colors(c) - variates(i + c)
      i += numColors

  def mvhgMarginals(bg: BitGenerator, total: Long, colors: Array[Long], nsample0: Long, numVariates: Int,
      variates: Array[Long]): Unit =
    val numColors = colors.length
    if total == 0 || nsample0 == 0 || numVariates == 0 then return
    val moreThanHalf = nsample0 > total / 2
    val nsample = if moreThanHalf then total - nsample0 else nsample0
    var i = 0
    while i < numVariates * numColors do
      var numToSample = nsample
      var remaining = total
      var j = 0
      while numToSample > 0 && j + 1 < numColors do
        remaining -= colors(j)
        val r = hypergeometric(bg, colors(j), remaining, numToSample)
        variates(i + j) = r
        numToSample -= r
        j += 1
      if numToSample > 0 then variates(i + numColors - 1) = numToSample
      if moreThanHalf then
        for c <- 0 until numColors do variates(i + c) = colors(c) - variates(i + c)
      i += numColors
