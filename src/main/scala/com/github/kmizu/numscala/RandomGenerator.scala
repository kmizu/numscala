package com.github.kmizu.numscala.random

import com.github.kmizu.numscala.*
import Cons.*

/** `numpy.random.Generator`: the modern sampling API on top of a [[BitGenerator]]
  * (bit-for-bit compatible with NumPy for the same bit generator state).
  *
  * {{{
  * val rng = np.random.default_rng(42)
  * rng.random(3)            // 0.77395605, 0.43887844, 0.85859792
  * rng.integers(0, 10, 5)
  * rng.normal(0.0, 1.0, 2, 3)
  * }}}
  */
class Generator(val bit_generator: BitGenerator) extends RandomDiscrete:
  private val binom = new BinomialState

  override def toString: String = s"Generator(${bit_generator.name})"

  /** `spawn(n_children)`: independent child generators. */
  def spawn(n_children: Int): Seq[Generator] = bit_generator.spawn(n_children).map(new Generator(_))

  // ---- primitives ----
  protected def pStdNormal(): Double = Dist.standardNormal(bg)
  protected def pStdExp(): Double = Dist.standardExponential(bg)
  protected def pStdGamma(shape: Double): Double = Dist.standardGamma(bg, shape)
  protected def pNormal(loc: Double, scale: Double): Double = Dist.normal(bg, loc, scale)
  protected def pExponential(scale: Double): Double = Dist.exponential(bg, scale)
  protected def pGamma(shape: Double, scale: Double): Double = Dist.gamma(bg, shape, scale)
  protected def pBeta(a: Double, b: Double): Double = Dist.beta(bg, a, b)
  protected def pChisquare(df: Double): Double = Dist.chisquare(bg, df)
  protected def pNoncentralChisquare(df: Double, nonc: Double): Double = Dist.noncentralChisquare(bg, df, nonc)
  protected def pF(dfnum: Double, dfden: Double): Double = Dist.f(bg, dfnum, dfden)
  protected def pNoncentralF(dfnum: Double, dfden: Double, nonc: Double): Double = Dist.noncentralF(bg, dfnum, dfden, nonc)
  protected def pStdCauchy(): Double = Dist.standardCauchy(bg)
  protected def pStdT(df: Double): Double = Dist.standardT(bg, df)
  protected def pVonmises(mu: Double, kappa: Double): Double = Dist.vonmises(bg, mu, kappa, false)
  protected def pPareto(a: Double): Double = Dist.pareto(bg, a)
  protected def pWeibull(a: Double): Double = Dist.weibull(bg, a)
  protected def pPower(a: Double): Double = Dist.power(bg, a)
  protected def pLognormal(mean: Double, sigma: Double): Double = Dist.lognormal(bg, mean, sigma)
  protected def pRayleigh(scale: Double): Double = Dist.rayleigh(bg, scale)
  protected def pWald(mean: Double, scale: Double): Double = Dist.wald(bg, mean, scale)
  protected def uniformRangeCons: Cons = NonNegative

  protected def pBinomial(p: Double, n: Long): Long = Dist.binomial(bg, p, n, binom)
  protected def binomialNCons: Cons = NonNegative
  protected def pNegativeBinomial(n: Double, p: Double): Long = Dist.negativeBinomial(bg, n, p)
  protected def negBinCheck(n: Array[Double], p: Array[Double], scalar: Boolean): Unit =
    RCommon.checkArr(n, "n", PositiveNotNan)
    if scalar then RCommon.check(p(0), "p", BoundedGt01) else RCommon.checkArr(p, "p", BoundedGt01)
    var i = 0
    while i < n.length do
      val maxLam = (1 - p(i)) / p(i) * (n(i) + 10 * math.sqrt(n(i)))
      if maxLam > Dist.PoissonLamMax then
        throw new IllegalArgumentException("n too large or p too small, see Generator.negative_binomial Notes")
      i += 1
  protected def pPoisson(lam: Double): Long = Dist.poisson(bg, lam)
  protected def poissonCons: Cons = Poisson
  protected def pZipf(a: Double): Long = Dist.zipf(bg, a)
  protected def pGeometric(p: Double): Long = Dist.geometric(bg, p)
  protected def pLogseries(p: Double): Long = Dist.logseries(bg, p)
  protected def pHypergeometric(good: Long, bad: Long, sample: Long): Long = Dist.hypergeometric(bg, good, bad, sample)
  protected def hyperCheck(good: Array[Long], bad: Array[Long], sample: Array[Long]): (Cons, Cons, Cons) =
    val mx = 1000000000L
    if good.exists(_ >= mx) || bad.exists(_ >= mx) then
      throw new IllegalArgumentException(s"both ngood and nbad must be less than $mx")
    if good.indices.exists(i => good(i) + bad(i) < sample(i)) then
      throw new IllegalArgumentException("ngood + nbad < nsample")
    (NonNegative, NonNegative, NonNegative)

  // ---------------- random ----------------
  /** `random()`: one float in `[0, 1)`. */
  def random(): Double = bg.nextDouble()
  /** `random(size*)`: floats in `[0, 1)`. */
  def random(size: Int*): NDArray[Double] = rd(size)(bg.nextDouble())
  /** `random(size)` with the shape given as a `Seq`. */
  @annotation.targetName("randomSeq") def random(size: Seq[Int]): NDArray[Double] = rd(size)(bg.nextDouble())
  /** `random(size, dtype)`: `dtype` is `DType.Float64` or `DType.Float32`. */
  def random[T](size: SizeArg, dtype: FloatDType[T]): NDArray[T] =
    floatFill(size, dtype)(bg.nextDouble())(Dist.nextFloat(bg))

  private def floatFill[T](size: SizeArg, dtype: FloatDType[T])(fd: => Double)(ff: => Float): NDArray[T] =
    val shp = RCommon.sizeShape(size)
    val shape = if shp == null then Array.empty[Int] else RCommon.checkShape(shp)
    dtype.name match
      case "float64" => RCommon.fillD(shape)(fd).asInstanceOf[NDArray[T]]
      case "float32" =>
        val n = Shape.size(shape)
        val out = new Array[Float](n)
        var i = 0
        while i < n do { out(i) = ff; i += 1 }
        NDArray.fromArray(out, shape).asInstanceOf[NDArray[T]]
      case other => throw new IllegalArgumentException(s"Unsupported dtype $other")

  /** `standard_normal(size, dtype)` (float64 or float32 ziggurat). */
  def standard_normal[T](size: SizeArg, dtype: FloatDType[T]): NDArray[T] =
    floatFill(size, dtype)(Dist.standardNormal(bg))(Dist.standardNormalF(bg))

  /** `standard_exponential(size, dtype, method)`; `method` is `"zig"` (default) or `"inv"`. */
  def standard_exponential[T](size: SizeArg, dtype: FloatDType[T], method: String): NDArray[T] =
    method match
      case "zig" => floatFill(size, dtype)(Dist.standardExponential(bg))(Dist.standardExponentialF(bg))
      case "inv" =>
        floatFill(size, dtype)(-math.log1p(-bg.nextDouble()))(-math.log1p(-Dist.nextFloat(bg).toDouble).toFloat)
      case _ => throw new IllegalArgumentException("method must be 'zig' or 'inv'")

  /** `standard_gamma(shape, size, dtype)` (float64 or float32). */
  def standard_gamma[T](shape: Param, size: SizeArg, dtype: FloatDType[T]): NDArray[T] =
    if dtype.name == "float64" then standard_gamma(shape, size).asInstanceOf[NDArray[T]]
    else if dtype.name == "float32" then
      val sh = RCommon.toD(shape)
      RCommon.checkArr(sh.toArray, "shape", NonNegative)
      val (shp, fl) = RCommon.broadcastAll(size, Seq(sh))
      val vals = fl(0)
      NDArray.fromArray(Array.tabulate(vals.length)(i => Dist.standardGammaF(bg, vals(i).toFloat)), shp)
        .asInstanceOf[NDArray[T]]
    else throw new IllegalArgumentException(s"Unsupported dtype ${dtype.name} for standard_gamma")

  // ---------------- integers ----------------
  /** `integers(high)`: one int64 in `[0, high)`. */
  def integers(high: Long): Long = integers(0L, high)
  /** `integers(low, high)`: one int64 in `[low, high)`. */
  def integers(low: Long, high: Long): Long =
    RandomInts.int64(bg, low, high, null, masked = false, endpoint = false).item
  /** `integers(low, high, size*)`: int64 samples in `[low, high)`. */
  def integers(low: Long, high: Long, size: Int*): NDArray[Long] =
    RandomInts.int64(bg, low, high, size, masked = false, endpoint = false)
  /** `integers(low, high, size, endpoint)` with broadcasting array bounds (int64 result).
    * With `high = null`, samples come from `[0, low)`.
    */
  def integers(low: Param, high: Param | Null = null, size: SizeArg = null, endpoint: Boolean = false): NDArray[Long] =
    if high == null then RandomInts.int64(bg, 0L, low, size, false, endpoint)
    else RandomInts.int64(bg, low, high, size, false, endpoint)
  /** `integers(low, high, size, dtype, endpoint)` for `dtype` in int64/int32/int16/int8/bool. */
  def integers[T](low: Param, high: Param | Null, size: SizeArg, dtype: DType[T], endpoint: Boolean): NDArray[T] =
    val (lo, hi) = if high == null then (0L: Param, low) else (low, high)
    RandomInts.generic(bg, lo, hi, size, dtype, false, endpoint)

  // ---------------- choice ----------------
  /** `choice(n)`: one integer drawn uniformly from `[0, n)`. */
  def choice(a: Int): Long = choiceImpl(a, null, true, null, 0, true).asInstanceOf[NDArray[Long]].item
  /** `choice(a)` for a 1-d array: one element. */
  def choice[T](a: NDArray[T]): T =
    if a.ndim != 1 then throw new IllegalArgumentException("choice(a) without size requires a 1-d array; pass size")
    choiceImpl(a, null, true, null, 0, true).asInstanceOf[NDArray[T]].item
  /** `choice(a, size, replace, p, axis, shuffle)`: `a` is a population size (`Int`, giving
    * `NDArray[Long]` indices) or an array (giving its elements, taken along `axis`).
    */
  def choice[A](a: A, size: SizeArg = null, replace: Boolean = true, p: Param | Null = null,
      axis: Int = 0, shuffle: Boolean = true)(using c: ChoiceOf[A]): NDArray[c.Out] =
    choiceImpl(a, size, replace, p, axis, shuffle).asInstanceOf[NDArray[c.Out]]

  private def choiceImpl(a: Any, size: SizeArg, replace: Boolean, p0: Param | Null, axis: Int,
      shuffle: Boolean): NDArray[?] =
    val shp0 = RCommon.sizeShape(size)
    val isScalar = shp0 == null
    val shape = if isScalar then Array.empty[Int] else shp0
    val n = Shape.size(shape)
    val popSize: Long = a match
      case i: Int =>
        if i <= 0 && n != 0 then throw new IllegalArgumentException("a must be a positive integer unless no samples are taken")
        i.toLong
      case arr: NDArray[?] =>
        if arr.ndim == 0 then throw new IllegalArgumentException("a must be a sequence or an integer")
        val ps = arr.shapeArr(Shape.normAxis(axis, arr.ndim))
        if ps == 0 && n != 0 then throw new IllegalArgumentException("a cannot be empty unless no samples are taken")
        ps.toLong
    val p: Array[Double] | Null =
      if p0 == null then null
      else
        val pa = RCommon.toD(p0)
        if pa.ndim != 1 then throw new IllegalArgumentException("p must be 1-dimensional")
        if pa.size != popSize then throw new IllegalArgumentException("a and p must have same size")
        val pv = pa.toArray
        val s = RCommon.kahanSum(pv, 0, pv.length)
        if s.isNaN then throw new IllegalArgumentException("Probabilities contain NaN")
        if pv.exists(_ < 0) then throw new IllegalArgumentException("Probabilities are not non-negative")
        if math.abs(s - 1.0) > math.sqrt(math.ulp(1.0)) then
          throw new IllegalArgumentException("Probabilities do not sum to 1. See Notes section of docstring for more information.")
        pv
    val idx: Array[Long] =
      if replace then
        if p != null then
          val cdf = RandomArrays.cumsum(p)
          val last = cdf(cdf.length - 1)
          for i <- cdf.indices do cdf(i) /= last
          Array.tabulate(n)(_ => RandomArrays.searchRight(cdf, bg.nextDouble()).toLong)
        else RandomInts.int64(bg, 0L, popSize, shape.toSeq, false, false).toArray
      else
        if n > popSize then
          throw new IllegalArgumentException("Cannot take a larger sample than population when replace is False")
        if p != null then
          if p.count(_ > 0) < n then throw new IllegalArgumentException("Fewer non-zero entries in p than size")
          RandomChoice.noReplaceP(this, p.clone(), n)
        else RandomChoice.noReplace(bg, popSize, n.toLong, shuffle)
    val idxArr = NDArray.fromArray(idx, shape)
    a match
      case _: Int => idxArr
      case arr: NDArray[?] =>
        val ar = arr.asInstanceOf[NDArray[Any]]
        val ax = Shape.normAxis(axis, ar.ndim)
        val taken = RandomArrays.take(ar, idx.map(_.toInt), ax)
        // a.take(idx, axis): the sampled axis is replaced by the index shape
        val outShape = ar.shapeArr.take(ax) ++ shape ++ ar.shapeArr.drop(ax + 1)
        taken.reshape(outShape.toSeq*)

  // ---------------- permuted ----------------
  /** `permuted(x, axis)`: a copy of `x` with each 1-d slice along `axis` shuffled independently;
    * with `axis = null` the flattened array is shuffled.
    */
  def permuted[T](x: NDArray[T], axis: Int | Null = null): NDArray[T] =
    val out = x.copy()
    axis match
      case null =>
        val flat = out.reshape(out.size)
        shuffle(flat)
        out
      case ax0: Int =>
        val ax = Shape.normAxis(ax0, out.ndim)
        val moved = out.moveaxis(ax, out.ndim - 1)
        val n = out.shapeArr(ax)
        val others = moved.shapeArr.dropRight(1)
        val cnt = Shape.size(others)
        val d = moved.data
        val st = moved.stridesArr
        val idx = new Array[Int](others.length)
        var k = 0
        while k < cnt do
          var off = moved.offset
          var q = 0
          while q < others.length do { off += idx(q) * st(q); q += 1 }
          val s = st(st.length - 1)
          var i = n - 1
          while i >= 0 do
            val j = Dist.interval(bg, i.toLong).toInt
            RandomArrays.swap(d, off + j * s, off + i * s)
            i -= 1
          var dd = others.length - 1
          while dd >= 0 && { idx(dd) += 1; idx(dd) == others(dd) } do
            idx(dd) = 0
            dd -= 1
          k += 1
        out

  // ---------------- multivariate ----------------
  /** `multinomial(n, pvals, size)`: counts of `n` trials over `len(pvals)` categories. */
  def multinomial(n: Param, pvals: Param, size: SizeArg = null): NDArray[Long] =
    RandomMulti.multinomial(bg, binom, n, pvals, size, legacy = false)

  /** `multivariate_hypergeometric(colors, nsample, size, method)` (`"marginals"` or `"count"`). */
  def multivariate_hypergeometric(colors: Param, nsample: Long, size: SizeArg = null,
      method: String = "marginals"): NDArray[Long] =
    RandomMulti.mvhg(bg, colors, nsample, size, method)

  /** `dirichlet(alpha, size)`. */
  def dirichlet(alpha: Param, size: SizeArg = null): NDArray[Double] = RandomMulti.dirichlet(bg, alpha, size)

  /** `multivariate_normal(mean, cov, size, check_valid, tol, method)`; `method` is `"svd"`,
    * `"eigh"` or `"cholesky"` (factorisations are computed internally).
    */
  def multivariate_normal(mean: Param, cov: Param, size: SizeArg = null, check_valid: String = "warn",
      tol: Double = 1e-8, method: String = "svd"): NDArray[Double] =
    RandomMulti.multivariateNormal(mean, cov, size, check_valid, tol, method, shp => standard_normal(shp.toSeq), legacy = false)

/** Element type of `choice` results: `Long` indices for an integer population, the array's
  * element type for an array population.
  */
trait ChoiceOf[A]:
  type Out

object ChoiceOf:
  given intPopulation: ChoiceOf[Int] with
    type Out = Long
  given arrayPopulation[T]: ChoiceOf[NDArray[T]] with
    type Out = T
