package com.github.kmizu.numscala.random

import com.github.kmizu.numscala.*
import Cons.*

/** The legacy state tuple `('MT19937', key, pos, has_gauss, cached_gaussian)`. */
type LegacyMTState = (String, NDArray[Long], Int, Int, Double)

/** `numpy.random.RandomState`: the legacy Mersenne-Twister API, stream-compatible with NumPy's
  * `np.random.seed(...)` / `np.random.rand(...)` etc.
  */
class RandomState private (initial: BitGenerator) extends RandomDiscrete:
  private var bitGen: BitGenerator = initial
  private val aug = new AugState(initial)
  private val binom = new BinomialState

  /** Seeds a new MT19937 with NumPy's legacy seeding (an `Int`/`Long` in `[0, 2**32)`, an
    * array of such values, or `null` for OS entropy).
    */
  def this(seed: Any = null) =
    this(RandomState.legacyMT(seed))

  /** The underlying bit generator (`_bit_generator`). */
  def bit_generator: BitGenerator = bitGen

  /** Replaces the bit generator (used by `np.random.set_bit_generator`). */
  private[numscala] def initializeBitGenerator(b: BitGenerator): Unit =
    bitGen = b
    aug.bg = b
    resetGauss()

  private def resetGauss(): Unit =
    aug.hasGauss = false
    aug.gauss = 0.0

  override def toString: String = s"RandomState(${bitGen.name})"

  /** `seed(seed)`: re-seeds the MT19937 with NumPy's legacy seeding. */
  def seed(seed: Any = null): Unit =
    bitGen match
      case mt: MT19937 => mt._legacy_seeding(seed)
      case _ => throw new UnsupportedOperationException("can only re-seed a MT19937 BitGenerator")
    resetGauss()

  /** `get_state()`: the legacy tuple `('MT19937', key, pos, has_gauss, cached_gaussian)`. */
  def get_state(): LegacyMTState =
    bitGen match
      case mt: MT19937 =>
        ("MT19937", NDArray.fromArray(mt.key.map(_ & 0xffffffffL), Array(624)), mt.pos, if aug.hasGauss then 1 else 0, aug.gauss)
      case _ => throw new IllegalArgumentException("legacy can only be True when the underlying bitgenerator is an instance of MT19937.")

  /** `get_state(legacy)`: the tuple form when `legacy`, else the bit generator's state map plus
    * `has_gauss`/`gauss` entries.
    */
  def get_state(legacy: Boolean): LegacyMTState | Map[String, Any] =
    if legacy && bitGen.isInstanceOf[MT19937] then get_state()
    else bitGen.state ++ Map("has_gauss" -> (if aug.hasGauss then 1 else 0), "gauss" -> aug.gauss)

  /** `set_state(state)` from a legacy tuple. */
  def set_state(state: LegacyMTState): Unit =
    if state._1 != "MT19937" then throw new IllegalArgumentException("set_state can only be used with legacy MT19937 state instances.")
    set_state(Map[String, Any]("bit_generator" -> state._1, "state" -> Map("key" -> state._2, "pos" -> state._3),
      "has_gauss" -> state._4, "gauss" -> state._5))

  /** `set_state(state)` from a state map (as returned by `get_state(legacy = false)`). */
  def set_state(state: Map[String, Any]): Unit =
    if !state.contains("bit_generator") || !state.contains("state") then
      throw new IllegalArgumentException("state dictionary is not valid.")
    aug.gauss = RCommon.scalarToDouble(state.getOrElse("gauss", 0.0))
    aug.hasGauss = RandomUtil.anyToLong(state.getOrElse("has_gauss", 0)) != 0
    bitGen.state = state

  // ---- primitives (legacy) ----
  protected def pStdNormal(): Double = LegacyDist.gauss(aug)
  protected def pStdExp(): Double = LegacyDist.standardExponential(aug)
  protected def pStdGamma(shape: Double): Double = LegacyDist.standardGamma(aug, shape)
  protected def pNormal(loc: Double, scale: Double): Double = LegacyDist.normal(aug, loc, scale)
  protected def pExponential(scale: Double): Double = LegacyDist.exponential(aug, scale)
  protected def pGamma(shape: Double, scale: Double): Double = LegacyDist.gamma(aug, shape, scale)
  protected def pBeta(a: Double, b: Double): Double = LegacyDist.beta(aug, a, b)
  protected def pChisquare(df: Double): Double = LegacyDist.chisquare(aug, df)
  protected def pNoncentralChisquare(df: Double, nonc: Double): Double = LegacyDist.noncentralChisquare(aug, df, nonc)
  protected def pF(dfnum: Double, dfden: Double): Double = LegacyDist.f(aug, dfnum, dfden)
  protected def pNoncentralF(dfnum: Double, dfden: Double, nonc: Double): Double =
    LegacyDist.noncentralF(aug, dfnum, dfden, nonc)
  protected def pStdCauchy(): Double = LegacyDist.standardCauchy(aug)
  protected def pStdT(df: Double): Double = LegacyDist.standardT(aug, df)
  protected def pVonmises(mu: Double, kappa: Double): Double = Dist.vonmises(bg, mu, kappa, true)
  protected def pPareto(a: Double): Double = LegacyDist.pareto(aug, a)
  protected def pWeibull(a: Double): Double = LegacyDist.weibull(aug, a)
  protected def pPower(a: Double): Double = LegacyDist.power(aug, a)
  protected def pLognormal(mean: Double, sigma: Double): Double = LegacyDist.lognormal(aug, mean, sigma)
  protected def pRayleigh(scale: Double): Double = LegacyDist.rayleigh(aug, scale)
  protected def pWald(mean: Double, scale: Double): Double = LegacyDist.wald(aug, mean, scale)
  protected def uniformRangeCons: Cons = NoCons

  protected def pBinomial(p: Double, n: Long): Long = Dist.legacyBinomial(bg, p, n, binom)
  protected def binomialNCons: Cons = LegacyNonNegInboundsLong
  protected def pNegativeBinomial(n: Double, p: Double): Long = LegacyDist.negativeBinomial(aug, n, p)
  protected def negBinCheck(n: Array[Double], p: Array[Double], scalar: Boolean): Unit =
    if scalar then
      RCommon.check(n(0), "n", Positive); RCommon.check(p(0), "p", Bounded01)
    else
      RCommon.checkArr(n, "n", Positive); RCommon.checkArr(p, "p", Bounded01)
  protected def pPoisson(lam: Double): Long = Dist.poisson(bg, lam)
  protected def poissonCons: Cons = LegacyPoisson
  protected def pZipf(a: Double): Long = LegacyDist.zipf(bg, a)
  protected def pGeometric(p: Double): Long = LegacyDist.geometric(bg, p)
  protected def pLogseries(p: Double): Long = LegacyDist.logseries(bg, p)
  protected def pHypergeometric(good: Long, bad: Long, sample: Long): Long = LegacyDist.hypergeometric(bg, good, bad, sample)
  protected def hyperCheck(good: Array[Long], bad: Array[Long], sample: Array[Long]): (Cons, Cons, Cons) =
    if good.indices.exists(i => good(i) + bad(i) < sample(i)) then
      throw new IllegalArgumentException("ngood + nbad < nsample")
    (LegacyNonNegInboundsLong, NonNegative, Gte1)

  // ---------------- uniform floats ----------------
  /** `random_sample()`: one float in `[0, 1)`. */
  def random_sample(): Double = bg.nextDouble()
  /** `random_sample(size*)`. */
  def random_sample(size: Int*): NDArray[Double] = rd(size)(bg.nextDouble())
  /** `random_sample(size)` with the shape as a `Seq`. */
  @annotation.targetName("random_sampleSeq") def random_sample(size: Seq[Int]): NDArray[Double] = rd(size)(bg.nextDouble())
  /** `random()`: alias of [[random_sample]]. */
  def random(): Double = random_sample()
  /** `random(size*)`: alias of [[random_sample]]. */
  def random(size: Int*): NDArray[Double] = random_sample(size)
  /** `random(size)` with the shape as a `Seq`. */
  @annotation.targetName("randomSeq") def random(size: Seq[Int]): NDArray[Double] = random_sample(size)
  /** `ranf()`: alias of [[random_sample]]. */
  def ranf(): Double = random_sample()
  /** `ranf(size*)`: alias of [[random_sample]]. */
  def ranf(size: Int*): NDArray[Double] = random_sample(size)
  /** `sample()`: alias of [[random_sample]]. */
  def sample(): Double = random_sample()
  /** `sample(size*)`: alias of [[random_sample]]. */
  def sample(size: Int*): NDArray[Double] = random_sample(size)
  /** `rand()`: one float in `[0, 1)`. */
  def rand(): Double = random_sample()
  /** `rand(d0, d1, ...)`: floats in `[0, 1)` of shape `(d0, d1, ...)`. */
  def rand(d0: Int, dims: Int*): NDArray[Double] = random_sample(d0 +: dims)
  /** `randn()`: one standard normal sample (polar Box–Muller with cached value). */
  def randn(): Double = standard_normal()
  /** `randn(d0, d1, ...)`. */
  def randn(d0: Int, dims: Int*): NDArray[Double] = standard_normal(d0 +: dims)

  /** `tomaxint()`: one non-negative int64 (`next_uint64 >> 1`). */
  def tomaxint(): Long = bg.nextUInt64() >>> 1
  /** `tomaxint(size*)`. */
  def tomaxint(size: Int*): NDArray[Long] = rl(size)(bg.nextUInt64() >>> 1)

  // ---------------- randint ----------------
  /** `randint(high)`: one int64 in `[0, high)`. */
  def randint(high: Long): Long = randint(0L, high)
  /** `randint(low, high)`: one int64 in `[low, high)`. */
  def randint(low: Long, high: Long): Long = RandomInts.int64(bg, low, high, null, masked = true, endpoint = false).item
  /** `randint(low, high, size*)`. */
  def randint(low: Long, high: Long, size: Int*): NDArray[Long] = RandomInts.int64(bg, low, high, size, true, false)
  /** `randint(low, high, size)` with broadcasting bounds; with `high = null` samples `[0, low)`. */
  def randint(low: Param, high: Param | Null = null, size: SizeArg = null): NDArray[Long] =
    if high == null then RandomInts.int64(bg, 0L, low, size, true, false)
    else RandomInts.int64(bg, low, high, size, true, false)
  /** `randint(low, high, size, dtype)` for int64/int32/int16/int8/bool. */
  def randint[T](low: Param, high: Param | Null, size: SizeArg, dtype: DType[T]): NDArray[T] =
    val (lo, hi) = if high == null then (0L: Param, low) else (low, high)
    RandomInts.generic(bg, lo, hi, size, dtype, true, false)

  /** `random_integers(low, high)` (deprecated in NumPy): one int in the closed range `[low, high]`. */
  def random_integers(low: Long, high: Long): Long = randint(low, high + 1)
  /** `random_integers(high)`: one int in `[1, high]`. */
  def random_integers(high: Long): Long = randint(1L, high + 1)
  /** `random_integers(low, high, size*)`. */
  def random_integers(low: Long, high: Long, size: Int*): NDArray[Long] = randint(low, high + 1, size*)

  // ---------------- choice ----------------
  /** `choice(n)`: one integer from `[0, n)`. */
  def choice(a: Int): Long = choiceImpl(a, null, true, null).asInstanceOf[NDArray[Long]].item
  /** `choice(a)` for a 1-d array: one element. */
  def choice[T](a: NDArray[T]): T = choiceImpl(a, null, true, null).asInstanceOf[NDArray[T]].item
  /** `choice(a, size, replace, p)`: `a` is a population size or a 1-d array. */
  def choice[A](a: A, size: SizeArg = null, replace: Boolean = true,
      p: Param | Null = null)(using c: ChoiceOf[A]): NDArray[c.Out] =
    choiceImpl(a, size, replace, p).asInstanceOf[NDArray[c.Out]]

  private def choiceImpl(a: Any, size: SizeArg, replace: Boolean, p0: Param | Null): NDArray[?] =
    val shp0 = RCommon.sizeShape(size)
    val shape = if shp0 == null then Array.empty[Int] else shp0
    val n = Shape.size(shape)
    val popSize: Int = a match
      case i: Int =>
        if i <= 0 && n != 0 then throw new IllegalArgumentException("a must be greater than 0 unless no samples are taken")
        i
      case arr: NDArray[?] =>
        if arr.ndim != 1 then throw new IllegalArgumentException("a must be 1-dimensional")
        if arr.size == 0 && n != 0 then throw new IllegalArgumentException("'a' cannot be empty unless no samples are taken")
        arr.size
    val p: Array[Double] | Null =
      if p0 == null then null
      else
        val pa = RCommon.toD(p0)
        if pa.ndim != 1 then throw new IllegalArgumentException("'p' must be 1-dimensional")
        if pa.size != popSize then throw new IllegalArgumentException("'a' and 'p' must have same size")
        val pv = pa.toArray
        val s = RCommon.kahanSum(pv, 0, pv.length)
        if s.isNaN then throw new IllegalArgumentException("probabilities contain NaN")
        if pv.exists(_ < 0) then throw new IllegalArgumentException("probabilities are not non-negative")
        if math.abs(s - 1.0) > math.sqrt(math.ulp(1.0)) then throw new IllegalArgumentException("probabilities do not sum to 1")
        pv
    val idx: Array[Long] =
      if replace then
        if p != null then
          val cdf = RandomArrays.cumsum(p)
          val last = cdf(cdf.length - 1)
          for i <- cdf.indices do cdf(i) /= last
          Array.tabulate(n)(_ => RandomArrays.searchRight(cdf, bg.nextDouble()).toLong)
        else RandomInts.int64(bg, 0L, popSize.toLong, shape.toSeq, true, false).toArray
      else
        if n > popSize then
          throw new IllegalArgumentException("Cannot take a larger sample than population when 'replace=False'")
        if p != null then
          if p.count(_ > 0) < n then throw new IllegalArgumentException("Fewer non-zero entries in p than size")
          RandomChoice.noReplaceP(this, p.clone(), n)
        else permutation(popSize).toArray.take(n)
    a match
      case _: Int => NDArray.fromArray(idx, shape)
      case arr: NDArray[?] =>
        val ar = arr.asInstanceOf[NDArray[Any]]
        RandomArrays.take(ar, idx.map(_.toInt), 0).reshape(shape.toSeq*)

  // ---------------- multivariate ----------------
  /** `multinomial(n, pvals, size)`. */
  def multinomial(n: Long, pvals: Param, size: SizeArg = null): NDArray[Long] =
    RandomMulti.multinomial(bg, binom, n, pvals, size, legacy = true)

  /** `dirichlet(alpha, size)`. */
  def dirichlet(alpha: Param, size: SizeArg = null): NDArray[Double] = RandomMulti.dirichletLegacy(aug, alpha, size)

  /** `multivariate_normal(mean, cov, size, check_valid, tol)` (SVD factorisation). */
  def multivariate_normal(mean: Param, cov: Param, size: SizeArg = null, check_valid: String = "warn",
      tol: Double = 1e-8): NDArray[Double] =
    RandomMulti.multivariateNormal(mean, cov, size, check_valid, tol, "svd", shp => standard_normal(shp.toSeq), legacy = true)

object RandomState:
  private def legacyMT(seed: Any): BitGenerator = seed match
    case b: BitGenerator => b
    case null => new MT19937()
    case s =>
      val mt = new MT19937()
      mt._legacy_seeding(s)
      mt
