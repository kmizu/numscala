package numscala

import NpReduceImpl.*

/** Order statistics: `np.median`, `np.percentile`, `np.quantile` and their NaN-ignoring variants.
  *
  * `q` may be a `Double`, a `Seq[Double]` or an `NDArray`; the dimensions of `q` are prepended
  * to the output shape. All 13 NumPy `method`s are supported. Results are always floating
  * point (NumPy keeps the input dtype for the discrete methods on integer input).
  */
trait NpReduceQuantile:

  /** Median of all elements (`np.median(a)`); NaN if any element is NaN. */
  def median[T](a: NDArray[T])(using t: ToFloat[T]): t.Out =
    t.dtype.fromDouble(NpReduceQuantileImpl.medianOf(doublesOf(a), false))

  /** Median over `axis` (`np.median(a, axis, keepdims=...)`). */
  def median[T](a: NDArray[T], axis: Axis | Null = null, keepdims: Boolean = false)(using
      t: ToFloat[T]
  ): NDArray[t.Out] =
    NpReduceQuantileImpl.lanes(a, axis, keepdims, Array.emptyIntArray, 1, t.dtype)((lane, out) =>
      out(0) = NpReduceQuantileImpl.medianOf(lane, false)
    )

  /** Median ignoring NaNs (`np.nanmedian(a)`). */
  def nanmedian[T](a: NDArray[T])(using t: ToFloat[T]): t.Out =
    t.dtype.fromDouble(NpReduceQuantileImpl.medianOf(doublesOf(a), true))

  /** Median over `axis` ignoring NaNs (`np.nanmedian(a, axis, keepdims=...)`). */
  def nanmedian[T](a: NDArray[T], axis: Axis | Null = null, keepdims: Boolean = false)(using
      t: ToFloat[T]
  ): NDArray[t.Out] =
    NpReduceQuantileImpl.lanes(a, axis, keepdims, Array.emptyIntArray, 1, t.dtype)((lane, out) =>
      out(0) = NpReduceQuantileImpl.medianOf(lane, true)
    )

  /** The `q`-th quantile (`0 <= q <= 1`) of all elements (`np.quantile(a, q)`). */
  def quantile[T](a: NDArray[T], q: Double)(using t: ToFloat[T]): t.Out =
    NpReduceQuantileImpl.scalar(a, q, "linear", false, false, t.dtype)

  /** The `q`-th quantile with an interpolation `method` (`np.quantile(a, q, method=...)`). */
  def quantile[T](a: NDArray[T], q: Double, method: String)(using t: ToFloat[T]): t.Out =
    NpReduceQuantileImpl.scalar(a, q, method, false, false, t.dtype)

  /** Quantiles over `axis` (`np.quantile(a, q, axis, method=..., keepdims=...)`). */
  def quantile[T](
      a: NDArray[T],
      q: Double | Seq[Double] | NDArray[?],
      axis: Axis | Null = null,
      method: String = "linear",
      keepdims: Boolean = false
  )(using t: ToFloat[T]): NDArray[t.Out] =
    NpReduceQuantileImpl.general(a, q, axis, method, keepdims, false, false, t.dtype)

  /** The `q`-th percentile (`0 <= q <= 100`) of all elements (`np.percentile(a, q)`). */
  def percentile[T](a: NDArray[T], q: Double)(using t: ToFloat[T]): t.Out =
    NpReduceQuantileImpl.scalar(a, q, "linear", true, false, t.dtype)

  /** The `q`-th percentile with an interpolation `method` (`np.percentile(a, q, method=...)`). */
  def percentile[T](a: NDArray[T], q: Double, method: String)(using t: ToFloat[T]): t.Out =
    NpReduceQuantileImpl.scalar(a, q, method, true, false, t.dtype)

  /** Percentiles over `axis` (`np.percentile(a, q, axis, method=..., keepdims=...)`). */
  def percentile[T](
      a: NDArray[T],
      q: Double | Seq[Double] | NDArray[?],
      axis: Axis | Null = null,
      method: String = "linear",
      keepdims: Boolean = false
  )(using t: ToFloat[T]): NDArray[t.Out] =
    NpReduceQuantileImpl.general(a, q, axis, method, keepdims, true, false, t.dtype)

  /** Quantile ignoring NaNs (`np.nanquantile(a, q)`). */
  def nanquantile[T](a: NDArray[T], q: Double)(using t: ToFloat[T]): t.Out =
    NpReduceQuantileImpl.scalar(a, q, "linear", false, true, t.dtype)

  /** Quantile ignoring NaNs with a `method` (`np.nanquantile(a, q, method=...)`). */
  def nanquantile[T](a: NDArray[T], q: Double, method: String)(using t: ToFloat[T]): t.Out =
    NpReduceQuantileImpl.scalar(a, q, method, false, true, t.dtype)

  /** Quantiles over `axis` ignoring NaNs (`np.nanquantile(a, q, axis, method=..., keepdims=...)`). */
  def nanquantile[T](
      a: NDArray[T],
      q: Double | Seq[Double] | NDArray[?],
      axis: Axis | Null = null,
      method: String = "linear",
      keepdims: Boolean = false
  )(using t: ToFloat[T]): NDArray[t.Out] =
    NpReduceQuantileImpl.general(a, q, axis, method, keepdims, false, true, t.dtype)

  /** Percentile ignoring NaNs (`np.nanpercentile(a, q)`). */
  def nanpercentile[T](a: NDArray[T], q: Double)(using t: ToFloat[T]): t.Out =
    NpReduceQuantileImpl.scalar(a, q, "linear", true, true, t.dtype)

  /** Percentile ignoring NaNs with a `method` (`np.nanpercentile(a, q, method=...)`). */
  def nanpercentile[T](a: NDArray[T], q: Double, method: String)(using t: ToFloat[T]): t.Out =
    NpReduceQuantileImpl.scalar(a, q, method, true, true, t.dtype)

  /** Percentiles over `axis` ignoring NaNs (`np.nanpercentile(a, q, axis, method=..., keepdims=...)`). */
  def nanpercentile[T](
      a: NDArray[T],
      q: Double | Seq[Double] | NDArray[?],
      axis: Axis | Null = null,
      method: String = "linear",
      keepdims: Boolean = false
  )(using t: ToFloat[T]): NDArray[t.Out] =
    NpReduceQuantileImpl.general(a, q, axis, method, keepdims, true, true, t.dtype)

/** Quantile kernels (NumPy's `_quantile`, Hyndman & Fan methods). */
private[numscala] object NpReduceQuantileImpl:

  val methods: Set[String] = Set(
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "nearest",
    "midpoint"
  )

  /** Sorted copy without NaNs (`dropNaN`) or with NaNs last. */
  private def prepare(lane: Array[Double], dropNaN: Boolean): Array[Double] =
    val s = if dropNaN then lane.filterNot(_.isNaN) else lane.clone()
    java.util.Arrays.sort(s)
    s

  def medianOf(lane: Array[Double], dropNaN: Boolean): Double =
    val s = prepare(lane, dropNaN)
    val n = s.length
    if n == 0 then Double.NaN
    else if s(n - 1).isNaN then Double.NaN
    else if n % 2 == 1 then s(n / 2)
    else (s(n / 2 - 1) + s(n / 2)) / 2.0

  private def lerp(a: Double, b: Double, t: Double): Double =
    val diff = b - a
    if t >= 0.5 then b - diff * (1 - t) else a + diff * t

  private def discrete(index: Double, cond: (Double, Double) => Boolean): Int =
    val prev = math.floor(index)
    val gamma = index - prev
    val r = if cond(gamma, index) then prev else prev + 1
    if r < 0 then 0 else r.toInt

  /** Quantile `q` of the sorted, NaN-free values `s` (length `n > 0`). */
  def quantileSorted(s: Array[Double], n: Int, q: Double, method: String): Double =
    def at(i: Int): Double = s(if i < 0 then n + i else i)
    def cvi(alpha: Double, beta: Double): Double = n * q + (alpha + q * (1 - alpha - beta)) - 1
    method match
      case "inverted_cdf" => at(discrete(n * q - 1, (g, _) => g == 0))
      case "closest_observation" =>
        at(discrete(n * q - 1 - 0.5, (g, idx) => g == 0 && pyMod(math.floor(idx), 2) == 1))
      case "lower" => at(math.floor((n - 1) * q).toInt)
      case "higher" => at(math.ceil((n - 1) * q).toInt)
      case "nearest" => at(math.rint((n - 1) * q).toInt)
      case _ =>
        val vi = method match
          case "averaged_inverted_cdf" => n * q - 1
          case "interpolated_inverted_cdf" => cvi(0, 1)
          case "hazen" => cvi(0.5, 0.5)
          case "weibull" => cvi(0, 0)
          case "median_unbiased" => cvi(1.0 / 3, 1.0 / 3)
          case "normal_unbiased" => cvi(3.0 / 8.0, 3.0 / 8.0)
          case "linear" => (n - 1) * q
          case "midpoint" => 0.5 * (math.floor((n - 1) * q) + math.ceil((n - 1) * q))
          case other => throw invalidMethod(other)
        var prev = math.floor(vi)
        var next = prev + 1
        if vi >= n - 1 || vi.isNaN then
          prev = -1
          next = -1
        else if vi < 0 then
          prev = 0
          next = 0
        var gamma = vi - prev
        method match
          case "averaged_inverted_cdf" => gamma = if gamma == 0 then 0.5 else 1.0
          case "midpoint" => gamma = if vi % 1 == 0 then 0.0 else 0.5
          case _ => ()
        lerp(at(prev.toInt), at(next.toInt), gamma)

  def invalidMethod(m: String) = new IllegalArgumentException(
    s"'$m' is not a valid method. Use one of: ${methods.toSeq.sorted.mkString(", ")}"
  )

  /** Quantiles of one lane, writing `out(i)` for each `qs(i)`. */
  def lane(lane: Array[Double], qs: Array[Double], method: String, dropNaN: Boolean, out: Array[Double]): Unit =
    val s = prepare(lane, dropNaN)
    val n = s.length
    var i = 0
    while i < qs.length do
      out(i) =
        if n == 0 then
          if dropNaN then Double.NaN
          else throw new IndexOutOfBoundsException("index -1 is out of bounds for axis 0 with size 0")
        else if s(n - 1).isNaN then Double.NaN
        else quantileSorted(s, n, qs(i), method)
      i += 1

  /** Normalizes and validates `q` (as fractions); returns values and shape. */
  def qValues(q: Double | Seq[Double] | NDArray[?], percent: Boolean): (Array[Double], Array[Int]) =
    val (vals, shape) = q match
      case d: Double => (Array(d), Array.emptyIntArray)
      case arr: NDArray[?] => (doublesOf(arr, "q"), arr.shapeArr.clone())
      case s: Seq[?] => (s.map(x => x.asInstanceOf[Matchable] match
          case v: Double => v
          case v: Int => v.toDouble
          case v: Long => v.toDouble
          case v: Float => v.toDouble
          case v => throw new IllegalArgumentException(s"invalid q value $v")
        ).toArray, Array(s.length))
    if percent then
      if vals.exists(v => !(v >= 0 && v <= 100)) then
        throw new IllegalArgumentException("Percentiles must be in the range [0, 100]")
      (vals.map(_ / 100), shape)
    else
      if vals.exists(v => !(v >= 0 && v <= 1)) then
        throw new IllegalArgumentException("Quantiles must be in the range [0, 1]")
      (vals, shape)

  /** Runs `f(lane, out)` (writing `nq` values) on every lane over `axis`; output shape `qShape ++ reduced`. */
  def lanes[T, U](a: NDArray[T], axis: Axis | Null, keepdims: Boolean, qShape: Array[Int], nq: Int, od: FloatDType[U])(
      f: (Array[Double], Array[Double]) => Unit
  ): NDArray[U] =
    if a.dtype.isComplex then throw new IllegalArgumentException("a must be an array of real numbers")
    val ax = axesOf(axis, a.ndim)
    val (buf, m, l) = gather(a, ax)
    val vals = toDoubles(a.dtype, buf)
    val res = od.newArray(nq * m)
    val laneBuf = new Array[Double](l)
    val out = new Array[Double](nq)
    var r = 0
    while r < m do
      System.arraycopy(vals, r * l, laneBuf, 0, l)
      f(laneBuf, out)
      var i = 0
      while i < nq do
        res(i * m + r) = od.fromDouble(out(i))
        i += 1
      r += 1
    NDArray.fromArray(res, qShape ++ reducedShape(a.shapeArr, ax, keepdims))(using od)

  def general[T, U](
      a: NDArray[T],
      q: Double | Seq[Double] | NDArray[?],
      axis: Axis | Null,
      method: String,
      keepdims: Boolean,
      percent: Boolean,
      dropNaN: Boolean,
      od: FloatDType[U]
  ): NDArray[U] =
    if !methods(method) then throw invalidMethod(method)
    val (qs, qShape) = qValues(q, percent)
    lanes(a, axis, keepdims, qShape, qs.length, od)((ln, out) => lane(ln, qs, method, dropNaN, out))

  def scalar[T, U](a: NDArray[T], q: Double, method: String, percent: Boolean, dropNaN: Boolean, od: FloatDType[U]): U =
    general(a, q, null, method, false, percent, dropNaN, od).item
