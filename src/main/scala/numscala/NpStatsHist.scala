package numscala

import NpReduceImpl.*

/** Histograms and binning: `np.histogram`, `np.histogram_bin_edges`, `np.histogram2d`,
  * `np.histogramdd`, `np.bincount`, `np.digitize`.
  *
  * `bins` may be a bin count, an estimator name (`"auto"`, `"fd"`, `"doane"`, `"scott"`,
  * `"stone"`, `"rice"`, `"sturges"`, `"sqrt"`) or the bin edges (`NDArray` or `Seq[Double]`).
  */
trait NpStatsHist:

  type Bins = Int | String | NDArray[?] | Seq[Double]

  /** Histogram with 10 equal-width bins (`np.histogram(a)`): `(counts, bin_edges)`. */
  def histogram[T](a: NDArray[T]): (NDArray[Long], NDArray[Double]) = NpHist.counts(a, 10, null)

  /** Histogram with the given `bins` (`np.histogram(a, bins)`): `(counts, bin_edges)`. */
  def histogram[T](a: NDArray[T], bins: Bins): (NDArray[Long], NDArray[Double]) = NpHist.counts(a, bins, null)

  /** Histogram over `range` (`np.histogram(a, bins, range)`): `(counts, bin_edges)`. */
  def histogram[T](a: NDArray[T], bins: Bins, range: (Double, Double)): (NDArray[Long], NDArray[Double]) =
    NpHist.counts(a, bins, range)

  /** Weighted and/or normalized histogram (`np.histogram(a, bins, range, density, weights)`).
    * Returns floating-point values (sums of weights, or a probability density if `density`).
    */
  def histogram[T](
      a: NDArray[T],
      bins: Bins = 10,
      range: (Double, Double) | Null = null,
      density: Boolean = false,
      weights: NDArray[?] | Null = null
  ): (NDArray[Double], NDArray[Double]) =
    NpHist.weighted(a, bins, range, density, weights)

  /** The bin edges `np.histogram` would use (`np.histogram_bin_edges(a, bins, range, weights)`). */
  def histogram_bin_edges[T](
      a: NDArray[T],
      bins: Bins = 10,
      range: (Double, Double) | Null = null,
      weights: NDArray[?] | Null = null
  ): NDArray[Double] =
    val av = NpHist.values(a)
    if weights != null && weights.size != a.size then
      throw new IllegalArgumentException("weights should have the same shape as a.")
    vecD(NpHist.binEdges(av, a.dtype.kind == 'i' || a.dtype.kind == 'b', bins, range, weights != null)._1)

  /** Multidimensional histogram of `sample` (shape `(N, D)`, or `(N,)` for `D = 1`)
    * (`np.histogramdd(sample, bins, range, density, weights)`): `(hist, edges)`.
    * `bins` is a count for every dimension or one count/edge array per dimension.
    */
  def histogramdd[T](
      sample: NDArray[T],
      bins: Int | Seq[Int | NDArray[?] | Seq[Double]] = 10,
      range: Seq[(Double, Double) | Null] | Null = null,
      density: Boolean = false,
      weights: NDArray[?] | Null = null
  ): (NDArray[Double], Seq[NDArray[Double]]) =
    val s = if sample.ndim == 1 then sample.reshape(sample.shapeArr(0), 1) else sample
    if s.ndim != 2 then throw new IllegalArgumentException("sample must be 1-d or 2-d")
    val n = s.shapeArr(0)
    val d = s.shapeArr(1)
    val all = doublesOf(s, "sample")
    val cols = Array.tabulate(d)(j => Array.tabulate(n)(i => all(i * d + j)))
    NpHist.dd(cols, bins, range, density, weights)

  /** Bi-dimensional histogram of points `(x, y)` (`np.histogram2d(x, y, bins, range, density, weights)`):
    * `(H, xedges, yedges)`. `bins` is a count, a pair of counts, one edge array for both
    * axes, or a pair of edge arrays.
    */
  def histogram2d(
      x: NDArray[?],
      y: NDArray[?],
      bins: Int | (Int, Int) | NDArray[?] | Seq[Double] | (NDArray[?], NDArray[?]) = 10,
      range: ((Double, Double), (Double, Double)) | Null = null,
      density: Boolean = false,
      weights: NDArray[?] | Null = null
  ): (NDArray[Double], NDArray[Double], NDArray[Double]) =
    val xs = doublesOf(x, "x")
    val ys = doublesOf(y, "y")
    if xs.length != ys.length then throw new IllegalArgumentException("x and y must have the same length")
    val bs: Seq[Int | NDArray[?] | Seq[Double]] = bins match
      case i: Int => Seq(i, i)
      case a: NDArray[?] => Seq(a, a)
      case s: Seq[?] => Seq(s.asInstanceOf[Seq[Double]], s.asInstanceOf[Seq[Double]])
      case (b0, b1) => Seq(b0.asInstanceOf[Int | NDArray[?]], b1.asInstanceOf[Int | NDArray[?]])
    val rs: Seq[(Double, Double) | Null] | Null = range match
      case null => null
      case (r0, r1) => Seq(r0.asInstanceOf[(Double, Double)], r1.asInstanceOf[(Double, Double)])
    val (h, edges) = NpHist.dd(Array(xs, ys), bs, rs, density, weights)
    (h, edges(0), edges(1))

  /** Number of occurrences of each non-negative integer (`np.bincount(x, minlength=...)`). */
  def bincount[T](x: NDArray[T], minlength: Int = 0): NDArray[Long] =
    val idx = NpHist.bincountIdx(x, minlength)
    val out = new Array[Long](NpHist.bincountLen(idx, minlength))
    idx.foreach(i => out(i) += 1)
    NDArray.fromArray(out, Array(out.length))

  /** Weighted bin counts (`np.bincount(x, weights)`). */
  def bincount[T](x: NDArray[T], weights: NDArray[?]): NDArray[Double] = bincount(x, weights, 0)

  /** Weighted bin counts with a minimum length (`np.bincount(x, weights, minlength)`). */
  def bincount[T](x: NDArray[T], weights: NDArray[?], minlength: Int): NDArray[Double] =
    val idx = NpHist.bincountIdx(x, minlength)
    val w = doublesOf(weights, "weights")
    if w.length != idx.length then throw new IllegalArgumentException("The weights and list don't have the same length.")
    val out = new Array[Double](NpHist.bincountLen(idx, minlength))
    var i = 0
    while i < idx.length do
      out(idx(i)) += w(i)
      i += 1
    NDArray.fromArray(out, Array(out.length))

  /** Indices of the bins to which each value belongs (`np.digitize(x, bins, right)`). */
  def digitize(x: NDArray[?], bins: NDArray[?] | Seq[Double], right: Boolean = false): NDArray[Int] =
    if x.dtype.isComplex then throw new IllegalArgumentException("x may not be complex")
    val xs = doublesOf(x, "x")
    val b = NpHist.edgesOf(bins, "bins")
    val f = NpHist.digitizer(b, right)
    NDArray.fromArray(xs.map(f), x.shapeArr.clone())

  /** Bin index of a single value (`np.digitize(x, bins)` with scalar `x`). */
  def digitize(x: Double, bins: NDArray[?] | Seq[Double]): Int = NpHist.digitizer(NpHist.edgesOf(bins, "bins"), false)(x)

  /** Bin index of a single value (`np.digitize(x, bins, right)` with scalar `x`). */
  def digitize(x: Double, bins: NDArray[?] | Seq[Double], right: Boolean): Int =
    NpHist.digitizer(NpHist.edgesOf(bins, "bins"), right)(x)

/** Histogram kernels (port of `numpy/lib/_histograms_impl.py`). */
private[numscala] object NpHist:

  def values(a: NDArray[?]): Array[Double] = doublesOf(a, "a")

  def edgesOf(b: NDArray[?] | Seq[Double], what: String): Array[Double] = b match
    case arr: NDArray[?] =>
      if arr.ndim != 1 then throw new IllegalArgumentException(s"$what must be 1-dimensional")
      doublesOf(arr, what)
    case s: Seq[?] => s.asInstanceOf[Seq[Double]].toArray

  def digitizer(bins: Array[Double], right: Boolean): Double => Int =
    val n = bins.length
    val inc = (0 until n - 1).forall(i => bins(i) <= bins(i + 1))
    val dec = (0 until n - 1).forall(i => bins(i) >= bins(i + 1))
    if !inc && !dec then throw new IllegalArgumentException("bins must be monotonically increasing or decreasing")
    // NumPy: side = 'left' if right else 'right'
    if inc then x => searchSorted(bins, x, !right)
    else
      val rev = bins.reverse
      x => n - searchSorted(rev, x, !right)

  def bincountIdx[T](x: NDArray[T], minlength: Int): Array[Int] =
    if minlength < 0 then throw new IllegalArgumentException("'minlength' must not be negative")
    if x.ndim == 0 then throw new IllegalArgumentException("object of too small depth for desired array")
    if x.ndim > 1 then throw new IllegalArgumentException("object too deep for desired array")
    val k = x.dtype.kind
    if k != 'i' && k != 'b' && !(x.size == 0 && k == 'f') then
      throw new IllegalArgumentException(
        s"Cannot cast array data from dtype('${x.dtype.name}') to dtype('int64') according to the rule 'safe'"
      )
    val idx = x.toArray.map(v => x.dtype.toLong(v))
    if idx.exists(_ < 0) then throw new IllegalArgumentException("'list' argument must have no negative elements")
    idx.map(_.toInt)

  def bincountLen(idx: Array[Int], minlength: Int): Int =
    math.max(if idx.isEmpty then 0 else idx.max + 1, minlength)

  // ---------------------------------------------------------------- bin edges

  def outerEdges(a: Array[Double], range: (Double, Double) | Null): (Double, Double) =
    var (first, last) =
      if range != null then
        val (f, l) = range.asInstanceOf[(Double, Double)]
        if f > l then throw new IllegalArgumentException("max must be larger than min in range parameter.")
        if f.isNaN || l.isNaN || f.isInfinite || l.isInfinite then
          throw new IllegalArgumentException(s"supplied range of [${fmt(f)}, ${fmt(l)}] is not finite")
        (f, l)
      else if a.isEmpty then (0.0, 1.0)
      else
        var mn = a(0)
        var mx = a(0)
        a.foreach { v =>
          if v.isNaN || mn.isNaN then mn = Double.NaN else if v < mn then mn = v
          if v.isNaN || mx.isNaN then mx = Double.NaN else if v > mx then mx = v
        }
        if mn.isNaN || mx.isNaN || mn.isInfinite || mx.isInfinite then
          throw new IllegalArgumentException(s"autodetected range of [${fmt(mn)}, ${fmt(mx)}] is not finite")
        (mn, mx)
    if first == last then
      first = first - 0.5
      last = last + 0.5
    (first, last)

  private def fmt(d: Double): String = Format.formatFloatShort(d)

  /** `np.linspace(start, stop, num, endpoint=True)` exactly as NumPy computes it. */
  def linspace(start: Double, stop: Double, num: Int): Array[Double] =
    val div = num - 1
    val delta = stop - start
    val out = new Array[Double](num)
    if div > 0 then
      val step = delta / div
      var i = 0
      while i < num do
        out(i) = (if step == 0 then (i.toDouble / div) * delta else i * step) + start
        i += 1
      out(num - 1) = stop
    else if num == 1 then out(0) = start
    out

  private def ptp(x: Array[Double]): Double = x.max - x.min

  private def std(x: Array[Double]): Double =
    val m = Reduce.pairwiseSum(x, 0, x.length) / x.length
    val d = x.map(v => (v - m) * (v - m))
    math.sqrt(Reduce.pairwiseSum(d, 0, d.length) / x.length)

  private def log2(v: Double): Double = math.log(v) / math.log(2.0)

  /** Bin width estimators (`_hist_bin_*`). */
  def estimatorWidth(name: String, x: Array[Double], range: (Double, Double)): Double =
    val n = x.length
    name match
      case "sqrt" => ptp(x) / math.sqrt(n.toDouble)
      case "sturges" => ptp(x) / (log2(n.toDouble) + 1.0)
      case "rice" => ptp(x) / (2.0 * math.pow(n.toDouble, 1.0 / 3))
      case "scott" => math.pow(24.0 * math.pow(math.Pi, 0.5) / n, 1.0 / 3.0) * std(x)
      case "stone" =>
        val p = ptp(x)
        if n <= 1 || p == 0 then 0.0
        else
          def jhat(nbins: Int): Double =
            val hh = p / nbins
            val (cnt, _) = uniformCounts(x, null, range._1, range._2, nbins)
            var dot = 0.0
            cnt.foreach { c =>
              val pk = c / n
              dot += pk * pk
            }
            (2 - (n + 1) * dot) / hh
          val upper = math.max(100, math.sqrt(n.toDouble).toInt)
          var best = 1
          var bestV = jhat(1)
          var k = 2
          while k <= upper do
            val v = jhat(k)
            if v < bestV then
              best = k
              bestV = v
            k += 1
          p / best
      case "doane" =>
        if n > 2 then
          val sg1 = math.sqrt(6.0 * (n - 2) / ((n + 1.0) * (n + 3)))
          val sigma = std(x)
          if sigma > 0.0 then
            val m = Reduce.pairwiseSum(x, 0, n) / n
            val t = x.map { v => val z = (v - m) / sigma; z * z * z }
            val g1 = Reduce.pairwiseSum(t, 0, n) / n
            ptp(x) / (1.0 + log2(n.toDouble) + log2(1.0 + math.abs(g1) / sg1))
          else 0.0
        else 0.0
      case "fd" =>
        val s = x.clone()
        java.util.Arrays.sort(s)
        val iqr = NpQuantile.quantileSorted(s, n, 0.75, "linear") - NpQuantile.quantileSorted(s, n, 0.25, "linear")
        2.0 * iqr * math.pow(n.toDouble, -1.0 / 3.0)
      case "auto" =>
        val fd = estimatorWidth("fd", x, range)
        val st = estimatorWidth("sturges", x, range)
        if fd != 0.0 && !fd.isNaN then math.min(fd, st) else if fd.isNaN then fd else st
      case other => throw new IllegalArgumentException(s"'$other' is not a valid estimator for `bins`")

  val estimators: Set[String] = Set("auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt")

  /** Returns `(edges, Some((first, last, nEqualBins)))` for uniform bins, or `(edges, None)`. */
  def binEdges(
      a: Array[Double],
      isInt: Boolean,
      bins: NpStatsHist#Bins,
      range: (Double, Double) | Null,
      hasWeights: Boolean
  ): (Array[Double], Option[(Double, Double, Int)]) =
    def uniform(first: Double, last: Double, nb: Int) =
      val e = linspace(first, last, nb + 1)
      if (0 until nb).exists(i => !(e(i) < e(i + 1))) then
        throw new IllegalArgumentException(s"Too many bins for data range. Cannot create $nb finite-sized bins.")
      (e, Some((first, last, nb)))
    bins match
      case name: String =>
        if !estimators(name) then throw new IllegalArgumentException(s"'$name' is not a valid estimator for `bins`")
        if hasWeights then
          throw new IllegalArgumentException("Automated estimation of the number of bins is not supported for weighted data")
        val (first, last) = outerEdges(a, range)
        val data = if range != null then a.filter(v => v >= first && v <= last) else a
        val nb =
          if data.isEmpty then 1
          else
            var width = estimatorWidth(name, data, (first, last))
            if width != 0.0 && !width.isNaN then
              if isInt && width < 1 then width = 1
              math.ceil((last - first) / width).toInt
            else 1
        uniform(first, last, nb)
      case nb: Int =>
        if nb < 1 then throw new IllegalArgumentException("`bins` must be positive, when an integer")
        val (first, last) = outerEdges(a, range)
        uniform(first, last, nb)
      case other =>
        val e = other match
          case arr: NDArray[?] =>
            if arr.ndim != 1 then throw new IllegalArgumentException("`bins` must be 1d, when an array")
            doublesOf(arr, "bins")
          case s: Seq[?] => s.asInstanceOf[Seq[Double]].toArray
          case x => throw new IllegalArgumentException(s"invalid bins: $x")
        if (0 until e.length - 1).exists(i => e(i) > e(i + 1)) then
          throw new IllegalArgumentException("`bins` must increase monotonically, when an array")
        (e, None)

  /** NumPy's fast path for equal-width bins: per-bin sums of `w` (or counts). */
  def uniformCounts(a: Array[Double], w: Array[Double] | Null, first: Double, last: Double, nb: Int): (Array[Double], Array[Double]) =
    val edges = linspace(first, last, nb + 1)
    val out = new Array[Double](nb)
    val denom = last - first
    var i = 0
    while i < a.length do
      val x = a(i)
      if x >= first && x <= last then
        var idx = (((x - first) / denom) * nb).toInt
        if idx == nb then idx -= 1
        if x < edges(idx) then idx -= 1
        else if idx != nb - 1 && x >= edges(idx + 1) then idx += 1
        out(idx) += (if w == null then 1.0 else w.asInstanceOf[Array[Double]](i))
      i += 1
    (out, edges)

  /** General path for explicit edges: sort + searchsorted (with weight cumsums). */
  def edgeCounts(a: Array[Double], w: Array[Double] | Null, edges: Array[Double]): Array[Double] =
    val ne = edges.length
    if ne < 2 then return Array.empty[Double]
    val order = a.indices.sortWith((i, j) => Format.compareDouble(a(i), a(j)) < 0).toArray
    val sa = order.map(a(_))
    val cum = new Array[Double](ne)
    val cw: Array[Double] | Null =
      if w == null then null
      else
        val c = new Array[Double](sa.length + 1)
        var k = 0
        while k < sa.length do
          c(k + 1) = c(k) + w.asInstanceOf[Array[Double]](order(k))
          k += 1
        c
    var k = 0
    while k < ne do
      val bi = searchSorted(sa, edges(k), right = k == ne - 1)
      cum(k) = if cw == null then bi.toDouble else cw.asInstanceOf[Array[Double]](bi)
      k += 1
    Array.tabulate(ne - 1)(i => cum(i + 1) - cum(i))

  private def histCore(
      a: NDArray[?],
      bins: NpStatsHist#Bins,
      range: (Double, Double) | Null,
      weights: NDArray[?] | Null
  ): (Array[Double], Array[Double]) =
    val av = values(a)
    val wv: Array[Double] | Null =
      if weights == null then null
      else
        if weights.size != a.size || !weights.shapeArr.sameElements(a.shapeArr) then
          throw new IllegalArgumentException("weights should have the same shape as a.")
        doublesOf(weights, "weights")
    val k = a.dtype.kind
    val (edges, uni) = binEdges(av, k == 'i' || k == 'b', bins, range, weights != null)
    uni match
      case Some((first, last, nb)) => (uniformCounts(av, wv, first, last, nb)._1, edges)
      case None => (edgeCounts(av, wv, edges), edges)

  def counts(a: NDArray[?], bins: NpStatsHist#Bins, range: (Double, Double) | Null): (NDArray[Long], NDArray[Double]) =
    val (c, e) = histCore(a, bins, range, null)
    (NDArray.fromArray(c.map(_.toLong), Array(c.length)), vecD(e))

  def weighted(
      a: NDArray[?],
      bins: NpStatsHist#Bins,
      range: (Double, Double) | Null,
      density: Boolean,
      weights: NDArray[?] | Null
  ): (NDArray[Double], NDArray[Double]) =
    val (c, e) = histCore(a, bins, range, weights)
    if density then
      val total = c.sum
      (vecD(Array.tabulate(c.length)(i => c(i) / (e(i + 1) - e(i)) / total)), vecD(e))
    else (vecD(c), vecD(e))

  /** `np.histogramdd` on `D` columns of equal length. */
  def dd(
      cols: Array[Array[Double]],
      bins: Int | Seq[Int | NDArray[?] | Seq[Double]],
      range: Seq[(Double, Double) | Null] | Null,
      density: Boolean,
      weights: NDArray[?] | Null
  ): (NDArray[Double], Seq[NDArray[Double]]) =
    val d = cols.length
    val n = if d == 0 then 0 else cols(0).length
    val bs: Seq[Int | NDArray[?] | Seq[Double]] = bins match
      case i: Int => Seq.fill(d)(i)
      case s: Seq[?] =>
        if s.length != d then
          throw new IllegalArgumentException("The dimension of bins must be equal to the dimension of the sample x.")
        s.asInstanceOf[Seq[Int | NDArray[?] | Seq[Double]]]
    val rs: Seq[(Double, Double) | Null] =
      if range == null then Seq.fill(d)(null)
      else
        val r = range.asInstanceOf[Seq[(Double, Double) | Null]]
        if r.length != d then throw new IllegalArgumentException("range argument must have one entry per dimension")
        r
    val edges: Array[Array[Double]] = Array.tabulate(d) { i =>
      bs(i) match
        case k: Int =>
          if k < 1 then throw new IllegalArgumentException(s"`bins[$i]` must be positive, when an integer")
          val (mn, mx) = outerEdges(cols(i), rs(i))
          linspace(mn, mx, k + 1)
        case other =>
          val e = other match
            case arr: NDArray[?] => doublesOf(arr, "bins")
            case s: Seq[?] => s.asInstanceOf[Seq[Double]].toArray
            case x => throw new IllegalArgumentException(s"invalid bins: $x")
          if (0 until e.length - 1).exists(j => e(j) > e(j + 1)) then
            throw new IllegalArgumentException(s"`bins[$i]` must be monotonically increasing, when an array")
          e
    }
    val nbin = edges.map(_.length + 1)
    val w: Array[Double] | Null =
      if weights == null then null
      else
        val wv = doublesOf(weights, "weights")
        if wv.length != n then throw new IllegalArgumentException("weights must have the same length as the sample")
        wv
    val total = nbin.foldLeft(1)(_ * _)
    val hist = new Array[Double](total)
    var p = 0
    while p < n do
      var flat = 0
      var i = 0
      while i < d do
        val x = cols(i)(p)
        val e = edges(i)
        var c = searchSorted(e, x, right = true)
        if x == e(e.length - 1) then c -= 1
        flat = flat * nbin(i) + c
        i += 1
      hist(flat) += (if w == null then 1.0 else w.asInstanceOf[Array[Double]](p))
      p += 1
    // strip the outlier bins
    val coreShape = nbin.map(_ - 2)
    val core = new Array[Double](Shape.size(coreShape))
    val idx = new Array[Int](d)
    var k = 0
    while k < core.length do
      var rem = k
      var j = d - 1
      while j >= 0 do
        idx(j) = rem % coreShape(j)
        rem /= coreShape(j)
        j -= 1
      var flat = 0
      j = 0
      while j < d do
        flat = flat * nbin(j) + idx(j) + 1
        j += 1
      core(k) = hist(flat)
      k += 1
    if density then
      val s = core.sum
      k = 0
      while k < core.length do
        var rem = k
        var j = d - 1
        var v = core(k)
        val pos = new Array[Int](d)
        while j >= 0 do
          pos(j) = rem % coreShape(j)
          rem /= coreShape(j)
          j -= 1
        j = 0
        while j < d do
          v = v / (edges(j)(pos(j) + 1) - edges(j)(pos(j)))
          j += 1
        core(k) = v / s
        k += 1
    (NDArray.fromArray(core, coreShape), edges.toSeq.map(vecD))
