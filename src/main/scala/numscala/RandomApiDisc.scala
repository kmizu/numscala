package numscala.random

import numscala.*
import Cons.*

/** The discrete distributions and permutation helpers shared by [[Generator]] and [[RandomState]]. */
abstract class RandomDiscrete extends RandomContinuous:
  protected def pBinomial(p: Double, n: Long): Long
  protected def binomialNCons: Cons
  protected def pNegativeBinomial(n: Double, p: Double): Long
  /** Validates negative-binomial parameters (flattened, broadcast). */
  protected def negBinCheck(n: Array[Double], p: Array[Double], scalar: Boolean): Unit
  protected def pPoisson(lam: Double): Long
  protected def poissonCons: Cons
  protected def pZipf(a: Double): Long
  protected def pGeometric(p: Double): Long
  protected def pLogseries(p: Double): Long
  protected def pHypergeometric(good: Long, bad: Long, sample: Long): Long
  /** Validates hypergeometric parameters; returns the per-argument constraints to apply afterwards. */
  protected def hyperCheck(good: Array[Long], bad: Array[Long], sample: Array[Long]): (Cons, Cons, Cons)

  private def d1(size: SizeArg, a: Param, an: String, ac: Cons)(f: Double => Long): NDArray[Long] =
    RCommon.disc(size, Seq((a, an, ac)))(v => f(v(0)))

  // ---------------- binomial ----------------
  /** `binomial(n, p)`: one sample. */
  def binomial(n: Long, p: Double): Long =
    RCommon.check(p, "p", Bounded01); RCommon.check(n.toDouble, "n", binomialNCons)
    pBinomial(p, n)
  /** `binomial(n, p, size*)`. */
  def binomial(n: Long, p: Double, size: Int*): NDArray[Long] = binomialA(n, p, size)
  /** `binomial(n, p, size)` with broadcasting. */
  def binomial(n: Param, p: Param, size: SizeArg = null): NDArray[Long] = binomialA(n, p, size)
  private def binomialA(n: Param, p: Param, size: SizeArg): NDArray[Long] =
    val pa = RCommon.toD(p)
    val na = RCommon.toL(n)
    if pa.ndim == 0 && na.ndim == 0 then
      val pv = pa.item; val nv = na.item
      RCommon.check(pv, "p", Bounded01); RCommon.check(nv.toDouble, "n", binomialNCons)
      rl(size)(pBinomial(pv, nv))
    else
      RCommon.checkArr(pa.toArray, "p", Bounded01)
      RCommon.checkArr(na.toArray.map(_.toDouble), "n", binomialNCons)
      val (shp, fl) = RCommon.broadcastAll(size, Seq(pa, na.asType(using DType.Float64).asInstanceOf[NDArray[Double]]))
      val nflat = na.broadcastTo(shp*).toArray
      val pflat = fl(0)
      NDArray.fromArray(Array.tabulate(pflat.length)(i => pBinomial(pflat(i), nflat(i))), shp)

  protected def rl(s: SizeArg)(f: => Long): NDArray[Long] =
    val shp = RCommon.sizeShape(s)
    if shp == null then NDArray.scalar(f) else RCommon.fillL(RCommon.checkShape(shp))(f)

  // ---------------- negative_binomial ----------------
  /** `negative_binomial(n, p)`: one sample. */
  def negative_binomial(n: Double, p: Double): Long =
    negBinCheck(Array(n), Array(p), true)
    pNegativeBinomial(n, p)
  /** `negative_binomial(n, p, size*)`. */
  def negative_binomial(n: Double, p: Double, size: Int*): NDArray[Long] = negBinA(n, p, size)
  /** `negative_binomial(n, p, size)` with broadcasting. */
  def negative_binomial(n: Param, p: Param, size: SizeArg = null): NDArray[Long] = negBinA(n, p, size)
  private def negBinA(n: Param, p: Param, size: SizeArg): NDArray[Long] =
    val na = RCommon.toD(n); val pa = RCommon.toD(p)
    val scalar = na.ndim == 0 && pa.ndim == 0
    if scalar then negBinCheck(na.toArray, pa.toArray, true)
    else
      val (_, fl) = RCommon.broadcastAll(null, Seq(na, pa))
      negBinCheck(fl(0), fl(1), false)
    RCommon.disc(size, Seq((na, "n", NoCons), (pa, "p", NoCons)))(v => pNegativeBinomial(v(0), v(1)))

  // ---------------- poisson ----------------
  /** `poisson(lam)`: one sample. */
  def poisson(lam: Double): Long = { RCommon.check(lam, "lam", poissonCons); pPoisson(lam) }
  /** `poisson()`: one sample with `lam = 1`. */
  def poisson(): Long = poisson(1.0)
  /** `poisson(lam, size*)`. */
  def poisson(lam: Double, size: Int*): NDArray[Long] = poissonA(lam, size)
  /** `poisson(lam, size)` with broadcasting. */
  def poisson(lam: Param = 1.0, size: SizeArg = null): NDArray[Long] = poissonA(lam, size)
  private def poissonA(lam: Param, size: SizeArg) = d1(size, lam, "lam", poissonCons)(pPoisson)

  // ---------------- zipf / geometric / logseries ----------------
  /** `zipf(a)`: one sample. */
  def zipf(a: Double): Long = { RCommon.check(a, "a", Gt1); pZipf(a) }
  /** `zipf(a, size*)`. */
  def zipf(a: Double, size: Int*): NDArray[Long] = zipfA(a, size)
  /** `zipf(a, size)` with broadcasting. */
  def zipf(a: Param, size: SizeArg = null): NDArray[Long] = zipfA(a, size)
  private def zipfA(a: Param, size: SizeArg) = d1(size, a, "a", Gt1)(pZipf)

  /** `geometric(p)`: one sample (number of trials to the first success). */
  def geometric(p: Double): Long = { RCommon.check(p, "p", BoundedGt01); pGeometric(p) }
  /** `geometric(p, size*)`. */
  def geometric(p: Double, size: Int*): NDArray[Long] = geometricA(p, size)
  /** `geometric(p, size)` with broadcasting. */
  def geometric(p: Param, size: SizeArg = null): NDArray[Long] = geometricA(p, size)
  private def geometricA(p: Param, size: SizeArg) = d1(size, p, "p", BoundedGt01)(pGeometric)

  /** `logseries(p)`: one sample. */
  def logseries(p: Double): Long = { RCommon.check(p, "p", BoundedLt01); pLogseries(p) }
  /** `logseries(p, size*)`. */
  def logseries(p: Double, size: Int*): NDArray[Long] = logseriesA(p, size)
  /** `logseries(p, size)` with broadcasting. */
  def logseries(p: Param, size: SizeArg = null): NDArray[Long] = logseriesA(p, size)
  private def logseriesA(p: Param, size: SizeArg) = d1(size, p, "p", BoundedLt01)(pLogseries)

  // ---------------- hypergeometric ----------------
  /** `hypergeometric(ngood, nbad, nsample)`: one sample. */
  def hypergeometric(ngood: Long, nbad: Long, nsample: Long): Long =
    hypergeometricA(ngood, nbad, nsample, null).item
  /** `hypergeometric(ngood, nbad, nsample, size*)`. */
  def hypergeometric(ngood: Long, nbad: Long, nsample: Long, size: Int*): NDArray[Long] =
    hypergeometricA(ngood, nbad, nsample, size)
  /** `hypergeometric(ngood, nbad, nsample, size)` with broadcasting. */
  def hypergeometric(ngood: Param, nbad: Param, nsample: Param, size: SizeArg = null): NDArray[Long] =
    hypergeometricA(ngood, nbad, nsample, size)
  private def hypergeometricA(ngood: Param, nbad: Param, nsample: Param, size: SizeArg): NDArray[Long] =
    val g = RCommon.toL(ngood); val b = RCommon.toL(nbad); val s = RCommon.toL(nsample)
    val (_, fl) = RCommon.broadcastAll(null, Seq(g, b, s))
    val (gc, bc, sc) = hyperCheck(fl(0), fl(1), fl(2))
    RCommon.discL(size, Seq((g, "ngood", gc), (b, "nbad", bc), (s, "nsample", sc)))(v => pHypergeometric(v(0), v(1), v(2)))

  // ---------------- bytes / shuffle / permutation ----------------
  /** `bytes(length)`: random bytes. */
  def bytes(length: Int): Array[Byte] =
    if length < 0 then throw new IllegalArgumentException("negative dimensions are not allowed")
    val n = if length == 0 then 0 else (length - 1) / 4 + 1
    val out = new Array[Byte](n * 4)
    var i = 0
    while i < n do
      val w = bg.nextUInt32()
      out(4 * i) = w.toByte; out(4 * i + 1) = (w >>> 8).toByte
      out(4 * i + 2) = (w >>> 16).toByte; out(4 * i + 3) = (w >>> 24).toByte
      i += 1
    out.take(length)

  /** `shuffle(x, axis)`: shuffles `x` in place along `axis` (Fisher–Yates, like NumPy). */
  def shuffle(x: NDArray[?], axis: Int = 0): Unit =
    val xa = x.asInstanceOf[NDArray[Any]]
    if xa.ndim == 0 then throw new IllegalArgumentException("len() of unsized object")
    val ax = Shape.normAxis(axis, xa.ndim)
    if xa.size == 0 then return
    if xa.ndim == 1 then
      val n = xa.shapeArr(0)
      val st = xa.stridesArr(0)
      val off = xa.offset
      val d: Array[?] = x.data
      var i = n - 1
      while i >= 1 do
        val j = Dist.interval(bg, i.toLong).toInt
        RandomArrays.swap(d, off + j * st, off + i * st)
        i -= 1
    else
      val v = xa.swapaxes(0, ax)
      var i = v.shapeArr(0) - 1
      while i >= 1 do
        val j = Dist.interval(bg, i.toLong).toInt
        if i != j then
          val bj = v.subArray(j)
          val tmp = bj.copy()
          bj := v.subArray(i)
          v.subArray(i) := tmp
        i -= 1

  /** Shuffles a mutable Scala buffer in place (NumPy's generic-sequence path). */
  def shuffle[T](x: scala.collection.mutable.IndexedSeq[T]): Unit =
    var i = x.length - 1
    while i >= 1 do
      val j = Dist.interval(bg, i.toLong).toInt
      val t = x(i); x(i) = x(j); x(j) = t
      i -= 1

  /** `permutation(n)`: a shuffled `arange(n)`. */
  def permutation(x: Int): NDArray[Long] =
    if x < 0 then throw new IllegalArgumentException("negative dimensions are not allowed")
    val arr = NDArray.fromArray(Array.tabulate(x)(_.toLong), Array(x))
    shuffle(arr)
    arr

  /** `permutation(x, axis)`: a shuffled copy of `x` (permuted along `axis`). */
  def permutation[T](x: NDArray[T], axis: Int): NDArray[T] =
    if x.ndim < 1 then throw new IndexError("x must be an integer or at least 1-dimensional")
    val ax = Shape.normAxis(axis, x.ndim)
    if x.ndim == 1 then
      val c = x.copy()
      shuffle(c)
      c
    else
      val idx = NDArray.fromArray(Array.tabulate(x.shapeArr(ax))(i => i), Array(x.shapeArr(ax)))
      shuffle(idx)
      RandomArrays.take(x, idx.toArray, ax)

  /** `permutation(x)`: a shuffled copy of `x` along the first axis. */
  def permutation[T](x: NDArray[T]): NDArray[T] = permutation(x, 0)

/** Exception mirroring Python's `IndexError`. */
class IndexError(msg: String) extends IndexOutOfBoundsException(msg)

private[numscala] object RandomArrays:
  /** Swaps two elements of a (possibly primitive) array. */
  def swap(d: Array[?], i: Int, j: Int): Unit = d match
    case a: Array[Double] => val t = a(i); a(i) = a(j); a(j) = t
    case a: Array[Long] => val t = a(i); a(i) = a(j); a(j) = t
    case a: Array[Int] => val t = a(i); a(i) = a(j); a(j) = t
    case a: Array[Float] => val t = a(i); a(i) = a(j); a(j) = t
    case a: Array[Short] => val t = a(i); a(i) = a(j); a(j) = t
    case a: Array[Byte] => val t = a(i); a(i) = a(j); a(j) = t
    case a: Array[Boolean] => val t = a(i); a(i) = a(j); a(j) = t
    case a: Array[AnyRef] => val t = a(i); a(i) = a(j); a(j) = t
    case a => throw new IllegalArgumentException(s"unsupported array $a")

  /** `a.take(idx, axis)`. */
  def take[T](a: NDArray[T], idx: Array[Int], axis: Int): NDArray[T] =
    val ax = Shape.normAxis(axis, a.ndim)
    val n = a.shapeArr(ax)
    val moved = a.moveaxis(ax, 0)
    val inner = moved.shapeArr.drop(1)
    val innerSize = Shape.size(inner)
    given DType[T] = a.dtype
    val out = a.dtype.newArray(idx.length * innerSize)
    var k = 0
    while k < idx.length do
      val i = idx(k)
      if i < -n || i >= n then
        throw new IndexOutOfBoundsException(s"index $i is out of bounds for axis $ax with size $n")
      val sub = moved.subArray(if i < 0 then i + n else i).toArray
      System.arraycopy(sub, 0, out, k * innerSize, innerSize)
      k += 1
    NDArray.fromArray(out, idx.length +: inner).moveaxis(0, ax).copy()

  /** `cdf.searchsorted(u, side='right')` for a sorted cdf. */
  def searchRight(cdf: Array[Double], u: Double): Int =
    var lo = 0
    var hi = cdf.length
    while lo < hi do
      val mid = (lo + hi) >>> 1
      if cdf(mid) <= u then lo = mid + 1 else hi = mid
    lo

  /** `np.cumsum(p)` (sequential, like NumPy for 1-d float arrays). */
  def cumsum(p: Array[Double]): Array[Double] =
    val out = new Array[Double](p.length)
    var s = 0.0
    var i = 0
    while i < p.length do { s += p(i); out(i) = s; i += 1 }
    out
