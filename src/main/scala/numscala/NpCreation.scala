package numscala

/** Array creation routines (`np.array`, `np.zeros`, `np.arange`, `np.linspace`, `np.eye`, ...). */
trait NpCreation:

  // ------------------------------------------------------------------ from data

  /** Creates an array from a scalar or (nested) `Seq`/`Array`/`NDArray`:
    * `np.array(Seq(Seq(1.0, 2.0), Seq(3.0, 4.0)))`. A scalar gives a 0-d array.
    */
  def array[A, E](a: A)(using n: Nested[A, E]): NDArray[E] =
    val shape = n.shapeOf(a).toArray
    val out = n.dtype.newArray(Shape.size(shape))
    n.write(a, out, 0)
    NDArray.fromArray(out, shape)(using n.dtype)

  /** `np.array(1, 2, 3)` or `np.array(Seq(1, 2), Seq(3, 4))`: the arguments become the first axis. */
  def array[A, E](x0: A, x1: A, rest: A*)(using n: Nested[A, E]): NDArray[E] =
    array[Seq[A], E](Seq(x0, x1) ++ rest)(using Nested.seq[A, E](using n))

  /** Converts to an array without copying if `a` already is one (`np.asarray`). */
  def asarray[A, E](a: A)(using n: Nested[A, E]): NDArray[E] = a match
    case arr: NDArray[?] => arr.asInstanceOf[NDArray[E]]
    case _ => array(a)

  /** Array from data converted to a dtype: `np.arrayOf(Seq(1, 2), DType.Float64)` (`np.array(x, dtype=...)`). */
  def arrayOf[A, E, U](a: A, dtype: DType[U])(using n: Nested[A, E]): NDArray[U] = array(a).asType(using dtype)

  def ascontiguousarray[T](a: NDArray[T]): NDArray[T] = if a.ndim == 0 then a.reshape(1) else a.contiguous
  def asfortranarray[T](a: NDArray[T]): NDArray[T] = a.copy()
  def copy[T](a: NDArray[T]): NDArray[T] = a.copy()

  /** Wraps a flat Scala array (no copy) with an optional shape. */
  def fromArray[T](data: Array[T], shape: Int*)(using d: DType[T]): NDArray[T] =
    NDArray.fromArray(data, if shape.isEmpty then Array(data.length) else Shape.resolve(shape.toArray, data.length))

  /** `np.fromiter`. */
  def fromiter[T](it: IterableOnce[T])(using d: DType[T]): NDArray[T] =
    val a = it.iterator.toArray(using d.classTag)
    NDArray.fromArray(a)

  /** `np.fromfunction`: element value computed from its multi-index. */
  def fromfunction[T](shape: Int*)(f: Seq[Int] => T)(using d: DType[T]): NDArray[T] =
    val sh = shape.toArray
    val n = Shape.size(sh)
    val out = d.newArray(n)
    val idx = new Array[Int](sh.length)
    var k = 0
    while k < n do
      out(k) = f(scala.collection.immutable.ArraySeq.unsafeWrapArray(idx.clone()))
      var ax = sh.length - 1
      var carry = true
      while carry && ax >= 0 do
        idx(ax) += 1
        if idx(ax) < sh(ax) then carry = false
        else
          idx(ax) = 0
          ax -= 1
      k += 1
    NDArray.fromArray(out, sh)

  /** `np.fromstring(s, sep=...)` for text data. */
  def fromstring[T](s: String, sep: String = " ")(using d: DType[T]): NDArray[T] =
    val parts =
      if sep.trim.isEmpty then s.trim.split("\\s+").filter(_.nonEmpty)
      else s.split(java.util.regex.Pattern.quote(sep)).map(_.trim).filter(_.nonEmpty)
    NDArray.fromArray(parts.map(d.fromString)(using d.classTag))

  // ------------------------------------------------------------------ filled arrays

  def zeros[T](shape: Int*)(using d: DefaultDType[T]): NDArray[T] = NDArray.zerosOf(d.dtype, shape.toArray)
  def ones[T](shape: Int*)(using d: DefaultDType[T]): NDArray[T] =
    NDArray.fillOf(d.dtype, shape.toArray, d.dtype.one)
  /** NumPy's `empty` (initialised to zeros here, as the JVM always zeroes memory). */
  def empty[T](shape: Int*)(using d: DefaultDType[T]): NDArray[T] = zeros[T](shape*)
  def full[T](shape: Seq[Int], fillValue: T)(using d: DType[T]): NDArray[T] =
    NDArray.fillOf(d, shape.toArray, fillValue)

  def zeros_like[T](a: NDArray[T]): NDArray[T] = NDArray.zerosOf(a.dtype, a.shapeArr.clone())
  def ones_like[T](a: NDArray[T]): NDArray[T] = NDArray.fillOf(a.dtype, a.shapeArr.clone(), a.dtype.one)
  def empty_like[T](a: NDArray[T]): NDArray[T] = zeros_like(a)
  def full_like[T](a: NDArray[T], fillValue: T): NDArray[T] = NDArray.fillOf(a.dtype, a.shapeArr.clone(), fillValue)
  def zerosLikeOf[T, U](a: NDArray[T])(using u: DType[U]): NDArray[U] = NDArray.zerosOf(u, a.shapeArr.clone())

  // ------------------------------------------------------------------ ranges

  /** `np.arange(stop)` for integers: `[0, 1, ..., stop-1]`. */
  def arange(stop: Int): NDArray[Int] = arange(0, stop, 1)
  def arange(start: Int, stop: Int): NDArray[Int] = arange(start, stop, 1)
  def arange(start: Int, stop: Int, step: Int): NDArray[Int] =
    if step == 0 then throw new IllegalArgumentException("Maximum allowed size exceeded")
    val n = math.max(0L, math.ceil((stop.toLong - start).toDouble / step).toLong).toInt
    NDArray.tabulate(n)(i => start + i * step)(using DType.Int32)
  def arange(stop: Long): NDArray[Long] = arange(0L, stop, 1L)
  def arange(start: Long, stop: Long, step: Long): NDArray[Long] =
    if step == 0 then throw new IllegalArgumentException("Maximum allowed size exceeded")
    val n = math.max(0L, math.ceil((stop - start).toDouble / step).toLong).toInt
    NDArray.tabulate(n)(i => start + i * step)(using DType.Int64)
  /** `np.arange(stop)` for floats. */
  def arange(stop: Double): NDArray[Double] = arange(0.0, stop, 1.0)
  def arange(start: Double, stop: Double): NDArray[Double] = arange(start, stop, 1.0)
  def arange(start: Double, stop: Double, step: Double): NDArray[Double] =
    if step == 0.0 then throw new IllegalArgumentException("Maximum allowed size exceeded")
    val n = math.max(0.0, math.ceil((stop - start) / step)).toInt
    NDArray.tabulate(n)(i => start + i * step)(using DType.Float64)

  /** `num` evenly spaced samples over `[start, stop]` (`np.linspace`). */
  def linspace(start: Double, stop: Double, num: Int = 50, endpoint: Boolean = true): NDArray[Double] =
    linspaceWithStep(start, stop, num, endpoint)._1

  /** `np.linspace(..., retstep=True)`. */
  def linspaceWithStep(start: Double, stop: Double, num: Int, endpoint: Boolean = true): (NDArray[Double], Double) =
    if num < 0 then throw new IllegalArgumentException(s"Number of samples, $num, must be non-negative.")
    val div = if endpoint then num - 1 else num
    val delta = stop - start
    val step = if div > 0 then delta / div else Double.NaN
    val out = new Array[Double](num)
    var i = 0
    while i < num do
      out(i) =
        if div > 0 then
          if step == 0.0 then (i.toDouble / div) * delta + start else start + i * step
        else start + i * delta
      i += 1
    if endpoint && num > 1 then out(num - 1) = stop
    (NDArray.fromArray(out), if div > 0 then step else Double.NaN)

  /** `np.linspace` between two arrays: samples stacked along a new first axis. */
  def linspace(start: NDArray[Double], stop: NDArray[Double], num: Int, endpoint: Boolean): NDArray[Double] =
    val t = linspace(0.0, 1.0, num, endpoint)
    val sh = Shape.broadcast(start.shapeArr, stop.shapeArr)
    val s = start.broadcastTo(sh*)
    val e = stop.broadcastTo(sh*)
    val tt = t.reshape((num +: Array.fill(sh.length)(1).toSeq)*)
    val res = tt * (e - s) + s
    // force exact endpoints like NumPy
    if endpoint && num > 1 then res(num - 1, ---) = e.copy()
    res

  def logspace(start: Double, stop: Double, num: Int = 50, endpoint: Boolean = true, base: Double = 10.0): NDArray[Double] =
    linspace(start, stop, num, endpoint).map(x => math.pow(base, x))

  def geomspace(start: Double, stop: Double, num: Int = 50, endpoint: Boolean = true): NDArray[Double] =
    if start == 0.0 || stop == 0.0 then throw new IllegalArgumentException("Geometric sequence cannot include zero")
    val flip = start < 0 && stop < 0
    val (s, e) = if flip then (-start, -stop) else (start, stop)
    if s < 0 || e < 0 then
      throw new IllegalArgumentException("geomspace of mixed-sign real endpoints requires complex numbers")
    val r = logspace(math.log10(s), math.log10(e), num, endpoint).toArray
    if num > 0 then r(0) = s
    if num > 1 && endpoint then r(num - 1) = e
    val out = NDArray.fromArray(r)
    if flip then out.map(-_) else out

  // ------------------------------------------------------------------ matrices

  /** 2-D array with ones on the `k`-th diagonal (`np.eye`). */
  def eye[T](n: Int, m: Int = -1, k: Int = 0)(using d: DefaultDType[T]): NDArray[T] =
    val cols = if m < 0 then n else m
    val a = NDArray.zerosOf(d.dtype, Array(n, cols))
    var i = 0
    while i < n do
      val j = i + k
      if j >= 0 && j < cols then a.data(i * cols + j) = d.dtype.one
      i += 1
    a

  def identity[T](n: Int)(using d: DefaultDType[T]): NDArray[T] = eye[T](n)

  /** Extracts a diagonal (2-D input) or builds a diagonal matrix (1-D input) (`np.diag`). */
  def diag[T](v: NDArray[T], k: Int = 0): NDArray[T] =
    v.ndim match
      case 1 =>
        val n = v.shapeArr(0) + math.abs(k)
        val out = NDArray.zerosOf(v.dtype, Array(n, n))
        val src = v.toArray
        var i = 0
        while i < src.length do
          val (r, c) = if k >= 0 then (i, i + k) else (i - k, i)
          out.data(r * n + c) = src(i)
          i += 1
        out
      case 2 => v.diagonal(k).copy()
      case _ => throw new IllegalArgumentException("Input must be 1- or 2-d.")

  /** 2-D array with the flattened input as a diagonal (`np.diagflat`). */
  def diagflat[T](v: NDArray[T], k: Int = 0): NDArray[T] = diag(v.ravel(), k)

  /** Lower-triangular matrix of ones (`np.tri`). */
  def tri[T](n: Int, m: Int = -1, k: Int = 0)(using d: DefaultDType[T]): NDArray[T] =
    val cols = if m < 0 then n else m
    val a = NDArray.zerosOf(d.dtype, Array(n, cols))
    for i <- 0 until n; j <- 0 until cols if j <= i + k do a.data(i * cols + j) = d.dtype.one
    a

  /** Lower triangle of an array (`np.tril`); works on the last two axes. */
  def tril[T](m: NDArray[T], k: Int = 0): NDArray[T] = triMask(m, k, lower = true)
  /** Upper triangle of an array (`np.triu`). */
  def triu[T](m: NDArray[T], k: Int = 0): NDArray[T] = triMask(m, k, lower = false)

  private def triMask[T](m: NDArray[T], k: Int, lower: Boolean): NDArray[T] =
    val a = if m.ndim == 1 then m.reshape(1, m.shapeArr(0)).broadcastTo(m.shapeArr(0), m.shapeArr(0)).copy() else m.copy()
    val rows = a.shapeArr(a.ndim - 2)
    val cols = a.shapeArr(a.ndim - 1)
    val n = a.size
    val z = a.dtype.zero
    var p = 0
    while p < n do
      val c = p % cols
      val r = (p / cols) % rows
      val keep = if lower then c <= r + k else c >= r + k
      if !keep then a.data(p) = z
      p += 1
    a

  /** Vandermonde matrix (`np.vander`); decreasing powers by default. */
  def vander[T](x: NDArray[T], n: Int = -1, increasing: Boolean = false)(using d: NumDType[T]): NDArray[T] =
    if x.ndim != 1 then throw new IllegalArgumentException("x must be a one-dimensional array or sequence.")
    val len = x.shapeArr(0)
    val cols = if n < 0 then len else n
    val xs = x.toArray
    val out = d.newArray(len * cols)
    for i <- 0 until len do
      var acc = d.one
      for j <- 0 until cols do
        val col = if increasing then j else cols - 1 - j
        out(i * cols + col) = acc
        acc = d.times(acc, xs(i))
    NDArray.fromArray(out, Array(len, cols))

  // ------------------------------------------------------------------ grids

  /** Coordinate matrices from coordinate vectors (`np.meshgrid`). `indexing` is "xy" or "ij". */
  def meshgrid[T](xs: Seq[NDArray[T]], indexing: String = "xy", sparse: Boolean = false, copy: Boolean = true): Seq[NDArray[T]] =
    val nd = xs.length
    if indexing != "xy" && indexing != "ij" then throw new IllegalArgumentException("Valid values for `indexing` are 'xy' and 'ij'.")
    val lens = xs.map(_.size).toArray
    val outputShape = lens.clone()
    if indexing == "xy" && nd > 1 then
      outputShape(0) = lens(1)
      outputShape(1) = lens(0)
    xs.zipWithIndex.map { (x, i) =>
      val sh = Array.fill(nd)(1)
      val pos = if indexing == "xy" && nd > 1 && i < 2 then 1 - i else i
      sh(pos) = x.size
      val r = x.ravel().reshape(sh.toSeq*)
      if sparse then (if copy then r.copy() else r)
      else
        val b = r.broadcastTo(outputShape.toSeq*)
        if copy then b.copy() else b
    }

  def meshgrid[T](x: NDArray[T], y: NDArray[T]): (NDArray[T], NDArray[T]) =
    val r = meshgrid(Seq(x, y))
    (r(0), r(1))

  /** Dense multi-dimensional "meshgrid" of integer ranges (`np.mgrid[0:a, 0:b]`). */
  def mgrid(ranges: Range*): NDArray[Int] =
    val vs = ranges.map(r => NDArray.fromArray(r.toArray))
    stackFirst(meshgrid(vs, indexing = "ij"))

  /** `np.mgrid` with float ranges given as `(start, stop, step)` triples. */
  def mgridD(ranges: (Double, Double, Double)*): NDArray[Double] =
    val vs = ranges.map((a, b, s) => arange(a, b, s))
    stackFirst(meshgrid(vs, indexing = "ij"))

  /** Open (sparse) grid (`np.ogrid`). */
  def ogrid(ranges: Range*): Seq[NDArray[Int]] =
    meshgrid(ranges.map(r => NDArray.fromArray(r.toArray)), indexing = "ij", sparse = true)

  /** Grid indices (`np.indices`): shape `(len(dims), *dims)`. */
  def indices(dims: Int*): NDArray[Int] =
    stackFirst(meshgrid(dims.map(n => arange(n)), indexing = "ij"))

  private[numscala] def stackFirst[T](arrays: Seq[NDArray[T]]): NDArray[T] =
    if arrays.isEmpty then throw new IllegalArgumentException("need at least one array to stack")
    val d = arrays.head.dtype
    val sh = arrays.head.shapeArr
    val n = Shape.size(sh)
    val out = d.newArray(n * arrays.length)
    arrays.zipWithIndex.foreach { (a, i) =>
      if !java.util.Arrays.equals(a.shapeArr, sh) then throw new IllegalArgumentException("all input arrays must have the same shape")
      System.arraycopy(a.toArray, 0, out, i * n, n)
    }
    NDArray.fromArray(out, arrays.length +: sh)(using d)
