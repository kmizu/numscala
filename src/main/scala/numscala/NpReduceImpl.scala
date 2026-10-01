package numscala

/** Internal helpers shared by the reduction and statistics modules. */
private[numscala] object NpReduceImpl:

  /** Normalized reduction axes; `null` means every axis. */
  def axesOf(axis: Axis | Null, ndim: Int): Array[Int] =
    if axis == null then Array.tabulate(ndim)(identity)
    else normAxes(axis.asInstanceOf[Axis], ndim)

  /** `Lanes.reduceAxes` with an optional axis. */
  def reduceL[T, U](a: NDArray[T], axis: Axis | Null, keepdims: Boolean)(f: (Array[T], Int) => U)(using
      u: DType[U]
  ): NDArray[U] =
    Lanes.reduceAxes(a, axesOf(axis, a.ndim), keepdims)(f)

  /** Output shape of a reduction over `ax`. */
  def reducedShape(shape: Array[Int], ax: Array[Int], keepdims: Boolean): Array[Int] =
    val s = ax.toSet
    if keepdims then shape.indices.map(i => if s(i) then 1 else shape(i)).toArray
    else shape.indices.filterNot(s).map(shape(_)).toArray

  /** Gathers the lanes of `a` over the axes `ax` into a row-major `(m, l)` buffer.
    * Returns `(buffer, m, l)`; row `i` holds lane `i` in C order of the remaining axes.
    */
  def gather[T](a: NDArray[T], ax: Array[Int]): (Array[T], Int, Int) =
    val s = ax.toSet
    val keep = (0 until a.ndim).filterNot(s)
    val perm = keep ++ ax.toSeq
    val t = if perm == (0 until a.ndim) then a else a.transpose(perm*)
    val l = Shape.size(ax.map(a.shapeArr(_)))
    val m = Shape.size(keep.map(a.shapeArr(_)).toArray)
    (t.toArray, m, l)

  /** Converts a buffer to doubles. */
  def toDoubles[T](d: DType[T], arr: Array[T]): Array[Double] =
    arr match
      case x: Array[Double] => x
      case _ =>
        val out = new Array[Double](arr.length)
        var i = 0
        while i < arr.length do
          out(i) = d.toDouble(arr(i))
          i += 1
        out

  /** All elements of `a` as doubles in C order (rejects complex input). */
  def doublesOf(a: NDArray[?], what: String = "array"): Array[Double] =
    val d = a.dtype.asInstanceOf[DType[Any]]
    if d.isComplex then throw new IllegalArgumentException(s"$what must be real, got complex")
    if d.isString then throw new IllegalArgumentException(s"$what must be numeric, got str")
    a.asInstanceOf[NDArray[Any]].astypeDyn(DType.Float64).asInstanceOf[NDArray[Double]].toArray

  /** Copies the non-NaN entries of `buf(0 until n)` into `dst`; returns their count. */
  def compactNonNaN[T](d: DType[T], buf: Array[T], n: Int, dst: Array[T]): Int =
    var k = 0
    var i = 0
    while i < n do
      val x = buf(i)
      if !d.isNaN(x) then
        dst(k) = x
        k += 1
      i += 1
    k

  /** Python-style floor modulo on doubles. */
  def pyMod(a: Double, b: Double): Double =
    val r = a % b
    if r != 0.0 && ((r < 0) != (b < 0)) then r + b else r

  /** First index `i` with `arr(i) >= x` (`side='left'`) or `arr(i) > x` (`'right'`), NaN sorted last. */
  def searchSorted(arr: Array[Double], x: Double, right: Boolean): Int =
    var lo = 0
    var hi = arr.length
    while lo < hi do
      val mid = (lo + hi) >>> 1
      val c = Format.compareDouble(arr(mid), x)
      if (if right then c <= 0 else c < 0) then lo = mid + 1 else hi = mid
    lo

  /** Concatenates two arrays along `axis` (shapes must agree elsewhere). */
  def concat[T](a: NDArray[T], b: NDArray[T], axis: Int): NDArray[T] =
    val ax = Shape.normAxis(axis, a.ndim)
    if a.ndim != b.ndim then
      throw new IllegalArgumentException(
        "all the input array dimensions except for the concatenation axis must match exactly"
      )
    var i = 0
    while i < a.ndim do
      if i != ax && a.shapeArr(i) != b.shapeArr(i) then
        throw new IllegalArgumentException(
          s"all the input array dimensions except for the concatenation axis must match exactly, but along dimension $i, the array at index 0 has size ${a.shapeArr(i)} and the array at index 1 has size ${b.shapeArr(i)}"
        )
      i += 1
    val na = a.shapeArr(ax)
    val nb = b.shapeArr(ax)
    val sh = a.shapeArr.clone()
    sh(ax) = na + nb
    val outer = Shape.size(a.shapeArr.take(ax))
    val innerA = Shape.size(a.shapeArr.drop(ax))
    val innerB = Shape.size(b.shapeArr.drop(ax))
    val da = a.toArray
    val db = b.toArray
    val out = a.dtype.newArray(Shape.size(sh))
    var k = 0
    var o = 0
    while o < outer do
      System.arraycopy(da, o * innerA, out, k, innerA)
      k += innerA
      System.arraycopy(db, o * innerB, out, k, innerB)
      k += innerB
      o += 1
    NDArray.fromArray(out, sh)(using a.dtype)

  /** Broadcasts a 0-d `p` to `a`'s shape with length 1 along `axis` (used by `diff`). */
  def edgeArray[T](p: NDArray[T], like: NDArray[T], axis: Int): NDArray[T] =
    if p.ndim == 0 then
      val sh = like.shapeArr.clone()
      sh(axis) = 1
      p.broadcastTo(sh*).copy()
    else p

  /** A 1-d double array. */
  def vecD(xs: Array[Double]): NDArray[Double] = NDArray.fromArray(xs, Array(xs.length))
