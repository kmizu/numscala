package numscala

/** Core sorting and searching kernels. */
private[numscala] object Sorting:

  /** Sorts `out(0 until n)` in place in NumPy order (NaN last). */
  def sortBuffer[T](d: DType[T], out: Array[T], n: Int): Unit =
    (out: Any) match
      case x: Array[Double] => java.util.Arrays.sort(x, 0, n)
      case x: Array[Float] => java.util.Arrays.sort(x, 0, n)
      case x: Array[Int] => java.util.Arrays.sort(x, 0, n)
      case x: Array[Long] => java.util.Arrays.sort(x, 0, n)
      case x: Array[Short] => java.util.Arrays.sort(x, 0, n)
      case x: Array[Byte] => java.util.Arrays.sort(x, 0, n)
      case x: Array[Boolean] =>
        var f = 0
        var i = 0
        while i < n do
          if !x(i) then f += 1
          i += 1
        i = 0
        while i < n do
          x(i) = i >= f
          i += 1
      case x: Array[AnyRef] =>
        java.util.Arrays.sort(x, 0, n, d.ordering.asInstanceOf[Ordering[AnyRef]])
      case _ => throw new IllegalStateException("unexpected buffer type")

  def sort[T](a: NDArray[T], axis: Int): NDArray[T] =
    given DType[T] = a.dtype
    if a.ndim == 0 then a.copy()
    else
      Lanes.transform(a, axis) { (in: Array[T], n: Int, out: Array[T]) =>
        System.arraycopy(in, 0, out, 0, n)
        sortBuffer(a.dtype, out, n)
      }

  /** Stable argsort of `keys(0 until n)` into `out`. */
  def argsortBuffer[T](d: DType[T], keys: Array[T], n: Int, out: Array[Int]): Unit =
    var i = 0
    while i < n do
      out(i) = i
      i += 1
    val tmp = new Array[Int](n)
    (keys: Any) match
      case k: Array[Double] => mergeSortD(out, tmp, 0, n, k)
      case _ =>
        d.kind match
          case 'i' | 'b' =>
            val lk = new Array[Long](n)
            var j = 0
            while j < n do
              lk(j) = d.toLong(keys(j))
              j += 1
            mergeSortL(out, tmp, 0, n, lk)
          case 'f' =>
            val dk = new Array[Double](n)
            var j = 0
            while j < n do
              dk(j) = d.toDouble(keys(j))
              j += 1
            mergeSortD(out, tmp, 0, n, dk)
          case _ =>
            val ord = d.ordering
            mergeSortG(out, tmp, 0, n, (x: Int, y: Int) => ord.compare(keys(x), keys(y)))

  private inline def mergeSortImpl(idx: Array[Int], tmp: Array[Int], lo: Int, hi: Int)(inline cmp: (Int, Int) => Int): Unit =
    // bottom-up stable merge sort with insertion sort for small runs
    val run = 16
    var s = lo
    while s < hi do
      val e = math.min(s + run, hi)
      var i = s + 1
      while i < e do
        val v = idx(i)
        var j = i - 1
        while j >= s && cmp(idx(j), v) > 0 do
          idx(j + 1) = idx(j)
          j -= 1
        idx(j + 1) = v
        i += 1
      s = e
    var width = run
    var src = idx
    var dst = tmp
    while width < hi - lo do
      var left = lo
      while left < hi do
        val mid = math.min(left + width, hi)
        val right = math.min(left + 2 * width, hi)
        var i = left
        var j = mid
        var k = left
        while i < mid && j < right do
          if cmp(src(j), src(i)) < 0 then
            dst(k) = src(j)
            j += 1
          else
            dst(k) = src(i)
            i += 1
          k += 1
        while i < mid do
          dst(k) = src(i)
          i += 1
          k += 1
        while j < right do
          dst(k) = src(j)
          j += 1
          k += 1
        left = right
      val t = src
      src = dst
      dst = t
      width *= 2
    if !(src eq idx) then System.arraycopy(src, lo, idx, lo, hi - lo)

  private def mergeSortD(idx: Array[Int], tmp: Array[Int], lo: Int, hi: Int, k: Array[Double]): Unit =
    mergeSortImpl(idx, tmp, lo, hi)((x, y) => Format.compareDouble(k(x), k(y)))
  private def mergeSortL(idx: Array[Int], tmp: Array[Int], lo: Int, hi: Int, k: Array[Long]): Unit =
    mergeSortImpl(idx, tmp, lo, hi)((x, y) => java.lang.Long.compare(k(x), k(y)))
  private def mergeSortG(idx: Array[Int], tmp: Array[Int], lo: Int, hi: Int, c: (Int, Int) => Int): Unit =
    mergeSortImpl(idx, tmp, lo, hi)((x, y) => c(x, y))

  def argsort[T](a: NDArray[T], axis: Int): NDArray[Int] =
    if a.ndim == 0 then NDArray.fromArray(Array(0), Array.emptyIntArray)
    else
      Lanes.transform(a, axis) { (in: Array[T], n: Int, out: Array[Int]) =>
        argsortBuffer(a.dtype, in, n, out)
      }(using DType.Int32)

private[numscala] object Searching:
  /** Indices of the non-zero elements, one array per dimension (`np.nonzero`). */
  def nonzero[T](a: NDArray[T]): Seq[NDArray[Int]] =
    val d = a.dtype
    if a.ndim == 0 then
      val n = if d.toBoolean(a.item) then 1 else 0
      return Seq(NDArray.fromArray(new Array[Int](n), Array(n)))
    val nd = a.ndim
    val buf = Array.fill(nd)(scala.collection.mutable.ArrayBuilder.make[Int])
    val idx = new Array[Int](nd)
    var k = 0
    val data = a.data
    a.foreachOffset { o =>
      if d.toBoolean(data(o)) then
        var r = k
        var ax = nd - 1
        while ax >= 0 do
          idx(ax) = r % a.shapeArr(ax)
          r /= a.shapeArr(ax)
          ax -= 1
        var j = 0
        while j < nd do
          buf(j) += idx(j)
          j += 1
      k += 1
    }
    buf.toSeq.map(b => NDArray.fromArray(b.result())(using DType.Int32))
