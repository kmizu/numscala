package numscala

/** Core sorting and searching kernels. */
private[numscala] object Sorting:
  def sort[T](a: NDArray[T], axis: Int): NDArray[T] =
    given DType[T] = a.dtype
    given scala.reflect.ClassTag[T] = a.dtype.classTag
    val ord = a.dtype.ordering
    if a.ndim == 0 then a.copy()
    else
      Lanes.transform(a, axis) { (in: Array[T], n: Int, out: Array[T]) =>
        System.arraycopy(in, 0, out, 0, n)
        scala.util.Sorting.stableSort(out, (x: T, y: T) => ord.compare(x, y) < 0)
      }

  def argsort[T](a: NDArray[T], axis: Int): NDArray[Int] =
    val ord = a.dtype.ordering
    if a.ndim == 0 then NDArray.fromArray(Array(0), Array.emptyIntArray)
    else
      Lanes.transform(a, axis) { (in: Array[T], n: Int, out: Array[Int]) =>
        val idx = Array.tabulate(n)(identity).map(Integer.valueOf)
        java.util.Arrays.sort(idx, (x: Integer, y: Integer) => ord.compare(in(x), in(y)))
        var i = 0
        while i < n do
          out(i) = idx(i).intValue
          i += 1
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
    val total = a.size
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
    val _ = total
    buf.toSeq.map(b => NDArray.fromArray(b.result())(using DType.Int32))
