package com.github.kmizu.numscala

/** Internal kernels shared by the sorting, searching, set and indexing routines. */
private[numscala] object NpSortImpl:

  /** `Some(axis)` for an integer axis, `None` for NumPy's `axis=None`. */
  def axisOpt(axis: Int | None.type): Option[Int] = axis match
    case i: Int => Some(i)
    case _ => None

  /** Index-normalization mode (`mode=` of `take`, `put`, `choose`, `ravel_multi_index`). */
  enum Mode:
    case Raise, Wrap, Clip

  def parseMode(mode: String): Mode = mode match
    case "raise" => Mode.Raise
    case "wrap" => Mode.Wrap
    case "clip" => Mode.Clip
    case other => throw new IllegalArgumentException(s"clipmode must be one of 'clip', 'raise', or 'wrap' (got '$other')")

  /** Normalizes index `i` for an axis of length `n` according to `mode`; `raise` accepts
    * negative indices (Python style) when `allowNeg` is set.
    */
  def normIndex(i: Int, n: Int, mode: Mode, allowNeg: Boolean, axis: Int = 0): Int = mode match
    case Mode.Raise =>
      val j = if allowNeg && i < 0 then i + n else i
      if j < 0 || j >= n then
        throw new IndexOutOfBoundsException(s"index $i is out of bounds for axis $axis with size $n")
      j
    case Mode.Wrap =>
      if n == 0 then throw new IndexOutOfBoundsException("cannot do a non-empty take from an empty axes.")
      Math.floorMod(i, n)
    case Mode.Clip =>
      if n == 0 then throw new IndexOutOfBoundsException("cannot do a non-empty take from an empty axes.")
      if i < 0 then 0 else if i >= n then n - 1 else i

  /** Comparison used by the set routines: like NumPy's sort order, where complex values
    * containing NaN go last, in the order `R+nanj < nan+Rj < nan+nanj`.
    */
  def setCompare[T](d: DType[T]): (T, T) => Int =
    if d.isComplex then
      (x: T, y: T) =>
        val a = x.asInstanceOf[Complex]
        val b = y.asInstanceOf[Complex]
        def cls(c: Complex) = (if c.re.isNaN then 2 else 0) + (if c.im.isNaN then 1 else 0)
        val ca = cls(a)
        val cb = cls(b)
        if ca != cb then Integer.compare(ca, cb)
        else
          val r = Format.compareDouble(a.re, b.re)
          if r != 0 then r else Format.compareDouble(a.im, b.im)
    else (x: T, y: T) => d.compare(x, y)

  /** Stable argsort of a flat buffer with an explicit comparison. */
  def argsortBy[T](vals: Array[T], cmp: (T, T) => Int): Array[Int] =
    val n = vals.length
    val idx = new Array[Integer](n)
    var i = 0
    while i < n do
      idx(i) = Integer.valueOf(i)
      i += 1
    java.util.Arrays.sort(idx, (x: Integer, y: Integer) => cmp(vals(x.intValue), vals(y.intValue)))
    val out = new Array[Int](n)
    i = 0
    while i < n do
      out(i) = idx(i).intValue
      i += 1
    out

  /** Stable in-place sort of a buffer with an explicit comparison. */
  def sortInPlace[T](arr: Array[T], cmp: (T, T) => Int): Unit =
    val perm = argsortBy(arr, cmp)
    val copy = arr.clone()
    var i = 0
    while i < perm.length do
      arr(i) = copy(perm(i))
      i += 1

  /** Calls `f(base, stride)` for each lane along axis `ax` of a C-contiguous buffer of shape `sh`. */
  def foreachLane(sh: Array[Int], ax: Int)(f: (Int, Int) => Unit): Unit =
    val st = Shape.cStrides(sh)
    val keep = (0 until sh.length).filter(_ != ax).toArray
    val s = st(ax)
    Strided.foreach1(keep.map(sh(_)), keep.map(st(_)), 0)(b => f(b, s))

  // ------------------------------------------------------------------ selection (partition)

  /** Introselect: rearranges `arr(lo0 until hi0)` so that `arr(k)` holds the element that
    * would be there if the range were sorted, with no larger element before it and no
    * smaller element after it.  Falls back to sorting when recursion gets too deep.
    */
  def select[T](arr: Array[T], lo0: Int, hi0: Int, k: Int, cmp: (T, T) => Int): Unit =
    var lo = lo0
    var hi = hi0 - 1
    var depth = 2 * (32 - Integer.numberOfLeadingZeros(math.max(hi0 - lo0, 1)))
    var done = false
    while !done && hi > lo do
      if hi - lo < 16 then
        insertionSort(arr, lo, hi, cmp)
        done = true
      else if depth == 0 then
        val slice = arr.slice(lo, hi + 1)
        sortInPlace(slice, cmp)
        System.arraycopy(slice, 0, arr, lo, slice.length)
        done = true
      else
        depth -= 1
        val mid = (lo + hi) >>> 1
        if cmp(arr(mid), arr(lo)) < 0 then swap(arr, mid, lo)
        if cmp(arr(hi), arr(lo)) < 0 then swap(arr, hi, lo)
        if cmp(arr(hi), arr(mid)) < 0 then swap(arr, hi, mid)
        val pivot = arr(mid)
        var i = lo
        var j = hi
        while i <= j do
          while cmp(arr(i), pivot) < 0 do i += 1
          while cmp(arr(j), pivot) > 0 do j -= 1
          if i <= j then
            swap(arr, i, j)
            i += 1
            j -= 1
        if k <= j then hi = j
        else if k >= i then lo = i
        else done = true

  private def insertionSort[T](arr: Array[T], lo: Int, hi: Int, cmp: (T, T) => Int): Unit =
    var i = lo + 1
    while i <= hi do
      val x = arr(i)
      var j = i - 1
      while j >= lo && cmp(arr(j), x) > 0 do
        arr(j + 1) = arr(j)
        j -= 1
      arr(j + 1) = x
      i += 1

  private inline def swap[T](arr: Array[T], i: Int, j: Int): Unit =
    val t = arr(i)
    arr(i) = arr(j)
    arr(j) = t

  /** Normalized, sorted, distinct `kth` values for an axis of length `n`. */
  def kthList(kth: Int | Seq[Int], n: Int): Array[Int] =
    val raw: Seq[Int] = kth match
      case i: Int => Seq(i)
      case s: Seq[?] => s.asInstanceOf[Seq[Int]]
    raw.map { k =>
      val j = if k < 0 then k + n else k
      if j < 0 || j >= n then throw new IllegalArgumentException(s"kth(=$k) out of bounds ($n)")
      j
    }.distinct.sorted.toArray

  /** Partitions `arr(0 until n)` around every kth. */
  def multiSelect[T](arr: Array[T], n: Int, ks: Array[Int], cmp: (T, T) => Int): Unit =
    var lo = 0
    for k <- ks do
      select(arr, lo, n, k, cmp)
      lo = k + 1

  // ------------------------------------------------------------------ binary search

  /** Insertion point of `v` in the sorted sequence `get(0 until n)`. */
  inline def bisect[T](n: Int, get: Int => T, v: T, right: Boolean, cmp: (T, T) => Int): Int =
    var lo = 0
    var hi = n
    while lo < hi do
      val mid = (lo + hi) >>> 1
      val c = cmp(get(mid), v)
      if (if right then c <= 0 else c < 0) then lo = mid + 1 else hi = mid
    lo

  /** Truthiness of each element of any array, broadcast to `shape` (C order). */
  def truthy(a: NDArray[?], shape: Array[Int]): Array[Boolean] = truthyT(a, shape)

  private def truthyT[T](a0: NDArray[T], shape: Array[Int]): Array[Boolean] =
    val d = a0.dtype
    val v = a0.broadcastTo(shape*)
    val out = new Array[Boolean](Shape.size(shape))
    var k = 0
    val data = v.data
    v.foreachOffset { o =>
      out(k) = d.toBoolean(data(o))
      k += 1
    }
    out

  /** Element accessor (C order, broadcast to `shape`) and comparison for an array of unknown dtype. */
  def keyAccess(a: NDArray[?], shape: Array[Int], flatShape: Array[Int]): (Int, Int) => Int = keyAccessT(a, shape, flatShape)

  private def keyAccessT[T](a: NDArray[T], shape: Array[Int], flatShape: Array[Int]): (Int, Int) => Int =
    val arr: Array[T] = a.broadcastTo(shape*).reshapeArr(flatShape).toArray
    val d = a.dtype
    (i, j) => d.compare(arr(i), arr(j))
