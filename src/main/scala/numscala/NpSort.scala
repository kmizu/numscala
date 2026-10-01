package numscala

import NpSortImpl.*

/** Sorting and searching routines (`np.sort`, `np.argsort`, `np.partition`,
  * `np.searchsorted`, `np.where`, `np.nonzero`, ...).  Index-picking routines
  * (`take`, `put`, `choose`, ...) live in [[NpSortTake]].
  */
trait NpSort extends NpSortTake:

  // ------------------------------------------------------------------ sorting

  /** Sorted copy of an array along `axis` (`np.sort`); `axis = None` sorts the flattened array.
    * NaNs sort last. The sort is always stable, so `kind` is accepted only for compatibility.
    */
  def sort[T](a: NDArray[T], axis: Int | None.type = -1, kind: String = "quicksort"): NDArray[T] =
    checkKind(kind)
    axisOpt(axis) match
      case None => Sorting.sort(a.ravel(), 0)
      case Some(ax) =>
        if a.ndim == 0 then throw new IndexOutOfBoundsException(s"axis $ax is out of bounds for array of dimension 0")
        Sorting.sort(a, Shape.normAxis(ax, a.ndim))

  /** Indices that would sort an array along `axis` (`np.argsort`); stable, NaNs last. */
  def argsort[T](a: NDArray[T], axis: Int | None.type = -1, kind: String = "quicksort"): NDArray[Int] =
    checkKind(kind)
    axisOpt(axis) match
      case None => Sorting.argsort(a.ravel(), 0)
      case Some(ax) =>
        if a.ndim == 0 then throw new IndexOutOfBoundsException(s"axis $ax is out of bounds for array of dimension 0")
        Sorting.argsort(a, Shape.normAxis(ax, a.ndim))

  private def checkKind(kind: String): Unit = kind match
    case "quicksort" | "mergesort" | "heapsort" | "stable" => ()
    case other => throw new IllegalArgumentException(s"sort kind must be one of 'quick', 'heap', or 'stable' (got '$other')")

  /** Indirect stable sort on several keys (`np.lexsort`): the '''last''' key is the primary one. */
  def lexsort(keys: Seq[NDArray[?]], axis: Int = -1): NDArray[Int] =
    if keys.isEmpty then throw new IllegalArgumentException("need sequence of keys with len > 0 in lexsort")
    val sh0 = Shape.broadcast(keys.map(_.shapeArr)*)
    val sh = if sh0.isEmpty then Array(1) else sh0
    val ax = Shape.normAxis(axis, sh.length)
    val ks = keys.map(k => keyAccess(k, sh0, sh)).toArray
    val nk = ks.length
    val n = sh(ax)
    val out = new Array[Int](Shape.size(sh))
    foreachLane(sh, ax) { (base, st) =>
      val idx = new Array[Integer](n)
      var j = 0
      while j < n do
        idx(j) = Integer.valueOf(j)
        j += 1
      java.util.Arrays.sort(
        idx,
        (x: Integer, y: Integer) =>
          var c = 0
          var k = nk - 1
          while c == 0 && k >= 0 do
            c = ks(k)(base + x.intValue * st, base + y.intValue * st)
            k -= 1
          c
      )
      j = 0
      while j < n do
        out(base + j * st) = idx(j).intValue
        j += 1
    }
    NDArray.fromArray(out, sh0)

  /** `np.lexsort` with the keys given as the rows of a 2-D array. */
  def lexsort(keys: NDArray[?]): NDArray[Int] = lexsort(keys, -1)

  /** `np.lexsort(keys, axis)` with the keys stacked along the first axis of `keys`. */
  def lexsort(keys: NDArray[?], axis: Int): NDArray[Int] =
    if keys.ndim == 0 then throw new IllegalArgumentException("need sequence of keys with len > 0 in lexsort")
    lexsort(keys.iterator.toSeq, axis)

  /** Sorts a complex array by real part, then imaginary part (`np.sort_complex`). */
  def sort_complex[T](a: NDArray[T]): NDArray[Complex] =
    val c = a.asType(using DType.Complex128)
    if c.ndim == 0 then c.copy() else Sorting.sort(c, -1)

  /** Partial sort (`np.partition`): the element at each `kth` position is where it would be in
    * sorted order; smaller-or-equal elements come before it and greater-or-equal after it.
    * Uses introselect (linear expected time).
    */
  def partition[T](a: NDArray[T], kth: Int | Seq[Int], axis: Int | None.type = -1): NDArray[T] =
    val (src, ax) = axisOpt(axis) match
      case None => (a.ravel(), 0)
      case Some(x) =>
        if a.ndim == 0 then throw new IllegalArgumentException("Cannot partition a 0-d array")
        (a, Shape.normAxis(x, a.ndim))
    val n = src.shapeArr(ax)
    val ks = kthList(kth, n)
    val ord = src.dtype.ordering
    val cmp: (T, T) => Int = ord.compare
    given DType[T] = src.dtype
    Lanes.transform(src, ax) { (in: Array[T], len: Int, out: Array[T]) =>
      System.arraycopy(in, 0, out, 0, len)
      multiSelect(out, len, ks, cmp)
    }

  /** Indices that would partition an array (`np.argpartition`). */
  def argpartition[T](a: NDArray[T], kth: Int | Seq[Int], axis: Int | None.type = -1): NDArray[Int] =
    val (src, ax) = axisOpt(axis) match
      case None => (a.ravel(), 0)
      case Some(x) =>
        if a.ndim == 0 then throw new IllegalArgumentException("Cannot partition a 0-d array")
        (a, Shape.normAxis(x, a.ndim))
    val n = src.shapeArr(ax)
    val ks = kthList(kth, n)
    val ord = src.dtype.ordering
    Lanes.transform(src, ax) { (in: Array[T], len: Int, out: Array[Int]) =>
      var j = 0
      while j < len do
        out(j) = j
        j += 1
      multiSelect(out, len, ks, (x: Int, y: Int) => ord.compare(in(x), in(y)))
    }(using DType.Int32)

  // ------------------------------------------------------------------ searching

  /** Indices where elements of `v` should be inserted into the sorted 1-D array `a` to keep it
    * sorted (`np.searchsorted`). `side` is `"left"` or `"right"`; `sorter` optionally gives the
    * indices that sort `a`. Values are compared in the promoted dtype.
    */
  def searchsorted[T, U](
      a: NDArray[T],
      v: NDArray[U],
      side: String = "left",
      sorter: NDArray[Int] | Null = null
  )(using p: Promote[T, U]): NDArray[Int] =
    val od = p.dtype
    val sorted = sortedView(a, sorter, od)
    val right = parseSide(side)
    val cmp: (p.Out, p.Out) => Int = od.compare
    val vd = v.dtype
    v.map(x => bisect(sorted.length, sorted(_), od.castFrom(vd, x), right, cmp))(using DType.Int32)

  /** `np.searchsorted` for a single value; returns the insertion index. */
  def searchsorted[T](a: NDArray[T], v: T): Int = searchsorted(a, v, "left", null)

  /** `np.searchsorted(a, v, side)` for a single value. */
  def searchsorted[T](a: NDArray[T], v: T, side: String): Int = searchsorted(a, v, side, null)

  /** `np.searchsorted(a, v, side, sorter)` for a single value. */
  def searchsorted[T](a: NDArray[T], v: T, side: String, sorter: NDArray[Int] | Null): Int =
    val sorted = sortedView(a, sorter, a.dtype)
    val cmp: (T, T) => Int = a.dtype.compare
    bisect(sorted.length, sorted(_), v, parseSide(side), cmp)

  private def parseSide(side: String): Boolean = side match
    case "left" => false
    case "right" => true
    case other => throw new IllegalArgumentException(s"side must be 'left' or 'right' (got '$other')")

  private def sortedView[T, O](a: NDArray[T], sorter: NDArray[Int] | Null, od: DType[O]): Array[O] =
    if a.ndim != 1 then throw new IllegalArgumentException("object too deep for desired array (a must be 1-D)")
    val vals = a.toArray
    val ad = a.dtype
    sorter match
      case null => vals.map(x => od.castFrom(ad, x))(using od.classTag)
      case s: NDArray[Int] @unchecked =>
        if s.ndim != 1 || s.size != vals.length then throw new IllegalArgumentException("sorter.size must equal a.size")
        s.toArray.map { i =>
          if i < 0 || i >= vals.length then throw new IllegalArgumentException("Sorter index out of range.")
          od.castFrom(ad, vals(i))
        }(using od.classTag)

  /** Elements chosen from `x` where `condition` is true and from `y` elsewhere, with
    * broadcasting and dtype promotion (`np.where(condition, x, y)`).
    */
  def where[A, B](condition: NDArray[?], x: NDArray[A], y: NDArray[B])(using p: Promote[A, B]): NDArray[p.Out] =
    val od = p.dtype
    val c = condition.asInstanceOf[NDArray[Any]]
    val cd = c.dtype
    val xd = x.dtype
    val yd = y.dtype
    if (xd eq od) && (yd eq od) then
      NDArray.zipMap3(c, x.asInstanceOf[NDArray[p.Out]], y.asInstanceOf[NDArray[p.Out]])((k, a, b) =>
        if cd.toBoolean(k) then a else b
      )(using od)
    else
      NDArray.zipMap3(c, x, y)((k, a, b) => if cd.toBoolean(k) then od.castFrom(xd, a) else od.castFrom(yd, b))(using od)

  /** `np.where(condition, x, scalar)`. */
  def where[A, B](condition: NDArray[?], x: NDArray[A], y: B)(using p: Promote[A, B], db: DType[B]): NDArray[p.Out] =
    where(condition, x, NDArray.scalar(y))

  /** `np.where(condition, scalar, y)`. */
  def where[A, B](condition: NDArray[?], x: A, y: NDArray[B])(using p: Promote[A, B], da: DType[A]): NDArray[p.Out] =
    where(condition, NDArray.scalar(x), y)

  /** `np.where(condition, scalar, scalar)`. */
  def where[A, B](condition: NDArray[?], x: A, y: B)(using p: Promote[A, B], da: DType[A], db: DType[B]): NDArray[p.Out] =
    where(condition, NDArray.scalar(x), NDArray.scalar(y))

  /** One-argument `np.where(condition)`: the same as [[nonzero]]. */
  def where(condition: NDArray[?]): Seq[NDArray[Int]] = nonzero(condition)

  /** Indices of the non-zero (true) elements, one array per dimension (`np.nonzero`). */
  def nonzero(a: NDArray[?]): Seq[NDArray[Int]] =
    if a.ndim == 0 then
      throw new IllegalArgumentException(
        "Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead."
      )
    Searching.nonzero(a)

  /** Indices of non-zero elements grouped by element: shape `(N, a.ndim)` (`np.argwhere`). */
  def argwhere(a: NDArray[?]): NDArray[Int] =
    if a.ndim == 0 then
      val n = if truthy(a, Array.emptyIntArray)(0) then 1 else 0
      NDArray.fromArray(new Array[Int](0), Array(n, 0))
    else
      val nz = Searching.nonzero(a).map(_.toArray)
      val nd = a.ndim
      val n = nz.head.length
      NDArray.tabulate(n, nd)(k => nz(k % nd)(k / nd))(using DType.Int32)

  /** Indices of the non-zero elements of the flattened array (`np.flatnonzero`). */
  def flatnonzero(a: NDArray[?]): NDArray[Int] =
    Searching.nonzero(a.asInstanceOf[NDArray[Any]].ravel()).head

  /** Elements of `arr` (flattened) where `condition` (flattened) is true (`np.extract`). */
  def extract[T](condition: NDArray[?], arr: NDArray[T]): NDArray[T] =
    val c = truthy(condition, condition.shapeArr)
    val v = arr.toArray
    val b = scala.collection.mutable.ArrayBuilder.make[T](using arr.dtype.classTag)
    var i = 0
    while i < c.length do
      if c(i) then
        if i >= v.length then throw new IndexOutOfBoundsException(s"index $i is out of bounds for axis 0 with size ${v.length}")
        b += v(i)
      i += 1
    NDArray.fromArray(b.result())(using arr.dtype)

  /** Elements from `choicelist` chosen by the first true condition in `condlist`, `default`
    * where none holds (`np.select`).
    */
  def select[T](condlist: Seq[NDArray[Boolean]], choicelist: Seq[NDArray[T]], default: T): NDArray[T] =
    if condlist.length != choicelist.length then
      throw new IllegalArgumentException("list of cases must be same length as list of conditions")
    if condlist.isEmpty then throw new IllegalArgumentException("select with an empty condition list is not possible")
    val d = choicelist.head.dtype
    val sh = Shape.broadcast((condlist.map(_.shapeArr) ++ choicelist.map(_.shapeArr))*)
    val n = Shape.size(sh)
    val out = d.newArray(n)
    var i = 0
    while i < n do
      out(i) = default
      i += 1
    val done = new Array[Boolean](n)
    for (c, ch) <- condlist.zip(choicelist) do
      val cv = c.broadcastTo(sh*).toArray
      val vv = ch.broadcastTo(sh*).toArray
      var k = 0
      while k < n do
        if !done(k) && cv(k) then
          out(k) = vv(k)
          done(k) = true
        k += 1
    NDArray.fromArray(out, sh)(using d)

  /** `np.select` with the default value `0` of the choices' dtype. */
  def select[T](condlist: Seq[NDArray[Boolean]], choicelist: Seq[NDArray[T]]): NDArray[T] =
    if choicelist.isEmpty then throw new IllegalArgumentException("select with an empty condition list is not possible")
    select(condlist, choicelist, choicelist.head.dtype.zero)
