package numscala

import NpSortImpl.*

/** Index-based picking and placing routines (`np.take`, `np.put`, `np.choose`,
  * `np.take_along_axis`, `np.compress`, `np.place`, `np.putmask`, `np.fill_diagonal`, ...).
  */
trait NpSortTake:

  /** Constructs an array from an index array and a list of arrays to choose from
    * (`np.choose`); all operands are broadcast together. `mode` is `raise`, `wrap` or `clip`.
    */
  def choose[T](a: NDArray[Int], choices: Seq[NDArray[T]], mode: String = "raise"): NDArray[T] =
    if choices.isEmpty then throw new IllegalArgumentException("0-length sequence.")
    val m = parseMode(mode)
    val d = choices.head.dtype
    val sh = Shape.broadcast((a.shapeArr +: choices.map(_.shapeArr))*)
    val idx = a.broadcastTo(sh*).toArray
    val cs = choices.map(_.broadcastTo(sh*).toArray).toIndexedSeq
    val nc = cs.length
    val n = idx.length
    val out = d.newArray(n)
    var k = 0
    while k < n do
      val i = idx(k)
      val j = m match
        case Mode.Raise =>
          if i < 0 || i >= nc then throw new IllegalArgumentException("invalid entry in choice array")
          i
        case _ => normIndex(i, nc, m, allowNeg = false)
      out(k) = cs(j)(k)
      k += 1
    NDArray.fromArray(out, sh)(using d)

  /** Takes elements along an axis (`np.take`); with `axis = None` the flattened array is used.
    * The result has shape `a.shape[:axis] + indices.shape + a.shape[axis+1:]`.
    */
  def take[T](a: NDArray[T], indices: NDArray[Int], axis: Int | None.type = None, mode: String = "raise"): NDArray[T] =
    val m = parseMode(mode)
    val (src, ax) = axisOpt(axis) match
      case None => (a.ravel(), 0)
      case Some(x) => (a, Shape.normAxis(x, a.ndim))
    val sh = src.shapeArr
    val outer = Shape.size(sh.take(ax))
    val n = sh(ax)
    val inner = Shape.size(sh.drop(ax + 1))
    val idx = indices.toArray.map(i => normIndex(i, n, m, allowNeg = true, axis = ax))
    val vals = src.toArray
    val d = src.dtype
    val ni = idx.length
    val out = d.newArray(outer * ni * inner)
    var o = 0
    var k = 0
    while o < outer do
      var j = 0
      while j < ni do
        System.arraycopy(vals, (o * n + idx(j)) * inner, out, k, inner)
        k += inner
        j += 1
      o += 1
    NDArray.fromArray(out, sh.take(ax) ++ indices.shapeArr ++ sh.drop(ax + 1))(using d)

  /** A single element of the flattened array (`np.take(a, i)`). */
  def take[T](a: NDArray[T], index: Int): T = a.flatGet(index)

  /** Replaces `a.flat[ind]` with the values `v`, repeated as needed (`np.put`); in place. */
  def put[T](a: NDArray[T], ind: NDArray[Int], v: NDArray[T], mode: String = "raise"): Unit =
    val m = parseMode(mode)
    val vals = v.toArray
    if vals.isEmpty then return
    val n = a.size
    val idx = ind.toArray
    var k = 0
    while k < idx.length do
      a.flatSet(normIndex(idx(k), n, m, allowNeg = true), vals(k % vals.length))
      k += 1

  /** `np.put` with a scalar value. */
  def put[T](a: NDArray[T], ind: NDArray[Int], v: T): Unit = put(a, ind, NDArray.scalar(v)(using a.dtype), "raise")

  /** `np.put` with a scalar value and an index mode. */
  def put[T](a: NDArray[T], ind: NDArray[Int], v: T, mode: String): Unit =
    put(a, ind, NDArray.scalar(v)(using a.dtype), mode)

  /** Takes values from `arr` by matching 1-D index and data slices along `axis`
    * (`np.take_along_axis`); `indices` must have the same number of dimensions as `arr`.
    */
  def take_along_axis[T](arr: NDArray[T], indices: NDArray[Int], axis: Int | None.type): NDArray[T] =
    axisOpt(axis) match
      case None =>
        if indices.ndim != 1 then
          throw new IllegalArgumentException("when axis=None, `indices` must have a single dimension.")
        take(arr, indices, None, "raise")
      case Some(x) =>
        val ax = Shape.normAxis(x, arr.ndim)
        val (sh, ast, ist) = alongLayout(arr, indices, ax)
        val n = arr.shapeArr(ax)
        val sAx = arr.stridesArr(ax)
        val d = arr.dtype
        val out = d.newArray(Shape.size(sh))
        val ad = arr.data
        val id = indices.data
        var k = 0
        Strided.foreach2(sh, ast, arr.offset, ist, indices.offset) { (oa, oi) =>
          out(k) = ad(oa + normIndex(id(oi), n, Mode.Raise, allowNeg = true, axis = ax) * sAx)
          k += 1
        }
        NDArray.fromArray(out, sh)(using d)

  /** Puts `values` into `arr` at the positions given by matching index slices along `axis`
    * (`np.put_along_axis`); in place.
    */
  def put_along_axis[T](arr: NDArray[T], indices: NDArray[Int], values: NDArray[T], axis: Int | None.type): Unit =
    axisOpt(axis) match
      case None =>
        if indices.ndim != 1 then
          throw new IllegalArgumentException("when axis=None, `indices` must have a single dimension.")
        val idx = indices.toArray
        val v = values.broadcastTo(indices.shapeArr*).toArray
        val n = arr.size
        var k = 0
        while k < idx.length do
          arr.flatSet(normIndex(idx(k), n, Mode.Raise, allowNeg = true), v(k))
          k += 1
      case Some(x) =>
        val ax = Shape.normAxis(x, arr.ndim)
        val (sh, ast, ist) = alongLayout(arr, indices, ax)
        val vst = Strided.broadcastStrides(values.shapeArr, values.stridesArr, sh)
        val vals = if values.data eq arr.data then values.copy() else values
        val vst2 = if vals eq values then vst else Strided.broadcastStrides(vals.shapeArr, vals.stridesArr, sh)
        val n = arr.shapeArr(ax)
        val sAx = arr.stridesArr(ax)
        val ad = arr.data
        val id = indices.data
        val vd = vals.data
        Strided.foreach3(sh, ast, arr.offset, ist, indices.offset, vst2, vals.offset) { (oa, oi, ov) =>
          ad(oa + normIndex(id(oi), n, Mode.Raise, allowNeg = true, axis = ax) * sAx) = vd(ov)
        }

  /** `np.put_along_axis` with a scalar value. */
  def put_along_axis[T](arr: NDArray[T], indices: NDArray[Int], values: T, axis: Int | None.type): Unit =
    put_along_axis(arr, indices, NDArray.scalar(values)(using arr.dtype), axis)

  /** Broadcast shape and strides (arr with a zero stride on `ax`, indices broadcast). */
  private def alongLayout(arr: NDArray[?], indices: NDArray[Int], ax: Int): (Array[Int], Array[Int], Array[Int]) =
    if indices.ndim != arr.ndim then
      throw new IllegalArgumentException("`indices` and `arr` must have the same number of dimensions")
    val nd = arr.ndim
    val sh = Array.tabulate(nd) { d =>
      if d == ax then indices.shapeArr(d)
      else
        val p = arr.shapeArr(d)
        val q = indices.shapeArr(d)
        if p == q || q == 1 then p
        else if p == 1 then q
        else
          throw new IllegalArgumentException(
            s"shape mismatch: indexing arrays could not be broadcast together with shapes " +
              s"${Shape.str(arr.shapeArr)} ${Shape.str(indices.shapeArr)}"
          )
    }
    val ast = Array.tabulate(nd)(d => if d == ax || arr.shapeArr(d) != sh(d) then 0 else arr.stridesArr(d))
    val ist = Array.tabulate(nd)(d => if indices.shapeArr(d) != sh(d) then 0 else indices.stridesArr(d))
    (sh, ast, ist)

  /** Selects slices of `a` along `axis` where `condition` is true (`np.compress`);
    * `axis = None` works on the flattened array.
    */
  def compress[T](condition: NDArray[Boolean], a: NDArray[T], axis: Int | None.type = None): NDArray[T] =
    if condition.ndim != 1 then throw new IllegalArgumentException("condition must be a 1-d array")
    val idx = Searching.nonzero(condition).head
    take(a, idx, axis, "raise")

  /** Changes elements of `arr` where `mask` is true to successive values of `vals`, repeated
    * as needed (`np.place`); in place.
    */
  def place[T](arr: NDArray[T], mask: NDArray[Boolean], vals: NDArray[T]): Unit =
    if mask.size != arr.size then throw new IllegalArgumentException("place: mask and data must be the same size")
    val mk = mask.toArray
    val v = vals.toArray
    var j = 0
    var i = 0
    while i < mk.length do
      if mk(i) then
        if v.isEmpty then throw new IllegalArgumentException("Cannot insert from an empty array!")
        arr.flatSet(i, v(j % v.length))
        j += 1
      i += 1

  /** `np.place` with a scalar value. */
  def place[T](arr: NDArray[T], mask: NDArray[Boolean], vals: T): Unit =
    place(arr, mask, NDArray.scalar(vals)(using arr.dtype))

  /** Sets `a.flat[n] = values[n % len(values)]` wherever `mask.flat[n]` is true (`np.putmask`). */
  def putmask[T](a: NDArray[T], mask: NDArray[Boolean], values: NDArray[T]): Unit =
    if mask.size != a.size then throw new IllegalArgumentException("putmask: mask and data must be the same size")
    val mk = mask.toArray
    val v = values.toArray
    if v.isEmpty then return
    var i = 0
    while i < mk.length do
      if mk(i) then a.flatSet(i, v(i % v.length))
      i += 1

  /** `np.putmask` with a scalar value. */
  def putmask[T](a: NDArray[T], mask: NDArray[Boolean], values: T): Unit =
    putmask(a, mask, NDArray.scalar(values)(using a.dtype))

  /** Fills the main diagonal of an array of at least 2 dimensions, in place
    * (`np.fill_diagonal`). For tall 2-D matrices, `wrap = true` continues the diagonal
    * after an empty row, like NumPy.
    */
  def fill_diagonal[T](a: NDArray[T], v: T, wrap: Boolean = false): Unit =
    fill_diagonal(a, NDArray.scalar(v)(using a.dtype), wrap)

  /** `np.fill_diagonal` with an array of values (used cyclically). */
  def fill_diagonal[T](a: NDArray[T], v: NDArray[T]): Unit = fill_diagonal(a, v, false)

  /** `np.fill_diagonal(a, values, wrap)` with an array of values (used cyclically). */
  def fill_diagonal[T](a: NDArray[T], v: NDArray[T], wrap: Boolean): Unit =
    if a.ndim < 2 then throw new IllegalArgumentException("array must be at least 2-d")
    val (step, end) =
      if a.ndim == 2 then
        val cols = a.shapeArr(1)
        (cols + 1, if wrap then a.size else math.min(a.size, cols * cols))
      else
        if a.shapeArr.exists(_ != a.shapeArr(0)) then
          throw new IllegalArgumentException("All dimensions of input must be of equal length")
        var acc = 0
        var p = 1
        var i = 0
        while i < a.ndim - 1 do
          p *= a.shapeArr(i)
          acc += p
          i += 1
        (1 + acc, a.size)
    val vals = v.toArray
    if vals.isEmpty then return
    var pos = 0
    var k = 0
    while pos < end do
      a.flatSet(pos, vals(k % vals.length))
      pos += step
      k += 1

  /** Specified diagonals (`np.diagonal`): a view of the diagonal of the 2-D sub-arrays
    * defined by `axis1` and `axis2`, appended as the last axis.
    */
  def diagonal[T](a: NDArray[T], offset: Int = 0, axis1: Int = 0, axis2: Int = 1): NDArray[T] =
    Ops.diagonal(a, offset, axis1, axis2)
