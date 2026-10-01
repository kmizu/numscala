package numscala

/** `np.ma` reductions, joining, sorting, selection and 2-D helpers. */
trait MAFunctions:
  import MA.toMA

  // ------------------------------------------------------------------ counting / compressing

  /** `np.ma.compressed(x)`: the unmasked values as a 1-D array. */
  def compressed[T](x: MaskedArrayLike[T]): NDArray[T] = toMA(x).compressed()
  /** `np.ma.count(a)`: number of unmasked elements. */
  def count[T](a: MaskedArrayLike[T]): Int = toMA(a).count()
  def count[T](a: MaskedArrayLike[T], axis: Axis, keepdims: Boolean = false): NDArray[Int] = toMA(a).count(axis, keepdims)
  /** `np.ma.count_masked(arr)`: number of masked elements. */
  def count_masked[T](arr: MaskedArrayLike[T]): Int = toMA(arr).size - toMA(arr).count()
  def count_masked[T](arr: MaskedArrayLike[T], axis: Axis): NDArray[Int] =
    Lanes.reduce(toMA(arr).maskArray, axis, false) { (b, n) =>
      var c = 0
      var i = 0
      while i < n do
        if b(i) then c += 1
        i += 1
      c
    }(using DType.Int32)

  // ------------------------------------------------------------------ reductions

  /** `np.ma.sum(a)`: sum of unmasked values (`masked` if all are masked). */
  def sum[T](a: MaskedArrayLike[T])(using s: SumOf[T]): s.Out | MaskedConstant.type = toMA(a).sum()
  def sum[T](a: MaskedArrayLike[T], axis: Axis, keepdims: Boolean = false)(using s: SumOf[T]): MaskedArray[s.Out] =
    toMA(a).sum(axis, keepdims)
  /** `np.ma.prod(a)`. */
  def prod[T](a: MaskedArrayLike[T])(using s: SumOf[T]): s.Out | MaskedConstant.type = toMA(a).prod()
  def prod[T](a: MaskedArrayLike[T], axis: Axis, keepdims: Boolean = false)(using s: SumOf[T]): MaskedArray[s.Out] =
    toMA(a).prod(axis, keepdims)
  /** `np.ma.mean(a)`. */
  def mean[T](a: MaskedArrayLike[T])(using t: ToInexact[T]): t.Out | MaskedConstant.type = toMA(a).mean()
  def mean[T](a: MaskedArrayLike[T], axis: Axis, keepdims: Boolean = false)(using t: ToInexact[T]): MaskedArray[t.Out] =
    toMA(a).mean(axis, keepdims)
  /** `np.ma.var(a)`: variance of the unmasked values. */
  def `var`[T](a: MaskedArrayLike[T])(using r: RealOf[T]): r.Out | MaskedConstant.type = toMA(a).variance(0)
  /** `np.ma.var(a, axis, ddof, keepdims)`. */
  def `var`[T](a: MaskedArrayLike[T], axis: Axis, ddof: Int = 0, keepdims: Boolean = false)(using r: RealOf[T]): MaskedArray[r.Out] =
    toMA(a).variance(axis, ddof, keepdims)
  /** `np.ma.var(a, axis=None, ddof=ddof)`: pass `None` as the axis. */
  def `var`[T](a: MaskedArrayLike[T], axis: None.type, ddof: Int)(using r: RealOf[T]): r.Out | MaskedConstant.type =
    toMA(a).variance(ddof)
  /** Alias of `np.ma.var` usable without backquotes. */
  def variance[T](a: MaskedArrayLike[T])(using r: RealOf[T]): r.Out | MaskedConstant.type = toMA(a).variance(0)
  def variance[T](a: MaskedArrayLike[T], axis: Axis, ddof: Int = 0, keepdims: Boolean = false)(using r: RealOf[T]): MaskedArray[r.Out] =
    toMA(a).variance(axis, ddof, keepdims)
  def variance[T](a: MaskedArrayLike[T], axis: None.type, ddof: Int)(using r: RealOf[T]): r.Out | MaskedConstant.type =
    toMA(a).variance(ddof)
  /** `np.ma.std(a)`: standard deviation of the unmasked values. */
  def std[T](a: MaskedArrayLike[T])(using r: RealOf[T]): r.Out | MaskedConstant.type = toMA(a).std(0)
  /** `np.ma.std(a, axis, ddof, keepdims)`. */
  def std[T](a: MaskedArrayLike[T], axis: Axis, ddof: Int = 0, keepdims: Boolean = false)(using r: RealOf[T]): MaskedArray[r.Out] =
    toMA(a).std(axis, ddof, keepdims)
  /** `np.ma.std(a, axis=None, ddof=ddof)`: pass `None` as the axis. */
  def std[T](a: MaskedArrayLike[T], axis: None.type, ddof: Int)(using r: RealOf[T]): r.Out | MaskedConstant.type = toMA(a).std(ddof)
  /** `np.ma.min(a)`. */
  def min[T](a: MaskedArrayLike[T]): T | MaskedConstant.type = toMA(a).min()
  def min[T](a: MaskedArrayLike[T], axis: Axis, keepdims: Boolean = false): MaskedArray[T] = toMA(a).min(axis, keepdims)
  /** `np.ma.max(a)`. */
  def max[T](a: MaskedArrayLike[T]): T | MaskedConstant.type = toMA(a).max()
  def max[T](a: MaskedArrayLike[T], axis: Axis, keepdims: Boolean = false): MaskedArray[T] = toMA(a).max(axis, keepdims)
  /** `np.ma.ptp(a)`. */
  def ptp[T](a: MaskedArrayLike[T])(using d: NumDType[T]): T | MaskedConstant.type = toMA(a).ptp()
  def ptp[T](a: MaskedArrayLike[T], axis: Axis)(using d: NumDType[T]): MaskedArray[T] = toMA(a).ptp(axis)
  /** `np.ma.argmin(a)` (masked values ignored). */
  def argmin[T](a: MaskedArrayLike[T]): Int = toMA(a).argmin()
  def argmin[T](a: MaskedArrayLike[T], axis: Int): NDArray[Int] = toMA(a).argmin(axis)
  /** `np.ma.argmax(a)`. */
  def argmax[T](a: MaskedArrayLike[T]): Int = toMA(a).argmax()
  def argmax[T](a: MaskedArrayLike[T], axis: Int): NDArray[Int] = toMA(a).argmax(axis)
  /** `np.ma.all(a)`. */
  def all[T](a: MaskedArrayLike[T]): Boolean | MaskedConstant.type = toMA(a).all()
  def all[T](a: MaskedArrayLike[T], axis: Axis, keepdims: Boolean = false): MaskedArray[Boolean] = toMA(a).all(axis, keepdims)
  /** `np.ma.any(a)`. */
  def any[T](a: MaskedArrayLike[T]): Boolean | MaskedConstant.type = toMA(a).any()
  def any[T](a: MaskedArrayLike[T], axis: Axis, keepdims: Boolean = false): MaskedArray[Boolean] = toMA(a).any(axis, keepdims)
  /** `np.ma.cumsum(a)` (flattened). */
  def cumsum[T](a: MaskedArrayLike[T])(using s: SumOf[T]): MaskedArray[s.Out] = toMA(a).cumsum()
  def cumsum[T](a: MaskedArrayLike[T], axis: Int)(using s: SumOf[T]): MaskedArray[s.Out] = toMA(a).cumsum(axis)
  /** `np.ma.cumprod(a)` (flattened). */
  def cumprod[T](a: MaskedArrayLike[T])(using s: SumOf[T]): MaskedArray[s.Out] = toMA(a).cumprod()
  def cumprod[T](a: MaskedArrayLike[T], axis: Int)(using s: SumOf[T]): MaskedArray[s.Out] = toMA(a).cumprod(axis)
  /** `np.ma.anom(a)`: deviations from the mean. */
  def anom[T](a: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] = toMA(a).anom()
  def anom[T](a: MaskedArrayLike[T], axis: Int)(using t: ToInexact[T]): MaskedArray[t.Out] = toMA(a).anom(axis)

  /** `np.ma.median(a)`: median of the unmasked values (`masked` if all are masked). */
  def median[T](a: MaskedArrayLike[T])(using t: ToInexact[T]): t.Out | MaskedConstant.type =
    MAReduce.full(toMA(a), (b, n) => MAReduce.medianSorted(toMA(a).dtype, b, n, t.dtype))
  def median[T](a: MaskedArrayLike[T], axis: Axis, keepdims: Boolean = false)(using t: ToInexact[T]): MaskedArray[t.Out] =
    val m = toMA(a)
    MAReduce.lanes(m, axis, keepdims, t.dtype, t.dtype.zero)((b, n) => MAReduce.medianSorted(m.dtype, b, n, t.dtype))

  /** `np.ma.average(a)`: mean of the unmasked values. */
  def average[T](a: MaskedArrayLike[T])(using t: ToInexact[T]): t.Out | MaskedConstant.type = toMA(a).mean()
  def average[T](a: MaskedArrayLike[T], axis: Axis)(using t: ToInexact[T]): MaskedArray[t.Out] =
    val r = toMA(a).mean(axis)
    if r._mask == null then r.withMask(NDArray.zerosOf(DType.Bool, r._data.shapeArr.clone())) else r
  /** `np.ma.average(a, weights=w)`: weighted mean ignoring masked values (`w` broadcast to `a`). */
  def average[T](a: MaskedArrayLike[T], weights: NDArray[Double]): Double | MaskedConstant.type =
    val (x, w, mask) = weighted(toMA(a), weights)
    var num = 0.0
    var den = 0.0
    var any = false
    var i = 0
    while i < x.length do
      if !mask(i) then
        num += x(i) * w(i)
        den += w(i)
        any = true
      i += 1
    if !any then MaskedConstant else num / den
  /** `np.ma.average(a, axis, weights=w)`; `w` is either broadcastable to `a` or 1-D along `axis`. */
  def average[T](a: MaskedArrayLike[T], axis: Int, weights: NDArray[Double]): MaskedArray[Double] =
    val m = toMA(a)
    val ax = Shape.normAxis(axis, m.ndim)
    val w0 =
      if weights.ndim == 1 && m.ndim > 1 && weights.size == m._data.shapeArr(ax) then
        val sh = Array.fill(m.ndim)(1)
        sh(ax) = weights.size
        weights.reshape(sh.toIndexedSeq*)
      else weights
    val (x, w, mask) = weighted(m, w0)
    val sh = m._data.shapeArr.clone()
    val mk = NDArray.fromArray(mask, sh.clone())
    val num = new MaskedArray(NDArray.fromArray(x.indices.map(i => x(i) * w(i)).toArray, sh.clone()), mk, None, false).sum(ax)
    val den = new MaskedArray(NDArray.fromArray(w, sh.clone()), mk.copy(), None, false).sum(ax)
    MaskedArray.domainedF(num, den, DType.Float64)(_ / _)

  private def weighted[T](m: MaskedArray[T], weights: NDArray[Double]): (Array[Double], Array[Double], Array[Boolean]) =
    (m._data.astype(using DType.Float64).toArray, weights.broadcastTo(m.shape*).toArray, m.maskArray.toArray)

  // ------------------------------------------------------------------ joining

  private def concatND[T](arrs: Seq[NDArray[T]], axis: Int): NDArray[T] =
    if arrs.isEmpty then throw new IllegalArgumentException("need at least one array to concatenate")
    val nd = arrs.head.ndim
    if nd == 0 then throw new IllegalArgumentException("zero-dimensional arrays cannot be concatenated")
    val ax = Shape.normAxis(axis, nd)
    arrs.foreach { a =>
      if a.ndim != nd then
        throw new IllegalArgumentException(
          s"all the input array dimensions except for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has ${nd} dimension(s) and another has ${a.ndim} dimension(s)"
        )
      for k <- 0 until nd if k != ax && a.shapeArr(k) != arrs.head.shapeArr(k) do
        throw new IllegalArgumentException(
          s"all the input array dimensions except for the concatenation axis must match exactly, but along dimension $k, the array at index 0 has size ${arrs.head.shapeArr(k)} and another has size ${a.shapeArr(k)}"
        )
    }
    val sh = arrs.head.shapeArr.clone()
    sh(ax) = arrs.map(_.shapeArr(ax)).sum
    val d = arrs.head.dtype
    val out = NDArray.zerosOf(d, sh)
    var start = 0
    arrs.foreach { a =>
      val len = a.shapeArr(ax)
      val items = (0 until nd).map(k => if k == ax then Index.Slice(Some(start), Some(start + len), 1) else Index.All)
      out.index(items) := a
      start += len
    }
    out

  /** `np.ma.concatenate(arrays, axis)`: joins data and masks (mask shrunk to `nomask` when empty). */
  def concatenate[T](arrays: Seq[MaskedArrayLike[T]], axis: Int = 0): MaskedArray[T] =
    val ms = arrays.map(toMA)
    val d = concatND(ms.map(_._data), axis)
    if ms.forall(_._mask == null) then MaskedArray.wrap(d)
    else
      val m = concatND(ms.map(_.maskArray), axis)
      new MaskedArray(d, if m.any() then m else null, None, false)

  private def joinFull[T](ms: Seq[MaskedArray[T]], f: [X] => Seq[NDArray[X]] => NDArray[X]): MaskedArray[T] =
    new MaskedArray(f(ms.map(_._data)), f(ms.map(_.maskArray)), None, false)

  /** `np.ma.stack(arrays, axis)`: joins along a new axis (always with a full mask). */
  def stack[T](arrays: Seq[MaskedArrayLike[T]], axis: Int = 0): MaskedArray[T] =
    val ms = arrays.map(toMA)
    if ms.isEmpty then throw new IllegalArgumentException("need at least one array to stack")
    val nd = ms.head.ndim
    if ms.exists(m => m.shape != ms.head.shape) then throw new IllegalArgumentException("all input arrays must have the same shape")
    val ax = Shape.normAxis(axis, nd + 1)
    joinFull(ms, [X] => (xs: Seq[NDArray[X]]) => concatND(xs.map(_.expandDims(ax)), ax))

  private def atLeast2d[X](a: NDArray[X]): NDArray[X] =
    if a.ndim == 0 then a.reshape(1, 1) else if a.ndim == 1 then a.expandDims(0) else a
  private def atLeast1d[X](a: NDArray[X]): NDArray[X] = if a.ndim == 0 then a.reshape(1) else a

  /** `np.ma.vstack(arrays)`: stacks row-wise. */
  def vstack[T](arrays: Seq[MaskedArrayLike[T]]): MaskedArray[T] =
    joinFull(arrays.map(toMA), [X] => (xs: Seq[NDArray[X]]) => concatND(xs.map(atLeast2d), 0))
  /** `np.ma.row_stack` (alias of [[vstack]]). */
  def row_stack[T](arrays: Seq[MaskedArrayLike[T]]): MaskedArray[T] = vstack(arrays)
  /** `np.ma.hstack(arrays)`: stacks column-wise. */
  def hstack[T](arrays: Seq[MaskedArrayLike[T]]): MaskedArray[T] =
    joinFull(arrays.map(toMA), [X] => (xs: Seq[NDArray[X]]) =>
      val ys = xs.map(atLeast1d)
      concatND(ys, if ys.head.ndim == 1 then 0 else 1)
    )
  /** `np.ma.column_stack(arrays)`: 1-D arrays become columns. */
  def column_stack[T](arrays: Seq[MaskedArrayLike[T]]): MaskedArray[T] =
    joinFull(arrays.map(toMA), [X] => (xs: Seq[NDArray[X]]) =>
      concatND(xs.map(x => if x.ndim < 2 then atLeast1d(x).expandDims(1) else x), 1)
    )

  // ------------------------------------------------------------------ selection

  private def whereImpl[T](condition: MaskedArrayLike[Boolean], xd: NDArray[T], xm: NDArray[Boolean], yd: NDArray[T],
      ym: NDArray[Boolean]): MaskedArray[T] =
    val c = toMA(condition)
    val cf = c.filled(false)
    val cm = c.maskArray
    val data = NDArray.zipMap3(cf, xd, yd)((b, x, y) => if b then x else y)(using xd.dtype)
    val mask0 = NDArray.zipMap3(cf, xm, ym)((b, x, y) => if b then x else y)(using DType.Bool)
    val mask = NDArray.zipMap(cm, mask0)(_ || _)
    new MaskedArray(data, if mask.any() then MaskedArray.fitMask(mask, data.shapeArr) else null, None, false)

  /** `np.ma.where(condition, x, y)`: elements of `x` where the condition holds, else `y`;
    * masked where the condition or the chosen element is masked.
    */
  def where[T](condition: MaskedArrayLike[Boolean], x: MaskedArrayLike[T], y: MaskedArrayLike[T]): MaskedArray[T] =
    val (mx, my) = (toMA(x), toMA(y))
    whereImpl(condition, mx._data, mx.maskArray, my._data, my.maskArray)
  def where[T](condition: MaskedArrayLike[Boolean], x: MaskedArrayLike[T], y: MaskedConstant.type): MaskedArray[T] =
    val mx = toMA(x)
    whereImpl(condition, mx._data, mx.maskArray, NDArray.scalar(mx.dtype.zero)(using mx.dtype), NDArray.scalar(true))
  def where[T](condition: MaskedArrayLike[Boolean], x: MaskedConstant.type, y: MaskedArrayLike[T]): MaskedArray[T] =
    val my = toMA(y)
    whereImpl(condition, NDArray.scalar(my.dtype.zero)(using my.dtype), NDArray.scalar(true), my._data, my.maskArray)
  /** `np.ma.where(condition)`: indices of the unmasked true elements. */
  def where(condition: MaskedArrayLike[Boolean]): Seq[NDArray[Int]] = toMA(condition).filled(false).nonzero
  /** `np.ma.nonzero(a)`. */
  def nonzero[T](a: MaskedArrayLike[T]): Seq[NDArray[Int]] = toMA(a).nonzero

  /** `np.ma.clip(a, lo, hi)`: clips the data, keeping the mask. */
  def clip[T](a: MaskedArrayLike[T], a_min: T, a_max: T)(using d: RealDType[T]): MaskedArray[T] = toMA(a).clip(a_min, a_max)

  /** `np.ma.sort(a, axis)`: sorted copy, masked values last (or first when `endwith = false`). */
  def sort[T](a: MaskedArrayLike[T], axis: Int = -1, endwith: Boolean = true): MaskedArray[T] =
    MAReduce.sorted(toMA(a), axis, endwith)
  /** `np.ma.argsort(a, axis)`: indices that sort `a`, masked values last. */
  def argsort[T](a: MaskedArrayLike[T], axis: Int = -1, endwith: Boolean = true): NDArray[Int] = toMA(a).argsort(axis, endwith)

  /** `np.ma.unique(a)`: sorted unique unmasked values, followed by one masked entry if any. */
  def unique[T](a: MaskedArrayLike[T]): MaskedArray[T] =
    val m = toMA(a)
    val c = m.compressed().toArray
    Sorting.sortBuffer(m.dtype, c, c.length)
    val buf = scala.collection.mutable.ArrayBuffer.empty[T]
    var i = 0
    while i < c.length do
      if buf.isEmpty || !(m.dtype.equiv(buf.last, c(i)) || (m.dtype.isNaN(buf.last) && m.dtype.isNaN(c(i)))) then buf += c(i)
      i += 1
    val anyMasked = m._mask != null && m._mask.any()
    if anyMasked then
      val firstMasked = m._data.toArray.zip(m._mask.toArray).collectFirst { case (x, true) => x }.get
      buf += firstMasked
    val data = NDArray.fromArray(buf.toArray(using m.dtype.classTag), Array(buf.length))(using m.dtype)
    val mask = NDArray.tabulate(buf.length)(k => anyMasked && k == buf.length - 1)(using DType.Bool)
    new MaskedArray(data, mask, m._fill, false)

  /** `np.ma.apply_along_axis(func1d, axis, arr)` for scalar-valued `func1d` (best effort):
    * applies `func1d` to every 1-D lane along `axis`; a `masked` result masks that output.
    */
  def apply_along_axis[T, U](func1d: MaskedArray[T] => U | MaskedConstant.type, axis: Int, arr: MaskedArrayLike[T])(using
      u: DType[U]
  ): MaskedArray[U] =
    val m = toMA(arr)
    val ax = Shape.normAxis(axis, m.ndim)
    val moved = m.moveaxis(ax, -1)
    val n = m._data.shapeArr(ax)
    val outShape = moved._data.shapeArr.dropRight(1)
    val rows = moved.reshape(-1, n)
    val k = Shape.size(outShape)
    val data = u.newArray(k)
    val mask = new Array[Boolean](k)
    var i = 0
    while i < k do
      func1d(rows.subArray(i)) match
        case MaskedConstant =>
          mask(i) = true
          data(i) = u.zero
        case v => data(i) = v.asInstanceOf[U]
      i += 1
    new MaskedArray(NDArray.fromArray(data, outShape), if mask.exists(identity) then NDArray.fromArray(mask, outShape) else null, None, false)

  // ------------------------------------------------------------------ contiguous runs

  private def runs(mask: Array[Boolean], want: Boolean): Seq[Index.Slice] =
    val out = scala.collection.mutable.ArrayBuffer.empty[Index.Slice]
    var i = 0
    val n = mask.length
    while i < n do
      if mask(i) == want then
        val s = i
        while i < n && mask(i) == want do i += 1
        out += Index.Slice(Some(s), Some(i), 1)
      else i += 1
    out.toSeq

  private def flatMask[T](a: MaskedArrayLike[T]): Array[Boolean] = toMA(a).maskArray.toArray

  /** `np.ma.flatnotmasked_contiguous(a)`: slices of the contiguous unmasked runs of the flattened array. */
  def flatnotmasked_contiguous[T](a: MaskedArrayLike[T]): Seq[Index.Slice] = runs(flatMask(a), false)
  /** `np.ma.notmasked_contiguous(a)`: as [[flatnotmasked_contiguous]] (axis=None). */
  def notmasked_contiguous[T](a: MaskedArrayLike[T]): Seq[Index.Slice] = flatnotmasked_contiguous(a)
  /** `np.ma.notmasked_contiguous(a, axis)` for 2-D arrays: the runs of every 1-D lane along `axis`. */
  def notmasked_contiguous[T](a: MaskedArrayLike[T], axis: Int): Seq[Seq[Index.Slice]] =
    val m = toMA(a)
    if m.ndim > 2 then throw new IllegalArgumentException("Currently limited to at most 2D array.")
    if m.ndim == 1 then Seq(flatnotmasked_contiguous(a))
    else
      val ax = Shape.normAxis(axis, 2)
      val mk = m.maskArray
      val other = 1 - ax
      (0 until mk.shapeArr(other)).map { i =>
        val lane = if ax == 0 then mk(::, i) else mk(i, ::)
        runs(lane.toArray, false)
      }
  /** `np.ma.clump_masked(a)`: slices of the masked runs of a 1-D array. */
  def clump_masked[T](a: MaskedArrayLike[T]): Seq[Index.Slice] =
    val m = toMA(a)
    if m._mask == null then Seq.empty else runs(flatMask(a), true)
  /** `np.ma.clump_unmasked(a)`: slices of the unmasked runs of a 1-D array. */
  def clump_unmasked[T](a: MaskedArrayLike[T]): Seq[Index.Slice] = runs(flatMask(a), false)
  /** `np.ma.flatnotmasked_edges(a)`: flat indices of the first and last unmasked values (`None` if all masked). */
  def flatnotmasked_edges[T](a: MaskedArrayLike[T]): Option[(Int, Int)] =
    val m = flatMask(a)
    val first = m.indexWhere(!_)
    if first < 0 then None else Some((first, m.lastIndexWhere(!_)))
  /** `np.ma.notmasked_edges(a)` with axis=None: same as [[flatnotmasked_edges]]. */
  def notmasked_edges[T](a: MaskedArrayLike[T]): Option[(Int, Int)] = flatnotmasked_edges(a)

  // ------------------------------------------------------------------ 2-D helpers

  private def check2d(m: MaskedArray[?], fn: String): Unit =
    if m.ndim != 2 then throw new IllegalArgumentException(s"$fn works for 2D arrays only.")

  /** `np.ma.mask_rowcols(a)`: masks every row and column containing a masked value. */
  def mask_rowcols[T](a: MaskedArrayLike[T]): MaskedArray[T] = maskRowCols(toMA(a), rows = true, cols = true)
  /** `np.ma.mask_rowcols(a, axis)`: rows (`axis = 0`) or columns (`axis = 1`) only. */
  def mask_rowcols[T](a: MaskedArrayLike[T], axis: Int): MaskedArray[T] =
    val ax = Shape.normAxis(axis, 2)
    maskRowCols(toMA(a), rows = ax == 0, cols = ax == 1)
  /** `np.ma.mask_rows(a)`. */
  def mask_rows[T](a: MaskedArrayLike[T]): MaskedArray[T] = mask_rowcols(a, 0)
  /** `np.ma.mask_cols(a)`. */
  def mask_cols[T](a: MaskedArrayLike[T]): MaskedArray[T] = mask_rowcols(a, 1)

  private def maskRowCols[T](m: MaskedArray[T], rows: Boolean, cols: Boolean): MaskedArray[T] =
    check2d(m, "mask_rowcols")
    if m._mask == null || !m._mask.any() then return m
    val mk = m._mask
    val badRows = mk.any(1)
    val badCols = mk.any(0)
    val nm = NDArray.tabulate(m.shape*) { k =>
      val i = k / m._data.shapeArr(1)
      val j = k % m._data.shapeArr(1)
      (rows && badRows.flatGet(i)) || (cols && badCols.flatGet(j))
    }(using DType.Bool)
    new MaskedArray(m._data.copy(), nm, m._fill, m._hard)

  /** `np.ma.compress_rowcols(x)`: drops rows and columns containing masked values. */
  def compress_rowcols[T](x: MaskedArrayLike[T]): NDArray[T] = compressRC(toMA(x), true, true)
  /** `np.ma.compress_rowcols(x, axis)`: rows (`axis = 0`) or columns (`axis = 1`) only. */
  def compress_rowcols[T](x: MaskedArrayLike[T], axis: Int): NDArray[T] =
    val ax = Shape.normAxis(axis, 2)
    compressRC(toMA(x), ax == 0, ax == 1)
  /** `np.ma.compress_rows(a)`. */
  def compress_rows[T](a: MaskedArrayLike[T]): NDArray[T] = compress_rowcols(a, 0)
  /** `np.ma.compress_cols(a)`. */
  def compress_cols[T](a: MaskedArrayLike[T]): NDArray[T] = compress_rowcols(a, 1)

  private def compressRC[T](m: MaskedArray[T], rows: Boolean, cols: Boolean): NDArray[T] =
    check2d(m, "compress_rowcols")
    if m._mask == null then return m._data
    val mk = m._mask
    val keepR = (0 until m._data.shapeArr(0)).filter(i => !rows || !mk.subArray(i).any()).toArray
    val keepC = (0 until m._data.shapeArr(1)).filter(j => !cols || !mk(::, j).any()).toArray
    m._data(keepR, ::)(::, keepC)

  // ------------------------------------------------------------------ linear algebra / statistics

  private def propagate(m: NDArray[Boolean], axis: Int): NDArray[Boolean] =
    m.any(axis, keepdims = true).broadcastTo(m.shape*).copy()

  /** `np.ma.dot(a, b, strict)`: product treating masked values as 0; an output is masked when
    * no pair of unmasked inputs contributes to it (or, with `strict`, when any input is masked).
    */
  def dot[T](a: MaskedArrayLike[T], b: MaskedArrayLike[T], strict: Boolean = false)(using d: NumDType[T]): MaskedArray[T] =
    var ma = toMA(a)
    var mb = toMA(b)
    if strict && ma.ndim > 0 && mb.ndim > 0 then
      ma = ma.withMask(propagate(ma.maskArray, ma.ndim - 1))
      mb = mb.withMask(propagate(mb.maskArray, if mb.ndim == 1 then 0 else mb.ndim - 2))
    val data = LinAlgCore.dotD(ma.filled(d.zero), mb.filled(d.zero), d)
    val am = ma.maskArray.map(x => if x then 0 else 1)(using DType.Int32)
    val bm = mb.maskArray.map(x => if x then 0 else 1)(using DType.Int32)
    val cnt = LinAlgCore.dotD(am, bm, DType.Int32)
    new MaskedArray(data, cnt.map(_ == 0)(using DType.Bool), None, false)

  /** `np.ma.cov(x, rowvar, bias, ddof)` for one 1-D or 2-D variable set (pairwise-complete normalisation). */
  def cov[T](x: MaskedArrayLike[T], rowvar: Boolean = true, bias: Boolean = false, ddof: Int = Int.MinValue,
      allow_masked: Boolean = true): MaskedArray[Double] =
    val m0 = toMA(x)
    if m0.ndim > 2 then throw new IllegalArgumentException("x has more than 2 dimensions")
    val m = if m0.ndim < 2 then m0.reshape(1, m0.size) else m0
    val xmask = m.maskArray
    if !allow_masked && xmask.any() then throw new IllegalArgumentException("Cannot process masked data.")
    val dd = if ddof == Int.MinValue then (if bias then 0 else 1) else ddof
    val rv = rowvar || m._data.shapeArr(0) == 1
    val xs = if rv then m else m.T
    val (nv, no) = (xs._data.shapeArr(0), xs._data.shapeArr(1))
    val data = xs._data.astype(using DType.Float64).toArray
    val msk = xs.maskArray.toArray
    for i <- 0 until nv do
      var s = 0.0
      var c = 0
      for k <- 0 until no if !msk(i * no + k) do
        s += data(i * no + k)
        c += 1
      val mean = if c == 0 then 0.0 else s / c
      for k <- 0 until no do data(i * no + k) = if msk(i * no + k) then 0.0 else data(i * no + k) - mean
    val out = new Array[Double](nv * nv)
    val om = new Array[Boolean](nv * nv)
    for i <- 0 until nv; j <- 0 until nv do
      var s = 0.0
      var c = 0
      for k <- 0 until no do
        s += data(i * no + k) * data(j * no + k)
        if !msk(i * no + k) && !msk(j * no + k) then c += 1
      val fact = (c - dd).toDouble
      om(i * nv + j) = fact <= 0
      out(i * nv + j) = s / fact
    new MaskedArray(NDArray.fromArray(out, Array(nv, nv)), NDArray.fromArray(om, Array(nv, nv)), None, false).squeeze()

  /** `np.ma.corrcoef(x, rowvar)`: correlation coefficients from [[cov]]. */
  def corrcoef[T](x: MaskedArrayLike[T], rowvar: Boolean = true, allow_masked: Boolean = true): MaskedArray[Double] =
    val c = cov(x, rowvar, allow_masked = allow_masked)
    if c.ndim == 0 then MaskedArray.allMasked(c)
    else
      val n = c._data.shapeArr(0)
      val dg = (0 until n).map(i => c._data.getAt(Array(i, i)))
      val dm = (0 until n).map(i => c.maskArray.getAt(Array(i, i)))
      val std = dg.map(math.sqrt)
      val outer = NDArray.tabulate(n, n)(k => std(k / n) * std(k % n))(using DType.Float64)
      val om = NDArray.tabulate(n, n)(k => dm(k / n) || dm(k % n) || std(k / n).isNaN || std(k % n).isNaN)(using DType.Bool)
      MaskedArray.domainedF(c, new MaskedArray(outer, om, None, false), DType.Float64)(_ / _)
