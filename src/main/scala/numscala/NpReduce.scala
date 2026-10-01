package numscala

import NpReduceImpl.*

/** Reductions: `np.sum`, `np.mean`, `np.std`, `np.max`, `np.argmax`, `np.cumsum`, `np.average`, ...
  *
  * Each reduction has a scalar-returning full-reduction overload `f(a)` and an
  * array-returning overload `f(a, axis, ..., keepdims)` where `axis = null` (the default)
  * reduces over every axis (giving a 0-d array, or an all-ones shape with `keepdims`).
  */
trait NpReduce extends NpReduceNan, NpReduceQuantile:

  /** Sum of all elements (`np.sum(a)`); small integers accumulate in int64. */
  def sum[T](a: NDArray[T])(using s: SumOf[T]): s.Out = Reduce.sumAll(a, s.dtype)

  /** Sum over `axis` (`np.sum(a, axis, keepdims=...)`). */
  def sum[T](a: NDArray[T], axis: Axis | Null = null, keepdims: Boolean = false)(using
      s: SumOf[T]
  ): NDArray[s.Out] =
    reduceL(a, axis, keepdims)((b, n) => Reduce.sumBuf(a.dtype, b, n, s.dtype))(using s.dtype)

  /** Product of all elements (`np.prod(a)`). */
  def prod[T](a: NDArray[T])(using s: SumOf[T]): s.Out = Reduce.prodAll(a, s.dtype)

  /** Product over `axis` (`np.prod(a, axis, keepdims=...)`). */
  def prod[T](a: NDArray[T], axis: Axis | Null = null, keepdims: Boolean = false)(using
      s: SumOf[T]
  ): NDArray[s.Out] =
    reduceL(a, axis, keepdims)((b, n) => Reduce.prodBuf(a.dtype, b, n, s.dtype))(using s.dtype)

  /** Arithmetic mean of all elements (`np.mean(a)`); NaN for an empty array. */
  def mean[T](a: NDArray[T])(using t: ToInexact[T]): t.Out = Reduce.meanAll(a, t.dtype)

  /** Mean over `axis` (`np.mean(a, axis, keepdims=...)`). */
  def mean[T](a: NDArray[T], axis: Axis | Null = null, keepdims: Boolean = false)(using
      t: ToInexact[T]
  ): NDArray[t.Out] =
    reduceL(a, axis, keepdims)((b, n) => Reduce.meanBuf(a.dtype, b, n, t.dtype))(using t.dtype)

  /** Variance of all elements (`np.var(a)`); complex input gives a real result. */
  def `var`[T](a: NDArray[T])(using r: RealOf[T]): r.Out = Reduce.varAll(a, 0, r.dtype)

  /** Variance over `axis` with `ddof` delta degrees of freedom (`np.var(a, axis, ddof=...)`). */
  def `var`[T](a: NDArray[T], axis: Axis | Null = null, ddof: Int = 0, keepdims: Boolean = false)(using
      r: RealOf[T]
  ): NDArray[r.Out] =
    reduceL(a, axis, keepdims)((b, n) => Reduce.varBuf(a.dtype, b, n, ddof, r.dtype))(using r.dtype)

  /** Standard deviation of all elements (`np.std(a)`). */
  def std[T](a: NDArray[T])(using r: RealOf[T]): r.Out = r.dtype.sqrt(Reduce.varAll(a, 0, r.dtype))

  /** Standard deviation over `axis` (`np.std(a, axis, ddof=..., keepdims=...)`). */
  def std[T](a: NDArray[T], axis: Axis | Null = null, ddof: Int = 0, keepdims: Boolean = false)(using
      r: RealOf[T]
  ): NDArray[r.Out] =
    val d = r.dtype
    reduceL(a, axis, keepdims)((b, n) => d.sqrt(Reduce.varBuf(a.dtype, b, n, ddof, d)))(using d)

  /** Largest element (`np.max(a)`); NaN propagates. */
  def max[T](a: NDArray[T]): T = Reduce.maxAll(a)

  /** Maximum over `axis` (`np.max(a, axis, keepdims=...)`). */
  def max[T](a: NDArray[T], axis: Axis | Null = null, keepdims: Boolean = false): NDArray[T] =
    reduceL(a, axis, keepdims)((b, n) => Reduce.extremeBuf(a.dtype, b, n, true))(using a.dtype)

  /** Smallest element (`np.min(a)`); NaN propagates. */
  def min[T](a: NDArray[T]): T = Reduce.minAll(a)

  /** Minimum over `axis` (`np.min(a, axis, keepdims=...)`). */
  def min[T](a: NDArray[T], axis: Axis | Null = null, keepdims: Boolean = false): NDArray[T] =
    reduceL(a, axis, keepdims)((b, n) => Reduce.extremeBuf(a.dtype, b, n, false))(using a.dtype)

  /** Alias of [[max]] (`np.amax`). */
  def amax[T](a: NDArray[T]): T = max(a)

  /** Alias of [[max]] (`np.amax(a, axis)`). */
  def amax[T](a: NDArray[T], axis: Axis | Null = null, keepdims: Boolean = false): NDArray[T] =
    max(a, axis, keepdims)

  /** Alias of [[min]] (`np.amin`). */
  def amin[T](a: NDArray[T]): T = min(a)

  /** Alias of [[min]] (`np.amin(a, axis)`). */
  def amin[T](a: NDArray[T], axis: Axis | Null = null, keepdims: Boolean = false): NDArray[T] =
    min(a, axis, keepdims)

  /** Peak to peak (`max - min`) of all elements (`np.ptp(a)`). */
  def ptp[T](a: NDArray[T])(using d: NumDType[T]): T = d.minus(Reduce.maxAll(a), Reduce.minAll(a))

  /** Peak to peak over `axis` (`np.ptp(a, axis, keepdims=...)`). */
  def ptp[T](a: NDArray[T], axis: Axis | Null = null, keepdims: Boolean = false)(using d: NumDType[T]): NDArray[T] =
    reduceL(a, axis, keepdims) { (b, n) =>
      d.minus(Reduce.extremeBuf(a.dtype, b, n, true), Reduce.extremeBuf(a.dtype, b, n, false))
    }(using d)

  /** Flat index of the first maximum (`np.argmax(a)`); a NaN counts as the maximum. */
  def argmax[T](a: NDArray[T]): Int = Reduce.argmaxAll(a)

  /** Indices of the maxima along `axis` (`np.argmax(a, axis, keepdims=...)`); `null` = flattened. */
  def argmax[T](a: NDArray[T], axis: Int | Null = null, keepdims: Boolean = false): NDArray[Int] =
    argImpl(a, axis, keepdims)((b, n) => Reduce.argExtremeBuf(a.dtype, b, n, true))

  /** Flat index of the first minimum (`np.argmin(a)`). */
  def argmin[T](a: NDArray[T]): Int = Reduce.argminAll(a)

  /** Indices of the minima along `axis` (`np.argmin(a, axis, keepdims=...)`). */
  def argmin[T](a: NDArray[T], axis: Int | Null = null, keepdims: Boolean = false): NDArray[Int] =
    argImpl(a, axis, keepdims)((b, n) => Reduce.argExtremeBuf(a.dtype, b, n, false))

  private[numscala] def argImpl[T](a: NDArray[T], axis: Int | Null, keepdims: Boolean)(
      f: (Array[T], Int) => Int
  ): NDArray[Int] =
    if axis == null then
      val r = Lanes.reduceAll(a)(f)
      NDArray.fromArray(Array(r), if keepdims then Array.fill(a.ndim)(1) else Array.emptyIntArray)
    else Lanes.reduceAxes(a, Array(Shape.normAxis(axis.asInstanceOf[Int], a.ndim)), keepdims)(f)

  /** Whether every element is truthy (`np.all(a)`). */
  def all[T](a: NDArray[T]): Boolean = Reduce.allAll(a)

  /** Logical AND over `axis` (`np.all(a, axis, keepdims=...)`). */
  def all[T](a: NDArray[T], axis: Axis | Null = null, keepdims: Boolean = false): NDArray[Boolean] =
    reduceL(a, axis, keepdims) { (b, n) =>
      var i = 0
      while i < n && a.dtype.toBoolean(b(i)) do i += 1
      i == n
    }

  /** Whether any element is truthy (`np.any(a)`). */
  def any[T](a: NDArray[T]): Boolean = Reduce.anyAll(a)

  /** Logical OR over `axis` (`np.any(a, axis, keepdims=...)`). */
  def any[T](a: NDArray[T], axis: Axis | Null = null, keepdims: Boolean = false): NDArray[Boolean] =
    reduceL(a, axis, keepdims) { (b, n) =>
      var i = 0
      while i < n && !a.dtype.toBoolean(b(i)) do i += 1
      i < n
    }

  /** Number of non-zero (truthy) elements (`np.count_nonzero(a)`). */
  def count_nonzero[T](a: NDArray[T]): Long =
    var c = 0L
    a.foreach(x => if a.dtype.toBoolean(x) then c += 1)
    c

  /** Number of non-zero elements over `axis` (`np.count_nonzero(a, axis, keepdims=...)`). */
  def count_nonzero[T](a: NDArray[T], axis: Axis | Null = null, keepdims: Boolean = false): NDArray[Long] =
    reduceL(a, axis, keepdims) { (b, n) =>
      var c = 0L
      var i = 0
      while i < n do
        if a.dtype.toBoolean(b(i)) then c += 1
        i += 1
      c
    }

  private def cumImpl[T, U](a: NDArray[T], axis: Int | Null, d: NumDType[U], initial: Boolean, init: U)(
      f: (U, U) => U
  ): NDArray[U] =
    val (src, ax) = if axis == null then (a.ravel(), 0) else (a, axis.asInstanceOf[Int])
    if src.ndim == 0 then throw new IndexOutOfBoundsException(s"axis $ax is out of bounds for array of dimension 0")
    val sd = a.dtype
    val len = src.shapeArr(Shape.normAxis(ax, src.ndim))
    Lanes.transform(src, ax, if initial then len + 1 else len) { (in: Array[T], n: Int, out: Array[U]) =>
      var acc = init
      var o = 0
      if initial then
        out(0) = init
        o = 1
      var i = 0
      while i < n do
        acc = if i == 0 then d.castFrom(sd, in(0)) else f(acc, d.castFrom(sd, in(i)))
        out(o + i) = acc
        i += 1
    }(using d)

  /** Cumulative sum along `axis` (`np.cumsum`); `axis = null` works on the flattened array. */
  def cumsum[T](a: NDArray[T], axis: Int | Null = null)(using s: SumOf[T]): NDArray[s.Out] =
    cumImpl(a, axis, s.dtype, false, s.dtype.zero)(s.dtype.plus)

  /** Cumulative product along `axis` (`np.cumprod`); `axis = null` works on the flattened array. */
  def cumprod[T](a: NDArray[T], axis: Int | Null = null)(using s: SumOf[T]): NDArray[s.Out] =
    cumImpl(a, axis, s.dtype, false, s.dtype.one)(s.dtype.times)

  private def cumulativeArgs[T](x: NDArray[T], axis: Int | Null): (NDArray[T], Int) =
    if x.ndim == 0 then (x.reshape(1), 0)
    else if axis == null then
      if x.ndim > 1 then
        throw new IllegalArgumentException("For arrays which have more than one dimension ``axis`` argument is required.")
      (x, 0)
    else (x, axis.asInstanceOf[Int])

  /** Array-API cumulative sum (`np.cumulative_sum`); `include_initial` prepends a zero. */
  def cumulative_sum[T](x: NDArray[T], axis: Int | Null = null, include_initial: Boolean = false)(using
      s: SumOf[T]
  ): NDArray[s.Out] =
    val (src, ax) = cumulativeArgs(x, axis)
    cumImpl(src, ax, s.dtype, include_initial, s.dtype.zero)(s.dtype.plus)

  /** Array-API cumulative product (`np.cumulative_prod`); `include_initial` prepends a one. */
  def cumulative_prod[T](x: NDArray[T], axis: Int | Null = null, include_initial: Boolean = false)(using
      s: SumOf[T]
  ): NDArray[s.Out] =
    val (src, ax) = cumulativeArgs(x, axis)
    cumImpl(src, ax, s.dtype, include_initial, s.dtype.one)(s.dtype.times)

  // ---------------------------------------------------------------- average

  /** Mean of all elements (`np.average(a)` without weights). */
  def average[T](a: NDArray[T])(using t: ToInexact[T]): t.Out = Reduce.meanAll(a, t.dtype)

  /** Weighted average over `axis` (`np.average(a, axis, weights, keepdims=...)`).
    * `weights` has `a`'s shape, or is 1-d with the length of `a` along the single `axis`.
    */
  def average[T](
      a: NDArray[T],
      axis: Axis | Null = null,
      weights: NDArray[?] | Null = null,
      keepdims: Boolean = false
  )(using t: ToInexact[T]): NDArray[t.Out] =
    averageImpl(a, axis, weights, keepdims, t.dtype)._1

  /** Weighted average that also returns the sum of the weights (`np.average(..., returned=True)`). */
  def average[T](a: NDArray[T], axis: Axis | Null, weights: NDArray[?] | Null, returned: true)(using
      t: ToInexact[T]
  ): (NDArray[t.Out], NDArray[t.Out]) =
    averageImpl(a, axis, weights, false, t.dtype)

  private def averageImpl[T, U](
      a: NDArray[T],
      axis: Axis | Null,
      weights: NDArray[?] | Null,
      keepdims: Boolean,
      od: InexactDType[U]
  ): (NDArray[U], NDArray[U]) =
    val ax = axesOf(axis, a.ndim)
    val outShape = reducedShape(a.shapeArr, ax, keepdims)
    if weights == null then
      val avg = Lanes.reduceAxes(a, ax, keepdims)((b, n) => Reduce.meanBuf(a.dtype, b, n, od))(using od)
      val cnt = od.fromLong(Shape.size(ax.map(a.shapeArr(_))).toLong)
      (avg, NDArray.fillOf(od, outShape, cnt))
    else
      val wsrc = weights.asInstanceOf[NDArray[Any]]
      if wsrc.dtype.isComplex && !od.isComplex then
        throw new IllegalArgumentException("complex weights require a complex array")
      val w0: NDArray[U] = wsrc.asType(using od)
      val w: NDArray[U] =
        if w0.shapeArr.sameElements(a.shapeArr) then w0
        else
          if axis == null then
            throw new IllegalArgumentException("Axis must be specified when shapes of a and weights differ.")
          if w0.ndim != ax.length then
            throw new IllegalArgumentException("Shape of weights must be consistent with shape of a along specified axis.")
          if !ax.indices.forall(i => w0.shapeArr(i) == a.shapeArr(ax(i))) then
            throw new IllegalArgumentException("Shape of weights must be consistent with shape of a along specified axis.")
          // place the weights' axes at positions `ax` and broadcast
          val sh = Array.fill(a.ndim)(1)
          ax.indices.foreach(i => sh(ax(i)) = w0.shapeArr(i))
          w0.reshape(sh*).broadcastTo(a.shapeArr.toSeq*)
      val (va, m, l) = gather(a, ax)
      val (vw, _, _) = gather(w, ax)
      val avg = od.newArray(m)
      val scl = od.newArray(m)
      val prodBuf = od.newArray(l)
      val wBuf = od.newArray(l)
      var r = 0
      while r < m do
        var j = 0
        while j < l do
          val wv = vw(r * l + j)
          wBuf(j) = wv
          prodBuf(j) = od.times(od.castFrom(a.dtype, va(r * l + j)), wv)
          j += 1
        val ws = Reduce.sumBuf(od, wBuf, l, od)
        if od.equiv(ws, od.zero) then throw new ArithmeticException("Weights sum to zero, can't be normalized")
        scl(r) = ws
        avg(r) = od.div(Reduce.sumBuf(od, prodBuf, l, od), ws)
        r += 1
      (NDArray.fromArray(avg, outShape)(using od), NDArray.fromArray(scl, outShape)(using od))
