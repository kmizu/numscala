package com.github.kmizu.numscala

import NpReduceImpl.*

/** NaN-ignoring reductions: `np.nansum`, `np.nanmean`, `np.nanmax`, `np.nanargmin`, `np.nancumsum`, ... */
trait NpReduceNan:

  /** Reduces each lane after dropping NaNs; `f(buf, count, originalLength)`. */
  private def nanLanes[T, U](a: NDArray[T], ax: Array[Int], keepdims: Boolean)(f: (Array[T], Int, Int) => U)(using
      u: DType[U]
  ): NDArray[U] =
    var scratch: Array[T] = null
    Lanes.reduceAxes(a, ax, keepdims) { (b, n) =>
      if scratch == null || scratch.length < n then scratch = a.dtype.newArray(n)
      val k = compactNonNaN(a.dtype, b, n, scratch)
      f(scratch, k, n)
    }

  private def nanAll[T, U](a: NDArray[T])(f: (Array[T], Int, Int) => U): U =
    val arr = a.toArray
    val buf = a.dtype.newArray(arr.length)
    val k = compactNonNaN(a.dtype, arr, arr.length, buf)
    f(buf, k, arr.length)

  /** Sum treating NaN as zero (`np.nansum(a)`). */
  def nansum[T](a: NDArray[T])(using s: SumOf[T]): s.Out =
    nanAll(a)((b, k, _) => Reduce.sumBuf(a.dtype, b, k, s.dtype))

  /** Sum over `axis` treating NaN as zero (`np.nansum(a, axis, keepdims=...)`). */
  def nansum[T](a: NDArray[T], axis: Axis | Null = null, keepdims: Boolean = false)(using
      s: SumOf[T]
  ): NDArray[s.Out] =
    nanLanes(a, axesOf(axis, a.ndim), keepdims)((b, k, _) => Reduce.sumBuf(a.dtype, b, k, s.dtype))(using s.dtype)

  /** Product treating NaN as one (`np.nanprod(a)`). */
  def nanprod[T](a: NDArray[T])(using s: SumOf[T]): s.Out =
    nanAll(a)((b, k, _) => Reduce.prodBuf(a.dtype, b, k, s.dtype))

  /** Product over `axis` treating NaN as one (`np.nanprod(a, axis, keepdims=...)`). */
  def nanprod[T](a: NDArray[T], axis: Axis | Null = null, keepdims: Boolean = false)(using
      s: SumOf[T]
  ): NDArray[s.Out] =
    nanLanes(a, axesOf(axis, a.ndim), keepdims)((b, k, _) => Reduce.prodBuf(a.dtype, b, k, s.dtype))(using s.dtype)

  /** Mean ignoring NaNs (`np.nanmean(a)`); NaN for an all-NaN array. */
  def nanmean[T](a: NDArray[T])(using t: ToInexact[T]): t.Out =
    nanAll(a)((b, k, _) => Reduce.meanBuf(a.dtype, b, k, t.dtype))

  /** Mean over `axis` ignoring NaNs (`np.nanmean(a, axis, keepdims=...)`). */
  def nanmean[T](a: NDArray[T], axis: Axis | Null = null, keepdims: Boolean = false)(using
      t: ToInexact[T]
  ): NDArray[t.Out] =
    nanLanes(a, axesOf(axis, a.ndim), keepdims)((b, k, _) => Reduce.meanBuf(a.dtype, b, k, t.dtype))(using t.dtype)

  /** Variance ignoring NaNs (`np.nanvar(a)`). */
  def nanvar[T](a: NDArray[T])(using r: RealOf[T]): r.Out =
    nanAll(a)((b, k, _) => Reduce.varBuf(a.dtype, b, k, 0, r.dtype))

  /** Variance over `axis` ignoring NaNs (`np.nanvar(a, axis, ddof=..., keepdims=...)`). */
  def nanvar[T](a: NDArray[T], axis: Axis | Null = null, ddof: Int = 0, keepdims: Boolean = false)(using
      r: RealOf[T]
  ): NDArray[r.Out] =
    nanLanes(a, axesOf(axis, a.ndim), keepdims)((b, k, _) => Reduce.varBuf(a.dtype, b, k, ddof, r.dtype))(using
      r.dtype
    )

  /** Standard deviation ignoring NaNs (`np.nanstd(a)`). */
  def nanstd[T](a: NDArray[T])(using r: RealOf[T]): r.Out =
    r.dtype.sqrt(nanAll(a)((b, k, _) => Reduce.varBuf(a.dtype, b, k, 0, r.dtype)))

  /** Standard deviation over `axis` ignoring NaNs (`np.nanstd(a, axis, ddof=..., keepdims=...)`). */
  def nanstd[T](a: NDArray[T], axis: Axis | Null = null, ddof: Int = 0, keepdims: Boolean = false)(using
      r: RealOf[T]
  ): NDArray[r.Out] =
    val d = r.dtype
    nanLanes(a, axesOf(axis, a.ndim), keepdims)((b, k, _) => d.sqrt(Reduce.varBuf(a.dtype, b, k, ddof, d)))(using d)

  private def nanExtreme[T](d: DType[T], b: Array[T], k: Int, n: Int, wantMax: Boolean): T =
    if n == 0 then
      throw new IllegalArgumentException(
        s"zero-size array to reduction operation ${if wantMax then "fmax" else "fmin"} which has no identity"
      )
    if k == 0 then
      d match
        case i: InexactDType[T] => i.nan
        case _ => b(0)
    else Reduce.extremeBuf(d, b, k, wantMax)

  /** Maximum ignoring NaNs (`np.nanmax(a)`); NaN for an all-NaN array. */
  def nanmax[T](a: NDArray[T]): T = nanAll(a)((b, k, n) => nanExtreme(a.dtype, b, k, n, true))

  /** Maximum over `axis` ignoring NaNs (`np.nanmax(a, axis, keepdims=...)`). */
  def nanmax[T](a: NDArray[T], axis: Axis | Null = null, keepdims: Boolean = false): NDArray[T] =
    nanLanes(a, axesOf(axis, a.ndim), keepdims)((b, k, n) => nanExtreme(a.dtype, b, k, n, true))(using a.dtype)

  /** Minimum ignoring NaNs (`np.nanmin(a)`); NaN for an all-NaN array. */
  def nanmin[T](a: NDArray[T]): T = nanAll(a)((b, k, n) => nanExtreme(a.dtype, b, k, n, false))

  /** Minimum over `axis` ignoring NaNs (`np.nanmin(a, axis, keepdims=...)`). */
  def nanmin[T](a: NDArray[T], axis: Axis | Null = null, keepdims: Boolean = false): NDArray[T] =
    nanLanes(a, axesOf(axis, a.ndim), keepdims)((b, k, n) => nanExtreme(a.dtype, b, k, n, false))(using a.dtype)

  private def nanArg[T](d: DType[T], b: Array[T], n: Int, wantMax: Boolean): Int =
    if n == 0 then
      throw new IllegalArgumentException(
        s"attempt to get ${if wantMax then "argmax" else "argmin"} of an empty sequence"
      )
    var bi = -1
    var i = 0
    while i < n do
      val x = b(i)
      if !d.isNaN(x) then
        if bi < 0 then bi = i
        else
          val c = d.compare(x, b(bi))
          if (wantMax && c > 0) || (!wantMax && c < 0) then bi = i
      i += 1
    if bi < 0 then throw new IllegalArgumentException("All-NaN slice encountered")
    bi

  /** Flat index of the maximum ignoring NaNs (`np.nanargmax(a)`); fails on an all-NaN array. */
  def nanargmax[T](a: NDArray[T]): Int = Lanes.reduceAll(a)((b, n) => nanArg(a.dtype, b, n, true))

  /** Indices of the maxima along `axis` ignoring NaNs (`np.nanargmax(a, axis, keepdims=...)`). */
  def nanargmax[T](a: NDArray[T], axis: Int | Null = null, keepdims: Boolean = false): NDArray[Int] =
    nanArgImpl(a, axis, keepdims, true)

  /** Flat index of the minimum ignoring NaNs (`np.nanargmin(a)`). */
  def nanargmin[T](a: NDArray[T]): Int = Lanes.reduceAll(a)((b, n) => nanArg(a.dtype, b, n, false))

  /** Indices of the minima along `axis` ignoring NaNs (`np.nanargmin(a, axis, keepdims=...)`). */
  def nanargmin[T](a: NDArray[T], axis: Int | Null = null, keepdims: Boolean = false): NDArray[Int] =
    nanArgImpl(a, axis, keepdims, false)

  private def nanArgImpl[T](a: NDArray[T], axis: Int | Null, keepdims: Boolean, wantMax: Boolean): NDArray[Int] =
    if axis == null then
      val r = Lanes.reduceAll(a)((b, n) => nanArg(a.dtype, b, n, wantMax))
      NDArray.fromArray(Array(r), if keepdims then Array.fill(a.ndim)(1) else Array.emptyIntArray)
    else
      Lanes.reduceAxes(a, Array(Shape.normAxis(axis.asInstanceOf[Int], a.ndim)), keepdims)((b, n) =>
        nanArg(a.dtype, b, n, wantMax)
      )

  private def nanCum[T, U](a: NDArray[T], axis: Int | Null, d: NumDType[U], unit: U)(f: (U, U) => U): NDArray[U] =
    val (src, ax) = if axis == null then (a.ravel(), 0) else (a, axis.asInstanceOf[Int])
    val sd = a.dtype
    Lanes.transform(src, ax) { (in: Array[T], n: Int, out: Array[U]) =>
      var acc = unit
      var i = 0
      while i < n do
        val x = in(i)
        if !sd.isNaN(x) then acc = f(acc, d.castFrom(sd, x))
        out(i) = acc
        i += 1
    }(using d)

  /** Cumulative sum treating NaN as zero (`np.nancumsum`); `axis = null` flattens. */
  def nancumsum[T](a: NDArray[T], axis: Int | Null = null)(using s: SumOf[T]): NDArray[s.Out] =
    nanCum(a, axis, s.dtype, s.dtype.zero)(s.dtype.plus)

  /** Cumulative product treating NaN as one (`np.nancumprod`); `axis = null` flattens. */
  def nancumprod[T](a: NDArray[T], axis: Int | Null = null)(using s: SumOf[T]): NDArray[s.Out] =
    nanCum(a, axis, s.dtype, s.dtype.one)(s.dtype.times)
