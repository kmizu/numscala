package com.github.kmizu.numscala

/** `np.ma` constructors, mask construction/inspection and the `masked_*` family. */
trait MACreation:
  import MA.{toMA, internalMask}

  // ------------------------------------------------------------------ construction

  /** `np.ma.array(data)`: a masked array with no mask (an [[NDArray]] argument is not copied). */
  def array[A, E](data: A)(using n: Nested[A, E]): MaskedArray[E] =
    MaskedArray.wrap(np.asarray(data))

  /** `np.ma.array(data, mask=..., fill_value=..., hard_mask=...)`. `mask` may be a boolean
    * scalar (masks everything or nothing), nested booleans or an `NDArray[Boolean]`.
    */
  def array[A, E, M](data: A, mask: M, fill_value: Any = null, hard_mask: Boolean = false, copy: Boolean = false)(using
      n: Nested[A, E],
      mn: Nested[M, Boolean]
  ): MaskedArray[E] =
    val d0 = np.asarray(data)
    val d = if copy then d0.copy() else d0
    val m0 = np.asarray(mask)
    val m = if (m0 eq MA.nomask) then null else MaskedArray.fitMask(m0, d.shapeArr)
    val r = new MaskedArray(d, if m != null && (m eq m0) then m.copy() else m, None, hard_mask)
    if fill_value != null then r._fill = Some(d.dtype.coerce(fill_value))
    r

  /** `np.ma.array(masked_array)`: shares the data, copies the mask. */
  def array[T](data: MaskedArray[T]): MaskedArray[T] =
    new MaskedArray(data._data, if data._mask == null then null else data._mask.copy(), data._fill, data._hard)

  /** `np.ma.masked_array(data)` (alias of [[array]]). */
  def masked_array[A, E](data: A)(using n: Nested[A, E]): MaskedArray[E] = array(data)
  /** `np.ma.masked_array(data, mask, ...)` (alias of [[array]]). */
  def masked_array[A, E, M](data: A, mask: M, fill_value: Any = null, hard_mask: Boolean = false, copy: Boolean = false)(using
      n: Nested[A, E],
      mn: Nested[M, Boolean]
  ): MaskedArray[E] = array(data, mask, fill_value, hard_mask, copy)
  def masked_array[T](data: MaskedArray[T]): MaskedArray[T] = array(data)

  /** `np.ma.asarray`: views an array as a masked array without copying. */
  def asarray[T](a: MaskedArrayLike[T]): MaskedArray[T] = toMA(a)
  def asarray[A, E](a: A)(using n: Nested[A, E]): MaskedArray[E] = array(a)
  /** `np.ma.asanyarray` (same as [[asarray]]). */
  def asanyarray[T](a: MaskedArrayLike[T]): MaskedArray[T] = toMA(a)

  /** `np.ma.zeros(shape)` (no mask). */
  def zeros[T](shape: Int*)(using d: DefaultDType[T]): MaskedArray[T] = MaskedArray.wrap(np.zeros[T](shape*))
  /** `np.ma.ones(shape)`. */
  def ones[T](shape: Int*)(using d: DefaultDType[T]): MaskedArray[T] = MaskedArray.wrap(np.ones[T](shape*))
  /** `np.ma.empty(shape)` (zero-filled). */
  def empty[T](shape: Int*)(using d: DefaultDType[T]): MaskedArray[T] = MaskedArray.wrap(np.empty[T](shape*))
  /** `np.ma.masked_all(shape)`: every element masked. */
  def masked_all[T](shape: Int*)(using d: DefaultDType[T]): MaskedArray[T] =
    MaskedArray.allMasked(MaskedArray.wrap(np.zeros[T](shape*)))
  /** `np.ma.zeros_like(a)`: zeros, keeping the mask of `a`. */
  def zeros_like[T](a: MaskedArrayLike[T]): MaskedArray[T] = likeOf(a, toMA(a).dtype.zero)
  /** `np.ma.ones_like(a)`: ones, keeping the mask of `a`. */
  def ones_like[T](a: MaskedArrayLike[T]): MaskedArray[T] = likeOf(a, toMA(a).dtype.one)
  /** `np.ma.empty_like(a)`. */
  def empty_like[T](a: MaskedArrayLike[T]): MaskedArray[T] = zeros_like(a)
  /** `np.ma.masked_all_like(a)`. */
  def masked_all_like[T](a: MaskedArrayLike[T]): MaskedArray[T] = MaskedArray.allMasked(zeros_like(a))

  private def likeOf[T](a: MaskedArrayLike[T], v: T): MaskedArray[T] =
    val m = toMA(a)
    new MaskedArray(NDArray.fillOf(m.dtype, m._data.shapeArr.clone(), v), if m._mask == null then null else m._mask.copy(), m._fill, false)

  /** `np.ma.arange(stop)` and friends. */
  def arange(stop: Int): MaskedArray[Int] = MaskedArray.wrap(np.arange(stop))
  def arange(start: Int, stop: Int): MaskedArray[Int] = MaskedArray.wrap(np.arange(start, stop))
  def arange(start: Int, stop: Int, step: Int): MaskedArray[Int] = MaskedArray.wrap(np.arange(start, stop, step))
  def arange(stop: Double): MaskedArray[Double] = MaskedArray.wrap(np.arange(stop))
  def arange(start: Double, stop: Double): MaskedArray[Double] = MaskedArray.wrap(np.arange(start, stop))
  def arange(start: Double, stop: Double, step: Double): MaskedArray[Double] = MaskedArray.wrap(np.arange(start, stop, step))

  // ------------------------------------------------------------------ masks

  /** `np.ma.getmask(a)`: the mask, or `nomask`. */
  def getmask[T](a: MaskedArrayLike[T]): NDArray[Boolean] = toMA(a).mask
  /** `np.ma.getmaskarray(a)`: the mask as a full boolean array. */
  def getmaskarray[T](a: MaskedArrayLike[T]): NDArray[Boolean] = toMA(a).maskArray
  /** `np.ma.getdata(a)`: the underlying data. */
  def getdata[T](a: MaskedArrayLike[T]): NDArray[T] = toMA(a)._data
  /** `np.ma.is_masked(x)`: whether any element is masked. */
  def is_masked[T](x: MaskedArrayLike[T]): Boolean =
    val m = toMA(x)._mask
    m != null && m.any()
  /** `np.ma.isMaskedArray(x)`. */
  def isMaskedArray(x: Any): Boolean = x.isInstanceOf[MaskedArray[?]]
  /** `np.ma.isMA(x)`. */
  def isMA(x: Any): Boolean = isMaskedArray(x)
  /** `np.ma.isarray(x)`. */
  def isarray(x: Any): Boolean = isMaskedArray(x)
  /** `np.ma.is_mask(m)`: whether `m` is a valid (boolean) mask. */
  def is_mask(m: Any): Boolean = m match
    case a: NDArray[?] => a.dtype eq DType.Bool
    case _ => false

  /** `np.ma.make_mask(m)`: boolean mask from any array; `nomask` when nothing is true and `shrink`. */
  def make_mask[T](m: MaskedArrayLike[T], copy: Boolean = false, shrink: Boolean = true): NDArray[Boolean] =
    makeMask(m, copy, shrink)
  def make_mask[A, E](m: A)(using n: Nested[A, E]): NDArray[Boolean] = makeMask(np.asarray(m), false, true)

  private def makeMask[T](m: MaskedArrayLike[T], copy: Boolean, shrink: Boolean): NDArray[Boolean] =
    val r = m match
      case ma: MaskedArray[T @unchecked] => ma.filled(ma.dtype.fromBoolean(true)).map(ma.dtype.toBoolean)(using DType.Bool)
      case a: NDArray[T @unchecked] =>
        if (a.dtype eq DType.Bool) && !copy then a.asInstanceOf[NDArray[Boolean]] else a.map(a.dtype.toBoolean)(using DType.Bool)
    if shrink && !r.any() then MA.nomask else r

  /** `np.ma.make_mask_none(shape)`: an all-`False` mask. */
  def make_mask_none(shape: Int*): NDArray[Boolean] = NDArray.zerosOf(DType.Bool, shape.toArray)

  /** `np.ma.mask_or(m1, m2)`: logical or of two masks (`nomask` aware). */
  def mask_or(m1: NDArray[Boolean], m2: NDArray[Boolean], shrink: Boolean = true): NDArray[Boolean] =
    val r = MaskedArray.maskOr(internalMask(m1), internalMask(m2), shrink)
    if r == null then MA.nomask else r

  // ------------------------------------------------------------------ fill values

  /** `np.ma.filled(a)`: data with masked entries replaced by the fill value. */
  def filled[T](a: MaskedArrayLike[T]): NDArray[T] = toMA(a).filled()
  /** `np.ma.filled(a, fill_value)`. */
  def filled[T](a: MaskedArrayLike[T], fill_value: T): NDArray[T] = toMA(a).filled(fill_value)

  /** `np.ma.default_fill_value(dtype)`: 999999, 1e20, True, 'N/A' or 1e20+0j. */
  def default_fill_value[T](d: DType[T]): T = MaskedArray.defaultFill(d)
  def default_fill_value[T](a: MaskedArrayLike[T]): T = MaskedArray.defaultFill(toMA(a).dtype)
  /** `np.ma.minimum_fill_value`: the largest representable value (for `min`). */
  def minimum_fill_value[T](d: DType[T]): T = MaskedArray.minFill(d)
  def minimum_fill_value[T](a: MaskedArrayLike[T]): T = MaskedArray.minFill(toMA(a).dtype)
  /** `np.ma.maximum_fill_value`: the smallest representable value (for `max`). */
  def maximum_fill_value[T](d: DType[T]): T = MaskedArray.maxFill(d)
  def maximum_fill_value[T](a: MaskedArrayLike[T]): T = MaskedArray.maxFill(toMA(a).dtype)
  /** `np.ma.set_fill_value(a, v)` (no effect on plain arrays). */
  def set_fill_value[T](a: MaskedArrayLike[T], fill_value: T): Unit = a match
    case m: MaskedArray[T @unchecked] => m.fill_value = fill_value
    case _ => ()
  /** `np.ma.common_fill_value(a, b)`: the fill value if both agree. */
  def common_fill_value[T](a: MaskedArray[T], b: MaskedArray[T]): Option[T] =
    if a.dtype.equiv(a.fill_value, b.fill_value) then Some(a.fill_value) else None

  /** `np.ma.fix_invalid(a)`: masks NaN/inf and replaces them by the fill value in the data. */
  def fix_invalid[T](a: MaskedArrayLike[T], copy: Boolean = true, fill_value: Any = null): MaskedArray[T] =
    val m0 = toMA(a)
    val r = if copy then m0.copy() else new MaskedArray(m0._data, m0._mask, m0._fill, m0._hard)
    val invalid = r._data.map(x => !MaskedArray.isFinite(r.dtype, x))(using DType.Bool)
    if invalid.any() then
      r._mask = if r._mask == null then invalid else NDArray.zipMap(r._mask, invalid)(_ || _)
      val fv = if fill_value == null then r.fill_value else r.dtype.coerce(fill_value)
      r._data := NDArray.zipMap(r._data, invalid)((x, bad) => if bad then fv else x)(using r.dtype)
    r

  // ------------------------------------------------------------------ masked_* constructors

  /** `np.ma.masked_where(condition, a)`: masks `a` where `condition` holds. */
  def masked_where[T](condition: MaskedArrayLike[Boolean], a: MaskedArrayLike[T], copy: Boolean = true): MaskedArray[T] =
    val cond = condition match
      case c: MaskedArray[Boolean @unchecked] => c.filled(true)
      case c: NDArray[Boolean @unchecked] => c
    val in = toMA(a)
    if cond.ndim > 0 && !java.util.Arrays.equals(cond.shapeArr, in._data.shapeArr) then
      throw new IndexOutOfBoundsException(
        s"Inconsistent shape between the condition and the input (got ${Shape.str(cond.shapeArr)} and ${Shape.str(in._data.shapeArr)})"
      )
    val c = MaskedArray.fitMask(cond, in._data.shapeArr)
    val full = if in._mask == null then c.copy() else NDArray.zipMap(c, in._mask)(_ || _)
    val data = if copy then in._data.copy() else in._data
    val mask = if in._mask == null && !full.any() then null else full
    if !copy && (a.isInstanceOf[MaskedArray[?]]) then in._mask = mask
    new MaskedArray(data, mask, in._fill, in._hard)

  private def maskedBy[T](x: MaskedArrayLike[T], copy: Boolean)(pred: T => Boolean): MaskedArray[T] =
    val in = toMA(x)
    masked_where(in._data.map(pred)(using DType.Bool), x, copy)

  /** `np.ma.masked_equal(x, value)`; the fill value becomes `value`. */
  def masked_equal[T](x: MaskedArrayLike[T], value: T, copy: Boolean = true): MaskedArray[T] =
    val d = toMA(x).dtype
    val r = maskedBy(x, copy)(v => d.equiv(v, value))
    r._fill = Some(value)
    r
  /** `np.ma.masked_not_equal(x, value)`. */
  def masked_not_equal[T](x: MaskedArrayLike[T], value: T, copy: Boolean = true): MaskedArray[T] =
    val d = toMA(x).dtype
    maskedBy(x, copy)(v => !d.equiv(v, value))
  /** `np.ma.masked_greater(x, value)`. */
  def masked_greater[T](x: MaskedArrayLike[T], value: T, copy: Boolean = true): MaskedArray[T] =
    val d = toMA(x).dtype
    maskedBy(x, copy)(v => CmpOp.Gt.test(d, v, value))
  /** `np.ma.masked_greater_equal(x, value)`. */
  def masked_greater_equal[T](x: MaskedArrayLike[T], value: T, copy: Boolean = true): MaskedArray[T] =
    val d = toMA(x).dtype
    maskedBy(x, copy)(v => CmpOp.Ge.test(d, v, value))
  /** `np.ma.masked_less(x, value)`. */
  def masked_less[T](x: MaskedArrayLike[T], value: T, copy: Boolean = true): MaskedArray[T] =
    val d = toMA(x).dtype
    maskedBy(x, copy)(v => CmpOp.Lt.test(d, v, value))
  /** `np.ma.masked_less_equal(x, value)`. */
  def masked_less_equal[T](x: MaskedArrayLike[T], value: T, copy: Boolean = true): MaskedArray[T] =
    val d = toMA(x).dtype
    maskedBy(x, copy)(v => CmpOp.Le.test(d, v, value))
  /** `np.ma.masked_inside(x, v1, v2)`: masks the closed interval between `v1` and `v2`. */
  def masked_inside[T](x: MaskedArrayLike[T], v1: T, v2: T, copy: Boolean = true): MaskedArray[T] =
    val d = toMA(x).dtype
    val (lo, hi) = if d.compare(v2, v1) < 0 then (v2, v1) else (v1, v2)
    maskedBy(x, copy)(v => CmpOp.Ge.test(d, v, lo) && CmpOp.Le.test(d, v, hi))
  /** `np.ma.masked_outside(x, v1, v2)`: masks values outside the closed interval. */
  def masked_outside[T](x: MaskedArrayLike[T], v1: T, v2: T, copy: Boolean = true): MaskedArray[T] =
    val d = toMA(x).dtype
    val (lo, hi) = if d.compare(v2, v1) < 0 then (v2, v1) else (v1, v2)
    maskedBy(x, copy)(v => CmpOp.Lt.test(d, v, lo) || CmpOp.Gt.test(d, v, hi))
  /** `np.ma.masked_object(x, value)`: masks entries equal to `value` (any dtype, e.g. strings). */
  def masked_object[T](x: MaskedArrayLike[T], value: T, copy: Boolean = true): MaskedArray[T] = masked_equal(x, value, copy)

  /** `np.ma.masked_invalid(a)`: masks NaN and infinities (always returns a full mask). */
  def masked_invalid[T](a: MaskedArrayLike[T], copy: Boolean = true): MaskedArray[T] =
    val d = toMA(a).dtype
    val r = maskedBy(a, copy)(v => !MaskedArray.isFinite(d, v))
    if r._mask == null then r._mask = NDArray.zerosOf(DType.Bool, r._data.shapeArr.clone())
    r

  /** `np.ma.masked_values(x, value)`: masks values close to `value` (floats: `isclose`). */
  def masked_values[T](x: MaskedArrayLike[T], value: T, rtol: Double = 1e-5, atol: Double = 1e-8, copy: Boolean = true,
      shrink: Boolean = true): MaskedArray[T] =
    val in = toMA(x)
    val d = in.dtype
    val xnew = in.filled(value)
    val mask =
      if d.isFloating then
        val v = d.toDouble(value)
        xnew.map { e =>
          val a = d.toDouble(e)
          a == v || (!a.isInfinite && !v.isInfinite && math.abs(a - v) <= atol + rtol * math.abs(v))
        }(using DType.Bool)
      else xnew.map(e => d.equiv(e, value))(using DType.Bool)
    val data = if copy && (xnew eq in._data) then xnew.copy() else xnew
    val r = new MaskedArray(data, mask, Some(value), in._hard)
    if shrink then r.shrink_mask() else r
