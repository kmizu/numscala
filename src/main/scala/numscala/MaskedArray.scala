package numscala

import scala.collection.immutable.ArraySeq

/** `numpy.ma.masked`: the masked constant. Reading a masked element of a [[MaskedArray]]
  * (or a full reduction over only masked values) yields this singleton; assigning it to an
  * element masks that element.
  */
object MaskedConstant:
  /** `str(np.ma.masked)`. */
  override def toString: String = "--"
  /** `repr(np.ma.masked)`. */
  def repr: String = "masked"

/** An array with a boolean mask marking invalid entries — the Scala counterpart of
  * `numpy.ma.MaskedArray`.
  *
  * The data is an ordinary [[NDArray]]; the mask is either `nomask` (no element masked,
  * stored as `null` internally) or a boolean array of the same shape. Basic indexing,
  * reshaping and transposition return views sharing both data and mask, like NumPy.
  *
  * {{{
  * val a = np.ma.array(Seq(1, 2, 3), mask = Seq(false, true, false))
  * a.toString      // [1 -- 3]
  * a.sum()         // 4L
  * a(1)            // MaskedConstant (np.ma.masked)
  * a(0) = np.ma.masked
  * }}}
  */
final class MaskedArray[T] private[numscala] (
    private[numscala] val _data: NDArray[T],
    /** The mask, or `null` for `nomask`. Always the same shape as `_data` when present. */
    private[numscala] var _mask: NDArray[Boolean],
    private[numscala] var _fill: Option[T],
    private[numscala] var _hard: Boolean
):
  if _mask != null && !java.util.Arrays.equals(_mask.shapeArr, _data.shapeArr) then
    throw new IllegalArgumentException(
      s"Mask and data not compatible: data shape is ${Shape.str(_data.shapeArr)} and mask shape is ${Shape.str(_mask.shapeArr)}."
    )

  // ------------------------------------------------------------------ attributes

  /** The element dtype. */
  given dtype: DType[T] = _data.dtype
  def shape: Seq[Int] = _data.shape
  def ndim: Int = _data.ndim
  def size: Int = _data.size
  def length: Int = _data.length
  def itemsize: Int = _data.itemsize

  /** The underlying data (`a.data`), a view sharing memory with this array. */
  def data: NDArray[T] = _data

  /** The mask (`a.mask`): `np.ma.nomask` when nothing is masked, otherwise a boolean
    * array (shared, so writes to it change this array's mask).
    */
  def mask: NDArray[Boolean] = if _mask == null then MA.nomask else _mask

  /** Sets the mask (`a.mask = m`); `m` is applied in flat order, cycling if shorter. With a
    * hard mask, already-masked entries stay masked.
    */
  def mask_=(m: NDArray[Boolean]): Unit = setMaskValues(m)
  /** `a.mask = True/False`. */
  def mask_=(b: Boolean): Unit = setMaskValues(NDArray.scalar(b))

  /** A full boolean mask array (`np.ma.getmaskarray`). */
  def maskArray: NDArray[Boolean] =
    if _mask == null then NDArray.zerosOf(DType.Bool, _data.shapeArr.clone()) else _mask

  /** True when the mask is `nomask`. */
  def hasNoMask: Boolean = _mask == null

  /** Whether the mask is hard (`a.hardmask`). */
  def hardmask: Boolean = _hard
  /** Makes the mask hard: masked entries can no longer be unmasked by assignment. */
  def harden_mask(): this.type = { _hard = true; this }
  /** Makes the mask soft (the default). */
  def soften_mask(): this.type = { _hard = false; this }
  /** Copies the mask so that it is no longer shared with other views (`a.unshare_mask()`). */
  def unshare_mask(): this.type =
    if _mask != null then _mask = _mask.copy()
    this
  /** Reduces the mask to `nomask` when no element is masked (`a.shrink_mask()`). */
  def shrink_mask(): this.type =
    if _mask != null && !_mask.any() then _mask = null
    this

  /** The fill value (`a.fill_value`); defaults to `np.ma.default_fill_value(dtype)`. */
  def fill_value: T = _fill.getOrElse(MaskedArray.defaultFill(dtype))
  def fill_value_=(v: T): Unit = _fill = Some(v)
  /** `a.set_fill_value(v)`. */
  def set_fill_value(v: T): Unit = _fill = Some(v)

  private[numscala] def withMask(m: NDArray[Boolean]): MaskedArray[T] = new MaskedArray(_data, m, _fill, _hard)

  // ------------------------------------------------------------------ conversion

  /** Data with masked entries replaced by the fill value (`a.filled()`). Returns the data
    * itself (no copy) when nothing is masked.
    */
  def filled(): NDArray[T] = filled(fill_value)
  /** Data with masked entries replaced by `v` (`a.filled(v)`). */
  def filled(v: T): NDArray[T] =
    if _mask == null then _data
    else NDArray.zipMap(_data, _mask)((x, m) => if m then v else x)

  /** The unmasked values as a 1-D array (`a.compressed()`). */
  def compressed(): NDArray[T] =
    val d = _data.toArray
    if _mask == null then NDArray.fromArray(d, Array(d.length))
    else
      val m = _mask.toArray
      val out = dtype.newArray(m.count(!_))
      var k = 0
      var i = 0
      while i < d.length do
        if !m(i) then
          out(k) = d(i)
          k += 1
        i += 1
      NDArray.fromArray(out, Array(out.length))

  /** Number of unmasked elements (`a.count()`). */
  def count(): Int = if _mask == null then size else size - MaskedArray.countTrue(_mask)
  /** Number of unmasked elements along `axis` (`a.count(axis)`). */
  def count(axis: Axis, keepdims: Boolean = false): NDArray[Int] =
    if _mask == null then Lanes.reduce(_data, axis, keepdims)((_, n) => n)(using DType.Int32)
    else
      Lanes.reduce(_mask, axis, keepdims) { (b, n) =>
        var c = 0
        var i = 0
        while i < n do
          if !b(i) then c += 1
          i += 1
        c
      }(using DType.Int32)

  /** Nested lists with `null` for masked entries (`a.tolist()`); a 0-d array yields its element. */
  def tolist: Any =
    val m = if _mask == null then null else _mask
    def go(idx: Array[Int], axis: Int): Any =
      if axis == ndim then
        if m != null && m.getAt(idx) then null else _data.getAt(idx)
      else
        List.tabulate(_data.shapeArr(axis)) { i =>
          val j = idx.clone()
          j(axis) = i
          go(j, axis + 1)
        }
    go(new Array[Int](ndim), 0)

  /** Elements in C order, `None` for masked ones. */
  def toOptionList: List[Option[T]] =
    val d = _data.toArray
    if _mask == null then d.toList.map(Some(_))
    else d.toList.zip(_mask.toArray).map((x, m) => if m then None else Some(x))

  /** A copy with a different dtype (`a.astype(dtype)`). */
  def astype[U](using u: DType[U]): MaskedArray[U] =
    new MaskedArray(_data.astype[U], if _mask == null then null else _mask.copy(), MaskedArray.castFill(_fill, dtype, u), _hard)

  /** A deep copy of data and mask (`a.copy()`). */
  def copy(): MaskedArray[T] =
    new MaskedArray(_data.copy(), if _mask == null then null else _mask.copy(), _fill, _hard)

  // ------------------------------------------------------------------ indexing

  /** Element at a full multi-index: the value, or `MaskedConstant` when it is masked. */
  def apply(indices: Int*): T | MaskedConstant.type =
    val idx = indices.toArray
    val v = _data.getAt(idx)
    if _mask != null && _mask.getAt(idx) then MaskedConstant else v

  /** The single element of a size-1 array (`a.item()`), or `MaskedConstant`. */
  def item: T | MaskedConstant.type =
    if size != 1 then throw new IllegalArgumentException("can only convert an array of size 1 to a Scala scalar")
    if _mask != null && _mask.flatGet(0) then MaskedConstant else _data.flatGet(0)

  /** General indexing (slices, masks, integer arrays, new axes); basic indexing returns a
    * view sharing data and mask.
    */
  def apply(first: IndexLike, rest: IndexLike*): MaskedArray[T] = index((first +: rest).map(Index.from))

  /** Indexing with a masked boolean array (masked entries of the index count as `false`). */
  def apply(m: MaskedArray[Boolean]): MaskedArray[T] = index(Seq(Index.Mask(m.filled(false))))

  /** Indexing with a pre-built index sequence. */
  def index(items: Seq[Index]): MaskedArray[T] =
    new MaskedArray(_data.index(items), if _mask == null then null else _mask.index(items), _fill, _hard)

  /** Sub-array along the first axis (`a[i]`), a view. */
  def subArray(i: Int): MaskedArray[T] =
    new MaskedArray(_data.subArray(i), if _mask == null then null else _mask.subArray(i), _fill, _hard)

  /** Iterates over the sub-arrays along the first axis. */
  def iterator: Iterator[MaskedArray[T]] =
    if ndim == 0 then throw new IllegalArgumentException("iteration over a 0-d array")
    Iterator.range(0, _data.shapeArr(0)).map(subArray)

  def update(i0: Int, v: T | MaskedConstant.type): Unit = setValue(Seq(Index.At(i0)), v)
  def update(i0: Int, i1: Int, v: T | MaskedConstant.type): Unit = setValue(Seq(Index.At(i0), Index.At(i1)), v)
  def update(i0: Int, i1: Int, i2: Int, v: T | MaskedConstant.type): Unit =
    setValue(Seq(Index.At(i0), Index.At(i1), Index.At(i2)), v)
  def update(i0: IndexLike, v: T | MaskedConstant.type): Unit = set(Seq(i0))(v)
  /** `a[cond] = v` with a masked boolean condition (masked entries count as `false`). */
  def update(cond: MaskedArray[Boolean], v: T | MaskedConstant.type): Unit = set(Seq(cond.filled(false)))(v)
  def update(i0: IndexLike, v: MaskedArray[T]): Unit = setArray(Seq(i0))(v)
  def update(i0: IndexLike, v: NDArray[T]): Unit = setArray(Seq(i0))(MaskedArray.wrap(v))
  def update(i0: IndexLike, i1: IndexLike, v: T | MaskedConstant.type): Unit = set(Seq(i0, i1))(v)
  def update(i0: IndexLike, i1: IndexLike, v: MaskedArray[T]): Unit = setArray(Seq(i0, i1))(v)
  def update(i0: IndexLike, i1: IndexLike, v: NDArray[T]): Unit = setArray(Seq(i0, i1))(MaskedArray.wrap(v))
  def update(i0: IndexLike, i1: IndexLike, i2: IndexLike, v: T | MaskedConstant.type): Unit = set(Seq(i0, i1, i2))(v)

  /** `a[idx] = v` for a scalar or `np.ma.masked`. */
  def set(idx: Seq[IndexLike])(v: T | MaskedConstant.type): Unit = setValue(idx.map(Index.from), v)

  /** `a[idx] = values` (data and mask of `values` are broadcast to the selection). */
  def setArray(idx: Seq[IndexLike])(v: MaskedArray[T]): Unit =
    assign(idx.map(Index.from), v._data, v._mask)

  private def setValue(items: Seq[Index], v: T | MaskedConstant.type): Unit =
    if v.asInstanceOf[AnyRef] eq MaskedConstant then
      if _mask == null then _mask = NDArray.zerosOf(DType.Bool, _data.shapeArr.clone())
      _mask.set(items)(true)
    else assign(items, NDArray.scalar(v.asInstanceOf[T]), null)

  /** NumPy's `MaskedArray.__setitem__` for data `dval` and mask `mval` (`null` = nomask). */
  private def assign(items: Seq[Index], dval: NDArray[T], mval: NDArray[Boolean]): Unit =
    if _mask == null then
      _data.setArray(items)(dval)
      if mval != null then
        _mask = NDArray.zerosOf(DType.Bool, _data.shapeArr.clone())
        _mask.setArray(items)(mval)
    else if !_hard then
      _data.setArray(items)(dval)
      _mask.setArray(items)(if mval == null then NDArray.scalar(false) else mval)
    else
      val cur = _mask.index(items)
      val sel = _data.index(items)
      val newM = if mval == null then cur.copy() else NDArray.zipMap(cur, mval)(_ || _)
      val newD = NDArray.zipMap3(sel, dval, newM)((old, nv, m) => if m then old else nv)
      _data.setArray(items)(newD)
      _mask.setArray(items)(newM)

  private def setMaskValues(m: NDArray[Boolean]): Unit =
    if (m eq MA.nomask) && _mask == null then return
    if _mask == null then
      if !m.any() then return
      _mask = NDArray.zerosOf(DType.Bool, _data.shapeArr.clone())
    val vals = m.toArray
    val n = size
    if vals.isEmpty && n > 0 then throw new IllegalArgumentException("cannot set a mask from an empty array")
    var i = 0
    while i < n do
      val v = vals(i % vals.length)
      _mask.flatSet(i, if _hard then _mask.flatGet(i) || v else v)
      i += 1

  // ------------------------------------------------------------------ shape manipulation

  private def both(f: [X] => NDArray[X] => NDArray[X]): MaskedArray[T] =
    new MaskedArray(f(_data), if _mask == null then null else f(_mask), _fill, _hard)

  /** Gives a new shape (a view when possible), reshaping the mask alike. */
  def reshape(newShape: Int*): MaskedArray[T] = both([X] => (a: NDArray[X]) => a.reshape(newShape*))
  def reshape(newShape: Seq[Int], order: Char): MaskedArray[T] = both([X] => (a: NDArray[X]) => a.reshape(newShape, order))
  /** A 1-D view when possible (`a.ravel()`). */
  def ravel(): MaskedArray[T] = both([X] => (a: NDArray[X]) => a.ravel())
  /** A 1-D copy (`a.flatten()`). */
  def flatten(): MaskedArray[T] = both([X] => (a: NDArray[X]) => a.flatten())
  /** Permutes the axes (a view). */
  def transpose(axes: Int*): MaskedArray[T] = both([X] => (a: NDArray[X]) => a.transpose(axes*))
  /** The transposed array (`a.T`). */
  def T: MaskedArray[T] = transpose()
  def swapaxes(a1: Int, a2: Int): MaskedArray[T] = both([X] => (a: NDArray[X]) => a.swapaxes(a1, a2))
  def moveaxis(source: Int, destination: Int): MaskedArray[T] = both([X] => (a: NDArray[X]) => a.moveaxis(source, destination))
  def squeeze(axes: Int*): MaskedArray[T] = both([X] => (a: NDArray[X]) => a.squeeze(axes*))
  def expandDims(axis: Int): MaskedArray[T] = both([X] => (a: NDArray[X]) => a.expandDims(axis))
  /** Broadcast view to `shape` (`np.broadcast_to`). */
  def broadcastTo(newShape: Int*): MaskedArray[T] = both([X] => (a: NDArray[X]) => a.broadcastTo(newShape*))

  // ------------------------------------------------------------------ arithmetic

  import MaskedArray.{arith, domained, wrap, compareOp}

  def +[U](o: MaskedArray[U])(using p: NumPromote[T, U]): MaskedArray[p.Out] = arith(this, o, p.dtype, Arith.Add)
  def -[U](o: MaskedArray[U])(using p: NumPromote[T, U]): MaskedArray[p.Out] = arith(this, o, p.dtype, Arith.Sub)
  def *[U](o: MaskedArray[U])(using p: NumPromote[T, U]): MaskedArray[p.Out] = arith(this, o, p.dtype, Arith.Mul)
  def /[U](o: MaskedArray[U])(using p: DivPromote[T, U]): MaskedArray[p.Out] = domained(this, o, p.dtype, Arith.Div)
  def **[U](o: MaskedArray[U])(using p: NumPromote[T, U]): MaskedArray[p.Out] = MaskedArray.power(this, o, p.dtype)
  def %(o: MaskedArray[T])(using d: RealDType[T]): MaskedArray[T] = MaskedArray.domainedF(this, o, d)(d.mod)
  def floorDiv(o: MaskedArray[T])(using d: RealDType[T]): MaskedArray[T] = MaskedArray.domainedF(this, o, d)(d.floorDiv)

  def +[U](o: NDArray[U])(using p: NumPromote[T, U]): MaskedArray[p.Out] = arith(this, wrap(o), p.dtype, Arith.Add)
  def -[U](o: NDArray[U])(using p: NumPromote[T, U]): MaskedArray[p.Out] = arith(this, wrap(o), p.dtype, Arith.Sub)
  def *[U](o: NDArray[U])(using p: NumPromote[T, U]): MaskedArray[p.Out] = arith(this, wrap(o), p.dtype, Arith.Mul)
  def /[U](o: NDArray[U])(using p: DivPromote[T, U]): MaskedArray[p.Out] = domained(this, wrap(o), p.dtype, Arith.Div)
  def **[U](o: NDArray[U])(using p: NumPromote[T, U]): MaskedArray[p.Out] = MaskedArray.power(this, wrap(o), p.dtype)
  def %(o: NDArray[T])(using d: RealDType[T]): MaskedArray[T] = MaskedArray.domainedF(this, wrap(o), d)(d.mod)
  def floorDiv(o: NDArray[T])(using d: RealDType[T]): MaskedArray[T] = MaskedArray.domainedF(this, wrap(o), d)(d.floorDiv)

  def +(s: T)(using d: NumDType[T]): MaskedArray[T] = arith(this, MaskedArray.scalarOf(s), d, Arith.Add)
  def -(s: T)(using d: NumDType[T]): MaskedArray[T] = arith(this, MaskedArray.scalarOf(s), d, Arith.Sub)
  def *(s: T)(using d: NumDType[T]): MaskedArray[T] = arith(this, MaskedArray.scalarOf(s), d, Arith.Mul)
  def /(s: T)(using t: ToInexact[T]): MaskedArray[t.Out] = domained(this, MaskedArray.scalarOf(s), t.dtype, Arith.Div)
  def **(s: T)(using d: NumDType[T]): MaskedArray[T] = MaskedArray.power(this, MaskedArray.scalarOf(s), d)
  def %(s: T)(using d: RealDType[T]): MaskedArray[T] = MaskedArray.domainedF(this, MaskedArray.scalarOf(s), d)(d.mod)
  def floorDiv(s: T)(using d: RealDType[T]): MaskedArray[T] = MaskedArray.domainedF(this, MaskedArray.scalarOf(s), d)(d.floorDiv)

  def unary_-(using d: NumDType[T]): MaskedArray[T] = MaskedArray.unary(this, d, d.negate, None)
  def unary_+ : MaskedArray[T] = copy()

  // ------------------------------------------------------------------ comparisons

  def <[U](o: MaskedArray[U])(using p: Promote[T, U]): MaskedArray[Boolean] =
    compareOp(this, o, Ops.compare(_data, o._data, p.dtype, CmpOp.Lt), None)
  def <=[U](o: MaskedArray[U])(using p: Promote[T, U]): MaskedArray[Boolean] =
    compareOp(this, o, Ops.compare(_data, o._data, p.dtype, CmpOp.Le), None)
  def >[U](o: MaskedArray[U])(using p: Promote[T, U]): MaskedArray[Boolean] =
    compareOp(this, o, Ops.compare(_data, o._data, p.dtype, CmpOp.Gt), None)
  def >=[U](o: MaskedArray[U])(using p: Promote[T, U]): MaskedArray[Boolean] =
    compareOp(this, o, Ops.compare(_data, o._data, p.dtype, CmpOp.Ge), None)
  /** Elementwise `==` (masked where either side is masked). */
  def ===[U](o: MaskedArray[U])(using p: Promote[T, U]): MaskedArray[Boolean] =
    compareOp(this, o, Ops.equal(_data, o._data, p.dtype, true), Some(true))
  /** Elementwise `!=`. */
  def =!=[U](o: MaskedArray[U])(using p: Promote[T, U]): MaskedArray[Boolean] =
    compareOp(this, o, Ops.equal(_data, o._data, p.dtype, false), Some(false))

  def <[U](o: NDArray[U])(using p: Promote[T, U]): MaskedArray[Boolean] = this < wrap(o)
  def <=[U](o: NDArray[U])(using p: Promote[T, U]): MaskedArray[Boolean] = this <= wrap(o)
  def >[U](o: NDArray[U])(using p: Promote[T, U]): MaskedArray[Boolean] = this > wrap(o)
  def >=[U](o: NDArray[U])(using p: Promote[T, U]): MaskedArray[Boolean] = this >= wrap(o)
  def ===[U](o: NDArray[U])(using p: Promote[T, U]): MaskedArray[Boolean] = this === wrap(o)
  def =!=[U](o: NDArray[U])(using p: Promote[T, U]): MaskedArray[Boolean] = this =!= wrap(o)

  def <(s: T): MaskedArray[Boolean] = this < MaskedArray.scalarOf(s)
  def <=(s: T): MaskedArray[Boolean] = this <= MaskedArray.scalarOf(s)
  def >(s: T): MaskedArray[Boolean] = this > MaskedArray.scalarOf(s)
  def >=(s: T): MaskedArray[Boolean] = this >= MaskedArray.scalarOf(s)
  def ===(s: T): MaskedArray[Boolean] = this === MaskedArray.scalarOf(s)
  def =!=(s: T): MaskedArray[Boolean] = this =!= MaskedArray.scalarOf(s)

  // ------------------------------------------------------------------ reductions

  /** Sum of the unmasked elements; `MaskedConstant` if all are masked. */
  def sum()(using s: SumOf[T]): s.Out | MaskedConstant.type = MAReduce.full(this, MAReduce.sumF(dtype, s.dtype))
  /** Sum along `axis`, masked where every element of a lane is masked. */
  def sum(axis: Axis, keepdims: Boolean = false)(using s: SumOf[T]): MaskedArray[s.Out] =
    MAReduce.lanes(this, axis, keepdims, s.dtype, s.dtype.zero)(MAReduce.sumF(dtype, s.dtype))
  def prod()(using s: SumOf[T]): s.Out | MaskedConstant.type = MAReduce.full(this, MAReduce.prodF(dtype, s.dtype))
  def prod(axis: Axis, keepdims: Boolean = false)(using s: SumOf[T]): MaskedArray[s.Out] =
    MAReduce.lanes(this, axis, keepdims, s.dtype, s.dtype.zero)(MAReduce.prodF(dtype, s.dtype))
  /** Mean of the unmasked elements. */
  def mean()(using t: ToInexact[T]): t.Out | MaskedConstant.type =
    if _mask == null then _data.mean() else MAReduce.full(this, MAReduce.meanF(dtype, t.dtype))
  def mean(axis: Axis, keepdims: Boolean = false)(using t: ToInexact[T]): MaskedArray[t.Out] =
    MAReduce.lanes(this, axis, keepdims, t.dtype, t.dtype.zero)(MAReduce.meanF(dtype, t.dtype))
  /** Variance of the unmasked elements (`a.var(ddof=...)`). */
  def variance(ddof: Int = 0)(using r: RealOf[T]): r.Out | MaskedConstant.type =
    if _mask == null then _data.variance(ddof) else MAReduce.fullVar(this, ddof, r.dtype)
  def variance(axis: Axis, ddof: Int, keepdims: Boolean)(using r: RealOf[T]): MaskedArray[r.Out] =
    MAReduce.lanesVar(this, axis, ddof, keepdims, r.dtype)
  /** Standard deviation of the unmasked elements. */
  def std(ddof: Int = 0)(using r: RealOf[T]): r.Out | MaskedConstant.type =
    variance(ddof) match
      case MaskedConstant => MaskedConstant
      case v => r.dtype.sqrt(v.asInstanceOf[r.Out])
  def std(axis: Axis, ddof: Int, keepdims: Boolean)(using r: RealOf[T]): MaskedArray[r.Out] =
    val v = MAReduce.lanesVar(this, axis, ddof, keepdims, r.dtype)
    v._data.mapInPlace(r.dtype.sqrt)
    if _mask != null && v._mask == null then v._mask = NDArray.zerosOf(DType.Bool, v._data.shapeArr.clone())
    v
  /** Maximum of the unmasked elements. */
  def max(): T | MaskedConstant.type = MAReduce.full(this, (b, n) => Reduce.extremeBuf(dtype, b, n, true))
  def max(axis: Axis, keepdims: Boolean = false): MaskedArray[T] = MAReduce.extreme(this, axis, keepdims, true)
  /** Minimum of the unmasked elements. */
  def min(): T | MaskedConstant.type = MAReduce.full(this, (b, n) => Reduce.extremeBuf(dtype, b, n, false))
  def min(axis: Axis, keepdims: Boolean = false): MaskedArray[T] = MAReduce.extreme(this, axis, keepdims, false)
  /** Peak to peak of the unmasked elements. */
  def ptp()(using d: NumDType[T]): T | MaskedConstant.type =
    (max(), min()) match
      case (MaskedConstant, _) | (_, MaskedConstant) => MaskedConstant
      case (hi, lo) => d.minus(hi.asInstanceOf[T], lo.asInstanceOf[T])
  def ptp(axis: Axis)(using d: NumDType[T]): MaskedArray[T] = max(axis) - min(axis)
  /** Index (flat) of the maximum, ignoring masked values (`a.argmax()`). */
  def argmax(): Int = filled(MaskedArray.maxFill(dtype)).argmax()
  def argmax(axis: Int): NDArray[Int] = filled(MaskedArray.maxFill(dtype)).argmax(axis)
  /** Index (flat) of the minimum, ignoring masked values (`a.argmin()`). */
  def argmin(): Int = filled(MaskedArray.minFill(dtype)).argmin()
  def argmin(axis: Int): NDArray[Int] = filled(MaskedArray.minFill(dtype)).argmin(axis)
  /** True when all unmasked elements are true; `MaskedConstant` if all are masked. */
  def all(): Boolean | MaskedConstant.type =
    if _mask != null && _mask.all() then MaskedConstant else filled(dtype.fromBoolean(true)).all()
  def all(axis: Axis, keepdims: Boolean = false): MaskedArray[Boolean] =
    val d = filled(dtype.fromBoolean(true)).all(axis, keepdims)
    new MaskedArray(d, if _mask == null then null else _mask.all(axis, keepdims), None, false)
  /** True when any unmasked element is true; `MaskedConstant` if all are masked. */
  def any(): Boolean | MaskedConstant.type =
    if _mask != null && _mask.all() then MaskedConstant else filled(dtype.fromBoolean(false)).any()
  def any(axis: Axis, keepdims: Boolean = false): MaskedArray[Boolean] =
    val d = filled(dtype.fromBoolean(false)).any(axis, keepdims)
    new MaskedArray(d, if _mask == null then null else _mask.all(axis, keepdims), None, false)
  /** Cumulative sum of the flattened array, treating masked values as 0 (mask kept). */
  def cumsum()(using s: SumOf[T]): MaskedArray[s.Out] =
    new MaskedArray(filled(dtype.zero).cumsum(), if _mask == null then null else _mask.flatten(), None, _hard)
  def cumsum(axis: Int)(using s: SumOf[T]): MaskedArray[s.Out] =
    new MaskedArray(filled(dtype.zero).cumsum(axis), if _mask == null then null else _mask.copy(), None, _hard)
  /** Cumulative product, treating masked values as 1 (mask kept). */
  def cumprod()(using s: SumOf[T]): MaskedArray[s.Out] =
    new MaskedArray(filled(dtype.one).cumprod(), if _mask == null then null else _mask.flatten(), None, _hard)
  def cumprod(axis: Int)(using s: SumOf[T]): MaskedArray[s.Out] =
    new MaskedArray(filled(dtype.one).cumprod(axis), if _mask == null then null else _mask.copy(), None, _hard)
  /** Anomalies: deviations from the mean along `axis` (`a.anom(axis)`). */
  def anom()(using t: ToInexact[T]): MaskedArray[t.Out] =
    val x = astype(using t.dtype)
    mean() match
      case MaskedConstant => MaskedArray.allMasked(x)
      case m => arith(x, MaskedArray.scalarOf(m.asInstanceOf[t.Out])(using t.dtype), t.dtype, Arith.Sub)
  def anom(axis: Int)(using t: ToInexact[T]): MaskedArray[t.Out] =
    arith(astype(using t.dtype), mean(axis, keepdims = true), t.dtype, Arith.Sub)

  // ------------------------------------------------------------------ misc

  /** Sorted indices along `axis`; masked values go to the end (`endwith`) or start. */
  def argsort(axis: Int = -1, endwith: Boolean = true): NDArray[Int] =
    filled(MAReduce.sortFill(dtype, endwith)).argsort(axis)
  /** Sorts in place along `axis`, masked values last (`a.sort()`). */
  def sort(axis: Int = -1, endwith: Boolean = true): Unit =
    val s = MAReduce.sorted(this, axis, endwith)
    _data := s._data
    if s._mask != null then
      if _mask == null then _mask = s._mask.copy() else _mask := s._mask
  /** Indices of unmasked non-zero elements (`a.nonzero()`). */
  def nonzero: Seq[NDArray[Int]] = filled(dtype.zero).nonzero
  /** Clips the data to `[lo, hi]`, keeping the mask. */
  def clip(lo: T, hi: T)(using d: RealDType[T]): MaskedArray[T] =
    new MaskedArray(_data.clip(lo, hi), if _mask == null then null else _mask.copy(), _fill, _hard)
  /** Rounds the data, keeping the mask. */
  def round(decimals: Int = 0)(using d: NumDType[T]): MaskedArray[T] =
    new MaskedArray(_data.round(decimals), if _mask == null then null else _mask.copy(), _fill, _hard)
  /** Matrix product treating masked values as 0 (`a.dot(b)`). */
  def dot(o: MaskedArray[T])(using d: NumDType[T]): MaskedArray[T] = MA.dot(this, o)

  // ------------------------------------------------------------------ Object methods

  /** NumPy-style `str()`: masked entries print as `--`, e.g. `[1 -- 3]`. */
  override def toString: String = MAFormat.str(this)
  /** NumPy-style `repr()`: `masked_array(data=[1, --, 3], mask=[...], fill_value=...)`. */
  def repr: String = MAFormat.repr(this)

  /** Structural equality: same dtype, shape and mask, and equal unmasked data. */
  override def equals(other: Any): Boolean = other match
    case o: MaskedArray[?] =>
      (o.dtype eq dtype) && java.util.Arrays.equals(o._data.shapeArr, _data.shapeArr) &&
      maskArray == o.maskArray &&
      filled(dtype.zero) == o.asInstanceOf[MaskedArray[T]].filled(dtype.zero)
    case _ => false

  override def hashCode: Int = filled(dtype.zero).hashCode * 31 + maskArray.hashCode

/** Construction helpers and elementwise kernels for [[MaskedArray]]. */
object MaskedArray:
  /** Wraps data and an optional mask (`null` = nomask) without copying. */
  def apply[T](data: NDArray[T], mask: NDArray[Boolean] = null): MaskedArray[T] =
    new MaskedArray(data, if mask == null || (mask eq MA.nomask) then null else MaskedArray.fitMask(mask, data.shapeArr), None, false)

  /** An ndarray viewed as a masked array with `nomask` (no copy). */
  def wrap[T](a: NDArray[T]): MaskedArray[T] = new MaskedArray(a, null, None, false)
  private[numscala] def scalarOf[T](s: T)(using d: DType[T]): MaskedArray[T] = wrap(NDArray.scalar(s))
  private[numscala] def allMasked[T](a: MaskedArray[T]): MaskedArray[T] =
    new MaskedArray(a._data, NDArray.fillOf(DType.Bool, a._data.shapeArr.clone(), true), a._fill, a._hard)

  /** Broadcasts/reshapes a user mask to `shape` (a scalar mask fills the whole array). */
  private[numscala] def fitMask(m: NDArray[Boolean], shape: Array[Int]): NDArray[Boolean] =
    if java.util.Arrays.equals(m.shapeArr, shape) then m
    else if m.size == 1 then NDArray.fillOf(DType.Bool, shape.clone(), m.flatGet(0))
    else if m.size == Shape.size(shape) then m.reshape(shape.toIndexedSeq*).copy()
    else
      try m.broadcastTo(shape.toIndexedSeq*).copy()
      catch
        case _: IllegalArgumentException =>
          throw new IllegalArgumentException(
            s"Mask and data not compatible: data size is ${Shape.size(shape)}, mask size is ${m.size}."
          )

  private[numscala] def countTrue(m: NDArray[Boolean]): Int =
    var c = 0
    m.foreach(b => if b then c += 1)
    c

  // ---------------------------------------------------------------- fill values

  /** NumPy's `default_fill_value`: 999999 (int), 1e20 (float), True (bool), 'N/A' (str), 1e20+0j. */
  def defaultFill[T](d: DType[T]): T = d.kind match
    case 'b' => d.fromBoolean(true)
    case 'i' => d.fromLong(999999L)
    case 'f' => d.fromDouble(1e20)
    case 'c' => d.fromComplex(Complex(1e20, 0.0))
    case _ => d.fromString("N/A")

  /** NumPy's `minimum_fill_value`: the largest value of the dtype (used to ignore entries in `min`). */
  def minFill[T](d: DType[T]): T = d match
    case DType.Bool => d.fromBoolean(true)
    case i: IntDType[T] => i.maxValue
    case _: FloatDType[T] => d.fromDouble(Double.PositiveInfinity)
    case _: ComplexDType => d.fromComplex(Complex(Double.PositiveInfinity, Double.PositiveInfinity))
    case _ => throw new IllegalArgumentException(s"Unsuitable type ${d.name} for calculating minimum.")

  /** NumPy's `maximum_fill_value`: the smallest value of the dtype. */
  def maxFill[T](d: DType[T]): T = d match
    case DType.Bool => d.fromBoolean(false)
    case i: IntDType[T] => i.minValue
    case _: FloatDType[T] => d.fromDouble(Double.NegativeInfinity)
    case _: ComplexDType => d.fromComplex(Complex(Double.NegativeInfinity, Double.NegativeInfinity))
    case _ => throw new IllegalArgumentException(s"Unsuitable type ${d.name} for calculating maximum.")

  private[numscala] def castFill[A, U](f: Option[A], src: DType[A], dst: DType[U]): Option[U] =
    f.flatMap(x => scala.util.Try(dst.castFrom(src, x)).toOption)

  // ---------------------------------------------------------------- casting rules

  private def kindRank(d: DType[?]): Int = d.kind match
    case 'b' => 0
    case 'i' => 1
    case 'f' => 2
    case 'c' => 3
    case _ => 4

  /** `np.can_cast(src, dst, 'safe')`. */
  private[numscala] def canCastSafe(src: DType[?], dst: DType[?]): Boolean =
    (src eq dst) || (!src.isString && !dst.isString && (DType.promote(src, dst) eq dst))

  /** `np.can_cast(src, dst, 'same_kind')`. */
  private[numscala] def canCastSameKind(src: DType[?], dst: DType[?]): Boolean =
    (src eq dst) || (!src.isString && !dst.isString && kindRank(src) <= kindRank(dst))

  private[numscala] def isFinite[T](d: DType[T], x: T): Boolean = d match
    case i: InexactDType[T] => i.isFinite(x)
    case _ => true

  private[numscala] def absD[T](d: DType[T], x: T): Double =
    if d.isComplex then d.toComplex(x).abs else math.abs(d.toDouble(x))

  /** Full masks of `a` and `b` or'ed with broadcasting; `null` when both are nomask. */
  private[numscala] def orMasks(a: MaskedArray[?], b: MaskedArray[?]): NDArray[Boolean] =
    if a._mask == null && b._mask == null then null
    else NDArray.zipMap(a.maskArray, b.maskArray)(_ || _)

  /** `np.ma.mask_or` on internal masks (`null` = nomask), with shrinking. */
  private[numscala] def maskOr(m1: NDArray[Boolean], m2: NDArray[Boolean], shrink: Boolean = true): NDArray[Boolean] =
    val r =
      if m1 == null && m2 == null then null
      else if m1 == null then m2.copy()
      else if m2 == null then m1.copy()
      else NDArray.zipMap(m1, m2)(_ || _)
    if r != null && shrink && !r.any() then null else r

  /** Replaces `res` by `da` (cast) where `m` is true. */
  private def revert[A, U](res: NDArray[U], m: NDArray[Boolean], da: NDArray[A], out: DType[U]): NDArray[U] =
    val src = da.dtype
    NDArray.zipMap3(res, m, da)((r, mm, x) => if mm then out.castFrom(src, x) else r)(using out)

  // ---------------------------------------------------------------- elementwise kernels

  /** NumPy's `_MaskedBinaryOperation` (add, subtract, multiply, ...). */
  private[numscala] def binaryWith[A, B, U](a: MaskedArray[A], b: MaskedArray[B], out: DType[U])(
      compute: (NDArray[A], NDArray[B]) => NDArray[U]
  ): MaskedArray[U] =
    var res = compute(a._data, b._data)
    val m = orMasks(a, b)
    if m != null && m.any() then
      res = revert(res, m, a._data, out)
    val (fill, hard) = inherit(a, b, out)
    new MaskedArray(res, if m == null then null else fitMask(m, res.shapeArr), fill, hard)

  private[numscala] def inherit[A, B, U](a: MaskedArray[A], b: MaskedArray[B], out: DType[U]): (Option[U], Boolean) =
    (castFill(a._fill, a.dtype, out).orElse(castFill(b._fill, b.dtype, out)), a._hard)

  private[numscala] def arith[A, B, U](a: MaskedArray[A], b: MaskedArray[B], out: NumDType[U], op: Arith): MaskedArray[U] =
    binaryWith(a, b, out)((x, y) => Ops.arith(x, y, out, op))

  /** NumPy's `_DomainedBinaryOperation` with the safe-division domain (`divide`, `floor_divide`, `remainder`). */
  private[numscala] def domainedWith[A, B, U](a: MaskedArray[A], b: MaskedArray[B], out: DType[U])(
      compute: (NDArray[A], NDArray[B]) => NDArray[U]
  ): MaskedArray[U] =
    val res = compute(a._data, b._data)
    val tiny = java.lang.Double.MIN_NORMAL
    val (sa, sb) = (a.dtype, b.dtype)
    val dom = NDArray.zipMap(a._data, b._data)((x, y) => absD(sa, x) * tiny >= absD(sb, y))(using DType.Bool)
    val ma = a.maskArray
    val mb = b.maskArray
    val sh = res.shapeArr
    val base = NDArray.zipMap(res, dom)((r, d) => d || !isFinite(out, r))(using DType.Bool)
    val m = NDArray.zipMap3(base, ma, mb)(_ || _ || _)(using DType.Bool)
    val safe = canCastSafe(a.dtype, out)
    val src = a.dtype
    val data = NDArray.zipMap3(res, m, a._data)((r, mm, x) =>
      if !mm then r else if safe then out.castFrom(src, x) else out.zero
    )(using out)
    val (fill, hard) = inherit(a, b, out)
    new MaskedArray(data, fitMask(m, sh), fill, hard)

  private[numscala] def domained[A, B, U](a: MaskedArray[A], b: MaskedArray[B], out: NumDType[U], op: Arith): MaskedArray[U] =
    domainedWith(a, b, out)((x, y) => Ops.arith(x, y, out, op))

  private[numscala] def domainedF[T](a: MaskedArray[T], b: MaskedArray[T], d: DType[T])(f: (T, T) => T): MaskedArray[T] =
    domainedWith(a, b, d)((x, y) => NDArray.zipMap(x, y)(f)(using d))

  /** NumPy's `ma.power`: masks the inputs' masks and non-finite results (data there = fill value). */
  private[numscala] def power[A, B, U](a: MaskedArray[A], b: MaskedArray[B], out: NumDType[U]): MaskedArray[U] =
    val m0 = maskOr(a._mask, b._mask)
    var res = Ops.arith(a._data, b._data, out, Arith.Pow)
    val m = if m0 == null then null else fitMask(m0, res.shapeArr)
    if m != null then res = revert(res, m, a._data, out)
    val invalid = res.map(x => !isFinite(out, x))(using DType.Bool)
    val (fill, hard) = inherit(a, b, out)
    val result = new MaskedArray(res, if m == null then null else NDArray.zipMap(m, invalid)(_ || _), fill, hard)
    if invalid.any() then
      if result._mask == null then result._mask = invalid
      val fv = result.fill_value
      result._data := NDArray.zipMap(result._data, invalid)((x, bad) => if bad then fv else x)(using out)
    result

  /** NumPy's `_MaskedUnaryOperation`: optional `domain` (true = invalid input) plus non-finite results. */
  private[numscala] def unary[T, U](a: MaskedArray[T], out: DType[U], f: T => U, domain: Option[T => Boolean]): MaskedArray[U] =
    var res = a._data.map(f)(using out)
    val m: NDArray[Boolean] = domain match
      case None => if a._mask == null then null else a._mask.copy()
      case Some(dom) =>
        val bad = NDArray.zipMap(res, a._data)((r, x) => !isFinite(out, r) || dom(x))(using DType.Bool)
        if a._mask == null then bad else NDArray.zipMap(bad, a._mask)(_ || _)
    if m != null && canCastSameKind(a.dtype, out) then res = revert(res, m, a._data, out)
    new MaskedArray(res, m, castFill(a._fill, a.dtype, out), a._hard)

  /** NumPy's `MaskedArray._comparison`. `eqne` is `Some(true)` for `==`, `Some(false)` for `!=`. */
  private[numscala] def compareOp[A, B](a: MaskedArray[A], b: MaskedArray[B], check0: NDArray[Boolean], eqne: Option[Boolean]): MaskedArray[Boolean] =
    var check = check0
    var mask = maskOr(a._mask, b._mask)
    if mask != null then
      eqne.foreach { eq =>
        val sm = a.maskArray
        val om = b.maskArray
        check = NDArray.zipMap3(check, NDArray.zipMap(sm, om)((x, y) => (x == y) == eq), mask)((c, e, mm) => if mm then e else c)
      }
      if !java.util.Arrays.equals(mask.shapeArr, check.shapeArr) then mask = mask.broadcastTo(check.shapeArr.toIndexedSeq*).copy()
    new MaskedArray(check, mask, castFill(a._fill, a.dtype, DType.Bool), a._hard)
