package numscala

/** Reduction and lane kernels for [[MaskedArray]]: every reduction gathers the unmasked
  * values of a lane and applies an ordinary buffer reduction from [[Reduce]].
  */
private[numscala] object MAReduce:

  def sumF[T, U](src: DType[T], d: NumDType[U]): (Array[T], Int) => U = (b, n) => Reduce.sumBuf(src, b, n, d)
  def prodF[T, U](src: DType[T], d: NumDType[U]): (Array[T], Int) => U = (b, n) => Reduce.prodBuf(src, b, n, d)
  def meanF[T, U](src: DType[T], d: InexactDType[U]): (Array[T], Int) => U = (b, n) => Reduce.meanBuf(src, b, n, d)

  /** The unmasked values in C order and their number. */
  def gather[T](a: MaskedArray[T]): (Array[T], Int) =
    val d = a._data.toArray
    if a._mask == null then (d, d.length)
    else
      val m = a._mask.toArray
      var k = 0
      var i = 0
      while i < d.length do
        if !m(i) then
          d(k) = d(i)
          k += 1
        i += 1
      (d, k)

  /** Full reduction; `MaskedConstant` when a mask exists and every element is masked. */
  def full[T, U](a: MaskedArray[T], f: (Array[T], Int) => U): U | MaskedConstant.type =
    val (b, k) = gather(a)
    if a._mask != null && k == 0 then MaskedConstant else f(b, k)

  /** Flat C-order positions with the shape of `a`. */
  def positions(a: MaskedArray[?]): NDArray[Int] = NDArray.tabulate(a.shape*)(i => i)(using DType.Int32)

  /** Per-lane reduction over unmasked values: (results, unmasked counts). `empty` is used for
    * lanes without unmasked values.
    */
  def lanesRaw[T, U](a: MaskedArray[T], axis: Axis, keepdims: Boolean, out: DType[U], empty: U)(
      f: (Array[T], Int) => U
  ): (NDArray[U], NDArray[Int]) =
    val d = a._data.toArray
    val m = a.maskArray.toArray
    val pos = positions(a)
    var tmp = a.dtype.newArray(0)
    val res = Lanes.reduce(pos, axis, keepdims) { (b, n) =>
      if tmp.length < n then tmp = a.dtype.newArray(n)
      var k = 0
      var i = 0
      while i < n do
        val p = b(i)
        if !m(p) then
          tmp(k) = d(p)
          k += 1
        i += 1
      if k == 0 then empty else f(tmp, k)
    }(using out)
    val cnt = Lanes.reduce(pos, axis, keepdims) { (b, n) =>
      var k = 0
      var i = 0
      while i < n do
        if !m(b(i)) then k += 1
        i += 1
      k
    }(using DType.Int32)
    (res, cnt)

  /** Axis reduction masked where a whole lane is masked (`nomask` stays `nomask`). */
  def lanes[T, U](a: MaskedArray[T], axis: Axis, keepdims: Boolean, out: DType[U], empty: U)(
      f: (Array[T], Int) => U
  ): MaskedArray[U] =
    if a._mask == null then new MaskedArray(Lanes.reduce(a._data, axis, keepdims)(f)(using out), null, None, false)
    else
      val (res, cnt) = lanesRaw(a, axis, keepdims, out, empty)(f)
      new MaskedArray(res, cnt.map(_ == 0)(using DType.Bool), None, false)

  /** `max`/`min` along an axis; fully masked lanes hold the default fill value. */
  def extreme[T](a: MaskedArray[T], axis: Axis, keepdims: Boolean, wantMax: Boolean): MaskedArray[T] =
    val d = a.dtype
    if a._mask == null then
      val r = if wantMax then a._data.max(axis, keepdims) else a._data.min(axis, keepdims)
      new MaskedArray(r, null, None, false)
    else
      val (res, cnt) = lanesRaw(a, axis, keepdims, d, MaskedArray.defaultFill(d))((b, n) => Reduce.extremeBuf(d, b, n, wantMax))
      new MaskedArray(res, cnt.map(_ == 0)(using DType.Bool), None, false)

  /** Sum of squared deviations from the mean of `b(0 until n)` (complex: squared moduli). */
  private def ssq[T](src: DType[T], b: Array[T], n: Int): Double =
    if n == 0 then 0.0 else Reduce.varBuf(src, b, n, 0, DType.Float64) * n

  /** NumPy's masked `var`: sum of squares over `count - ddof`, masked when that is <= 0. */
  def fullVar[T, U](a: MaskedArray[T], ddof: Int, d: FloatDType[U]): U | MaskedConstant.type =
    val (b, k) = gather(a)
    val cnt = k - ddof
    if k == 0 || cnt == 0 then MaskedConstant else d.fromDouble(ssq(a.dtype, b, k) / cnt)

  def lanesVar[T, U](a: MaskedArray[T], axis: Axis, ddof: Int, keepdims: Boolean, d: FloatDType[U]): MaskedArray[U] =
    if a._mask == null then new MaskedArray(a._data.variance(axis, ddof, keepdims)(using realOf(d)), null, None, false)
    else
      val (res, cnt) = lanesRaw(a, axis, keepdims, d, d.zero) { (b, n) =>
        val c = n - ddof
        if c == 0 then d.zero else d.fromDouble(ssq(a.dtype, b, n) / c)
      }
      val m = cnt.map(_ - ddof <= 0)(using DType.Bool)
      new MaskedArray(res, if m.any() then m else null, None, false)

  private def realOf[T, U](d: FloatDType[U]): RealOf.Aux[T, U] =
    new RealOf[T]:
      type Out = U
      val dtype: FloatDType[U] = d

  /** The value masked entries are replaced with for sorting (`endwith`: masked last). */
  def sortFill[T](d: DType[T], endwith: Boolean): T =
    if endwith then (if d.isFloating then d.fromDouble(Double.NaN) else MaskedArray.minFill(d))
    else MaskedArray.maxFill(d)

  /** For each output element along `axis`, the flat position of the source element after a
    * stable sort of the filled keys.
    */
  def sortPositions[T](a: MaskedArray[T], axis: Int, endwith: Boolean, wantArgs: Boolean): NDArray[Int] =
    val keys = a.filled(sortFill(a.dtype, endwith)).toArray
    val pos = positions(a)
    var kb = a.dtype.newArray(0)
    var perm = new Array[Int](0)
    Lanes.transform(pos, axis) { (in: Array[Int], n: Int, out: Array[Int]) =>
      if kb.length < n then
        kb = a.dtype.newArray(n)
        perm = new Array[Int](n)
      var i = 0
      while i < n do
        kb(i) = keys(in(i))
        i += 1
      Sorting.argsortBuffer(a.dtype, kb, n, perm)
      i = 0
      while i < n do
        out(i) = if wantArgs then perm(i) else in(perm(i))
        i += 1
    }(using DType.Int32)

  /** A sorted copy along `axis` (masked values last when `endwith`). */
  def sorted[T](a: MaskedArray[T], axis: Int, endwith: Boolean): MaskedArray[T] =
    if a.ndim == 0 then a.copy()
    else
      val sp = sortPositions(a, axis, endwith, wantArgs = false)
      val d = a._data.toArray
      val data = sp.map(p => d(p))(using a.dtype)
      val mask =
        if a._mask == null then null
        else
          val m = a._mask.toArray
          sp.map(p => m(p))(using DType.Bool)
      new MaskedArray(data, mask, a._fill, a._hard)

  /** Median of sorted values `b(0 until n)` in dtype `out` (NaN if any NaN). */
  def medianSorted[T, U](src: DType[T], b: Array[T], n: Int, out: InexactDType[U]): U =
    if n == 0 then out.nan
    else
      var i = 0
      var hasNaN = false
      while i < n do
        if src.isNaN(b(i)) then hasNaN = true
        i += 1
      if hasNaN then out.nan
      else
        Sorting.sortBuffer(src, b, n)
        val h = n / 2
        if n % 2 == 1 then out.castFrom(src, b(h))
        else out.div(out.plus(out.castFrom(src, b(h - 1)), out.castFrom(src, b(h))), out.fromInt(2))
