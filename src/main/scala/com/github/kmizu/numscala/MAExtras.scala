package com.github.kmizu.numscala

/** `np.ma` aliases, shape manipulation, selection/placement and small array helpers. */
trait MAExtras:
  import MA.toMA

  /** Applies `f` to data and mask alike (the mask stays `nomask` when absent). */
  private def mapBoth[T](m: MaskedArray[T])(f: [X] => NDArray[X] => NDArray[X]): MaskedArray[T] =
    new MaskedArray(f(m._data), if m._mask == null then null else f(m._mask), m._fill, m._hard)

  /** NumPy's `_fromnxfunction`: applies `f` to the data and to the full mask array. */
  private def fullBoth[T](m: MaskedArray[T])(f: [X] => NDArray[X] => NDArray[X]): MaskedArray[T] =
    new MaskedArray(f(m._data), f(m.maskArray), None, false)

  private def sliceAxis(nd: Int, ax: Int, from: Int, until: Int): Seq[Index] =
    (0 until nd).map(k => if k == ax then Index.Slice(Some(from), Some(until), 1) else Index.All)

  // ------------------------------------------------------------------ aliases

  /** `np.ma.amax(a)` (alias of [[MA.max]]). */
  def amax[T](a: MaskedArrayLike[T]): T | MaskedConstant.type = toMA(a).max()
  /** `np.ma.amax(a, axis)`. */
  def amax[T](a: MaskedArrayLike[T], axis: Axis, keepdims: Boolean = false): MaskedArray[T] = toMA(a).max(axis, keepdims)
  /** `np.ma.amin(a)` (alias of [[MA.min]]). */
  def amin[T](a: MaskedArrayLike[T]): T | MaskedConstant.type = toMA(a).min()
  /** `np.ma.amin(a, axis)`. */
  def amin[T](a: MaskedArrayLike[T], axis: Axis, keepdims: Boolean = false): MaskedArray[T] = toMA(a).min(axis, keepdims)
  /** `np.ma.product(a)` (alias of [[MA.prod]]). */
  def product[T](a: MaskedArrayLike[T])(using s: SumOf[T]): s.Out | MaskedConstant.type = toMA(a).prod()
  /** `np.ma.product(a, axis)`. */
  def product[T](a: MaskedArrayLike[T], axis: Axis, keepdims: Boolean = false)(using s: SumOf[T]): MaskedArray[s.Out] =
    toMA(a).prod(axis, keepdims)
  /** `np.ma.anomalies(a)` (alias of [[MA.anom]]). */
  def anomalies[T](a: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] = toMA(a).anom()
  /** `np.ma.anomalies(a, axis)`. */
  def anomalies[T](a: MaskedArrayLike[T], axis: Int)(using t: ToInexact[T]): MaskedArray[t.Out] = toMA(a).anom(axis)
  /** `np.ma.round_(a, decimals)` (alias of [[MA.round]]). */
  def round_[T](a: MaskedArrayLike[T], decimals: Int = 0)(using d: NumDType[T]): MaskedArray[T] = toMA(a).round(decimals)

  /** `np.ma.alltrue(a)` (`logical_and.reduce`) for a 0-d or 1-D array: true when every unmasked
    * element is true; `masked` when everything is masked. Use `alltrue(a, axis)` for N-D arrays.
    */
  def alltrue[T](target: MaskedArrayLike[T]): Boolean | MaskedConstant.type =
    val m = toMA(target)
    if m.ndim > 1 then throw new IllegalArgumentException("alltrue(target) reduces axis 0; use alltrue(target, axis) for N-D arrays")
    m.all()
  /** `np.ma.alltrue(a, axis)`: masked where every element of a lane is masked. */
  def alltrue[T](target: MaskedArrayLike[T], axis: Int): MaskedArray[Boolean] = toMA(target).all(axis)
  /** `np.ma.sometrue(a)` (`logical_or.reduce`) for a 0-d or 1-D array. */
  def sometrue[T](target: MaskedArrayLike[T]): Boolean | MaskedConstant.type =
    val m = toMA(target)
    if m.ndim > 1 then throw new IllegalArgumentException("sometrue(target) reduces axis 0; use sometrue(target, axis) for N-D arrays")
    m.any()
  /** `np.ma.sometrue(a, axis)`. */
  def sometrue[T](target: MaskedArrayLike[T], axis: Int): MaskedArray[Boolean] = toMA(target).any(axis)

  /** `np.ma.angle(z, deg)`: the argument of complex numbers, keeping the mask. */
  def angle[T](z: MaskedArrayLike[T], deg: Boolean = false)(using r: RealOf[T]): MaskedArray[r.Out] =
    val m = toMA(z)
    new MaskedArray(np.angle(m._data, deg), if m._mask == null then null else m._mask.copy(), None, m._hard)

  // ------------------------------------------------------------------ attributes

  /** `np.ma.ndim(a)`. */
  def ndim[T](obj: MaskedArrayLike[T]): Int = toMA(obj).ndim
  /** `np.ma.shape(a)`. */
  def shape[T](obj: MaskedArrayLike[T]): Seq[Int] = toMA(obj).shape
  /** `np.ma.size(a)`. */
  def size[T](obj: MaskedArrayLike[T]): Int = toMA(obj).size
  /** `np.ma.size(a, axis)`: the length along `axis`. */
  def size[T](obj: MaskedArrayLike[T], axis: Int): Int =
    val m = toMA(obj)
    m.shape(Shape.normAxis(axis, m.ndim))
  /** `np.ma.ids(a)`: identity hashes of the data and of the mask (`nomask` when unmasked). */
  def ids[T](a: MaskedArrayLike[T]): (Int, Int) =
    val m = toMA(a)
    (System.identityHashCode(m._data), System.identityHashCode(m.mask))
  /** `np.ma.flatten_mask(mask)`: the mask flattened to 1-D (`nomask` gives `[False]`). */
  def flatten_mask(mask: NDArray[Boolean]): NDArray[Boolean] = mask.flatten()

  // ------------------------------------------------------------------ creation

  /** `np.ma.identity(n)`: the identity matrix as a masked array. */
  def identity[T](n: Int)(using d: DefaultDType[T]): MaskedArray[T] = MaskedArray.wrap(np.identity[T](n))
  /** `np.ma.indices(dims)`: grid indices as a masked array. */
  def indices(dims: Int*): MaskedArray[Int] = MaskedArray.wrap(np.indices(dims*))
  /** `np.ma.fromfunction(shape)(f)`. */
  def fromfunction[T](shape: Int*)(f: Seq[Int] => T)(using d: DType[T]): MaskedArray[T] =
    MaskedArray.wrap(np.fromfunction(shape*)(f))
  /** `np.ma.frombuffer(buffer, dtype, count, offset)`: little-endian bytes as a masked array. */
  def frombuffer[T](buffer: Array[Byte], dtype: DType[T] | Null = null, count: Int = -1, offset: Int = 0)(using
      dd: DefaultDType[T]
  ): MaskedArray[T] = MaskedArray.wrap(np.frombuffer(buffer, dtype, count, offset))
  /** `np.ma.copy(a)`: a deep copy of data and mask. */
  def copy[T](a: MaskedArrayLike[T]): MaskedArray[T] = toMA(a).copy()

  // ------------------------------------------------------------------ shape manipulation

  /** `np.ma.ravel(a)`. */
  def ravel[T](a: MaskedArrayLike[T]): MaskedArray[T] = toMA(a).ravel()
  /** `np.ma.reshape(a, newShape)`. */
  def reshape[T](a: MaskedArrayLike[T], newShape: Int*): MaskedArray[T] = toMA(a).reshape(newShape*)
  /** `np.ma.reshape(a, newShape, order)`. */
  def reshape[T](a: MaskedArrayLike[T], newShape: Seq[Int], order: Char): MaskedArray[T] = toMA(a).reshape(newShape, order)
  /** `np.ma.resize(x, newShape)`: repeats data and mask cyclically to fill `newShape`. */
  def resize[T](x: MaskedArrayLike[T], newShape: Int*): MaskedArray[T] =
    val m = toMA(x)
    val data = np.resize(m._data, newShape.toSeq)
    val mask = if m._mask == null || data.ndim == 0 then null else np.resize(m._mask, newShape.toSeq)
    new MaskedArray(data, mask, None, false)
  /** `np.ma.transpose(a, axes)`. */
  def transpose[T](a: MaskedArrayLike[T], axes: Int*): MaskedArray[T] = toMA(a).transpose(axes*)
  /** `np.ma.swapaxes(a, axis1, axis2)`. */
  def swapaxes[T](a: MaskedArrayLike[T], axis1: Int, axis2: Int): MaskedArray[T] = toMA(a).swapaxes(axis1, axis2)
  /** `np.ma.squeeze(a)`: removes all length-1 axes. */
  def squeeze[T](a: MaskedArrayLike[T]): MaskedArray[T] = toMA(a).squeeze()
  /** `np.ma.squeeze(a, axis)`. */
  def squeeze[T](a: MaskedArrayLike[T], axis: Axis): MaskedArray[T] =
    mapBoth(toMA(a))([X] => (x: NDArray[X]) => np.squeeze(x, axis))
  /** `np.ma.expand_dims(a, axis)`. */
  def expand_dims[T](a: MaskedArrayLike[T], axis: Axis): MaskedArray[T] =
    mapBoth(toMA(a))([X] => (x: NDArray[X]) => np.expand_dims(x, axis))
  /** `np.ma.atleast_1d(a)` (the mask becomes a full array). */
  def atleast_1d[T](a: MaskedArrayLike[T]): MaskedArray[T] = fullBoth(toMA(a))([X] => (x: NDArray[X]) => np.atleast_1d(x))
  def atleast_1d[T](arrays: Seq[MaskedArrayLike[T]]): Seq[MaskedArray[T]] = arrays.map(atleast_1d(_))
  /** `np.ma.atleast_2d(a)`. */
  def atleast_2d[T](a: MaskedArrayLike[T]): MaskedArray[T] = fullBoth(toMA(a))([X] => (x: NDArray[X]) => np.atleast_2d(x))
  def atleast_2d[T](arrays: Seq[MaskedArrayLike[T]]): Seq[MaskedArray[T]] = arrays.map(atleast_2d(_))
  /** `np.ma.atleast_3d(a)`. */
  def atleast_3d[T](a: MaskedArrayLike[T]): MaskedArray[T] = fullBoth(toMA(a))([X] => (x: NDArray[X]) => np.atleast_3d(x))
  def atleast_3d[T](arrays: Seq[MaskedArrayLike[T]]): Seq[MaskedArray[T]] = arrays.map(atleast_3d(_))

  /** `np.ma.dstack(arrays)`: stacks along the third axis (with a full mask). */
  def dstack[T](arrays: Seq[MaskedArrayLike[T]]): MaskedArray[T] =
    val ms = arrays.map(toMA)
    new MaskedArray(np.dstack(ms.map(_._data)), np.dstack(ms.map(_.maskArray)), None, false)
  /** `np.ma.hsplit(ary, indices_or_sections)`: splits data and mask column-wise. */
  def hsplit[T](ary: MaskedArrayLike[T], indices_or_sections: Int | Seq[Int]): Seq[MaskedArray[T]] =
    val m = toMA(ary)
    np.hsplit(m._data, indices_or_sections).zip(np.hsplit(m.maskArray, indices_or_sections))
      .map((d, k) => new MaskedArray(d, k, None, false))
  /** `np.ma.append(a, b, axis)`: concatenation, of the flattened inputs when `axis` is `None`. */
  def append[T](a: MaskedArrayLike[T], b: MaskedArrayLike[T], axis: Int | None.type = None): MaskedArray[T] = axis match
    case ax: Int => MA.concatenate(Seq(a, b), ax)
    case _ => MA.concatenate(Seq(toMA(a).ravel(), toMA(b).ravel()), 0)
  /** `np.ma.mr_[a, b, ...]`: concatenates the pieces (made at least 1-D) along the first axis. */
  def mr_[T](pieces: MaskedArrayLike[T]*): MaskedArray[T] =
    MA.concatenate(pieces.map(p => mapBoth(toMA(p))([X] => (x: NDArray[X]) => np.atleast_1d(x))), 0)

  /** `np.ma.apply_over_axes(func, a, axes)`: applies `func(a, axis)` over each axis in turn,
    * re-inserting reduced axes so the number of dimensions is preserved.
    */
  def apply_over_axes[T](func: (MaskedArray[T], Int) => MaskedArray[T], a: MaskedArrayLike[T], axes: Axis): MaskedArray[T] =
    val axs: Seq[Int] = axes match
      case i: Int => Seq(i)
      case s: Seq[?] => s.asInstanceOf[Seq[Int]]
    val start = toMA(a)
    val nd = start.ndim
    axs.foldLeft(start) { (v, axis0) =>
      val axis = if axis0 < 0 then axis0 + nd else axis0
      val res = func(v, axis)
      if res.ndim == v.ndim then res
      else
        val r = res.expandDims(axis)
        if r.ndim == v.ndim then r
        else throw new IllegalArgumentException("function is not returning an array of the correct shape")
    }

  // ------------------------------------------------------------------ diagonals

  /** `np.ma.diag(v, k)`: extracts a diagonal (2-D input) or builds a diagonal matrix (1-D input). */
  def diag[T](v: MaskedArrayLike[T], k: Int = 0): MaskedArray[T] =
    val m = toMA(v)
    new MaskedArray(np.diag(m._data, k), if m._mask == null then null else np.diag(m._mask, k), None, false)
  /** `np.ma.diagflat(v, k)`: a diagonal matrix from the flattened input (with a full mask). */
  def diagflat[T](v: MaskedArrayLike[T], k: Int = 0): MaskedArray[T] =
    fullBoth(toMA(v))([X] => (x: NDArray[X]) => np.diagflat(x, k))
  /** `np.ma.diagonal(a, offset, axis1, axis2)`. */
  def diagonal[T](a: MaskedArrayLike[T], offset: Int = 0, axis1: Int = 0, axis2: Int = 1): MaskedArray[T] =
    mapBoth(toMA(a))([X] => (x: NDArray[X]) => np.diagonal(x, offset, axis1, axis2))

  private def asDouble[T](d: DType[T], x: T): Double = if d.isComplex then d.toComplex(x).re else d.toDouble(x)

  /** `np.ma.trace(a, offset)` of a 2-D array: the sum of the unmasked diagonal entries, as a float. */
  def trace[T](a: MaskedArrayLike[T], offset: Int = 0): Double =
    val m = toMA(a)
    if m.ndim != 2 then throw new IllegalArgumentException("trace(a, offset) needs a 2-D array; use trace(a, offset, axis1, axis2)")
    trace(a, offset, 0, 1).flatGet(0)
  /** `np.ma.trace(a, offset, axis1, axis2)`: sums along the diagonals (masked entries count as 0). */
  def trace[T](a: MaskedArrayLike[T], offset: Int, axis1: Int, axis2: Int): NDArray[Double] =
    val dg = diagonal(a, offset, axis1, axis2)
    val d = dg.dtype
    val f = dg.filled(d.zero).map(x => asDouble(d, x))(using DType.Float64)
    Lanes.reduce(f, -1, false)((b, n) => Reduce.pairwiseSum(b, 0, n))(using DType.Float64)

  // ------------------------------------------------------------------ differences

  /** `np.ma.diff(a, n, axis, prepend, append)`: n-th discrete difference; a difference is masked
    * when either of its operands is masked. A 0-d `prepend`/`append` is broadcast along `axis`.
    */
  def diff[T](
      a: MaskedArrayLike[T],
      n: Int = 1,
      axis: Int = -1,
      prepend: MaskedArrayLike[T] | Null = null,
      append: MaskedArrayLike[T] | Null = null
  )(using d: NumDType[T]): MaskedArray[T] =
    val m0 = toMA(a)
    if n == 0 then return m0
    if n < 0 then throw new IllegalArgumentException(s"order must be non-negative but got $n")
    if m0.ndim == 0 then throw new IllegalArgumentException("diff requires input that is at least one dimensional")
    val ax = Shape.normAxis(axis, m0.ndim)
    def edge(e: MaskedArrayLike[T]): MaskedArray[T] =
      val em = toMA(e)
      if em.ndim > 0 then em
      else
        val sh = m0.shape.updated(ax, 1)
        mapBoth(em)([X] => (x: NDArray[X]) => x.broadcastTo(sh*))
    val parts = Seq(Option(prepend).map(p => edge(p.asInstanceOf[MaskedArrayLike[T]])), Some(m0),
      Option(append).map(p => edge(p.asInstanceOf[MaskedArrayLike[T]]))).flatten
    var x = if parts.length > 1 then MA.concatenate(parts, ax) else m0
    var k = 0
    while k < n do
      val len = x.shape(ax)
      val hi = x.index(sliceAxis(x.ndim, ax, math.min(1, len), len))
      val lo = x.index(sliceAxis(x.ndim, ax, 0, math.max(len - 1, 0)))
      x = MaskedArray.arith(hi, lo, d, Arith.Sub)
      k += 1
    x

  /** `np.ma.ediff1d(arr, to_end, to_begin)`: differences of the flattened array, optionally
    * framed by `to_begin` / `to_end` (then with a full mask).
    */
  def ediff1d[T](arr: MaskedArrayLike[T], to_end: MaskedArrayLike[T] | Null = null, to_begin: MaskedArrayLike[T] | Null = null)(using
      d: NumDType[T]
  ): MaskedArray[T] =
    val f = toMA(arr).ravel()
    val n = f.size
    val ed = MaskedArray.arith(f.index(sliceAxis(1, 0, math.min(1, n), n)), f.index(sliceAxis(1, 0, 0, math.max(n - 1, 0))), d, Arith.Sub)
    if to_end == null && to_begin == null then ed
    else
      val parts = Seq(Option(to_begin), Some(ed), Option(to_end)).flatten.map(p => toMA(p.asInstanceOf[MaskedArrayLike[T]]))
      MA.hstack(parts)

  // ------------------------------------------------------------------ selection

  /** `np.ma.choose(indices, choices, mode)`: picks from `choices`; masked where the chosen entry
    * or the index is masked (`nomask` when nothing is).
    */
  def choose[T](indices: MaskedArrayLike[Int], choices: Seq[MaskedArrayLike[T]], mode: String = "raise"): MaskedArray[T] =
    val mi = toMA(indices)
    val c = mi.filled(0)
    val ms = choices.map(toMA)
    val data = np.choose(c, ms.map(_._data), mode)
    val om = np.choose(c, ms.map(_.maskArray), mode)
    val mask = MaskedArray.maskOr(om, mi._mask)
    new MaskedArray(data, if mask == null then null else MaskedArray.fitMask(mask, data.shapeArr), None, false)

  /** `np.ma.compress(condition, a, axis)`: slices where the (data of the) condition holds. */
  def compress[T](condition: MaskedArrayLike[Boolean], a: MaskedArrayLike[T], axis: Int | None.type = None): MaskedArray[T] =
    val cond = toMA(condition)._data
    val m = toMA(a)
    new MaskedArray(np.compress(cond, m._data, axis), if m._mask == null then null else np.compress(cond, m._mask, axis), m._fill, m._hard)

  /** `np.ma.compress_nd(x, axis)`: the data with every slice (along each axis in `axis`, all
    * axes by default) that contains a masked value removed. Returns an empty 1-D array when
    * everything is masked.
    */
  def compress_nd[T](x: MaskedArrayLike[T], axis: Axis | None.type = None): NDArray[T] =
    val m = toMA(x)
    if m._mask == null || !m._mask.any() then return m._data
    if m._mask.all() then return NDArray.fromArray(m.dtype.newArray(0), Array(0))(using m.dtype)
    val nd = m.ndim
    val axes: Seq[Int] = axis match
      case None => 0 until nd
      case a => normAxes(a.asInstanceOf[Axis], nd).toSeq
    axes.foldLeft(m._data) { (data, ax) =>
      val others = (0 until nd).filterNot(_ == ax)
      val bad = if others.isEmpty then m._mask else m._mask.any(others)
      np.compress(bad.map(!_)(using DType.Bool), data, ax)
    }

  /** `np.ma.take(a, indices, axis, mode)`: masked indices select element 0 and mask the result. */
  def take[T](a: MaskedArrayLike[T], indices: MaskedArrayLike[Int], axis: Int | None.type = None, mode: String = "raise"): MaskedArray[T] =
    val m = toMA(a)
    val mi = toMA(indices)
    val ind = mi.filled(0)
    val data = np.take(m._data, ind, axis, mode)
    val mask0: NDArray[Boolean] =
      if m._mask == null then mi._mask
      else
        val t = np.take(m._mask, ind, axis, mode)
        if mi._mask == null then t else NDArray.zipMap(t, mi._mask)(_ || _)
    new MaskedArray(data, if mask0 == null then null else MaskedArray.fitMask(mask0, data.shapeArr), None, false)

  /** `np.ma.repeat(a, repeats, axis)`: repeats data and mask. */
  def repeat[T](a: MaskedArrayLike[T], repeats: Int | Seq[Int] | NDArray[Int], axis: Int | None.type = None): MaskedArray[T] =
    mapBoth(toMA(a))([X] => (x: NDArray[X]) => np.repeat(x, repeats, axis))

  /** `np.ma.vander(x, n)`: the Vandermonde matrix (an ndarray) with the rows of masked `x` zeroed. */
  def vander[T](x: MaskedArrayLike[T], n: Int = -1)(using d: NumDType[T]): NDArray[T] =
    val m = toMA(x)
    val v = np.vander(m._data, n)
    if m._mask != null then
      val cols = v.shapeArr(1)
      var i = 0
      while i < v.shapeArr(0) do
        if m._mask.flatGet(i) then
          var j = 0
          while j < cols do
            v.setAt(Array(i, j), d.zero)
            j += 1
        i += 1
    v

  // ------------------------------------------------------------------ in-place placement

  private def normIndex(i: Int, n: Int, mode: String): Int = mode match
    case "raise" =>
      if i < -n || i >= n then throw new IndexOutOfBoundsException(s"index $i is out of bounds for axis 0 with size $n")
      if i < 0 then i + n else i
    case "wrap" => if n == 0 then throw new IndexOutOfBoundsException("cannot do a non-empty take from an empty axes.") else Math.floorMod(i, n)
    case "clip" => if n == 0 then throw new IndexOutOfBoundsException("cannot do a non-empty take from an empty axes.") else math.max(0, math.min(n - 1, i))
    case other => throw new IllegalArgumentException(s"clipmode must be one of 'clip', 'raise', or 'wrap' (got '$other')")

  /** `np.ma.put(a, indices, values, mode)`: sets `a.flat[indices]` to `values` (cycled) in place,
    * copying their mask; with a hard mask, masked positions are left untouched.
    */
  def put[T](a: MaskedArrayLike[T], indices: NDArray[Int], values: MaskedArrayLike[T], mode: String = "raise"): Unit =
    val m = toMA(a)
    val n = m.size
    val vm = toMA(values)
    val vd = vm._data.toArray
    if vd.isEmpty then return
    val vk = if vm._mask == null then null else vm._mask.toArray
    var idx = indices.toArray.map(normIndex(_, n, mode))
    var pos = idx.indices.toArray
    if m._hard && m._mask != null then
      pos = pos.filter(k => !m._mask.flatGet(idx(k)))
    idx = pos.map(idx(_))
    pos.indices.foreach(k => m._data.flatSet(idx(k), vd(pos(k) % vd.length)))
    if m._mask == null && vk == null then return
    val mk = if m._mask == null then NDArray.zerosOf(DType.Bool, m._data.shapeArr.clone()) else m._mask
    pos.indices.foreach(k => mk.flatSet(idx(k), vk != null && vk(pos(k) % vk.length)))
    m._mask = if mk.any() then mk else null

  /** `np.ma.putmask(a, mask, values)`: sets `a` to `values` (broadcast) where `mask` is true, in
    * place, copying the mask of `values`; a hard mask only gains masked entries.
    */
  def putmask[T](a: MaskedArrayLike[T], mask: NDArray[Boolean], values: MaskedArrayLike[T]): Unit =
    val m = toMA(a)
    val sh = m._data.shapeArr.toIndexedSeq
    val where = mask.broadcastTo(sh*).toArray
    val vm = toMA(values)
    val vd = vm._data.broadcastTo(sh*).toArray
    val vk = if vm._mask == null then null else vm._mask.broadcastTo(sh*).toArray
    def copyMask(dst: NDArray[Boolean], src: Array[Boolean]): Unit =
      var i = 0
      while i < where.length do
        if where(i) then dst.flatSet(i, src(i))
        i += 1
    if m._mask == null then
      if vk != null then
        val mk = NDArray.zerosOf(DType.Bool, m._data.shapeArr.clone())
        copyMask(mk, vk)
        m._mask = mk
    else if m._hard then
      if vk != null then
        val mm = m._mask.copy()
        copyMask(mm, vk)
        var i = 0
        while i < where.length do
          if mm.flatGet(i) then m._mask.flatSet(i, true)
          i += 1
    else copyMask(m._mask, if vk == null then new Array[Boolean](where.length) else vk)
    var i = 0
    while i < where.length do
      if where(i) then m._data.flatSet(i, vd(i))
      i += 1
