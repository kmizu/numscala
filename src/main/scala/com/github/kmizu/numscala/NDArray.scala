package com.github.kmizu.numscala

import scala.collection.immutable.ArraySeq
import scala.reflect.ClassTag

/** An n-dimensional, homogeneously typed, strided array — the Scala counterpart of
  * `numpy.ndarray`.
  *
  * Like NumPy, basic slicing, transposition and (when possible) reshaping return
  * '''views''' that share memory with the original array; writes through a view are
  * visible in the base array.  Integer-array and boolean-mask indexing return copies.
  *
  * {{{
  * import numscala.*
  * val a = np.arange(12.0).reshape(3, 4)
  * a(1, ::)            // row 1 (a view)
  * a(::, "1:3")        // columns 1..2
  * a(a > 5.0)          // boolean mask -> 1-D copy
  * a(0, ::) := 0.0     // assignment through a view
  * (a + 1.0).sum()     // 78.0
  * }}}
  */
final class NDArray[T] private[numscala] (
    private[numscala] val data: Array[T],
    private[numscala] val shapeArr: Array[Int],
    private[numscala] val stridesArr: Array[Int],
    private[numscala] val offset: Int,
    private[numscala] val baseArray: NDArray[?] | Null
)(using val dtype: DType[T]):

  // ------------------------------------------------------------------ attributes

  /** The array dimensions. */
  def shape: Seq[Int] = ArraySeq.unsafeWrapArray(shapeArr.clone())
  def ndim: Int = shapeArr.length
  def size: Int = Shape.size(shapeArr)
  /** Strides in bytes, as in NumPy. */
  def strides: Seq[Int] = ArraySeq.unsafeWrapArray(stridesArr.map(_ * math.max(dtype.itemSize, 1)))
  /** Strides measured in elements. */
  def elementStrides: Seq[Int] = ArraySeq.unsafeWrapArray(stridesArr.clone())
  def itemsize: Int = dtype.itemSize
  def nbytes: Int = size * dtype.itemSize
  /** The array this one is a view of, if any. */
  def base: Option[NDArray[?]] = Option(baseArray)
  def owndata: Boolean = baseArray == null
  def length: Int = if ndim == 0 then throw new IllegalArgumentException("len() of unsized object") else shapeArr(0)

  def isCContiguous: Boolean =
    var expected = 1
    var i = ndim - 1
    var ok = true
    while ok && i >= 0 do
      if shapeArr(i) != 1 && stridesArr(i) != expected then ok = false
      expected *= shapeArr(i)
      i -= 1
    ok || size == 0

  def isFContiguous: Boolean =
    var expected = 1
    var i = 0
    var ok = true
    while ok && i < ndim do
      if shapeArr(i) != 1 && stridesArr(i) != expected then ok = false
      expected *= shapeArr(i)
      i += 1
    ok || size == 0

  /** True when the array is C-contiguous and starts at the beginning of its buffer. */
  private[numscala] def isPacked: Boolean = offset == 0 && isCContiguous && data.length == size

  // ------------------------------------------------------------------ raw element access

  private def checkedOffset(idx: Array[Int]): Int =
    if idx.length != ndim then
      throw new IndexOutOfBoundsException(
        s"expected $ndim indices for array of shape ${Shape.str(shapeArr)} but got ${idx.length}" +
          (if idx.length < ndim then " (use a(i, ::) or a(i, ---) to take a sub-array)" else "")
      )
    var off = offset
    var k = 0
    while k < idx.length do
      var i = idx(k)
      val n = shapeArr(k)
      if i < 0 then i += n
      if i < 0 || i >= n then
        throw new IndexOutOfBoundsException(s"index ${idx(k)} is out of bounds for axis $k with size $n")
      off += i * stridesArr(k)
      k += 1
    off

  /** Element at a full multi-index (negative indices count from the end). */
  def getAt(idx: Array[Int]): T = data(checkedOffset(idx))
  def setAt(idx: Array[Int], v: T): Unit = data(checkedOffset(idx)) = v

  /** Buffer offset of the `i`-th element in C (row-major) order. */
  private[numscala] def flatOffset(i: Int): Int =
    var rem = i
    var off = offset
    var k = ndim - 1
    while k >= 0 do
      val n = shapeArr(k)
      off += (rem % n) * stridesArr(k)
      rem /= n
      k -= 1
    off

  /** The `i`-th element in C order (like `a.flat[i]`). */
  def flatGet(i: Int): T =
    val n = size
    val j = if i < 0 then i + n else i
    if j < 0 || j >= n then throw new IndexOutOfBoundsException(s"index $i is out of bounds for size $n")
    data(flatOffset(j))
  def flatSet(i: Int, v: T): Unit =
    val n = size
    val j = if i < 0 then i + n else i
    if j < 0 || j >= n then throw new IndexOutOfBoundsException(s"index $i is out of bounds for size $n")
    data(flatOffset(j)) = v

  /** Calls `f` with the buffer offset of each element in C order. */
  private[numscala] inline def foreachOffset(inline f: Int => Unit): Unit =
    Strided.foreach1(shapeArr, stridesArr, offset)(f)

  /** Copies the elements, in C order, into a new Scala array. */
  def toArray: Array[T] =
    val n = size
    if isCContiguous && offset + n <= data.length && (n == 0 || stridesArr.forall(_ >= 0)) && isPackedRange then
      val out = dtype.newArray(n)
      System.arraycopy(data, offset, out, 0, n)
      out
    else
      val out = dtype.newArray(n)
      var k = 0
      val d = data
      foreachOffset { o =>
        out(k) = d(o)
        k += 1
      }
      out

  private def isPackedRange: Boolean =
    // C-contiguous arrays (ignoring length-1 axes) occupy [offset, offset+size)
    var ok = true
    var i = 0
    while i < ndim do
      if shapeArr(i) > 1 && stridesArr(i) <= 0 then ok = false
      i += 1
    ok

  def toSeq: IndexedSeq[T] = ArraySeq.unsafeWrapArray(toArray)
  def toList: List[T] = toArray.toList
  def toVector: Vector[T] = toArray.toVector
  /** Iterates over all elements in C order (`a.flat`). */
  def flat: Iterator[T] = toArray.iterator
  def foreach(f: T => Unit): Unit =
    val d = data
    foreachOffset(o => f(d(o)))

  /** Iterates over the sub-arrays along the first axis (`for row in a`). */
  def iterator: Iterator[NDArray[T]] =
    if ndim == 0 then throw new IllegalArgumentException("iteration over a 0-d array")
    Iterator.range(0, shapeArr(0)).map(i => subArray(i))

  /** Nested Scala lists, like `a.tolist()`; a 0-d array yields its element. */
  def tolist: Any =
    def go(axis: Int, off: Int): Any =
      if axis == ndim then data(off)
      else List.tabulate(shapeArr(axis))(i => go(axis + 1, off + i * stridesArr(axis)))
    go(0, offset)

  /** The single element of a size-1 array (`a.item()`). */
  def item: T =
    if size != 1 then throw new IllegalArgumentException("can only convert an array of size 1 to a Scala scalar")
    data(flatOffset(0))
  def item(i: Int): T = flatGet(i)

  // ------------------------------------------------------------------ construction helpers

  private[numscala] def view(shape: Array[Int], strides: Array[Int], off: Int): NDArray[T] =
    new NDArray[T](data, shape, strides, off, if baseArray == null then this else baseArray)

  /** A contiguous copy (`a.copy()`). */
  def copy(): NDArray[T] = NDArray.fromArray(toArray, shapeArr.clone())

  /** This array if C-contiguous, otherwise a contiguous copy (`np.ascontiguousarray`). */
  def contiguous: NDArray[T] = if isCContiguous then this else copy()

  /** New array with a different dtype (`a.astype(dtype)`). */
  def astype[U](using u: DType[U]): NDArray[U] =
    if u eq dtype then copy().asInstanceOf[NDArray[U]]
    else
      val src = dtype
      val out = u.newArray(size)
      var k = 0
      val d = data
      foreachOffset { o =>
        out(k) = u.castFrom(src, d(o))
        k += 1
      }
      NDArray.fromArray(out, shapeArr.clone())

  /** `astype` with an explicit dtype value: `a.astypeOf(DType.Int32)`. */
  def astypeOf[U](u: DType[U]): NDArray[U] = astype[U](using u)

  /** `astype` for a dtype only known at runtime. */
  def astypeDyn(u: DType[?]): NDArray[?] = astype(using u.asInstanceOf[DType[Any]])

  /** Converts without copying when the dtype already matches. */
  def asType[U](using u: DType[U]): NDArray[U] =
    if u eq dtype then this.asInstanceOf[NDArray[U]] else astype[U]

  // ------------------------------------------------------------------ elementwise helpers

  /** Applies `f` to every element, producing a new C-contiguous array. */
  def map[U](f: T => U)(using u: DType[U]): NDArray[U] =
    val out = u.newArray(size)
    var k = 0
    val d = data
    foreachOffset { o =>
      out(k) = f(d(o))
      k += 1
    }
    NDArray.fromArray(out, shapeArr.clone())

  /** Elementwise combination with broadcasting. */
  def zipMap[B, U](other: NDArray[B])(f: (T, B) => U)(using u: DType[U]): NDArray[U] =
    NDArray.zipMap(this, other)(f)

  /** In-place map. */
  def mapInPlace(f: T => T): this.type =
    val d = data
    foreachOffset(o => d(o) = f(d(o)))
    this

  /** Fills the array with a scalar value, in place (`a.fill(v)`). */
  def fill(v: T): Unit =
    val d = data
    foreachOffset(o => d(o) = v)

  /** Assigns `v` to every element (`a[...] = v`). */
  def :=(v: T): Unit = fill(v)

  /** Copies `src` (broadcast to this shape) into this array (`a[...] = src`). */
  def :=(src: NDArray[T]): Unit =
    val s0 = if src.data eq data then src.copy() else src
    val st = Strided.broadcastStrides(s0.shapeArr, s0.stridesArr, shapeArr)
    val d = data
    val sd = s0.data
    Strided.foreach2(shapeArr, stridesArr, offset, st, s0.offset)((o, so) => d(o) = sd(so))

  /** Assigns from an array of another dtype, converting elements (`a[...] = src`). */
  def assignFrom[S](src: NDArray[S]): Unit =
    if src.dtype eq dtype then this := src.asInstanceOf[NDArray[T]]
    else this := src.astype(using dtype)

  // ------------------------------------------------------------------ shape manipulation

  /** Gives a new shape without changing the data; one dimension may be `-1`.
    * Returns a view whenever NumPy would.
    */
  def reshape(newShape: Int*): NDArray[T] = reshapeArr(newShape.toArray)

  def reshape(newShape: Seq[Int], order: Char): NDArray[T] =
    order match
      case 'C' | 'A' => reshapeArr(newShape.toArray)
      case 'F' =>
        val ns = Shape.resolve(newShape.toArray, size)
        // F-order reshape == reverse axes, C reshape, reverse axes
        transpose().reshapeArr(ns.reverse).transpose()
      case other => throw new IllegalArgumentException(s"order must be one of 'C', 'F', 'A' (got '$other')")

  private[numscala] def reshapeArr(req: Array[Int]): NDArray[T] =
    val ns = Shape.resolve(req, size)
    if size == 0 then return NDArray.fromArray(dtype.newArray(0), ns)
    NDArray.attemptNoCopyReshape(shapeArr, stridesArr, ns) match
      case Some(st) => view(ns, st, offset)
      case None => NDArray.fromArray(toArray, ns)

  /** A 1-D view when the array is C-contiguous, otherwise a copy (`a.ravel()`, like NumPy). */
  def ravel(): NDArray[T] = if isCContiguous then reshapeArr(Array(size)) else flatten()
  def ravel(order: Char): NDArray[T] =
    if order == 'F' then transpose().ravel() else ravel()
  /** Always a 1-D copy (`a.flatten()`). */
  def flatten(): NDArray[T] = NDArray.fromArray(toArray, Array(size))
  def flatten(order: Char): NDArray[T] = if order == 'F' then transpose().flatten() else flatten()

  /** Permutes the axes (reverses them by default). Returns a view. */
  def transpose(axes: Int*): NDArray[T] =
    val perm =
      if axes.isEmpty then (ndim - 1 to 0 by -1).toArray
      else
        val p = axes.map(Shape.normAxis(_, ndim)).toArray
        if p.length != ndim || p.distinct.length != ndim then
          throw new IllegalArgumentException("axes don't match array")
        p
    view(perm.map(shapeArr(_)), perm.map(stridesArr(_)), offset)

  /** The transposed array (`a.T`). */
  def T: NDArray[T] = transpose()

  def swapaxes(a1: Int, a2: Int): NDArray[T] =
    val x = Shape.normAxis(a1, ndim)
    val y = Shape.normAxis(a2, ndim)
    val perm = (0 until ndim).toArray
    perm(x) = y
    perm(y) = x
    transpose(perm.toSeq*)

  def moveaxis(source: Int, destination: Int): NDArray[T] =
    val s = Shape.normAxis(source, ndim)
    val d = Shape.normAxis(destination, ndim)
    val order = (0 until ndim).filter(_ != s).toBuffer
    order.insert(d, s)
    transpose(order.toSeq*)

  /** Removes axes of length one (all, or just `axis`). */
  def squeeze(axes: Int*): NDArray[T] =
    val drop =
      if axes.isEmpty then (0 until ndim).filter(shapeArr(_) == 1).toSet
      else
        val s = axes.map(Shape.normAxis(_, ndim)).toSet
        s.foreach(a =>
          if shapeArr(a) != 1 then
            throw new IllegalArgumentException("cannot select an axis to squeeze out which has size not equal to one")
        )
        s
    val keep = (0 until ndim).filterNot(drop).toArray
    view(keep.map(shapeArr(_)), keep.map(stridesArr(_)), offset)

  /** Inserts a length-one axis at `axis` (`np.expand_dims`). */
  def expandDims(axis: Int): NDArray[T] =
    val a = Shape.normAxis(axis, ndim + 1)
    val sh = shapeArr.toBuffer
    val st = stridesArr.toBuffer
    sh.insert(a, 1)
    st.insert(a, 0)
    view(sh.toArray, st.toArray, offset)

  /** A read-mostly view broadcast to `shape` (`np.broadcast_to`). */
  def broadcastTo(newShape: Int*): NDArray[T] =
    val ns = newShape.toArray
    view(ns, Strided.broadcastStrides(shapeArr, stridesArr, ns), offset)

  // ------------------------------------------------------------------ indexing

  /** Sub-array along the first axis (`a[i]`), a view. */
  def subArray(i: Int): NDArray[T] =
    if ndim == 0 then throw new IndexOutOfBoundsException("too many indices for array: array is 0-dimensional")
    val n = shapeArr(0)
    val j = if i < 0 then i + n else i
    if j < 0 || j >= n then throw new IndexOutOfBoundsException(s"index $i is out of bounds for axis 0 with size $n")
    view(shapeArr.tail, stridesArr.tail, offset + j * stridesArr(0))

  /** Element access with one integer per axis: `a(i, j)`. */
  def apply(indices: Int*): T = getAt(indices.toArray)

  /** General indexing with slices, `::`, strings like `"1:-1"`, `None` (new axis),
    * `---` (ellipsis), integer arrays and boolean masks: `a(0, ::)`, `a("::2")`, `a(mask)`.
    */
  def apply(first: IndexLike, rest: IndexLike*): NDArray[T] =
    index((first +: rest).map(Index.from))

  /** Indexing with a pre-built index sequence. */
  def index(items: Seq[Index]): NDArray[T] =
    NDArray.resolveIndex(this, items) match
      case Left(v) => v
      case Right((sh, offs)) =>
        val out = dtype.newArray(offs.length)
        var k = 0
        while k < offs.length do
          out(k) = data(offs(k))
          k += 1
        NDArray.fromArray(out, sh)

  def update(i0: Int, v: T): Unit = setAt(Array(i0), v)
  def update(i0: Int, i1: Int, v: T): Unit = setAt(Array(i0, i1), v)
  def update(i0: Int, i1: Int, i2: Int, v: T): Unit = setAt(Array(i0, i1, i2), v)
  def update(i0: Int, i1: Int, i2: Int, i3: Int, v: T): Unit = setAt(Array(i0, i1, i2, i3), v)
  def update(i0: IndexLike, v: T): Unit = set(Seq(i0))(v)
  def update(i0: IndexLike, v: NDArray[T]): Unit = setArray(Seq(i0))(v)
  def update(i0: IndexLike, i1: IndexLike, v: T): Unit = set(Seq(i0, i1))(v)
  def update(i0: IndexLike, i1: IndexLike, v: NDArray[T]): Unit = setArray(Seq(i0, i1))(v)
  def update(i0: IndexLike, i1: IndexLike, i2: IndexLike, v: T): Unit = set(Seq(i0, i1, i2))(v)
  def update(i0: IndexLike, i1: IndexLike, i2: IndexLike, v: NDArray[T]): Unit = setArray(Seq(i0, i1, i2))(v)

  /** `a[idx] = v` for any index expression and a scalar. */
  def set(idx: Seq[IndexLike])(v: T): Unit =
    NDArray.resolveIndex(this, idx.map(Index.from)) match
      case Left(view) => view.fill(v)
      case Right((_, offs)) =>
        var k = 0
        while k < offs.length do
          data(offs(k)) = v
          k += 1

  /** `a[idx] = values` for any index expression; `values` is broadcast to the selection. */
  def setArray(idx: Seq[IndexLike])(v: NDArray[T]): Unit =
    NDArray.resolveIndex(this, idx.map(Index.from)) match
      case Left(view) => view := v
      case Right((sh, offs)) =>
        val src = v.broadcastTo(sh*).toArray
        var k = 0
        while k < offs.length do
          data(offs(k)) = src(k)
          k += 1

  // ------------------------------------------------------------------ arithmetic operators

  def +[U](o: NDArray[U])(using p: NumPromote[T, U]): NDArray[p.Out] = Ops.arith(this, o, p.dtype, Arith.Add)
  def -[U](o: NDArray[U])(using p: NumPromote[T, U]): NDArray[p.Out] = Ops.arith(this, o, p.dtype, Arith.Sub)
  def *[U](o: NDArray[U])(using p: NumPromote[T, U]): NDArray[p.Out] = Ops.arith(this, o, p.dtype, Arith.Mul)
  def /[U](o: NDArray[U])(using p: DivPromote[T, U]): NDArray[p.Out] = Ops.arith(this, o, p.dtype, Arith.Div)
  def **[U](o: NDArray[U])(using p: NumPromote[T, U]): NDArray[p.Out] = Ops.arith(this, o, p.dtype, Arith.Pow)
  def %(o: NDArray[T])(using d: RealDType[T]): NDArray[T] = Ops.binary(this, o, d)(d.mod)
  def floorDiv(o: NDArray[T])(using d: RealDType[T]): NDArray[T] = Ops.binary(this, o, d)(d.floorDiv)
  /** NumPy's `a // b` (floor division); `//` alone would start a comment, so write `` a `//` b ``. */
  def `//`(o: NDArray[T])(using d: RealDType[T]): NDArray[T] = floorDiv(o)

  def +(s: T)(using d: NumDType[T]): NDArray[T] = Ops.arith(this, NDArray.scalar(s), d, Arith.Add)
  def -(s: T)(using d: NumDType[T]): NDArray[T] = Ops.arith(this, NDArray.scalar(s), d, Arith.Sub)
  def *(s: T)(using d: NumDType[T]): NDArray[T] = Ops.arith(this, NDArray.scalar(s), d, Arith.Mul)
  def /(s: T)(using t: ToInexact[T]): NDArray[t.Out] =
    val od = t.dtype
    val sv = od.castFrom(dtype, s)
    map(x => od.div(od.castFrom(dtype, x), sv))(using od)
  def **(s: T)(using d: NumDType[T]): NDArray[T] = Ops.arith(this, NDArray.scalar(s), d, Arith.Pow)
  def %(s: T)(using d: RealDType[T]): NDArray[T] = map(x => d.mod(x, s))
  def floorDiv(s: T)(using d: RealDType[T]): NDArray[T] = map(x => d.floorDiv(x, s))
  /** NumPy's `a // s` with a scalar: `` a `//` 2 ``. */
  def `//`(s: T)(using d: RealDType[T]): NDArray[T] = floorDiv(s)

  def unary_-(using d: NumDType[T]): NDArray[T] = map(d.negate)
  def unary_+ : NDArray[T] = copy()

  // in-place operators (`+=` etc. are synthesized by Scala from these)
  def +=(o: NDArray[T])(using d: NumDType[T]): Unit = Ops.inPlace(this, o)(d.plus)
  def -=(o: NDArray[T])(using d: NumDType[T]): Unit = Ops.inPlace(this, o)(d.minus)
  def *=(o: NDArray[T])(using d: NumDType[T]): Unit = Ops.inPlace(this, o)(d.times)
  def /=(o: NDArray[T])(using d: InexactDType[T]): Unit = Ops.inPlace(this, o)(d.div)
  def +=(s: T)(using d: NumDType[T]): Unit = mapInPlace(d.plus(_, s))
  def -=(s: T)(using d: NumDType[T]): Unit = mapInPlace(d.minus(_, s))
  def *=(s: T)(using d: NumDType[T]): Unit = mapInPlace(d.times(_, s))
  def /=(s: T)(using d: InexactDType[T]): Unit = mapInPlace(d.div(_, s))
  /** NumPy's `a //= b` (in-place floor division): `` a `//=` b ``. */
  def `//=`(o: NDArray[T])(using d: RealDType[T]): Unit = Ops.inPlace(this, o)(d.floorDiv)
  def `//=`(s: T)(using d: RealDType[T]): Unit = mapInPlace(d.floorDiv(_, s))

  // ------------------------------------------------------------------ comparison operators

  def <[U](o: NDArray[U])(using p: Promote[T, U]): NDArray[Boolean] = Ops.compare(this, o, p.dtype, CmpOp.Lt)
  def <=[U](o: NDArray[U])(using p: Promote[T, U]): NDArray[Boolean] = Ops.compare(this, o, p.dtype, CmpOp.Le)
  def >[U](o: NDArray[U])(using p: Promote[T, U]): NDArray[Boolean] = Ops.compare(this, o, p.dtype, CmpOp.Gt)
  def >=[U](o: NDArray[U])(using p: Promote[T, U]): NDArray[Boolean] = Ops.compare(this, o, p.dtype, CmpOp.Ge)
  /** Elementwise equality (NumPy's `==`). */
  def ===[U](o: NDArray[U])(using p: Promote[T, U]): NDArray[Boolean] = Ops.equal(this, o, p.dtype, true)
  /** Elementwise inequality (NumPy's `!=`). */
  def =!=[U](o: NDArray[U])(using p: Promote[T, U]): NDArray[Boolean] = Ops.equal(this, o, p.dtype, false)

  def <(s: T): NDArray[Boolean] = map(x => CmpOp.Lt.test(dtype, x, s))
  def <=(s: T): NDArray[Boolean] = map(x => CmpOp.Le.test(dtype, x, s))
  def >(s: T): NDArray[Boolean] = map(x => CmpOp.Gt.test(dtype, x, s))
  def >=(s: T): NDArray[Boolean] = map(x => CmpOp.Ge.test(dtype, x, s))
  def ===(s: T): NDArray[Boolean] = map(x => dtype.equiv(x, s))
  def =!=(s: T): NDArray[Boolean] = map(x => !dtype.equiv(x, s))

  // ------------------------------------------------------------------ bitwise / logical operators

  def &(o: NDArray[T])(using b: BitOps[T]): NDArray[T] = Ops.binary(this, o, dtype)(b.and)
  def |(o: NDArray[T])(using b: BitOps[T]): NDArray[T] = Ops.binary(this, o, dtype)(b.or)
  def ^(o: NDArray[T])(using b: BitOps[T]): NDArray[T] = Ops.binary(this, o, dtype)(b.xor)
  def &(s: T)(using b: BitOps[T]): NDArray[T] = map(b.and(_, s))
  def |(s: T)(using b: BitOps[T]): NDArray[T] = map(b.or(_, s))
  def ^(s: T)(using b: BitOps[T]): NDArray[T] = map(b.xor(_, s))
  def unary_~(using b: BitOps[T]): NDArray[T] = map(b.not)
  def <<(o: NDArray[T])(using d: IntDType[T]): NDArray[T] = Ops.binary(this, o, d)(d.shiftLeft)
  def >>(o: NDArray[T])(using d: IntDType[T]): NDArray[T] = Ops.binary(this, o, d)(d.shiftRight)
  def <<(s: T)(using d: IntDType[T]): NDArray[T] = map(d.shiftLeft(_, s))
  def >>(s: T)(using d: IntDType[T]): NDArray[T] = map(d.shiftRight(_, s))

  // ------------------------------------------------------------------ linear algebra shortcuts

  /** Matrix product (NumPy's `@` / `np.matmul`). */
  def @@[U](o: NDArray[U])(using p: NumPromote[T, U]): NDArray[p.Out] = LinAlgCore.matmul(this, o)
  def matmul[U](o: NDArray[U])(using p: NumPromote[T, U]): NDArray[p.Out] = LinAlgCore.matmul(this, o)
  /** `np.dot`. */
  def dot[U](o: NDArray[U])(using p: NumPromote[T, U]): NDArray[p.Out] = LinAlgCore.dot(this, o)

  // ------------------------------------------------------------------ reductions

  /** Sum of all elements. */
  def sum()(using s: SumOf[T]): s.Out = Reduce.sumAll(this, s.dtype)
  def sum(axis: Axis, keepdims: Boolean = false)(using s: SumOf[T]): NDArray[s.Out] =
    Reduce.sum(this, axis, keepdims, s.dtype)
  def prod()(using s: SumOf[T]): s.Out = Reduce.prodAll(this, s.dtype)
  def prod(axis: Axis, keepdims: Boolean = false)(using s: SumOf[T]): NDArray[s.Out] =
    Reduce.prod(this, axis, keepdims, s.dtype)
  def mean()(using t: ToInexact[T]): t.Out = Reduce.meanAll(this, t.dtype)
  def mean(axis: Axis, keepdims: Boolean = false)(using t: ToInexact[T]): NDArray[t.Out] =
    Reduce.mean(this, axis, keepdims, t.dtype)
  def variance(ddof: Int = 0)(using r: RealOf[T]): r.Out = Reduce.varAll(this, ddof, r.dtype)
  def variance(axis: Axis, ddof: Int, keepdims: Boolean)(using r: RealOf[T]): NDArray[r.Out] =
    Reduce.variance(this, axis, ddof, keepdims, r.dtype)
  def std(ddof: Int = 0)(using r: RealOf[T]): r.Out = r.dtype.sqrt(Reduce.varAll(this, ddof, r.dtype))
  def std(axis: Axis, ddof: Int, keepdims: Boolean)(using r: RealOf[T]): NDArray[r.Out] =
    Reduce.variance(this, axis, ddof, keepdims, r.dtype).mapInPlace(r.dtype.sqrt)
  def max(): T = Reduce.maxAll(this)
  def max(axis: Axis, keepdims: Boolean = false): NDArray[T] = Reduce.max(this, axis, keepdims)
  def min(): T = Reduce.minAll(this)
  def min(axis: Axis, keepdims: Boolean = false): NDArray[T] = Reduce.min(this, axis, keepdims)
  /** Peak to peak (`max - min`). */
  def ptp()(using d: NumDType[T]): T = d.minus(max(), min())
  def ptp(axis: Axis)(using d: NumDType[T]): NDArray[T] = Ops.binary(max(axis), min(axis), d)(d.minus)
  def argmax(): Int = Reduce.argmaxAll(this)
  def argmax(axis: Int): NDArray[Int] = Reduce.argmax(this, axis, false)
  def argmin(): Int = Reduce.argminAll(this)
  def argmin(axis: Int): NDArray[Int] = Reduce.argmin(this, axis, false)
  def all(): Boolean = Reduce.allAll(this)
  def all(axis: Axis, keepdims: Boolean = false): NDArray[Boolean] = Reduce.all(this, axis, keepdims)
  def any(): Boolean = Reduce.anyAll(this)
  def any(axis: Axis, keepdims: Boolean = false): NDArray[Boolean] = Reduce.any(this, axis, keepdims)
  def cumsum()(using s: SumOf[T]): NDArray[s.Out] = Reduce.cumulate(this.ravel(), 0, s.dtype)(s.dtype.plus)
  def cumsum(axis: Int)(using s: SumOf[T]): NDArray[s.Out] = Reduce.cumulate(this, axis, s.dtype)(s.dtype.plus)
  def cumprod()(using s: SumOf[T]): NDArray[s.Out] = Reduce.cumulate(this.ravel(), 0, s.dtype)(s.dtype.times)
  def cumprod(axis: Int)(using s: SumOf[T]): NDArray[s.Out] = Reduce.cumulate(this, axis, s.dtype)(s.dtype.times)

  // ------------------------------------------------------------------ misc methods

  /** Clips values to `[lo, hi]`. */
  def clip(lo: T, hi: T)(using d: RealDType[T]): NDArray[T] = map(x => d.min(d.max(x, lo), hi))
  /** Returns a sorted copy (along the last axis). */
  def sorted: NDArray[T] = Sorting.sort(this, -1)
  /** Sorts in place along an axis. */
  def sort(axis: Int = -1): Unit = this := Sorting.sort(this, axis)
  def argsort(axis: Int = -1): NDArray[Int] = Sorting.argsort(this, axis)
  def nonzero: Seq[NDArray[Int]] = Searching.nonzero(this)
  def round(decimals: Int = 0)(using d: NumDType[T]): NDArray[T] = Ops.round(this, decimals)
  def conj: NDArray[T] =
    if dtype.isComplex then map(x => x.asInstanceOf[Complex].conj.asInstanceOf[T]) else copy()
  def diagonal(offset: Int = 0, axis1: Int = 0, axis2: Int = 1): NDArray[T] =
    Ops.diagonal(this, offset, axis1, axis2)
  def trace(offset: Int = 0)(using s: SumOf[T]): s.Out = Reduce.sumAll(diagonal(offset), s.dtype)

  /** Real part (for complex arrays; a copy for real arrays). */
  def real: NDArray[Double] =
    if dtype.isComplex then map(x => x.asInstanceOf[Complex].re)(using DType.Float64)
    else astype(using DType.Float64)
  /** Imaginary part (zeros for real arrays). */
  def imag: NDArray[Double] =
    if dtype.isComplex then map(x => x.asInstanceOf[Complex].im)(using DType.Float64)
    else NDArray.zerosOf(DType.Float64, shapeArr.clone())

  // ------------------------------------------------------------------ Object methods

  /** NumPy-style `str()` rendering, e.g. `[[1. 2.]\n [3. 4.]]`. */
  override def toString: String = Format.str(this)
  /** NumPy-style `repr()` rendering, e.g. `array([1, 2, 3], dtype=int32)`. */
  def repr: String = Format.repr(this)

  /** Structural equality: same dtype, same shape, equal elements (NaN == NaN). */
  override def equals(other: Any): Boolean = other match
    case o: NDArray[?] =>
      (o.dtype eq dtype) && java.util.Arrays.equals(shapeArr, o.shapeArr) && {
        val a = toArray
        val b = o.toArray.asInstanceOf[Array[T]]
        var i = 0
        var eq = true
        while eq && i < a.length do
          val x = a(i)
          val y = b(i)
          if !(dtype.equiv(x, y) || (dtype.isNaN(x) && dtype.isNaN(y))) then eq = false
          i += 1
        eq
      }
    case _ => false

  override def hashCode: Int =
    java.util.Arrays.hashCode(shapeArr) * 31 + toSeq.take(16).map(x => dtype.toDouble(x).##).hashCode

object NDArray:
  /** Wraps an existing buffer (C order) without copying. */
  def fromArray[T](data: Array[T], shape: Array[Int])(using d: DType[T]): NDArray[T] =
    if Shape.size(shape) != data.length then
      throw new IllegalArgumentException(
        s"cannot create array of shape ${Shape.str(shape)} from ${data.length} elements"
      )
    new NDArray[T](data, shape, Shape.cStrides(shape), 0, null)

  def fromArray[T](data: Array[T])(using d: DType[T]): NDArray[T] = fromArray(data, Array(data.length))

  /** A 0-d array holding one value. */
  def scalar[T](v: T)(using d: DType[T]): NDArray[T] =
    val a = d.newArray(1)
    a(0) = v
    fromArray(a, Array.emptyIntArray)

  def zerosOf[T](d: DType[T], shape: Array[Int]): NDArray[T] =
    val a = d.newArray(Shape.size(shape))
    if !d.classTag.runtimeClass.isPrimitive then java.util.Arrays.fill(a.asInstanceOf[Array[AnyRef]], d.zero.asInstanceOf[AnyRef])
    fromArray(a, shape)(using d)

  def fillOf[T](d: DType[T], shape: Array[Int], v: T): NDArray[T] =
    val n = Shape.size(shape)
    val a = d.newArray(n)
    var i = 0
    while i < n do
      a(i) = v
      i += 1
    fromArray(a, shape)(using d)

  /** Builds an array by evaluating `f` at each flat C-order index. */
  def tabulate[T](shape: Int*)(f: Int => T)(using d: DType[T]): NDArray[T] =
    val sh = shape.toArray
    val n = Shape.size(sh)
    val a = d.newArray(n)
    var i = 0
    while i < n do
      a(i) = f(i)
      i += 1
    fromArray(a, sh)

  /** Elementwise combination of two arrays with broadcasting. */
  def zipMap[A, B, U](a: NDArray[A], b: NDArray[B])(f: (A, B) => U)(using u: DType[U]): NDArray[U] =
    val sh = Shape.broadcast(a.shapeArr, b.shapeArr)
    val sa = Strided.broadcastStrides(a.shapeArr, a.stridesArr, sh)
    val sb = Strided.broadcastStrides(b.shapeArr, b.stridesArr, sh)
    val out = u.newArray(Shape.size(sh))
    val da = a.data
    val db = b.data
    var k = 0
    Strided.foreach2(sh, sa, a.offset, sb, b.offset) { (oa, ob) =>
      out(k) = f(da(oa), db(ob))
      k += 1
    }
    fromArray(out, sh)

  /** Elementwise combination of three arrays with broadcasting. */
  def zipMap3[A, B, C, U](a: NDArray[A], b: NDArray[B], c: NDArray[C])(f: (A, B, C) => U)(using
      u: DType[U]
  ): NDArray[U] =
    val sh = Shape.broadcast(a.shapeArr, b.shapeArr, c.shapeArr)
    val sa = Strided.broadcastStrides(a.shapeArr, a.stridesArr, sh)
    val sb = Strided.broadcastStrides(b.shapeArr, b.stridesArr, sh)
    val sc = Strided.broadcastStrides(c.shapeArr, c.stridesArr, sh)
    val out = u.newArray(Shape.size(sh))
    val da = a.data
    val db = b.data
    val dc = c.data
    var k = 0
    Strided.foreach3(sh, sa, a.offset, sb, b.offset, sc, c.offset) { (oa, ob, oc) =>
      out(k) = f(da(oa), db(ob), dc(oc))
      k += 1
    }
    fromArray(out, sh)

  /** NumPy's `_attempt_nocopy_reshape`: strides for a view with the new shape, if one exists. */
  private[numscala] def attemptNoCopyReshape(
      oldShape: Array[Int],
      oldStrides: Array[Int],
      newShape: Array[Int]
  ): Option[Array[Int]] =
    val keep = oldShape.indices.filter(oldShape(_) != 1)
    val od = keep.map(oldShape(_)).toArray
    val os = keep.map(oldStrides(_)).toArray
    val oldnd = od.length
    val newnd = newShape.length
    val ns = new Array[Int](newnd)
    var oi = 0
    var oj = 1
    var ni = 0
    var nj = 1
    var ok = true
    while ok && ni < newnd && oi < oldnd do
      var np0 = newShape(ni)
      var op = od(oi)
      while np0 != op do
        if np0 < op then
          np0 *= newShape(nj)
          nj += 1
        else
          op *= od(oj)
          oj += 1
      var ok2 = oi
      while ok && ok2 < oj - 1 do
        if os(ok2) != od(ok2 + 1) * os(ok2 + 1) then ok = false
        ok2 += 1
      if ok then
        ns(nj - 1) = os(oj - 1)
        var nk = nj - 1
        while nk > ni do
          ns(nk - 1) = ns(nk) * newShape(nk)
          nk -= 1
        ni = nj
        nj += 1
        oi = oj
        oj += 1
    if !ok then None
    else
      val last = if ni >= 1 then ns(ni - 1) else 1
      var nk = ni
      while nk < newnd do
        ns(nk) = last
        nk += 1
      Some(ns)

  /** Resolves an index expression into either a view (basic indexing) or the result
    * shape plus buffer offsets of every selected element in C order (advanced indexing).
    */
  private[numscala] def resolveIndex[T](a: NDArray[T], items0: Seq[Index]): Either[NDArray[T], (Array[Int], Array[Int])] =
    val nEll = items0.count(_ == Index.Ellipsis)
    if nEll > 1 then throw new IndexOutOfBoundsException("an index can only have a single ellipsis ('...')")
    val consumed = items0.map {
      case Index.NewAxis | Index.Ellipsis => 0
      case Index.Mask(m) => m.ndim
      case _ => 1
    }.sum
    if consumed > a.ndim then
      throw new IndexOutOfBoundsException(
        s"too many indices for array: array is ${a.ndim}-dimensional, but $consumed were indexed"
      )
    val fill = Seq.fill(a.ndim - consumed)(Index.All)
    val items1 =
      if nEll == 1 then
        val k = items0.indexOf(Index.Ellipsis)
        items0.take(k) ++ fill ++ items0.drop(k + 1)
      else items0 ++ fill
    val advanced = items1.exists {
      case _: Index.Take | _: Index.Mask => true
      case _ => false
    }
    // expand masks into integer index arrays; with advanced indexing, ints become 0-d arrays
    val items: Seq[Index] =
      if !advanced then items1
      else
        items1.flatMap {
          case Index.Mask(m) =>
            Searching.nonzero(m).map(Index.Take(_))
          case Index.At(i) => Seq(Index.Take(NDArray.scalar(i)))
          case other => Seq(other)
        }
    // basic pass: build a view, keeping advanced axes intact
    val sh = scala.collection.mutable.ArrayBuffer.empty[Int]
    val st = scala.collection.mutable.ArrayBuffer.empty[Int]
    var off = a.offset
    var axis = 0
    val advAxes = scala.collection.mutable.ArrayBuffer.empty[Int] // positions in the view
    val advIdx = scala.collection.mutable.ArrayBuffer.empty[NDArray[Int]]
    for it <- items do
      it match
        case Index.At(i) =>
          val n = a.shapeArr(axis)
          val j = if i < 0 then i + n else i
          if j < 0 || j >= n then
            throw new IndexOutOfBoundsException(s"index $i is out of bounds for axis $axis with size $n")
          off += j * a.stridesArr(axis)
          axis += 1
        case s: Index.Slice =>
          val (start, step, len) = s.resolve(a.shapeArr(axis))
          if len > 0 then off += start * a.stridesArr(axis)
          sh += len
          st += step * a.stridesArr(axis)
          axis += 1
        case Index.NewAxis =>
          sh += 1
          st += 0
        case Index.Take(ix) =>
          advAxes += sh.length
          advIdx += ix
          sh += a.shapeArr(axis)
          st += a.stridesArr(axis)
          axis += 1
        case Index.Ellipsis | Index.Mask(_) => () // already expanded
    val v = a.view(sh.toArray, st.toArray, off)
    if !advanced then Left(v)
    else
      val bshape = Shape.broadcast(advIdx.map(_.shapeArr).toSeq*)
      val bIdx = advIdx.map(ix => ix.broadcastTo(bshape*).toArray).toArray
      val nb = Shape.size(bshape)
      val advOff = new Array[Int](nb)
      var k = 0
      while k < advIdx.length do
        val ax = advAxes(k)
        val n = v.shapeArr(ax)
        val stride = v.stridesArr(ax)
        val ids = bIdx(k)
        var b = 0
        while b < nb do
          var i = ids(b)
          if i < 0 then i += n
          if i < 0 || i >= n then
            throw new IndexOutOfBoundsException(s"index ${ids(b)} is out of bounds for axis $ax with size $n")
          advOff(b) += i * stride
          b += 1
        k += 1
      val advSet = advAxes.toSet
      val remAxes = (0 until v.ndim).filterNot(advSet).toArray
      val remShape = remAxes.map(v.shapeArr(_))
      val remStrides = remAxes.map(v.stridesArr(_))
      val nr = Shape.size(remShape)
      val offs = new Array[Int](nb * nr)
      val remOffs = new Array[Int](nr)
      var r = 0
      Strided.foreach1(remShape, remStrides, 0) { o =>
        remOffs(r) = o
        r += 1
      }
      var b = 0
      while b < nb do
        val base = v.offset + advOff(b)
        var j = 0
        while j < nr do
          offs(b * nr + j) = base + remOffs(j)
          j += 1
        b += 1
      // like NumPy, advanced indices are adjacent only if no slice, Ellipsis (even an empty
      // one) or newaxis separates them in the index expression
      val isAdv = items0.map {
        case _: Index.Take | _: Index.Mask | _: Index.At => true
        case _ => false
      }
      val firstAdv = isAdv.indexOf(true)
      val adjacent = isAdv.lastIndexOf(true) - firstAdv + 1 == isAdv.count(identity)
      val p0 = advAxes.head
      if adjacent && p0 > 0 then
        // move broadcast dims to the position of the first advanced index
        val tmp = NDArray.fromArray(offs, bshape ++ remShape)
        val nbd = bshape.length
        val perm = (nbd until nbd + p0) ++ (0 until nbd) ++ (nbd + p0 until nbd + remShape.length)
        val t = tmp.transpose(perm*)
        Right((t.shapeArr, t.toArray))
      else Right((bshape ++ remShape, offs))
