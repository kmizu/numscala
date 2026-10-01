package com.github.kmizu.numscala

import scala.annotation.targetName
import NpShapeOps.*

/** Array shape manipulation (`np.reshape`, `np.transpose`, `np.flip`, `np.tile`, ...).
  *
  * Joining/splitting routines live in [[NpShapeJoin]], `np.pad` in [[NpShapePad]] and
  * `np.insert`/`np.delete`/`np.append` in [[NpShapeEdit]].
  */
trait NpShape extends NpShapeJoin, NpShapePad, NpShapeEdit:

  // ------------------------------------------------------------------ basic attributes

  /** Shape of an array (`np.shape`). */
  def shape[T](a: NDArray[T]): Seq[Int] = a.shape

  /** Number of dimensions (`np.ndim`). */
  def ndim[T](a: NDArray[T]): Int = a.ndim

  /** Number of elements (`np.size`). */
  def size[T](a: NDArray[T]): Int = a.size

  /** Number of elements along one axis (`np.size(a, axis)`). */
  def size[T](a: NDArray[T], axis: Int): Int = a.shapeArr(Shape.normAxis(axis, a.ndim))

  // ------------------------------------------------------------------ reshaping

  /** Gives a new shape without changing the data (`np.reshape`); one dimension may be `-1`. */
  def reshape[T](a: NDArray[T], newShape: Int*): NDArray[T] = a.reshapeArr(newShape.toArray)

  /** `np.reshape(a, shape, order)` with `order` in `'C'`, `'F'`, `'A'`. */
  def reshape[T](a: NDArray[T], newShape: Seq[Int], order: Char = 'C'): NDArray[T] =
    a.reshape(newShape, order)

  /** A flattened 1-D view when possible, otherwise a copy (`np.ravel`). Orders `'C'`, `'F'`, `'A'`, `'K'`. */
  def ravel[T](a: NDArray[T], order: Char = 'C'): NDArray[T] = order match
    case 'C' | 'A' | 'K' => a.ravel()
    case 'F' => a.ravel('F')
    case other => throw new IllegalArgumentException(s"order must be one of 'C', 'F', 'A', or 'K' (got '$other')")

  /** Returns `a` with a new shape, repeating its data cyclically as needed (`np.resize`). */
  def resize[T](a: NDArray[T], newShape: Int*): NDArray[T] = resize(a, newShape.toSeq)

  /** `np.resize(a, new_shape)` with the shape given as a sequence. */
  @targetName("resizeSeq")
  def resize[T](a: NDArray[T], newShape: Seq[Int]): NDArray[T] =
    val ns = newShape.toArray
    if ns.exists(_ < 0) then throw new IllegalArgumentException("all elements of `new_shape` must be non-negative")
    val d = a.dtype
    val n = Shape.size(ns)
    val src = a.toArray
    if src.length == 0 || n == 0 then NDArray.zerosOf(d, ns)
    else
      val out = d.newArray(n)
      var i = 0
      while i < n do
        val len = math.min(src.length, n - i)
        System.arraycopy(src, 0, out, i, len)
        i += len
      NDArray.fromArray(out, ns)(using d)

  // ------------------------------------------------------------------ axis permutations

  /** Reverses or permutes the axes (`np.transpose`); returns a view. */
  def transpose[T](a: NDArray[T], axes: Int*): NDArray[T] = a.transpose(axes*)

  /** `np.transpose(a, axes)` with the permutation given as a sequence. */
  @targetName("transposeSeq")
  def transpose[T](a: NDArray[T], axes: Seq[Int]): NDArray[T] = a.transpose(axes*)

  /** Array-API `permute_dims`: permutes the axes (a view). */
  def permute_dims[T](a: NDArray[T], axes: Seq[Int]): NDArray[T] = a.transpose(axes*)

  /** Array-API `permute_dims` with no axes: reverses them. */
  def permute_dims[T](a: NDArray[T]): NDArray[T] = a.transpose()

  /** Transposes the last two axes (`np.matrix_transpose`); a view. */
  def matrix_transpose[T](x: NDArray[T]): NDArray[T] =
    if x.ndim < 2 then
      throw new IllegalArgumentException(s"Input array must be at least 2-dimensional, but it is ${x.ndim}")
    x.swapaxes(-1, -2)

  /** Interchanges two axes (`np.swapaxes`); a view. */
  def swapaxes[T](a: NDArray[T], axis1: Int, axis2: Int): NDArray[T] = a.swapaxes(axis1, axis2)

  /** Moves one axis to a new position (`np.moveaxis`); a view. */
  def moveaxis[T](a: NDArray[T], source: Int, destination: Int): NDArray[T] =
    moveaxis(a, Seq(source), Seq(destination))

  /** Moves several axes to new positions (`np.moveaxis`); a view. */
  def moveaxis[T](a: NDArray[T], source: Seq[Int], destination: Seq[Int]): NDArray[T] =
    val nd = a.ndim
    val src = normAxisSeq(source, nd, "source")
    val dst = normAxisSeq(destination, nd, "destination")
    if src.length != dst.length then
      throw new IllegalArgumentException(
        "`source` and `destination` arguments must have the same number of elements"
      )
    val order = (0 until nd).filterNot(src.contains).toBuffer
    for (d, s) <- dst.zip(src).sortBy(_._1) do order.insert(d, s)
    a.transpose(order.toSeq*)

  /** Rolls `axis` backwards until it lies before position `start` (`np.rollaxis`); a view. */
  def rollaxis[T](a: NDArray[T], axis: Int, start: Int = 0): NDArray[T] =
    val n = a.ndim
    val ax = Shape.normAxis(axis, n)
    var st = if start < 0 then start + n else start
    if !(0 <= st && st < n + 1) then
      throw new IndexOutOfBoundsException(s"'start' arg requires ${-n} <= start < ${n + 1}, but $start was passed in")
    if ax < st then st -= 1
    if ax == st then a.view(a.shapeArr.clone(), a.stridesArr.clone(), a.offset)
    else
      val axes = (0 until n).filter(_ != ax).toBuffer
      axes.insert(st, ax)
      a.transpose(axes.toSeq*)

  // ------------------------------------------------------------------ adding / removing axes

  /** Inserts length-one axes at the given output positions (`np.expand_dims`); a view. */
  def expand_dims[T](a: NDArray[T], axis: Axis): NDArray[T] =
    val ax = axisSeq(axis)
    val outNd = a.ndim + ax.length
    insertOnes(a, normAxisSeq(ax, outNd).sorted)

  /** Removes all length-one axes (`np.squeeze`); a view. */
  def squeeze[T](a: NDArray[T]): NDArray[T] = a.squeeze()

  /** Removes the given length-one axes (`np.squeeze(a, axis)`); a view. */
  def squeeze[T](a: NDArray[T], axis: Axis): NDArray[T] = a.squeeze(axisSeq(axis)*)

  /** Views `a` with at least one dimension (`np.atleast_1d`). */
  def atleast_1d[T](a: NDArray[T]): NDArray[T] = prependOnes(a, 1)
  /** `np.atleast_1d` applied to several arrays. */
  def atleast_1d[T](arrays: Seq[NDArray[T]]): Seq[NDArray[T]] = arrays.map(atleast_1d(_))
  /** `np.atleast_1d(a, b, ...)`. */
  def atleast_1d[T](a: NDArray[T], b: NDArray[T], rest: NDArray[T]*): Seq[NDArray[T]] =
    atleast_1d((a +: b +: rest).toSeq)

  /** Views `a` with at least two dimensions (`np.atleast_2d`): `(N,)` becomes `(1, N)`. */
  def atleast_2d[T](a: NDArray[T]): NDArray[T] = prependOnes(a, 2)
  /** `np.atleast_2d` applied to several arrays. */
  def atleast_2d[T](arrays: Seq[NDArray[T]]): Seq[NDArray[T]] = arrays.map(atleast_2d(_))
  /** `np.atleast_2d(a, b, ...)`. */
  def atleast_2d[T](a: NDArray[T], b: NDArray[T], rest: NDArray[T]*): Seq[NDArray[T]] =
    atleast_2d((a +: b +: rest).toSeq)

  /** Views `a` with at least three dimensions (`np.atleast_3d`): `(N,)` -> `(1, N, 1)`, `(M, N)` -> `(M, N, 1)`. */
  def atleast_3d[T](a: NDArray[T]): NDArray[T] = atleast3(a)
  /** `np.atleast_3d` applied to several arrays. */
  def atleast_3d[T](arrays: Seq[NDArray[T]]): Seq[NDArray[T]] = arrays.map(atleast_3d(_))
  /** `np.atleast_3d(a, b, ...)`. */
  def atleast_3d[T](a: NDArray[T], b: NDArray[T], rest: NDArray[T]*): Seq[NDArray[T]] =
    atleast_3d((a +: b +: rest).toSeq)

  // ------------------------------------------------------------------ broadcasting

  /** A broadcast view of `a` with the given shape (`np.broadcast_to`). */
  def broadcast_to[T](a: NDArray[T], shape: Int*): NDArray[T] = broadcast_to(a, shape.toSeq)

  /** `np.broadcast_to(a, shape)` with the shape given as a sequence. */
  @targetName("broadcastToSeq")
  def broadcast_to[T](a: NDArray[T], shape: Seq[Int]): NDArray[T] =
    val ns = shape.toArray
    if ns.exists(_ < 0) then throw new IllegalArgumentException("all elements of broadcast shape must be non-negative")
    if ns.length < a.ndim then
      throw new IllegalArgumentException("input operand has more dimensions than allowed by the axis remapping")
    val off = ns.length - a.ndim
    var i = 0
    while i < a.ndim do
      val d = a.shapeArr(i)
      if d != 1 && d != ns(off + i) then
        throw new IllegalArgumentException(
          "operands could not be broadcast together with remapped shapes [original->remapped]: " +
            s"${Shape.str(a.shapeArr)} and requested shape ${Shape.str(ns)}"
        )
      i += 1
    a.broadcastTo(ns*)

  /** Broadcasts a sequence of shapes against each other (`np.broadcast_shapes`). */
  def broadcast_shapes(shapes: Seq[Int]*): Seq[Int] =
    val arrs = shapes.map(_.toArray)
    val nd = if arrs.isEmpty then 0 else arrs.map(_.length).max
    val out = Array.fill(nd)(1)
    val owner = Array.fill(nd)(-1)
    arrs.zipWithIndex.foreach { (s, k) =>
      val off = nd - s.length
      var i = 0
      while i < s.length do
        val d = s(i)
        val o = out(off + i)
        if o == 1 then
          out(off + i) = d
          if d != 1 then owner(off + i) = k
        else if d != 1 && d != o then
          val j = owner(off + i)
          throw new IllegalArgumentException(
            "shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between " +
              s"arg $j with shape ${Shape.str(arrs(j))} and arg $k with shape ${Shape.str(s)}."
          )
        i += 1
    }
    out.toSeq

  /** Broadcasts arrays against each other, returning views (`np.broadcast_arrays`). */
  def broadcast_arrays[T](arrays: Seq[NDArray[T]]): Seq[NDArray[T]] =
    val sh = broadcast_shapes(arrays.map(_.shape)*)
    arrays.map(_.broadcastTo(sh*))

  /** Broadcasts two arrays of possibly different dtypes (`np.broadcast_arrays(a, b)`). */
  def broadcast_arrays[A, B](a: NDArray[A], b: NDArray[B]): (NDArray[A], NDArray[B]) =
    val sh = broadcast_shapes(a.shape, b.shape)
    (a.broadcastTo(sh*), b.broadcastTo(sh*))

  /** Broadcasts three arrays of possibly different dtypes. */
  def broadcast_arrays[A, B, C](a: NDArray[A], b: NDArray[B], c: NDArray[C]): (NDArray[A], NDArray[B], NDArray[C]) =
    val sh = broadcast_shapes(a.shape, b.shape, c.shape)
    (a.broadcastTo(sh*), b.broadcastTo(sh*), c.broadcastTo(sh*))

  // ------------------------------------------------------------------ rearranging elements

  /** Reverses the order of elements along `axis` (all axes when `None`) (`np.flip`); a view. */
  def flip[T](m: NDArray[T], axis: Axis | None.type = None): NDArray[T] =
    val axes = axis match
      case None => (0 until m.ndim).toArray
      case ax: (Int | Seq[?]) => normAxisSeq(axisSeq(ax.asInstanceOf[Axis]), m.ndim)
    axes.foldLeft(m.view(m.shapeArr.clone(), m.stridesArr.clone(), m.offset))(flipAxis)

  /** Reverses the columns (axis 1) (`np.fliplr`); a view. */
  def fliplr[T](m: NDArray[T]): NDArray[T] =
    if m.ndim < 2 then throw new IllegalArgumentException("Input must be >= 2-d.")
    flipAxis(m, 1)

  /** Reverses the rows (axis 0) (`np.flipud`); a view. */
  def flipud[T](m: NDArray[T]): NDArray[T] =
    if m.ndim < 1 then throw new IllegalArgumentException("Input must be >= 1-d.")
    flipAxis(m, 0)

  /** Rotates by 90 degrees `k` times in the plane of `axes` (`np.rot90`); a view. */
  def rot90[T](m: NDArray[T], k: Int = 1, axes: (Int, Int) = (0, 1)): NDArray[T] =
    val nd = m.ndim
    val (a0, a1) = axes
    if a0 == a1 || math.abs(a0 - a1) == nd then throw new IllegalArgumentException("Axes must be different.")
    if a0 >= nd || a0 < -nd || a1 >= nd || a1 < -nd then
      throw new IllegalArgumentException(s"Axes=($a0, $a1) out of range for array of ndim=$nd.")
    val x0 = Shape.normAxis(a0, nd)
    val x1 = Shape.normAxis(a1, nd)
    val kk = Math.floorMod(k, 4)
    kk match
      case 0 => m.view(m.shapeArr.clone(), m.stridesArr.clone(), m.offset)
      case 2 => flipAxis(flipAxis(m, x0), x1)
      case _ =>
        val perm = (0 until nd).toArray
        perm(x0) = x1
        perm(x1) = x0
        if kk == 1 then flipAxis(m, x1).transpose(perm.toSeq*)
        else flipAxis(m.transpose(perm.toSeq*), x1)

  /** Rolls elements along the given axes, or over the flattened array when `axis` is `None` (`np.roll`). */
  def roll[T](a: NDArray[T], shift: Int | Seq[Int], axis: Axis | None.type = None): NDArray[T] =
    val shifts: Seq[Int] = shift match
      case i: Int => Seq(i)
      case s: Seq[?] => s.asInstanceOf[Seq[Int]]
    axis match
      case None =>
        val flat = a.ravel()
        roll(flat, shifts, 0).reshapeArr(a.shapeArr.clone())
      case ax: (Int | Seq[?]) =>
        val axes = axisSeq(ax.asInstanceOf[Axis]).map(Shape.normAxis(_, a.ndim))
        val (sh, axs) =
          if shifts.length == axes.length then (shifts, axes)
          else if shifts.length == 1 then (Seq.fill(axes.length)(shifts.head), axes)
          else if axes.length == 1 then (shifts, Seq.fill(shifts.length)(axes.head))
          else throw new IllegalArgumentException("'shift' and 'axis' should be scalars or 1D sequences")
        val total = new Array[Int](a.ndim)
        sh.zip(axs).foreach((s, x) => total(x) += s)
        var res: NDArray[T] = a.copy()
        var x = 0
        while x < a.ndim do
          val n = a.shapeArr(x)
          if n > 0 then
            val s = Math.floorMod(total(x), n)
            if s != 0 then
              res = Lanes.transform(res, x) { (in: Array[T], len: Int, out: Array[T]) =>
                System.arraycopy(in, 0, out, s, len - s)
                System.arraycopy(in, len - s, out, 0, s)
              }(using a.dtype)
          x += 1
        res

  /** Constructs an array by repeating `a` the number of times given by `reps` (`np.tile`). */
  def tile[T](a: NDArray[T], reps: Int | Seq[Int]): NDArray[T] =
    val tup0: Array[Int] = reps match
      case i: Int => Array(i)
      case s: Seq[?] => s.asInstanceOf[Seq[Int]].toArray
    if tup0.exists(_ < 0) then throw new IllegalArgumentException("negative dimensions are not allowed")
    val d = math.max(tup0.length, a.ndim)
    val c = prependOnes(a, d)
    val tup = Array.fill(d - tup0.length)(1) ++ tup0
    // view with shape (t0, s0, t1, s1, ...) and zero strides on the repeat axes
    val sh = new Array[Int](2 * d)
    val st = new Array[Int](2 * d)
    var i = 0
    while i < d do
      sh(2 * i) = tup(i)
      st(2 * i) = 0
      sh(2 * i + 1) = c.shapeArr(i)
      st(2 * i + 1) = c.stridesArr(i)
      i += 1
    val big = c.view(sh, st, c.offset)
    val outShape = Array.tabulate(d)(i => tup(i) * c.shapeArr(i))
    NDArray.fromArray(big.toArray, outShape)(using a.dtype)

  /** Repeats each element (`np.repeat`); `repeats` is a scalar or one count per element;
    * with `axis = None` the input is flattened first.
    */
  def repeat[T](a: NDArray[T], repeats: Int | Seq[Int] | NDArray[Int], axis: Int | None.type = None): NDArray[T] =
    val (src, ax) = axis match
      case None => (a.ravel(), 0)
      case i: Int => (a, Shape.normAxis(i, a.ndim))
    val n = src.shapeArr(ax)
    val reps: Array[Int] = repeats match
      case i: Int => Array.fill(n)(i)
      case arr: NDArray[?] => broadcastReps(arr.asInstanceOf[NDArray[Int]].toArray, n)
      case s: Seq[?] => broadcastReps(s.asInstanceOf[Seq[Int]].toArray, n)
    if reps.exists(_ < 0) then throw new IllegalArgumentException("negative dimensions are not allowed")
    val total = reps.foldLeft(0L)(_ + _)
    if total > Int.MaxValue then throw new IllegalArgumentException("array is too big")
    val idx = new Array[Int](total.toInt)
    var k = 0
    var i = 0
    while i < n do
      var r = 0
      while r < reps(i) do
        idx(k) = i
        k += 1
        r += 1
      i += 1
    takeAxis(src, ax, idx)

  private def broadcastReps(r: Array[Int], n: Int): Array[Int] =
    if r.length == n then r
    else if r.length == 1 then Array.fill(n)(r(0))
    else
      throw new IllegalArgumentException(
        s"operands could not be broadcast together with shape ($n,) (${r.length},)"
      )

  /** Trims leading (`"f"`) and/or trailing (`"b"`) zeros (`np.trim_zeros`); a view.
    * For n-d input, trims the bounding box of nonzero values along `axis` (all axes when `None`).
    */
  def trim_zeros[T](filt: NDArray[T], trim: String = "fb", axis: Axis | None.type = None): NDArray[T] =
    val t = trim.toLowerCase
    if t.isEmpty || !t.forall(c => c == 'f' || c == 'b') then
      throw new IllegalArgumentException(s"unexpected character(s) in `trim`: '$trim'")
    val front = t.contains('f')
    val back = t.contains('b')
    val nd = filt.ndim
    if nd == 0 then return filt
    val axes = axis match
      case None => (0 until nd).toArray
      case ax: (Int | Seq[?]) => normAxisSeq(axisSeq(ax.asInstanceOf[Axis]), nd)
    val d = filt.dtype
    // per-axis bounding box of the nonzero elements
    val lo = Array.fill(nd)(Int.MaxValue)
    val hi = Array.fill(nd)(-1)
    val idx = new Array[Int](nd)
    val data = filt.data
    Strided.foreach1(filt.shapeArr, filt.stridesArr, filt.offset) { o =>
      if d.toBoolean(data(o)) then
        var k = 0
        while k < nd do
          if idx(k) < lo(k) then lo(k) = idx(k)
          if idx(k) > hi(k) then hi(k) = idx(k)
          k += 1
      var k = nd - 1
      var carry = true
      while carry && k >= 0 do
        idx(k) += 1
        if idx(k) < filt.shapeArr(k) then carry = false
        else
          idx(k) = 0
          k -= 1
    }
    var res = filt
    for ax <- axes do
      val n = filt.shapeArr(ax)
      if hi(ax) < 0 then
        // all zero: NumPy returns an empty selection along this axis
        val start = if front then n else 0
        res = sliceAxis(res, ax, start, if front || back then 0 else n)
      else
        val start = if front then lo(ax) else 0
        val stop = if back then hi(ax) + 1 else n
        res = sliceAxis(res, ax, start, stop - start)
    res

  // ------------------------------------------------------------------ splitting along an axis

  /** Splits into sub-arrays (views) along `axis` (`np.unstack`). */
  def unstack[T](x: NDArray[T], axis: Int = 0): Seq[NDArray[T]] =
    if x.ndim == 0 then throw new IllegalArgumentException("Input array must be at least 1-d.")
    val ax = Shape.normAxis(axis, x.ndim)
    val keep = (0 until x.ndim).filter(_ != ax).toArray
    val sh = keep.map(x.shapeArr(_))
    val st = keep.map(x.stridesArr(_))
    (0 until x.shapeArr(ax)).map(i => x.view(sh.clone(), st.clone(), x.offset + i * x.stridesArr(ax)))

  // ------------------------------------------------------------------ copying

  /** Copies `src` into `dst` with broadcasting, optionally only where `where` is true (`np.copyto`).
    * `casting` is one of `"no"`, `"equiv"`, `"safe"`, `"same_kind"`, `"unsafe"`.
    */
  def copyto[T, S](
      dst: NDArray[T],
      src: NDArray[S],
      casting: String = "same_kind",
      where: NDArray[Boolean] | Boolean = true
  ): Unit =
    if !canCast(src.dtype, dst.dtype, casting) then
      throw new IllegalArgumentException(
        s"Cannot cast array data from ${dtypeRepr(src.dtype)} to ${dtypeRepr(dst.dtype)} according to the rule '$casting'"
      )
    val dd = dst.dtype
    val sd = src.dtype
    val s0: NDArray[S] = if src.data.asInstanceOf[AnyRef] eq dst.data.asInstanceOf[AnyRef] then src.copy() else src
    val sst = Strided.broadcastStrides(s0.shapeArr, s0.stridesArr, dst.shapeArr)
    val ddata = dst.data
    val sdata = s0.data
    val same = sd eq dd
    where match
      case b: Boolean =>
        if b then
          if same then dst := s0.asInstanceOf[NDArray[T]]
          else
            Strided.foreach2(dst.shapeArr, dst.stridesArr, dst.offset, sst, s0.offset) { (o, so) =>
              ddata(o) = dd.castFrom(sd, sdata(so))
            }
      case w: NDArray[?] =>
        val m = w.asInstanceOf[NDArray[Boolean]]
        val wst = Strided.broadcastStrides(m.shapeArr, m.stridesArr, dst.shapeArr)
        val wdata = m.data
        Strided.foreach3(dst.shapeArr, dst.stridesArr, dst.offset, sst, s0.offset, wst, m.offset) { (o, so, wo) =>
          if wdata(wo) then ddata(o) = dd.castFrom(sd, sdata(so))
        }

  /** Fills `dst` with a scalar (`np.copyto(dst, value)`). */
  def copyto[T](dst: NDArray[T], value: T): Unit = dst.fill(value)
