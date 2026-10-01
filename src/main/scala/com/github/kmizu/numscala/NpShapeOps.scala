package com.github.kmizu.numscala

/** Internal kernels shared by the shape-manipulation routines (`NpShape*`). */
private[numscala] object NpShapeOps:

  /** A view of `a` restricted to `len` elements along `axis`, starting at `start`, with `step`. */
  def sliceAxis[T](a: NDArray[T], axis: Int, start: Int, len: Int, step: Int = 1): NDArray[T] =
    val sh = a.shapeArr.clone()
    sh(axis) = len
    val st = a.stridesArr.clone()
    st(axis) = a.stridesArr(axis) * step
    a.view(sh, st, if len > 0 then a.offset + start * a.stridesArr(axis) else a.offset)

  /** A view reversed along `axis` (negative stride). */
  def flipAxis[T](a: NDArray[T], axis: Int): NDArray[T] =
    val n = a.shapeArr(axis)
    sliceAxis(a, axis, n - 1, n, -1)

  /** A view with ones prepended to the shape until it has `nd` dimensions. */
  def prependOnes[T](a: NDArray[T], nd: Int): NDArray[T] =
    if a.ndim >= nd then a
    else
      val k = nd - a.ndim
      a.view(Array.fill(k)(1) ++ a.shapeArr, Array.fill(k)(0) ++ a.stridesArr, a.offset)

  /** `np.atleast_3d` view: `(N,)` -> `(1, N, 1)`, `(M, N)` -> `(M, N, 1)`. */
  def atleast3[T](a: NDArray[T]): NDArray[T] = a.ndim match
    case 0 => a.view(Array(1, 1, 1), Array(0, 0, 0), a.offset)
    case 1 => a.view(Array(1, a.shapeArr(0), 1), Array(0, a.stridesArr(0), 0), a.offset)
    case 2 => a.view(a.shapeArr :+ 1, a.stridesArr :+ 0, a.offset)
    case _ => a

  /** A view with length-one axes inserted at the given positions of the output (sorted, distinct). */
  def insertOnes[T](a: NDArray[T], outAxes: Array[Int]): NDArray[T] =
    val nd = a.ndim + outAxes.length
    val set = outAxes.toSet
    val sh = new Array[Int](nd)
    val st = new Array[Int](nd)
    var j = 0
    var i = 0
    while i < nd do
      if set(i) then
        sh(i) = 1
        st(i) = 0
      else
        sh(i) = a.shapeArr(j)
        st(i) = a.stridesArr(j)
        j += 1
      i += 1
    a.view(sh, st, a.offset)

  /** Normalizes a sequence of axes against `ndim`, rejecting repeats (NumPy's `normalize_axis_tuple`). */
  def normAxisSeq(axes: Seq[Int], ndim: Int, argName: String = "axis"): Array[Int] =
    val out = axes.map(Shape.normAxis(_, ndim)).toArray
    if out.distinct.length != out.length then
      throw new IllegalArgumentException(s"repeated axis in `$argName` argument")
    out

  def axisSeq(axis: Axis): Seq[Int] = axis match
    case i: Int => Seq(i)
    case s: Seq[?] => s.asInstanceOf[Seq[Int]]

  /** Gathers positions `idx` along `axis` into a new contiguous array. */
  def takeAxis[T](a: NDArray[T], axis: Int, idx: Array[Int]): NDArray[T] =
    val m = idx.length
    Lanes.transform(a, axis, m) { (in: Array[T], _: Int, out: Array[T]) =>
      var k = 0
      while k < m do
        out(k) = in(idx(k))
        k += 1
    }(using a.dtype)

  /** Concatenates arrays of equal dimensionality along `axis` into a new contiguous array. */
  def concat[T](arrays: Seq[NDArray[T]], axis: Int): NDArray[T] =
    if arrays.isEmpty then throw new IllegalArgumentException("need at least one array to concatenate")
    val first = arrays.head
    val nd = first.ndim
    if nd == 0 then throw new IllegalArgumentException("zero-dimensional arrays cannot be concatenated")
    val ax = Shape.normAxis(axis, nd)
    var total = 0
    var i = 0
    for a <- arrays do
      if a.ndim != nd then
        throw new IllegalArgumentException(
          s"all the input arrays must have same number of dimensions, but the array at index 0 has $nd " +
            s"dimension(s) and the array at index $i has ${a.ndim} dimension(s)"
        )
      var k = 0
      while k < nd do
        if k != ax && a.shapeArr(k) != first.shapeArr(k) then
          throw new IllegalArgumentException(
            "all the input array dimensions except for the concatenation axis must match exactly, but along " +
              s"dimension $k, the array at index 0 has size ${first.shapeArr(k)} and the array at index $i " +
              s"has size ${a.shapeArr(k)}"
          )
        k += 1
      total += a.shapeArr(ax)
      i += 1
    val outShape = first.shapeArr.clone()
    outShape(ax) = total
    val d = first.dtype
    val out = NDArray.zerosOf(d, outShape)
    var pos = 0
    for a <- arrays do
      val len = a.shapeArr(ax)
      sliceAxis(out, ax, pos, len) := a
      pos += len
    out

  /** Converts `T | NDArray[T] | Seq[T]` user input into an array of dtype `d`. */
  def asArray[T](v: Any, d: DType[T]): NDArray[T] = v match
    case a: NDArray[?] =>
      if a.dtype eq d then a.asInstanceOf[NDArray[T]] else a.asInstanceOf[NDArray[Any]].astype(using d.asInstanceOf[DType[Any]]).asInstanceOf[NDArray[T]]
    case s: Seq[?] =>
      val arr = d.newArray(s.length)
      var i = 0
      for x <- s do
        arr(i) = d.coerce(x)
        i += 1
      NDArray.fromArray(arr, Array(arr.length))(using d)
    case x => NDArray.scalar(d.coerce(x))(using d)

  /** Calls `f(buf, n)` on every lane of `v` along `axis`, writing the (modified) buffer back. */
  def updateLanes[T](v: NDArray[T], axis: Int)(f: Array[T] => Unit): Unit =
    val n = v.shapeArr(axis)
    val keep = (0 until v.ndim).filter(_ != axis).toArray
    val sh = keep.map(v.shapeArr(_))
    val st = keep.map(v.stridesArr(_))
    val s = v.stridesArr(axis)
    val buf = v.dtype.newArray(n)
    val data = v.data
    if Shape.size(sh) > 0 && n > 0 then
      Strided.foreach1(sh, st, v.offset) { base =>
        var j = 0
        while j < n do
          buf(j) = data(base + j * s)
          j += 1
        f(buf)
        j = 0
        while j < n do
          data(base + j * s) = buf(j)
          j += 1
      }

  /** Whether `from` may be cast to `to` under NumPy's casting rule. */
  def canCast(from: DType[?], to: DType[?], casting: String): Boolean =
    def rank(d: DType[?]): Int = d.kind match
      case 'b' => 0
      case 'i' => 1
      case 'f' => 2
      case 'c' => 3
      case _ => 4
    def safe: Boolean =
      if from eq to then true
      else if to.isString then true
      else if from.isString then false
      else if from.isBool then true
      else if to.isBool then false
      else
        (from.kind, to.kind) match
          case ('i', 'i') | ('f', 'f') => from.itemSize <= to.itemSize
          case ('i', 'f') => to.itemSize == 8 || from.itemSize < to.itemSize
          case (_, 'c') => true
          case _ => false
    casting match
      case "unsafe" => true
      case "no" | "equiv" => from eq to
      case "safe" => safe
      case "same_kind" =>
        safe || (!from.isString && !to.isString && rank(from) <= rank(to) && !to.isBool)
      case other =>
        throw new IllegalArgumentException(
          s"casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe' (got '$other')"
        )

  def dtypeRepr(d: DType[?]): String = if d.isString then "dtype('<U')" else s"dtype('${d.name}')"
