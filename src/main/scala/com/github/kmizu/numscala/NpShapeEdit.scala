package com.github.kmizu.numscala

import NpShapeOps.*

/** `np.append`, `np.insert`, `np.delete`. */
trait NpShapeEdit:

  /** Appends values to the end of an array (`np.append`); with `axis = None` both inputs are
    * flattened first, otherwise they are concatenated along `axis` (and must have equal ndim).
    */
  def append[T](arr: NDArray[T], values: T | NDArray[T] | Seq[T], axis: Int | None.type = None): NDArray[T] =
    val v = asArray(values, arr.dtype)
    axis match
      case None => concat(Seq(arr.ravel(), v.ravel()), 0)
      case i: Int => concat(Seq(arr, v), i)

  /** Inserts `values` before the given index or indices along `axis` (`np.insert`).
    *
    * `obj` is an `Int`, a `Seq[Int]` / `NDArray[Int]` of indices, or a slice (a `Range` or
    * [[Index.Slice]], interpreted like a Python slice).  With `axis = None` the array is flattened.
    */
  def insert[T](
      arr: NDArray[T],
      obj: Int | Seq[Int] | Index | NDArray[?],
      values: T | NDArray[T] | Seq[T],
      axis: Int | None.type = None
  ): NDArray[T] =
    val d = arr.dtype
    val (a, ax) = axis match
      case None => (arr.ravel(), 0)
      case i: Int => (arr, Shape.normAxis(i, arr.ndim))
    val nd = a.ndim
    val n = a.shapeArr(ax)
    // (indices, isScalar)
    val (indices, scalar): (Array[Int], Boolean) = obj match
      case i: Int => (Array(i), true)
      case r: Range => (sliceIndices(Index.from(r), n), false)
      case s: Index.Slice => (sliceIndices(s, n), false)
      case _: Index => throw new IllegalArgumentException("invalid index object for insert")
      case s: Seq[?] => (s.map(intOf).toArray, false)
      case m: NDArray[?] =>
        if m.ndim > 1 then
          throw new IllegalArgumentException("index array argument obj to insert must be one dimensional or scalar")
        (m.toArray.map(intOf), m.ndim == 0)
    val vals0 = asArray(values, d)
    if indices.length == 1 then
      var index = indices(0)
      if index < -n || index > n then
        throw new IndexOutOfBoundsException(s"index ${indices(0)} is out of bounds for axis $ax with size $n")
      if index < 0 then index += n
      var vals = prependOnes(vals0, nd)
      if scalar then vals = vals.moveaxis(0, ax)
      val numnew = vals.shapeArr(ax)
      val newShape = a.shapeArr.clone()
      newShape(ax) += numnew
      val out = NDArray.zerosOf(d, newShape)
      sliceAxis(out, ax, 0, index) := sliceAxis(a, ax, 0, index)
      sliceAxis(out, ax, index, numnew) := vals
      sliceAxis(out, ax, index + numnew, n - index) := sliceAxis(a, ax, index, n - index)
      out
    else
      val idx = indices.map { i =>
        if i < -n || i > n then
          throw new IndexOutOfBoundsException(s"index $i is out of bounds for axis $ax with size ${n + 1}")
        if i < 0 then i + n else i
      }
      val numnew = idx.length
      val order = idx.indices.sortBy(idx(_)) // stable
      order.zipWithIndex.foreach((o, k) => idx(o) += k)
      val newShape = a.shapeArr.clone()
      newShape(ax) += numnew
      val out = NDArray.zerosOf(d, newShape)
      val selShape = a.shapeArr.clone()
      selShape(ax) = numnew
      val vb = vals0.broadcastTo(selShape*)
      val isNew = new Array[Boolean](newShape(ax))
      var k = 0
      while k < numnew do
        sliceAxis(out, ax, idx(k), 1) := sliceAxis(vb, ax, k, 1)
        isNew(idx(k)) = true
        k += 1
      var src = 0
      var pos = 0
      while pos < newShape(ax) do
        if !isNew(pos) then
          sliceAxis(out, ax, pos, 1) := sliceAxis(a, ax, src, 1)
          src += 1
        pos += 1
      out

  /** Removes the sub-arrays at the given index / indices / slice / boolean mask along `axis`
    * (`np.delete`); with `axis = None` the array is flattened first.
    */
  def delete[T](
      arr: NDArray[T],
      obj: Int | Seq[Int] | Seq[Boolean] | Index | NDArray[?],
      axis: Int | None.type = None
  ): NDArray[T] =
    val (a, ax) = axis match
      case None => (arr.ravel(), 0)
      case i: Int => (arr, Shape.normAxis(i, arr.ndim))
    val n = a.shapeArr(ax)
    val keep = Array.fill(n)(true)
    def drop(i: Int): Unit =
      if i < -n || i >= n then
        throw new IndexOutOfBoundsException(s"index $i is out of bounds for axis $ax with size $n")
      keep(if i < 0 then i + n else i) = false
    def mask(m: Array[Boolean]): Unit =
      if m.length != n then
        throw new IllegalArgumentException(
          s"boolean array argument obj to delete must be one dimensional and match the axis length of $n"
        )
      var i = 0
      while i < n do
        if m(i) then keep(i) = false
        i += 1
    obj match
      case i: Int => drop(i)
      case r: Range => sliceIndices(Index.from(r), n).foreach(drop)
      case s: Index.Slice => sliceIndices(s, n).foreach(drop)
      case _: Index => throw new IllegalArgumentException("invalid index object for delete")
      case s: Seq[?] =>
        if s.nonEmpty && s.forall(_.isInstanceOf[Boolean]) then mask(s.map(_.asInstanceOf[Boolean]).toArray)
        else s.foreach(x => drop(intOf(x)))
      case m: NDArray[?] =>
        if m.dtype.isBool then
          if m.ndim != 1 then mask(Array.empty) else mask(m.asInstanceOf[NDArray[Boolean]].toArray)
        else if m.dtype.isInteger then m.toArray.foreach(x => drop(intOf(x)))
        else throw new IllegalArgumentException("arrays used as indices must be of integer (or boolean) type")
    takeAxis(a, ax, (0 until n).filter(keep).toArray)

  private def sliceIndices(ix: Index, n: Int): Array[Int] = ix match
    case s: Index.Slice =>
      val (start, step, len) = s.resolve(n)
      Array.tabulate(len)(k => start + k * step)
    case _ => throw new IllegalArgumentException("expected a slice")

  private def intOf(x: Any): Int = x match
    case i: Int => i
    case l: Long => l.toInt
    case s: Short => s.toInt
    case b: Byte => b.toInt
    case b: Boolean => if b then 1 else 0
    case other => throw new IllegalArgumentException(s"invalid index $other")
