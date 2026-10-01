package com.github.kmizu.numscala

import NpSortImpl.*

/** Index-generation helpers (`np.diag_indices`, `np.tril_indices`, `np.ravel_multi_index`,
  * `np.unravel_index`, `np.ix_`, `np.ndindex`, `np.s_`, ...).  Functional helpers
  * (`apply_along_axis`, `vectorize`, `piecewise`, ...) live in [[NpIndexingFunc]].
  */
trait NpIndexing extends NpIndexingFunc:

  private def intArr(a: Array[Int]): NDArray[Int] = NDArray.fromArray(a)(using DType.Int32)

  /** Indices of the main diagonal of an `n`-wide, `ndim`-dimensional array (`np.diag_indices`). */
  def diag_indices(n: Int, ndim: Int = 2): Seq[NDArray[Int]] =
    Seq.fill(ndim)(intArr(Array.tabulate(n)(identity)))

  /** `np.diag_indices_from`: diagonal indices of an array whose dimensions are all equal. */
  def diag_indices_from(arr: NDArray[?]): Seq[NDArray[Int]] =
    if arr.ndim < 2 then throw new IllegalArgumentException("input array must be at least 2-d")
    if arr.shapeArr.exists(_ != arr.shapeArr(0)) then
      throw new IllegalArgumentException("All dimensions of input must be of equal length")
    diag_indices(arr.shapeArr(0), arr.ndim)

  private def triIdx(n: Int, k: Int, m: Int, lower: Boolean): (NDArray[Int], NDArray[Int]) =
    val cols = if m < 0 then n else m
    val rs = scala.collection.mutable.ArrayBuilder.make[Int]
    val cs = scala.collection.mutable.ArrayBuilder.make[Int]
    var i = 0
    while i < n do
      var j = 0
      while j < cols do
        if (lower && j - i <= k) || (!lower && j - i >= k) then
          rs += i
          cs += j
        j += 1
      i += 1
    (intArr(rs.result()), intArr(cs.result()))

  /** Row and column indices of the lower triangle of an `(n, m)` array (`np.tril_indices`);
    * `m = -1` means `m = n`.
    */
  def tril_indices(n: Int, k: Int = 0, m: Int = -1): (NDArray[Int], NDArray[Int]) = triIdx(n, k, m, lower = true)

  /** Row and column indices of the upper triangle of an `(n, m)` array (`np.triu_indices`). */
  def triu_indices(n: Int, k: Int = 0, m: Int = -1): (NDArray[Int], NDArray[Int]) = triIdx(n, k, m, lower = false)

  private def check2d(arr: NDArray[?]): Unit =
    if arr.ndim != 2 then throw new IllegalArgumentException("input array must be 2-d")

  /** `np.tril_indices_from`: lower-triangle indices for a 2-D array. */
  def tril_indices_from(arr: NDArray[?], k: Int = 0): (NDArray[Int], NDArray[Int]) =
    check2d(arr)
    tril_indices(arr.shapeArr(0), k, arr.shapeArr(1))

  /** `np.triu_indices_from`: upper-triangle indices for a 2-D array. */
  def triu_indices_from(arr: NDArray[?], k: Int = 0): (NDArray[Int], NDArray[Int]) =
    check2d(arr)
    triu_indices(arr.shapeArr(0), k, arr.shapeArr(1))

  /** Indices where `mask_func(ones((n, n)), k)` is non-zero (`np.mask_indices`),
    * e.g. `np.mask_indices(3, (m, k) => np.triu(m, k))`.
    */
  def mask_indices(n: Int, mask_func: (NDArray[Int], Int) => NDArray[?], k: Int = 0): (NDArray[Int], NDArray[Int]) =
    val m = mask_func(NDArray.fillOf(DType.Int32, Array(n, n), 1), k)
    val nz = Searching.nonzero(m)
    if nz.length != 2 then throw new IllegalArgumentException("mask_func must return a 2-d array")
    (nz(0), nz(1))

  /** Converts a tuple of index arrays into flat indices (`np.ravel_multi_index`). `mode` is one
    * of `raise`, `wrap`, `clip`, or one mode per dimension; `order` is `'C'` or `'F'`.
    */
  def ravel_multi_index(
      multi_index: Seq[NDArray[Int]],
      dims: Seq[Int],
      mode: String | Seq[String] = "raise",
      order: Char = 'C'
  ): NDArray[Int] =
    val nd = dims.length
    if multi_index.length != nd then
      throw new IllegalArgumentException(s"parameter multi_index must be a sequence of length $nd")
    val modes: Array[Mode] = mode match
      case s: String => Array.fill(nd)(parseMode(s))
      case ms: Seq[?] =>
        val a = ms.map(x => parseMode(x.asInstanceOf[String])).toArray
        if a.length != nd then throw new IllegalArgumentException("mode must have one entry per dimension")
        a
    val dimArr = dims.toArray
    if dimArr.exists(_ < 0) then throw new IllegalArgumentException("dimensions must be non-negative")
    val strides = orderStrides(dimArr, order)
    val sh = Shape.broadcast(multi_index.map(_.shapeArr)*)
    val idx = multi_index.map(_.broadcastTo(sh*).toArray).toArray
    val n = Shape.size(sh)
    val out = new Array[Int](n)
    var k = 0
    while k < n do
      var flat = 0L
      var d = 0
      while d < nd do
        val i = idx(d)(k)
        val j = modes(d) match
          case Mode.Raise =>
            if i < 0 || i >= dimArr(d) then throw new IllegalArgumentException("invalid entry in coordinates array")
            i
          case m => normIndex(i, dimArr(d), m, allowNeg = false)
        flat += j.toLong * strides(d)
        d += 1
      out(k) = flat.toInt
      k += 1
    NDArray.fromArray(out, sh)(using DType.Int32)

  /** `np.ravel_multi_index` for a single coordinate tuple. */
  def ravel_multi_index(multi_index: Seq[Int], dims: Seq[Int]): Int =
    ravel_multi_index(multi_index.map(i => NDArray.scalar(i)(using DType.Int32)), dims, "raise", 'C').item

  private def orderStrides(dims: Array[Int], order: Char): Array[Long] =
    val nd = dims.length
    val st = new Array[Long](nd)
    var acc = 1L
    order match
      case 'C' =>
        var d = nd - 1
        while d >= 0 do
          st(d) = acc
          acc *= dims(d)
          d -= 1
      case 'F' =>
        var d = 0
        while d < nd do
          st(d) = acc
          acc *= dims(d)
          d += 1
      case other => throw new IllegalArgumentException(s"only 'C' or 'F' order is permitted (got '$other')")
    if acc > Int.MaxValue then throw new IllegalArgumentException("invalid dims: array size defined by dims is larger than the maximum possible size.")
    st

  /** Converts flat indices into a tuple of coordinate arrays for an array of the given `shape`
    * (`np.unravel_index`).
    */
  def unravel_index(indices: NDArray[Int], shape: Seq[Int], order: Char = 'C'): Seq[NDArray[Int]] =
    val dims = shape.toArray
    val strides = orderStrides(dims, order)
    val total = Shape.size(dims)
    val idx = indices.toArray
    val nd = dims.length
    val outs = Array.fill(nd)(new Array[Int](idx.length))
    var k = 0
    while k < idx.length do
      val v = idx(k)
      if v < 0 || v >= total then
        throw new IllegalArgumentException(s"index $v is out of bounds for array with size $total")
      var d = 0
      while d < nd do
        outs(d)(k) = ((v / strides(d)) % dims(d)).toInt
        d += 1
      k += 1
    outs.toSeq.map(o => NDArray.fromArray(o, indices.shapeArr.clone())(using DType.Int32))

  /** `np.unravel_index` for a single flat index. */
  def unravel_index(index: Int, shape: Seq[Int]): Seq[Int] =
    unravel_index(NDArray.scalar(index)(using DType.Int32), shape, 'C').map(_.item)

  /** Open mesh from 1-D sequences (`np.ix_`): the `k`-th output has shape `(1, ..., n_k, ..., 1)`,
    * so that `a.index(np.ix_(rows, cols)...)` selects a cross product.
    */
  def ix_[T](args: NDArray[T]*): Seq[NDArray[T]] =
    val nd = args.length
    args.zipWithIndex.map { (a, k) =>
      if a.ndim != 1 then throw new IllegalArgumentException("Cross index must be 1 dimensional")
      val sh = Array.fill(nd)(1)
      sh(k) = a.size
      a.copy().reshapeArr(sh)
    }

  /** `np.ix_` for boolean masks: each mask is converted to the indices of its true entries. */
  @annotation.targetName("ix_bool")
  def ix_(args: NDArray[Boolean]*): Seq[NDArray[Int]] =
    args.foreach(a => if a.ndim != 1 then throw new IllegalArgumentException("Cross index must be 1 dimensional"))
    ix_(args.map(a => Searching.nonzero(a).head)*)

  /** Iterator over all multi-indices of `shape` in C order (`np.ndindex`). */
  def ndindex(shape: Int*): Iterator[Seq[Int]] =
    val sh = shape.toArray
    val n = Shape.size(sh)
    val nd = sh.length
    Iterator.range(0, n).map { k =>
      val idx = new Array[Int](nd)
      var r = k
      var d = nd - 1
      while d >= 0 do
        idx(d) = r % sh(d)
        r /= sh(d)
        d -= 1
      idx.toSeq
    }

  /** Iterator over `(multi-index, value)` pairs in C order (`np.ndenumerate`). */
  def ndenumerate[T](a: NDArray[T]): Iterator[(Seq[Int], T)] =
    val vals = a.toArray
    ndindex(a.shapeArr.toSeq*).zip(vals.iterator)

  /** Builds an index expression (`np.s_[...]`): each item is converted like an index of
    * `a(...)`, and strings may hold several comma-separated components, e.g. `np.s_("1:3, ::2")`.
    */
  def s_(items: IndexLike*): Seq[Index] =
    items.flatMap {
      case s: String if s.contains(',') => s.split(",").toSeq.map(t => Index.parse(t))
      case other => Seq(Index.from(other))
    }

  /** Same as [[s_]] (`np.index_exp[...]`). */
  def index_exp(items: IndexLike*): Seq[Index] = s_(items*)
