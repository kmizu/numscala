package numscala

/** Functional helpers: `np.apply_along_axis`, `np.apply_over_axes`, `np.vectorize`, `np.piecewise`. */
trait NpIndexingFunc:

  /** Applies `func1d` to the 1-D slices of `arr` along `axis` (`np.apply_along_axis`). The
    * result's dimensions replace `axis` in the output, so a function returning shape `s`
    * produces `arr.shape[:axis] + s + arr.shape[axis+1:]`.
    * Give the lambda's parameter type explicitly: `np.apply_along_axis((r: NDArray[Double]) => r * 2.0, 0, a)`.
    */
  def apply_along_axis[T, U](func1d: NDArray[T] => NDArray[U], axis: Int, arr: NDArray[T]): NDArray[U] =
    val (rows, outer) = lanesOf(arr, axis)
    val ax = Shape.normAxis(axis, arr.ndim)
    val first = func1d(rows.subArray(0))
    val rs = first.shapeArr.clone()
    val rsize = first.size
    val d = first.dtype
    val m = rows.shapeArr(0)
    val out = d.newArray(m * rsize)
    System.arraycopy(first.toArray, 0, out, 0, rsize)
    var i = 1
    while i < m do
      val r = func1d(rows.subArray(i))
      if !java.util.Arrays.equals(r.shapeArr, rs) then
        throw new IllegalArgumentException("all results of func1d must have the same shape")
      System.arraycopy(r.asType(using d).toArray, 0, out, i * rsize, rsize)
      i += 1
    val no = outer.length
    val nr = rs.length
    val res = NDArray.fromArray(out, outer ++ rs)(using d)
    val perm = (0 until ax) ++ (no until no + nr) ++ (ax until no)
    res.transpose(perm*).contiguous

  /** `np.apply_along_axis` for a function returning a scalar: the result drops `axis`. */
  @annotation.targetName("apply_along_axis_scalar")
  def apply_along_axis[T, U](func1d: NDArray[T] => U, axis: Int, arr: NDArray[T])(using u: DType[U]): NDArray[U] =
    val (rows, outer) = lanesOf(arr, axis)
    val m = rows.shapeArr(0)
    val out = u.newArray(m)
    var i = 0
    while i < m do
      out(i) = func1d(rows.subArray(i))
      i += 1
    NDArray.fromArray(out, outer)

  /** The lanes along `axis` as the rows of a 2-D array, plus the shape of the other axes. */
  private def lanesOf[T](arr: NDArray[T], axis: Int): (NDArray[T], Array[Int]) =
    val ax = Shape.normAxis(axis, arr.ndim)
    val moved = arr.moveaxis(ax, -1)
    val outer = moved.shapeArr.init
    val m = Shape.size(outer)
    if m == 0 then throw new IllegalArgumentException("Cannot apply_along_axis when any iteration dimensions are 0")
    (moved.reshapeArr(Array(m, arr.shapeArr(ax))), outer)

  /** Applies `func(a, axis)` repeatedly over several axes (`np.apply_over_axes`); results that
    * lost the axis get it re-inserted, so the number of dimensions is preserved.
    */
  def apply_over_axes[T](func: (NDArray[T], Int) => NDArray[T], a: NDArray[T], axes: Axis): NDArray[T] =
    val axs: Seq[Int] = axes match
      case i: Int => Seq(i)
      case s: Seq[?] => s.asInstanceOf[Seq[Int]]
    val nd = a.ndim
    axs.foldLeft(a) { (v, axis0) =>
      val axis = if axis0 < 0 then axis0 + nd else axis0
      val res = func(v, axis)
      if res.ndim == v.ndim then res
      else if res.ndim == v.ndim - 1 then res.expandDims(axis)
      else throw new IllegalArgumentException("function is not returning an array of the correct shape")
    }

  /** Turns a scalar function into an elementwise array function (`np.vectorize`). */
  def vectorize[A, U](f: A => U)(using u: DType[U]): NDArray[A] => NDArray[U] =
    a => a.map(f)

  /** `np.vectorize` for a two-argument function; the arguments are broadcast together. */
  def vectorize[A, B, U](f: (A, B) => U)(using u: DType[U]): (NDArray[A], NDArray[B]) => NDArray[U] =
    (a, b) => NDArray.zipMap(a, b)(f)

  /** `np.vectorize` for a three-argument function; the arguments are broadcast together. */
  def vectorize[A, B, C, U](f: (A, B, C) => U)(using u: DType[U]): (NDArray[A], NDArray[B], NDArray[C]) => NDArray[U] =
    (a, b, c) => NDArray.zipMap3(a, b, c)(f)

  /** Piecewise-defined function (`np.piecewise`): where `condlist(i)` holds, the result is
    * `funclist(i)` applied to `x` (or the constant `funclist(i)`). Later conditions take
    * precedence; with one extra entry in `funclist` it applies where no condition holds.
    * Elsewhere the result is zero.
    */
  def piecewise[T](x: NDArray[T], condlist: Seq[NDArray[Boolean]], funclist: Seq[(T => T) | T]): NDArray[T] =
    val nc = condlist.length
    val nf = funclist.length
    if nf != nc && nf != nc + 1 then
      throw new IllegalArgumentException(
        s"with $nc condition(s), either $nc or ${nc + 1} functions are expected"
      )
    val sh = x.shapeArr
    val conds0 = condlist.map(c => c.broadcastTo(sh*).toArray)
    val conds =
      if nf == nc + 1 then
        val n = Shape.size(sh)
        conds0 :+ Array.tabulate(n)(k => !conds0.exists(_(k)))
      else conds0
    val d = x.dtype
    val xs = x.toArray
    val out = d.newArray(xs.length)
    var k = 0
    while k < xs.length do
      out(k) = d.zero
      k += 1
    conds.zip(funclist).foreach { (c, f) =>
      val g: T => T = f match
        case fn: Function1[?, ?] => fn.asInstanceOf[T => T]
        case v => (_: T) => v.asInstanceOf[T]
      var j = 0
      while j < xs.length do
        if c(j) then out(j) = g(xs(j))
        j += 1
    }
    NDArray.fromArray(out, sh.clone())(using d)
