package numscala

/** Shared implementations of vector / tensor products used by `np` and `np.linalg`. */
private[numscala] object LinalgProducts:

  private def conjIfComplex[U](a: NDArray[U]): NDArray[U] = if a.dtype.isComplex then a.conj else a

  /** Outer product of the flattened inputs. */
  def outer[A, B, U](a: NDArray[A], b: NDArray[B], d: NumDType[U]): NDArray[U] =
    Ops.arith(a.ravel().reshape(a.size, 1), b.ravel().reshape(1, b.size), d, Arith.Mul)

  /** `np.inner`: sum product over the last axes. */
  def inner[A, B, U](a: NDArray[A], b: NDArray[B], d: NumDType[U]): NDArray[U] =
    if a.ndim == 0 || b.ndim == 0 then Ops.arith(a, b, d, Arith.Mul)
    else
      val ka = a.shapeArr(a.ndim - 1)
      val kb = b.shapeArr(b.ndim - 1)
      if ka != kb then
        throw new IllegalArgumentException(
          s"shapes ${Shape.str(a.shapeArr)} and ${Shape.str(b.shapeArr)} not aligned: $ka (dim ${a.ndim - 1}) != $kb (dim ${b.ndim - 1})"
        )
      LinAlgCore.tensordotD(a, b, Seq(a.ndim - 1), Seq(b.ndim - 1), d)

  /** `np.vdot`: conj(a) . b over the flattened inputs. */
  def vdot[A, B, U](a: NDArray[A], b: NDArray[B], d: NumDType[U]): U =
    if a.size != b.size then throw new IllegalArgumentException("vdot: vectors have different lengths")
    val x = conjIfComplex(a.asType(using d)).toArray
    val y = b.asType(using d).toArray
    var acc = d.zero
    var i = 0
    while i < x.length do
      acc = d.plus(acc, d.times(x(i), y(i)))
      i += 1
    acc

  /** `np.vecdot` / `np.linalg.vecdot`. */
  def vecdot[A, B, U](x1: NDArray[A], x2: NDArray[B], axis: Int, d: NumDType[U]): NDArray[U] =
    if x1.ndim == 0 || x2.ndim == 0 then
      throw new IllegalArgumentException("vecdot: Input operand does not have enough dimensions")
    val a = conjIfComplex(x1.asType(using d)).moveaxis(axis, -1)
    val b = x2.asType(using d).moveaxis(axis, -1)
    val na = a.shapeArr(a.ndim - 1)
    val nb = b.shapeArr(b.ndim - 1)
    if na != nb then
      throw new IllegalArgumentException(
        s"vecdot: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n),(n)->() (size $nb is different from $na)"
      )
    Reduce.sum(Ops.arith(a, b, d, Arith.Mul), -1, false, d)

  /** `np.kron`. */
  def kron[A, B, U](a: NDArray[A], b: NDArray[B], d: NumDType[U]): NDArray[U] =
    val nd = math.max(a.ndim, b.ndim)
    val sa = Array.fill(nd - a.ndim)(1) ++ a.shapeArr
    val sb = Array.fill(nd - b.ndim)(1) ++ b.shapeArr
    val ar = a.reshape(sa.flatMap(s => Seq(s, 1)).toSeq*)
    val br = b.reshape(sb.flatMap(s => Seq(1, s)).toSeq*)
    val prod = Ops.arith(ar, br, d, Arith.Mul)
    prod.reshape(sa.indices.map(i => sa(i) * sb(i))*)

  /** `np.cross`. */
  def cross[A, B, U](a0: NDArray[A], b0: NDArray[B], axisa: Int, axisb: Int, axisc: Int, d: NumDType[U]): NDArray[U] =
    if a0.ndim == 0 || b0.ndim == 0 then throw new IllegalArgumentException("cross: inputs must be at least 1-D")
    val a = a0.asType(using d).moveaxis(axisa, -1)
    val b = b0.asType(using d).moveaxis(axisb, -1)
    val la = a.shapeArr(a.ndim - 1)
    val lb = b.shapeArr(b.ndim - 1)
    if (la != 2 && la != 3) || (lb != 2 && lb != 3) then
      throw new IllegalArgumentException("incompatible dimensions for cross product\n(dimension must be 2 or 3)")
    def comp(x: NDArray[U], i: Int, len: Int): NDArray[U] =
      if i < len then x(---, i) else NDArray.scalar(d.zero)(using d)
    def mul(x: NDArray[U], y: NDArray[U]) = Ops.arith(x, y, d, Arith.Mul)
    def sub(x: NDArray[U], y: NDArray[U]) = Ops.arith(x, y, d, Arith.Sub)
    val a0c = comp(a, 0, la); val a1c = comp(a, 1, la); val a2c = comp(a, 2, la)
    val b0c = comp(b, 0, lb); val b1c = comp(b, 1, lb); val b2c = comp(b, 2, lb)
    val cp2 = sub(mul(a0c, b1c), mul(a1c, b0c))
    if la == 2 && lb == 2 then return cp2
    val cp0 = sub(mul(a1c, b2c), mul(a2c, b1c))
    val cp1 = sub(mul(a2c, b0c), mul(a0c, b2c))
    val lead = Shape.broadcast(cp0.shapeArr, cp1.shapeArr, cp2.shapeArr)
    val out = NDArray.zerosOf(d, lead :+ 3)
    out(---, 0) = cp0
    out(---, 1) = cp1
    out(---, 2) = cp2
    out.moveaxis(-1, axisc)

  /** `np.trace` over (axis1, axis2). */
  def trace[T, U](a: NDArray[T], offset: Int, axis1: Int, axis2: Int, d: NumDType[U]): NDArray[U] =
    Reduce.sum(Ops.diagonal(a, offset, axis1, axis2), -1, false, d)
