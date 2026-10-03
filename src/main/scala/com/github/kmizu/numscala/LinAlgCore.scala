package com.github.kmizu.numscala

/** Matrix products (`matmul`, `dot`), shared by operators and `np`/`np.linalg`. */
private[numscala] object LinAlgCore:

  /** `(m x k) @ (k x n)` on contiguous row-major buffers. */
  private def gemmD(a: Array[Double], aOff: Int, b: Array[Double], bOff: Int, out: Array[Double], oOff: Int, m: Int, k: Int, n: Int): Unit =
    var i = 0
    while i < m do
      val ai = aOff + i * k
      val oi = oOff + i * n
      var p = 0
      while p < k do
        // no `av == 0` shortcut: 0 * inf and 0 * nan must still produce NaN
        val av = a(ai + p)
        val bp = bOff + p * n
        var j = 0
        while j < n do
          out(oi + j) += av * b(bp + j)
          j += 1
        p += 1
      i += 1

  private def gemmGeneric[U](d: NumDType[U], a: Array[U], aOff: Int, b: Array[U], bOff: Int, out: Array[U], oOff: Int, m: Int, k: Int, n: Int): Unit =
    var i = 0
    while i < m do
      var j = 0
      while j < n do
        var acc = d.zero
        var p = 0
        while p < k do
          acc = d.plus(acc, d.times(a(aOff + i * k + p), b(bOff + p * n + j)))
          p += 1
        out(oOff + i * n + j) = acc
        j += 1
      i += 1

  /** `np.matmul` with NumPy's rules for 1-D operands and batch broadcasting. */
  def matmul[A, B, U](a0: NDArray[A], b0: NDArray[B])(using p: NumPromote[A, B]): NDArray[p.Out] =
    matmulD(a0, b0, p.dtype)

  def matmulD[A, B, U](a0: NDArray[A], b0: NDArray[B], d: NumDType[U]): NDArray[U] =
    if d eq DType.Float32 then
      val sel = cpu.F32Backend.default
      return matmulF32(a0, b0, new cpu.Workspace(), sel.kernels).asInstanceOf[NDArray[U]]
    val (a1, b1, batch, m, k, n) = matmulShapes(a0, b0)
    val aB = a1.asType(using d).broadcastTo((batch ++ Array(m, k)).toSeq*).contiguous
    val bB = b1.asType(using d).broadcastTo((batch ++ Array(k, n)).toSeq*).contiguous
    val nb = Shape.size(batch)
    // `contiguous` already produced C-order element runs starting at `offset`: no second copy
    val ad = aB.data
    val bd = bB.data
    val a0ff = aB.offset
    val b0ff = bB.offset
    val out = d.newArray(nb * m * n)
    (d: DType[?]) match
      case DType.Float64 =>
        val o = out.asInstanceOf[Array[Double]]
        var t = 0
        while t < nb do
          gemmD(ad.asInstanceOf[Array[Double]], a0ff + t * m * k, bd.asInstanceOf[Array[Double]], b0ff + t * k * n, o, t * m * n, m, k, n)
          t += 1
      case _ =>
        var t = 0
        while t < nb do
          gemmGeneric(d, ad, a0ff + t * m * k, bd, b0ff + t * k * n, out, t * m * n, m, k, n)
          t += 1
    NDArray.fromArray(out, resultShape(a0, b0, batch, m, n))(using d)

  /** `np.matmul` with both operands promoted to Float32, on the Float32 kernels (NS-CPU-001 §5.2). */
  private[numscala] def matmulF32(a0: NDArray[?], b0: NDArray[?], ws: cpu.Workspace, kernels: cpu.F32Kernels): NDArray[Float] =
    val (a1, b1, batch, m, k, n) = matmulShapes(a0, b0)
    val out = cpu.NDArrayF32Adapter.matmulBatched(
      cpu.NDArrayF32Adapter.toF32(a1, ws), cpu.NDArrayF32Adapter.toF32(b1, ws), batch, m, k, n, ws, kernels
    )
    NDArray.fromArray(out, resultShape(a0, b0, batch, m, n))

  /** 1-D promotion, core-dimension check and batch broadcast shape shared by the matmul paths. */
  private def matmulShapes[A, B](a0: NDArray[A], b0: NDArray[B]): (NDArray[A], NDArray[B], Array[Int], Int, Int, Int) =
    if a0.ndim == 0 || b0.ndim == 0 then
      throw new IllegalArgumentException("matmul: Input operand does not have enough dimensions")
    val a1 = if a0.ndim == 1 then a0.reshape(1, a0.shapeArr(0)) else a0
    val b1 = if b0.ndim == 1 then b0.reshape(b0.shapeArr(0), 1) else b0
    val m = a1.shapeArr(a1.ndim - 2)
    val k = a1.shapeArr(a1.ndim - 1)
    val k2 = b1.shapeArr(b1.ndim - 2)
    val n = b1.shapeArr(b1.ndim - 1)
    if k != k2 then
      throw new IllegalArgumentException(
        s"matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size $k2 is different from $k)"
      )
    val batch = Shape.broadcast(a1.shapeArr.dropRight(2), b1.shapeArr.dropRight(2))
    (a1, b1, batch, m, k, n)

  private def resultShape(a0: NDArray[?], b0: NDArray[?], batch: Array[Int], m: Int, n: Int): Array[Int] =
    var shape = batch ++ Array(m, n)
    if a0.ndim == 1 then shape = shape.patch(shape.length - 2, Nil, 1)
    if b0.ndim == 1 then shape = shape.dropRight(1)
    shape

  /** `np.dot`. */
  def dot[A, B, U](a: NDArray[A], b: NDArray[B])(using p: NumPromote[A, B]): NDArray[p.Out] =
    dotD(a, b, p.dtype)

  def dotD[A, B, U](a: NDArray[A], b: NDArray[B], d: NumDType[U]): NDArray[U] =
    if a.ndim == 0 || b.ndim == 0 then Ops.arith(a, b, d, Arith.Mul)
    else if a.ndim <= 2 && b.ndim <= 2 then matmulD(a, b, d)
    else
      // sum over last axis of a and second-to-last of b
      val bAxis = if b.ndim == 1 then 0 else b.ndim - 2
      tensordotD(a, b, Seq(a.ndim - 1), Seq(bAxis), d)

  /** `np.tensordot` over explicit axis lists. */
  def tensordotD[A, B, U](a: NDArray[A], b: NDArray[B], axesA: Seq[Int], axesB: Seq[Int], d: NumDType[U]): NDArray[U] =
    val aa = axesA.map(Shape.normAxis(_, a.ndim))
    val bb = axesB.map(Shape.normAxis(_, b.ndim))
    if aa.length != bb.length then throw new IllegalArgumentException("shape-mismatch for sum")
    aa.zip(bb).foreach((x, y) =>
      if a.shapeArr(x) != b.shapeArr(y) then throw new IllegalArgumentException("shape-mismatch for sum")
    )
    val freeA = (0 until a.ndim).filterNot(aa.contains)
    val freeB = (0 until b.ndim).filterNot(bb.contains)
    val k = aa.map(a.shapeArr(_)).product
    val ma = freeA.map(a.shapeArr(_)).product
    val nb = freeB.map(b.shapeArr(_)).product
    val at = a.transpose((freeA ++ aa)*).reshape(ma, k)
    val bt = b.transpose((bb ++ freeB)*).reshape(k, nb)
    val r = matmulD(at, bt, d)
    r.reshape((freeA.map(a.shapeArr(_)) ++ freeB.map(b.shapeArr(_)))*)
