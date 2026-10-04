package com.github.kmizu.numscala.cpu

import com.github.kmizu.numscala.{DType, LinAlgCore, NDArray, Shape, Strided}

/** Bridges `NDArray[Float]` views to the [[MatrixF32]] kernels without copying whenever possible.
  *
  * A 2-D (sub)array becomes a descriptor over the array's own buffer when its rows are contiguous
  * (C order, possibly with padded rows) or when it is the transpose of such a layout. Everything
  * else (negative or zero strides, non-unit column strides) is packed into the workspace and counted
  * in `bytesPacked`. Broadcast batch axes are walked by address and never materialised. `.toArray`,
  * `.copy()` and views keep their meaning.
  */
object NDArrayF32Adapter:

  private val RowMajor = 0
  private val Transposed = 1
  private val Packed = 2

  /** Layout of a non-empty `rows x cols` view with element strides `(rs, cs)`. */
  private def layoutOf(rows: Int, cols: Int, rs: Int, cs: Int): Int =
    if (cols == 1 || cs == 1) && (rows == 1 || rs >= cols) then RowMajor
    else if (rows == 1 || rs == 1) && (cols == 1 || cs >= rows) then Transposed
    else Packed

  /** Describes a non-empty 2-D Float32 array as a descriptor over its buffer, or `None` when it needs packing. */
  def describe(a: NDArray[Float]): Option[(MatrixF32, Transpose)] =
    if a.ndim != 2 then throw new IllegalArgumentException(s"describe: expected a 2-D array, got ${a.ndim}-D")
    val rows = a.shapeArr(0)
    val cols = a.shapeArr(1)
    if rows == 0 || cols == 0 then Some((MatrixF32(a.data, 0, rows, cols, math.max(cols, 1)), Transpose.No))
    else
      val rs = a.stridesArr(0)
      val cs = a.stridesArr(1)
      layoutOf(rows, cols, rs, cs) match
        case RowMajor => Some((MatrixF32(a.data, a.offset, rows, cols, if rows == 1 then cols else rs), Transpose.No))
        case Transposed => Some((MatrixF32(a.data, a.offset, cols, rows, if cols == 1 then rows else cs), Transpose.Yes))
        case _ => None

  /** `np.matmul` for operands promoted to Float32, through `kernels` and `workspace`.
    *
    * Same shape rules and results layout as `np.matmul`. Inputs of another dtype are converted first
    * (counted in `bytesConverted`); strided operands that cannot be described are packed (`bytesPacked`).
    */
  def matmul(a: NDArray[?], b: NDArray[?], workspace: Workspace, kernels: F32Kernels = ScalarF32Kernels): NDArray[Float] =
    LinAlgCore.matmulF32(a, b, workspace, kernels)

  /** Batched product of `a: [..., m, k]` and `b: [..., k, n]` broadcast to `batch`; returns the packed output buffer. */
  private[numscala] def matmulBatched(
      a: NDArray[Float], b: NDArray[Float], batch: Array[Int], m: Int, k: Int, n: Int,
      ws: Workspace, kernels: F32Kernels
  ): Array[Float] =
    val nb = Shape.size(batch)
    val total = nb.toLong * m * n
    if total > Int.MaxValue then throw new IllegalArgumentException("matmul: result is too large")
    val out = new Array[Float](total.toInt)
    if total == 0 || k == 0 then return out
    val nd = batch.length
    val aSt = Strided.broadcastStrides(a.shapeArr, a.stridesArr, batch ++ Array(m, k))
    val bSt = Strided.broadcastStrides(b.shapeArr, b.stridesArr, batch ++ Array(k, n))
    val aRs = aSt(nd); val aCs = aSt(nd + 1)
    val bRs = bSt(nd); val bCs = bSt(nd + 1)
    val la = layoutOf(m, k, aRs, aCs)
    val lb = layoutOf(k, n, bRs, bCs)
    val tA = if la == Transposed then Transpose.Yes else Transpose.No
    val tB = if lb == Transposed then Transpose.Yes else Transpose.No
    val packA = if la == Packed then Math.multiplyExact(m, k) else 0
    val packB = if lb == Packed then Math.multiplyExact(k, n) else 0
    val need = packA.toLong + packB + kernels.gemmWorkspaceFloats(m, n, k, tA, tB)
    if need > Int.MaxValue then throw new IllegalArgumentException("matmul: workspace would be too large")
    ws.enter()
    val mk = ws.mark
    try
      ws.reserveFloats(((mk >>> 32) + need).toInt)
      val pa = if la == Packed then ws.takeFloats(packA) else 0
      val pb = if lb == Packed then ws.takeFloats(packB) else 0
      val idx = new Array[Int](nd)
      var t = 0
      while t < nb do
        var offA = a.offset
        var offB = b.offset
        var ax = 0
        while ax < nd do
          offA += idx(ax) * aSt(ax)
          offB += idx(ax) * bSt(ax)
          ax += 1
        val am = la match
          case RowMajor => MatrixF32(a.data, offA, m, k, if m == 1 then k else aRs)
          case Transposed => MatrixF32(a.data, offA, k, m, if k == 1 then m else aCs)
          case _ => packStrided(a.data, offA, m, k, aRs, aCs, ws, pa)
        val bm = lb match
          case RowMajor => MatrixF32(b.data, offB, k, n, if k == 1 then n else bRs)
          case Transposed => MatrixF32(b.data, offB, n, k, if n == 1 then k else bCs)
          case _ => packStrided(b.data, offB, k, n, bRs, bCs, ws, pb)
        kernels.gemmInto(am, tA, bm, tB, MatrixF32(out, t * m * n, m, n, n), 1f, 0f, ws)
        // next batch multi-index (C order)
        ax = nd - 1
        while ax >= 0 && { idx(ax) += 1; idx(ax) == batch(ax) } do
          idx(ax) = 0
          ax -= 1
        t += 1
    finally
      ws.release(mk)
      ws.exit()
    out

  /** Copies an arbitrary-stride `rows x cols` view into the workspace at `dstOff` (row-major). */
  private def packStrided(src: Array[Float], off: Int, rows: Int, cols: Int, rs: Int, cs: Int, ws: Workspace, dstOff: Int): MatrixF32 =
    val dst = ws.floats
    var i = 0
    while i < rows do
      var j = 0
      val si = off + i * rs
      val di = dstOff + i * cols
      while j < cols do
        dst(di + j) = src(si + j * cs)
        j += 1
      i += 1
    ws.addPacked(rows.toLong * cols * 4)
    MatrixF32(dst, dstOff, rows, cols, cols)

  /** Converts `a` to Float32 following the usual promotion (no copy when it already is Float32). */
  private[numscala] def toF32(a: NDArray[?], ws: Workspace): NDArray[Float] =
    if a.dtype eq DType.Float32 then a.asInstanceOf[NDArray[Float]]
    else
      ws.addConverted(a.size.toLong * 4)
      a.asType(using DType.Float32)
