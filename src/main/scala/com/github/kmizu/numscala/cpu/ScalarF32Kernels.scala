package com.github.kmizu.numscala.cpu

/** The pure-Scala reference backend (always available, JDK 17+).
  *
  * Plain Float32 loops on primitive arrays: no boxing, closures or collections in inner loops and
  * no shortcut on zero operands. Other backends extend it and override the kernels they speed up,
  * so validation and the remaining kernels stay shared.
  */
open class ScalarF32Kernels protected () extends F32Kernels:
  import ScalarF32Kernels.*

  def name: String = "scalar"

  override def describe: String = s"scalar (KC=$KC, NC=$NC, MC=$MC)"

  // ------------------------------------------------------------------ GEMM

  protected def gemmImpl(
      a: MatrixF32, transA: Transpose, b: MatrixF32, transB: Transpose, c: MatrixF32,
      m: Int, n: Int, k: Int, alpha: Float, beta: Float, readsAB: Boolean, ws: Workspace, dispatchM: Int, dispatchN: Int
  ): Unit =
    scaleC(c, m, n, beta)
    if readsAB then
      if useScalarGemm(dispatchM, dispatchN, k) then scalarGemm(a, transA, b, transB, c, m, n, k, alpha, ws)
      else fastGemm(a, transA, b, transB, c, m, n, k, alpha, ws)

  /** Whether a product of logical shape `(m x k) * (k x n)` (the whole product, not a tile) uses the scalar loops. */
  protected def useScalarGemm(m: Int, n: Int, k: Int): Boolean = true

  /** The backend's own GEMM for products where [[useScalarGemm]] is false (C already scaled by beta). */
  protected def fastGemm(
      a: MatrixF32, transA: Transpose, b: MatrixF32, transB: Transpose, c: MatrixF32,
      m: Int, n: Int, k: Int, alpha: Float, ws: Workspace
  ): Unit = scalarGemm(a, transA, b, transB, c, m, n, k, alpha, ws)

  /** The reference loops, by layout (C already scaled by beta). */
  protected final def scalarGemm(
      a: MatrixF32, transA: Transpose, b: MatrixF32, transB: Transpose, c: MatrixF32,
      m: Int, n: Int, k: Int, alpha: Float, ws: Workspace
  ): Unit =
    (transA, transB) match
      case (Transpose.No, Transpose.No) => gemmNN(a, b, c, m, n, k, alpha)
      case (Transpose.No, Transpose.Yes) => gemmNT(a, b, c, m, n, k, alpha)
      case (Transpose.Yes, Transpose.No) => gemmTN(a, b, c, m, n, k, alpha)
      case (Transpose.Yes, Transpose.Yes) => gemmTN(a, packTransposed(b, k, n, ws), c, m, n, k, alpha)

  /** `C := beta * C`; `beta == 0` writes zeros without reading C. */
  protected final def scaleC(c: MatrixF32, m: Int, n: Int, beta: Float): Unit =
    if beta != 1f then
      val cd = c.data
      var i = 0
      while i < m do
        val ci = c.offset + i * c.rowStride
        if beta == 0f then java.util.Arrays.fill(cd, ci, ci + n, 0f)
        else
          var j = 0
          while j < n do
            cd(ci + j) = beta * cd(ci + j)
            j += 1
        i += 1

  /** Packs `op(B) = B^T` (B is `n x k` physically) into a `k x n` workspace tile; counted in `bytesPacked`. */
  protected final def packTransposed(b: MatrixF32, k: Int, n: Int, ws: Workspace): MatrixF32 =
    val off = ws.takeFloats(k * n)
    val dst = ws.floats
    val bd = b.data
    var j = 0
    while j < n do
      val bj = b.offset + j * b.rowStride
      var p = 0
      while p < k do
        dst(off + p * n + j) = bd(bj + p)
        p += 1
      j += 1
    ws.addPacked(k.toLong * n * 4)
    MatrixF32(dst, off, k, n, n)

  /** `C += alpha * A B`: blocked row-axpy (K and N blocked so the B panel stays in cache). */
  protected final def gemmNN(a: MatrixF32, b: MatrixF32, c: MatrixF32, m: Int, n: Int, k: Int, alpha: Float): Unit =
    val ad = a.data; val bd = b.data; val cd = c.data
    var p0 = 0
    while p0 < k do
      val p1 = math.min(p0 + KC, k)
      var j0 = 0
      while j0 < n do
        val j1 = math.min(j0 + NC, n)
        var i = 0
        while i < m do
          val ai = a.offset + i * a.rowStride
          val ci = c.offset + i * c.rowStride
          var p = p0
          while p < p1 do
            val av = alpha * ad(ai + p)
            val bp = b.offset + p * b.rowStride
            var j = j0
            while j < j1 do
              cd(ci + j) += av * bd(bp + j)
              j += 1
            p += 1
          i += 1
        j0 = j1
      p0 = p1

  /** `C += alpha * A B^T`: row-by-row dot products (four partial sums). */
  protected final def gemmNT(a: MatrixF32, b: MatrixF32, c: MatrixF32, m: Int, n: Int, k: Int, alpha: Float): Unit =
    val ad = a.data; val bd = b.data; val cd = c.data
    var i = 0
    while i < m do
      val ai = a.offset + i * a.rowStride
      val ci = c.offset + i * c.rowStride
      var j = 0
      while j < n do
        val bj = b.offset + j * b.rowStride
        var s0 = 0f; var s1 = 0f; var s2 = 0f; var s3 = 0f
        var p = 0
        while p + 3 < k do
          s0 += ad(ai + p) * bd(bj + p)
          s1 += ad(ai + p + 1) * bd(bj + p + 1)
          s2 += ad(ai + p + 2) * bd(bj + p + 2)
          s3 += ad(ai + p + 3) * bd(bj + p + 3)
          p += 4
        while p < k do
          s0 += ad(ai + p) * bd(bj + p)
          p += 1
        cd(ci + j) += alpha * ((s0 + s1) + (s2 + s3))
        j += 1
      i += 1

  /** `C += alpha * A^T B` (A is `k x m` physically): blocked over rows of C, axpy over rows of B. */
  protected final def gemmTN(a: MatrixF32, b: MatrixF32, c: MatrixF32, m: Int, n: Int, k: Int, alpha: Float): Unit =
    val ad = a.data; val bd = b.data; val cd = c.data
    var i0 = 0
    while i0 < m do
      val i1 = math.min(i0 + MC, m)
      var p = 0
      while p < k do
        val ap = a.offset + p * a.rowStride
        val bp = b.offset + p * b.rowStride
        var i = i0
        while i < i1 do
          val av = alpha * ad(ap + i)
          val ci = c.offset + i * c.rowStride
          var j = 0
          while j < n do
            cd(ci + j) += av * bd(bp + j)
            j += 1
          i += 1
        p += 1
      i0 = i1

  // ------------------------------------------------------------------ rows, elementwise, scan

  protected def gatherImpl(table: MatrixF32, ids: Array[Int], idsOffset: Int, count: Int, out: MatrixF32): Unit =
    RowOps.gather(table, ids, idsOffset, count, out)
  protected def scatterAddImpl(dst: MatrixF32, ids: Array[Int], idsOffset: Int, count: Int, values: MatrixF32): Unit =
    RowOps.scatterAdd(dst, ids, idsOffset, count, values)
  protected def copyImpl(src: MatrixF32, dst: MatrixF32): Unit = ElementOps.copy(src, dst)
  protected def fillImpl(dst: MatrixF32, value: Float): Unit = ElementOps.fill(dst, value)
  protected def axpyImpl(alpha: Float, x: MatrixF32, y: MatrixF32): Unit = ElementOps.axpy(alpha, x, y)
  protected def mulImpl(x: MatrixF32, y: MatrixF32, out: MatrixF32): Unit = ElementOps.mul(x, y, out)
  protected def sigmoidImpl(x: MatrixF32, out: MatrixF32): Unit = ElementOps.sigmoid(x, out)
  protected def siluImpl(x: MatrixF32, out: MatrixF32): Unit = ElementOps.silu(x, out)
  protected def rowLogSumExpImpl(x: MatrixF32, out: Array[Float], outOffset: Int): Unit =
    ElementOps.rowLogSumExp(x, out, outOffset)
  protected def rowSumSquaresImpl(x: MatrixF32, out: Array[Float], outOffset: Int): Unit =
    ElementOps.rowSumSquares(x, out, outOffset)
  protected def affineScanImpl(a: MatrixF32, b: MatrixF32, initial: MatrixF32, out: MatrixF32, steps: Int, batch: Int): Unit =
    AffineScan.scan(a, b, initial, out, steps, batch)

/** The shared scalar backend instance. */
object ScalarF32Kernels extends ScalarF32Kernels:
  /** K-direction block of the NN kernel. */
  final val KC = 128
  /** N-direction block of the NN kernel. */
  final val NC = 512
  /** M-direction block of the TN kernel. */
  final val MC = 64
