package com.github.kmizu.numscala.cpu.vector25

import com.github.kmizu.numscala.cpu.{MatrixF32, ScalarF32Kernels, Transpose, Workspace}
import jdk.incubator.vector.{FloatVector, VectorMask}

/** Float32 kernels on the JDK 25 Vector API (`FloatVector.SPECIES_PREFERRED`, FMA).
  *
  * Shares validation and the remaining kernels with [[ScalarF32Kernels]]. Rounding differs from the
  * scalar path (FMA, lane-wise partial sums), within the contract's tolerances, but for a fixed
  * backend instance and shape the result is reproducible: every reduction uses a fixed order that
  * does not depend on JIT compilation (no `reduceLanes`). GEMMs whose
  * `m * n * k` is below `smallGemm` stay on the scalar loops; the threshold is fixed per instance.
  */
class VectorF32Kernels(val smallGemm: Long) extends ScalarF32Kernels:
  // static final (Java) fields: the JIT must see the species as a constant
  import Species.{L, S}

  override def name: String = "vector25"

  override def describe: String = s"vector25 (species=${S}, lanes=$L, smallGemm=$smallGemm)"

  private inline def small(m: Int, n: Int, k: Int): Boolean = m.toLong * n * k < smallGemm

  /** Sum of the lanes in lane order 0, 1, ..., L-1.
    *
    * `reduceLanes(ADD)` leaves the float addition order unspecified, and the interpreter and the C2
    * intrinsic do differ, so results would change once a method gets JIT-compiled. Lane extraction and
    * lanewise `fma` are exact, so this keeps each backend/shape run-to-run reproducible.
    */
  private inline def sumLanes(v: FloatVector): Float =
    var s = v.lane(0)
    var l = 1
    while l < L do
      s += v.lane(l)
      l += 1
    s

  // ------------------------------------------------------------------ GEMM

  override protected def useScalarGemm(m: Int, n: Int, k: Int): Boolean = small(m, n, k)

  override protected def fastGemm(
      a: MatrixF32, transA: Transpose, b: MatrixF32, transB: Transpose, c: MatrixF32,
      m: Int, n: Int, k: Int, alpha: Float, ws: Workspace
  ): Unit =
    (transA, transB) match
      case (Transpose.No, Transpose.No) => panels(a.data, a.offset, a.rowStride, 1, b, c, m, n, k, alpha)
      case (Transpose.Yes, Transpose.No) => panels(a.data, a.offset, 1, a.rowStride, b, c, m, n, k, alpha)
      case (Transpose.No, Transpose.Yes) => nt(a, b, c, m, n, k, alpha)
      case (Transpose.Yes, Transpose.Yes) => panels(a.data, a.offset, 1, a.rowStride, packTransposed(b, k, n, ws), c, m, n, k, alpha)

  /** `C += alpha * A B` where `A(i, p) = ad(ao + i * ai + p * ap)` and B is row-major.
    * Column panels of width `2L` outermost (the B panel stays in L1/L2), 4-row register blocks inside.
    */
  private def panels(ad: Array[Float], ao: Int, ai: Int, ap: Int, b: MatrixF32, c: MatrixF32, m: Int, n: Int, k: Int, alpha: Float): Unit =
    val av = FloatVector.broadcast(S, alpha)
    val full2 = n - n % (2 * L)
    var j = 0
    while j < full2 do
      var i = 0
      while i + 3 < m do
        block4x2(ad, ao, ai, ap, b, c, i, j, k, av)
        i += 4
      while i < m do
        block1x2(ad, ao, ai, ap, b, c, i, j, k, av)
        i += 1
      j += 2 * L
    while j < n do
      val mask = S.indexInRange(j, n)
      var i = 0
      while i + 3 < m do
        block4x1(ad, ao, ai, ap, b, c, i, j, k, av, mask)
        i += 4
      while i < m do
        block1x1(ad, ao, ai, ap, b, c, i, j, k, av, mask)
        i += 1
      j += L

  private def block4x2(ad: Array[Float], ao: Int, ai: Int, ap: Int, b: MatrixF32, c: MatrixF32, i: Int, j: Int, k: Int, alpha: FloatVector): Unit =
    val bd = b.data
    var c00 = FloatVector.zero(S); var c01 = FloatVector.zero(S)
    var c10 = FloatVector.zero(S); var c11 = FloatVector.zero(S)
    var c20 = FloatVector.zero(S); var c21 = FloatVector.zero(S)
    var c30 = FloatVector.zero(S); var c31 = FloatVector.zero(S)
    val r0 = ao + i * ai
    var p = 0
    while p < k do
      val bo = b.offset + p * b.rowStride + j
      val b0 = FloatVector.fromArray(S, bd, bo)
      val b1 = FloatVector.fromArray(S, bd, bo + L)
      val ab = r0 + p * ap
      val a0 = FloatVector.broadcast(S, ad(ab))
      val a1 = FloatVector.broadcast(S, ad(ab + ai))
      val a2 = FloatVector.broadcast(S, ad(ab + 2 * ai))
      val a3 = FloatVector.broadcast(S, ad(ab + 3 * ai))
      c00 = a0.fma(b0, c00); c01 = a0.fma(b1, c01)
      c10 = a1.fma(b0, c10); c11 = a1.fma(b1, c11)
      c20 = a2.fma(b0, c20); c21 = a2.fma(b1, c21)
      c30 = a3.fma(b0, c30); c31 = a3.fma(b1, c31)
      p += 1
    store2(c, i, j, c00, c01, alpha)
    store2(c, i + 1, j, c10, c11, alpha)
    store2(c, i + 2, j, c20, c21, alpha)
    store2(c, i + 3, j, c30, c31, alpha)

  private def block1x2(ad: Array[Float], ao: Int, ai: Int, ap: Int, b: MatrixF32, c: MatrixF32, i: Int, j: Int, k: Int, alpha: FloatVector): Unit =
    val bd = b.data
    var c0 = FloatVector.zero(S); var c1 = FloatVector.zero(S)
    val r0 = ao + i * ai
    var p = 0
    while p < k do
      val bo = b.offset + p * b.rowStride + j
      val a0 = FloatVector.broadcast(S, ad(r0 + p * ap))
      c0 = a0.fma(FloatVector.fromArray(S, bd, bo), c0)
      c1 = a0.fma(FloatVector.fromArray(S, bd, bo + L), c1)
      p += 1
    store2(c, i, j, c0, c1, alpha)

  private def block4x1(ad: Array[Float], ao: Int, ai: Int, ap: Int, b: MatrixF32, c: MatrixF32, i: Int, j: Int, k: Int,
                       alpha: FloatVector, mask: VectorMask[java.lang.Float]): Unit =
    val bd = b.data
    var c0 = FloatVector.zero(S); var c1 = FloatVector.zero(S); var c2 = FloatVector.zero(S); var c3 = FloatVector.zero(S)
    val r0 = ao + i * ai
    var p = 0
    while p < k do
      val b0 = FloatVector.fromArray(S, bd, b.offset + p * b.rowStride + j, mask)
      val ab = r0 + p * ap
      c0 = FloatVector.broadcast(S, ad(ab)).fma(b0, c0)
      c1 = FloatVector.broadcast(S, ad(ab + ai)).fma(b0, c1)
      c2 = FloatVector.broadcast(S, ad(ab + 2 * ai)).fma(b0, c2)
      c3 = FloatVector.broadcast(S, ad(ab + 3 * ai)).fma(b0, c3)
      p += 1
    store1(c, i, j, c0, alpha, mask)
    store1(c, i + 1, j, c1, alpha, mask)
    store1(c, i + 2, j, c2, alpha, mask)
    store1(c, i + 3, j, c3, alpha, mask)

  private def block1x1(ad: Array[Float], ao: Int, ai: Int, ap: Int, b: MatrixF32, c: MatrixF32, i: Int, j: Int, k: Int,
                       alpha: FloatVector, mask: VectorMask[java.lang.Float]): Unit =
    val bd = b.data
    var c0 = FloatVector.zero(S)
    val r0 = ao + i * ai
    var p = 0
    while p < k do
      c0 = FloatVector.broadcast(S, ad(r0 + p * ap)).fma(FloatVector.fromArray(S, bd, b.offset + p * b.rowStride + j, mask), c0)
      p += 1
    store1(c, i, j, c0, alpha, mask)

  /** `C(i, j until j + 2L) += alpha * (acc0, acc1)`. */
  private inline def store2(c: MatrixF32, i: Int, j: Int, acc0: FloatVector, acc1: FloatVector, alpha: FloatVector): Unit =
    val o = c.offset + i * c.rowStride + j
    acc0.fma(alpha, FloatVector.fromArray(S, c.data, o)).intoArray(c.data, o)
    acc1.fma(alpha, FloatVector.fromArray(S, c.data, o + L)).intoArray(c.data, o + L)

  private inline def store1(c: MatrixF32, i: Int, j: Int, acc: FloatVector, alpha: FloatVector, mask: VectorMask[java.lang.Float]): Unit =
    val o = c.offset + i * c.rowStride + j
    acc.fma(alpha, FloatVector.fromArray(S, c.data, o, mask)).intoArray(c.data, o, mask)

  /** `C += alpha * A B^T`: two rows of A against four rows of B, lane-wise FMA partial sums over k, lanes summed
    * in a fixed order. Every element of C sees the same operation sequence whichever block computes it.
    */
  private def nt(a: MatrixF32, b: MatrixF32, c: MatrixF32, m: Int, n: Int, k: Int, alpha: Float): Unit =
    var i = 0
    while i + 1 < m do
      ntRows2(a, b, c, i, n, k, alpha)
      i += 2
    if i < m then ntRow1(a, b, c, i, n, k, alpha)

  private def ntRows2(a: MatrixF32, b: MatrixF32, c: MatrixF32, i: Int, n: Int, k: Int, alpha: Float): Unit =
    val ad = a.data; val bd = b.data; val cd = c.data
    val kb = S.loopBound(k)
    val tail = S.indexInRange(kb, k)
    val ar0 = a.offset + i * a.rowStride
    val ar1 = ar0 + a.rowStride
    val cr0 = c.offset + i * c.rowStride
    val cr1 = cr0 + c.rowStride
    var j = 0
    while j + 3 < n do
      val b0 = b.offset + j * b.rowStride
      val b1 = b0 + b.rowStride; val b2 = b1 + b.rowStride; val b3 = b2 + b.rowStride
      var s00 = FloatVector.zero(S); var s01 = FloatVector.zero(S); var s02 = FloatVector.zero(S); var s03 = FloatVector.zero(S)
      var s10 = FloatVector.zero(S); var s11 = FloatVector.zero(S); var s12 = FloatVector.zero(S); var s13 = FloatVector.zero(S)
      var p = 0
      while p < kb do
        val x0 = FloatVector.fromArray(S, ad, ar0 + p)
        val x1 = FloatVector.fromArray(S, ad, ar1 + p)
        val y0 = FloatVector.fromArray(S, bd, b0 + p)
        val y1 = FloatVector.fromArray(S, bd, b1 + p)
        val y2 = FloatVector.fromArray(S, bd, b2 + p)
        val y3 = FloatVector.fromArray(S, bd, b3 + p)
        s00 = x0.fma(y0, s00); s01 = x0.fma(y1, s01); s02 = x0.fma(y2, s02); s03 = x0.fma(y3, s03)
        s10 = x1.fma(y0, s10); s11 = x1.fma(y1, s11); s12 = x1.fma(y2, s12); s13 = x1.fma(y3, s13)
        p += L
      if kb < k then
        val x0 = FloatVector.fromArray(S, ad, ar0 + kb, tail)
        val x1 = FloatVector.fromArray(S, ad, ar1 + kb, tail)
        val y0 = FloatVector.fromArray(S, bd, b0 + kb, tail)
        val y1 = FloatVector.fromArray(S, bd, b1 + kb, tail)
        val y2 = FloatVector.fromArray(S, bd, b2 + kb, tail)
        val y3 = FloatVector.fromArray(S, bd, b3 + kb, tail)
        s00 = x0.fma(y0, s00); s01 = x0.fma(y1, s01); s02 = x0.fma(y2, s02); s03 = x0.fma(y3, s03)
        s10 = x1.fma(y0, s10); s11 = x1.fma(y1, s11); s12 = x1.fma(y2, s12); s13 = x1.fma(y3, s13)
      cd(cr0 + j) += alpha * sumLanes(s00)
      cd(cr0 + j + 1) += alpha * sumLanes(s01)
      cd(cr0 + j + 2) += alpha * sumLanes(s02)
      cd(cr0 + j + 3) += alpha * sumLanes(s03)
      cd(cr1 + j) += alpha * sumLanes(s10)
      cd(cr1 + j + 1) += alpha * sumLanes(s11)
      cd(cr1 + j + 2) += alpha * sumLanes(s12)
      cd(cr1 + j + 3) += alpha * sumLanes(s13)
      j += 4
    while j < n do
      cd(cr0 + j) += alpha * dot(ad, ar0, bd, b.offset + j * b.rowStride, k, kb, tail)
      cd(cr1 + j) += alpha * dot(ad, ar1, bd, b.offset + j * b.rowStride, k, kb, tail)
      j += 1

  private def ntRow1(a: MatrixF32, b: MatrixF32, c: MatrixF32, i: Int, n: Int, k: Int, alpha: Float): Unit =
    val ad = a.data; val bd = b.data; val cd = c.data
    val kb = S.loopBound(k)
    val tail = S.indexInRange(kb, k)
    val ar = a.offset + i * a.rowStride
    val cr = c.offset + i * c.rowStride
    var j = 0
    while j < n do
      cd(cr + j) += alpha * dot(ad, ar, bd, b.offset + j * b.rowStride, k, kb, tail)
      j += 1

  /** Lane-wise FMA dot product with the fixed-order lane sum (the per-element recipe of the NT kernel). */
  private def dot(ad: Array[Float], ao: Int, bd: Array[Float], bo: Int, k: Int, kb: Int, tail: VectorMask[java.lang.Float]): Float =
    var s0 = FloatVector.zero(S)
    var p = 0
    while p < kb do
      s0 = FloatVector.fromArray(S, ad, ao + p).fma(FloatVector.fromArray(S, bd, bo + p), s0)
      p += L
    if kb < k then s0 = FloatVector.fromArray(S, ad, ao + kb, tail).fma(FloatVector.fromArray(S, bd, bo + kb, tail), s0)
    sumLanes(s0)

  // ------------------------------------------------------------------ rows / elementwise / scan

  /** `y(yo until yo + n) += alpha * x(xo until xo + n)` with FMA. */
  private def axpyRow(alpha: Float, x: Array[Float], xo: Int, y: Array[Float], yo: Int, n: Int): Unit =
    val va = FloatVector.broadcast(S, alpha)
    val bound = S.loopBound(n)
    var j = 0
    while j < bound do
      FloatVector.fromArray(S, x, xo + j).fma(va, FloatVector.fromArray(S, y, yo + j)).intoArray(y, yo + j)
      j += L
    while j < n do
      y(yo + j) = Math.fma(alpha, x(xo + j), y(yo + j))
      j += 1

  override protected def scatterAddImpl(dst: MatrixF32, ids: Array[Int], idsOffset: Int, count: Int, values: MatrixF32): Unit =
    var r = 0
    while r < count do
      val vi = values.offset + r * values.rowStride
      val di = dst.offset + ids(idsOffset + r) * dst.rowStride
      val bound = S.loopBound(dst.cols)
      var j = 0
      while j < bound do
        FloatVector.fromArray(S, dst.data, di + j).add(FloatVector.fromArray(S, values.data, vi + j)).intoArray(dst.data, di + j)
        j += L
      while j < dst.cols do
        dst.data(di + j) += values.data(vi + j)
        j += 1
      r += 1

  override protected def axpyImpl(alpha: Float, x: MatrixF32, y: MatrixF32): Unit =
    var i = 0
    while i < x.rows do
      axpyRow(alpha, x.data, x.offset + i * x.rowStride, y.data, y.offset + i * y.rowStride, x.cols)
      i += 1

  override protected def mulImpl(x: MatrixF32, y: MatrixF32, out: MatrixF32): Unit =
    val bound = S.loopBound(x.cols)
    var i = 0
    while i < x.rows do
      val xi = x.offset + i * x.rowStride
      val yi = y.offset + i * y.rowStride
      val oi = out.offset + i * out.rowStride
      var j = 0
      while j < bound do
        FloatVector.fromArray(S, x.data, xi + j).mul(FloatVector.fromArray(S, y.data, yi + j)).intoArray(out.data, oi + j)
        j += L
      while j < x.cols do
        out.data(oi + j) = x.data(xi + j) * y.data(yi + j)
        j += 1
      i += 1

  override protected def rowSumSquaresImpl(x: MatrixF32, out: Array[Float], outOffset: Int): Unit =
    val bound = S.loopBound(x.cols)
    var i = 0
    while i < x.rows do
      val xi = x.offset + i * x.rowStride
      var acc = FloatVector.zero(S)
      var j = 0
      while j < bound do
        val v = FloatVector.fromArray(S, x.data, xi + j)
        acc = v.fma(v, acc)
        j += L
      var s = sumLanes(acc)
      while j < x.cols do
        val v = x.data(xi + j)
        s = Math.fma(v, v, s)
        j += 1
      out(outOffset + i) = s
      i += 1

  override protected def affineScanImpl(a: MatrixF32, b: MatrixF32, initial: MatrixF32, out: MatrixF32, steps: Int, batch: Int): Unit =
    val d = a.cols
    val bound = S.loopBound(d)
    var t = 0
    while t < steps do
      var j = 0
      while j < batch do
        val row = t * batch + j
        val ai = a.offset + row * a.rowStride
        val bi = b.offset + row * b.rowStride
        val oi = out.offset + row * out.rowStride
        val pd = if t == 0 then initial.data else out.data
        val pi = if t == 0 then initial.offset + j * initial.rowStride else out.offset + (row - batch) * out.rowStride
        var f = 0
        while f < bound do
          FloatVector.fromArray(S, a.data, ai + f)
            .fma(FloatVector.fromArray(S, pd, pi + f), FloatVector.fromArray(S, b.data, bi + f))
            .intoArray(out.data, oi + f)
          f += L
        while f < d do
          out.data(oi + f) = Math.fma(a.data(ai + f), pd(pi + f), b.data(bi + f))
          f += 1
        j += 1
      t += 1

/** The default `vector25` backend: GEMMs below 2048 multiply-adds use the scalar loops. */
object VectorF32Kernels extends VectorF32Kernels(2048L)
