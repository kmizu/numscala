package com.github.kmizu.numscala.cpu

import java.util.Random

/** Shared fixtures for the NS-CPU-KERNEL-1 contract; every backend runs exactly these tests. */
object F32Fixtures:
  /** A `rows x cols` matrix of uniform values in [-1, 1) at `offset` with `pad` spare floats per row (garbage in the gaps). */
  def randomMatrix(rnd: Random, rows: Int, cols: Int, offset: Int = 0, pad: Int = 0): MatrixF32 =
    val rs = math.max(cols + pad, 1)
    val data = Array.fill(offset + rows * rs + 3)(Float.NaN) // NaN in every unused slot
    var i = 0
    while i < rows do
      var j = 0
      while j < cols do
        data(offset + i * rs + j) = rnd.nextFloat() * 2f - 1f
        j += 1
      i += 1
    MatrixF32(data, offset, rows, cols, rs)

  /** `op(m)(i, j)` as a Double. */
  def at(m: MatrixF32, t: Transpose, i: Int, j: Int): Double = if t == Transpose.No then m(i, j) else m(j, i)

  /** Checks `c` against `alpha * op(A) op(B) + beta * c0` with the componentwise bound of NS-CPU-001 §8.1. */
  def checkGemm(
      a: MatrixF32, ta: Transpose, b: MatrixF32, tb: Transpose, c: MatrixF32, c0: Array[Float],
      alpha: Float, beta: Float, clue: String
  ): Unit =
    val m = c.rows; val n = c.cols; val k = a.logicalCols(ta)
    val u = math.pow(2, -24)
    val q = 2.0 * k + 4
    val gamma = q * u / (1 - q * u)
    var i = 0
    while i < m do
      var j = 0
      while j < n do
        var s = 0.0; var sa = 0.0
        var p = 0
        while p < k do
          val v = at(a, ta, i, p) * at(b, tb, p, j)
          s += v; sa += math.abs(v)
          p += 1
        val before = if beta == 0f then 0.0 else c0(i * n + j).toDouble
        val ref = alpha * s + beta * before
        val scale = math.abs(alpha) * sa + math.abs(beta * before)
        val bound = 1e-6 + 4 * gamma * scale
        val got = c(i, j)
        if !(math.abs(got - ref) <= bound) then
          throw new AssertionError(s"$clue: C($i,$j) = $got, reference $ref, bound $bound")
        j += 1
      i += 1

abstract class F32KernelContract(kernels: F32Kernels) extends munit.FunSuite:
  import F32Fixtures.*

  private val T = Transpose
  private val transposes = Seq(T.No -> T.No, T.No -> T.Yes, T.Yes -> T.No, T.Yes -> T.Yes)

  private def phys(rows: Int, cols: Int, t: Transpose): (Int, Int) = if t == T.No then (rows, cols) else (cols, rows)

  private def runGemm(rnd: Random, m: Int, n: Int, k: Int, ta: Transpose, tb: Transpose, alpha: Float, beta: Float,
                      offset: Int, pad: Int): Unit =
    val (ar, ac) = phys(m, k, ta)
    val (br, bc) = phys(k, n, tb)
    val a = randomMatrix(rnd, ar, ac, offset, pad)
    val b = randomMatrix(rnd, br, bc, offset + 1, pad)
    val c = randomMatrix(rnd, m, n, offset + 2, pad)
    if beta == 0f then fill(c, Float.NaN) // beta == 0 must not read C
    val c0 = c.toArray
    val ws = new Workspace(kernels.gemmWorkspaceFloats(m, n, k, ta, tb))
    kernels.gemmInto(a, ta, b, tb, c, alpha, beta, ws)
    checkGemm(a, ta, b, tb, c, c0, alpha, beta, s"${kernels.name} m=$m n=$n k=$k $ta$tb alpha=$alpha beta=$beta off=$offset pad=$pad")
    // padding between rows is never written
    if pad > 0 && m > 1 then assert(c.data(c.offset + n).isNaN, "row padding of C was written")

  private def fill(m: MatrixF32, v: Float): Unit =
    for i <- 0 until m.rows; j <- 0 until m.cols do m(i, j) = v

  // ------------------------------------------------------------------ K1 GEMM

  test("gemm: small and odd shapes, all transposes, offsets and padded rows") {
    val rnd = new Random(1)
    val dims = Seq(0, 1, 2, 3, 5, 7, 16, 17, 33)
    for (ta, tb) <- transposes; m <- dims; n <- Seq(1, 3, 17); k <- dims do
      runGemm(rnd, m, n, k, ta, tb, 1f, 0f, (m + k) % 3, (n + k) % 2)
  }

  test("gemm: alpha/beta combinations") {
    val rnd = new Random(2)
    for (ta, tb) <- transposes; (alpha, beta) <- Seq((1f, 0f), (0.5f, 2f), (-1.5f, 1f), (1f, -0.25f), (0f, 3f), (2f, 0f)) do
      runGemm(rnd, 9, 11, 13, ta, tb, alpha, beta, 1, 2)
  }

  test("gemm: representative shapes incl. tail features 383/385 and blocks larger than KC/NC") {
    val rnd = new Random(3)
    runGemm(rnd, 1, 384, 384, T.No, T.No, 1f, 0f, 0, 0)
    runGemm(rnd, 8, 768, 384, T.No, T.No, 1f, 0f, 0, 0)
    runGemm(rnd, 3, 385, 383, T.No, T.Yes, 1f, 0f, 1, 0)
    runGemm(rnd, 5, 383, 385, T.Yes, T.No, 1f, 1f, 0, 3)
    runGemm(rnd, 4, 129, 257, T.Yes, T.Yes, 1f, 0f, 2, 1)
    runGemm(rnd, 70, 600, 130, T.No, T.No, 1f, 0f, 0, 0)
  }

  test("gemm: beta = 0 over a NaN-filled C gives finite results") {
    val rnd = new Random(4)
    val a = randomMatrix(rnd, 4, 6); val b = randomMatrix(rnd, 6, 5)
    val c = MatrixF32.wrap(Array.fill(20)(Float.NaN), 4, 5)
    kernels.gemmInto(a, T.No, b, T.No, c, new Workspace())
    assert(c.toArray.forall(v => !v.isNaN && !v.isInfinite))
  }

  test("gemm: 0 * inf and 0 * nan are not skipped") {
    for (ta, tb) <- transposes; bad <- Seq(Float.PositiveInfinity, Float.NaN) do
      // op(A) = [[0, 1]], op(B) = [[bad], [1]]
      val a = if ta == T.No then MatrixF32.wrap(Array(0f, 1f), 1, 2) else MatrixF32.wrap(Array(0f, 1f), 2, 1)
      val b = if tb == T.No then MatrixF32.wrap(Array(bad, 1f), 2, 1) else MatrixF32.wrap(Array(bad, 1f), 1, 2)
      val c = MatrixF32.zeros(1, 1)
      kernels.gemmInto(a, ta, b, tb, c, new Workspace(2))
      assert(c(0, 0).isNaN, s"$ta$tb with $bad gave ${c(0, 0)}")
  }

  test("gemm: K == 0 or alpha == 0 only scales C and never reads A/B") {
    val c = MatrixF32.wrap(Array(1f, 2f, 3f, 4f), 2, 2)
    kernels.gemmInto(MatrixF32.wrap(new Array[Float](0), 2, 0), T.No, MatrixF32.wrap(new Array[Float](0), 0, 2), T.No, c, 1f, 2f, new Workspace())
    assertEquals(c.toArray.toSeq, Seq(2f, 4f, 6f, 8f))
    val nanA = MatrixF32.wrap(Array.fill(4)(Float.NaN), 2, 2)
    kernels.gemmInto(nanA, T.No, nanA, T.Yes, c, 0f, 0.5f, new Workspace())
    assertEquals(c.toArray.toSeq, Seq(1f, 2f, 3f, 4f))
    kernels.gemmInto(nanA, T.Yes, nanA, T.Yes, c, 0f, 0f, new Workspace()) // no workspace needed either
    assertEquals(c.toArray.toSeq, Seq(0f, 0f, 0f, 0f))
  }

  test("gemm: empty outputs are not touched") {
    val buf = Array(7f, 7f, 7f)
    val c = MatrixF32(buf, 1, 0, 5, 5)
    kernels.gemmInto(MatrixF32.zeros(0, 3), T.No, MatrixF32.zeros(3, 5), T.No, c, new Workspace())
    val c2 = MatrixF32(buf, 1, 4, 0, 1)
    kernels.gemmInto(MatrixF32.zeros(4, 3), T.No, MatrixF32.zeros(3, 0), T.No, c2, new Workspace())
    assertEquals(buf.toSeq, Seq(7f, 7f, 7f))
  }

  test("gemm: validation happens before any write") {
    val ws = new Workspace()
    val a = MatrixF32.zeros(2, 3)
    val c = MatrixF32.wrap(Array.fill(4)(9f), 2, 2)
    def untouched() = assertEquals(c.toArray.toSeq, Seq.fill(4)(9f))
    intercept[IllegalArgumentException](kernels.gemmInto(a, T.No, MatrixF32.zeros(2, 2), T.No, c, ws)) // inner dims
    intercept[IllegalArgumentException](kernels.gemmInto(a, T.No, MatrixF32.zeros(3, 3), T.No, c, ws)) // C shape
    // descriptor whose end overflows Int must be rejected using Long arithmetic
    val huge = MatrixF32(new Array[Float](16), 8, 3, 2, Int.MaxValue / 2 + 10)
    intercept[IllegalArgumentException](kernels.gemmInto(huge, T.No, MatrixF32.zeros(2, 2), T.No, MatrixF32.zeros(3, 2), ws))
    intercept[IllegalArgumentException](kernels.gemmInto(MatrixF32(new Array[Float](6), 0, 2, 3, 2), T.No, MatrixF32.zeros(3, 2), T.No, c, ws))
    intercept[IllegalArgumentException](kernels.gemmInto(MatrixF32(new Array[Float](6), -1, 2, 3, 3), T.No, MatrixF32.zeros(3, 2), T.No, c, ws))
    // C partially overlapping A
    val shared = Array.fill(8)(1f)
    val aa = MatrixF32(shared, 0, 2, 2, 2)
    val cc = MatrixF32(shared, 3, 2, 2, 2)
    intercept[IllegalArgumentException](kernels.gemmInto(aa, T.No, MatrixF32.zeros(2, 2), T.No, cc, ws))
    assertEquals(shared.toSeq, Seq.fill(8)(1f))
    // TT needs a packing workspace: shortfall detected before C is written
    intercept[IllegalStateException](kernels.gemmInto(MatrixF32.zeros(3, 2), T.Yes, MatrixF32.zeros(2, 3), T.Yes, c, ws))
    untouched()
  }

  test("gemm: A and B may overlap (read-only), e.g. A * A^T") {
    val rnd = new Random(5)
    val a = randomMatrix(rnd, 6, 7)
    val c = MatrixF32.zeros(6, 6)
    kernels.gemmInto(a, T.No, a, T.Yes, c, new Workspace())
    checkGemm(a, T.No, a, T.Yes, c, c.toArray, 1f, 0f, "A A^T")
  }

  test("gemm: TT packs into the workspace and counts bytesPacked") {
    val rnd = new Random(6)
    val a = randomMatrix(rnd, 5, 4); val b = randomMatrix(rnd, 3, 5)
    val ws = new Workspace(kernels.gemmWorkspaceFloats(4, 3, 5, T.Yes, T.Yes))
    val c = MatrixF32.zeros(4, 3)
    kernels.gemmInto(a, T.Yes, b, T.Yes, c, ws)
    checkGemm(a, T.Yes, b, T.Yes, c, c.toArray, 1f, 0f, "TT")
    assert(ws.bytesPacked > 0)
    val before = ws.allocatedBytes
    kernels.gemmInto(a, T.Yes, b, T.Yes, c, ws) // reused without growing
    assertEquals(ws.allocatedBytes, before)
    assertEquals(ws.growCount, 0)
  }

  test("workspace: debug mode rejects use from a second thread") {
    val ws = new Workspace(debug = true)
    kernels.gemmInto(MatrixF32.zeros(1, 1), T.No, MatrixF32.zeros(1, 1), T.No, MatrixF32.zeros(1, 1), ws)
    var err: Throwable | Null = null
    val th = new Thread(() =>
      try kernels.gemmInto(MatrixF32.zeros(1, 1), T.No, MatrixF32.zeros(1, 1), T.No, MatrixF32.zeros(1, 1), ws)
      catch case e: Throwable => err = e
    )
    th.start(); th.join()
    assert(err.isInstanceOf[IllegalStateException], s"got $err")
  }

  // ------------------------------------------------------------------ K2 rows

  test("rows: gather by id order, padded rows") {
    val rnd = new Random(7)
    val table = randomMatrix(rnd, 10, 5, 2, 3)
    val ids = Array(9, 0, 3, 3, 7)
    val out = randomMatrix(rnd, 5, 5, 1, 1)
    kernels.gatherRowsInto(table, ids, out)
    for r <- ids.indices; j <- 0 until 5 do assertEquals(out(r, j), table(ids(r), j))
    val sub = MatrixF32.zeros(2, 5)
    kernels.gatherRowsInto(table, ids, 1, 2, sub)
    assertEquals(sub(1, 4), table(3, 4))
  }

  test("rows: repeated ids accumulate; out-of-range and negative ids throw before writing") {
    val dst = MatrixF32.zeros(4, 3)
    val values = MatrixF32.wrap(Array(1f, 2f, 3f, 10f, 20f, 30f, 100f, 200f, 300f, 5f, 5f, 5f), 4, 3)
    kernels.scatterAddRowsInto(dst, Array(2, 2, 2, 0), values)
    assertEquals(dst.row(2).toArray.toSeq, Seq(111f, 222f, 333f))
    assertEquals(dst.row(0).toArray.toSeq, Seq(5f, 5f, 5f))
    val snapshot = dst.toArray.toSeq
    intercept[IndexOutOfBoundsException](kernels.scatterAddRowsInto(dst, Array(1, 1, 4, 0), values))
    intercept[IndexOutOfBoundsException](kernels.scatterAddRowsInto(dst, Array(1, -1, 0, 0), values))
    assertEquals(dst.toArray.toSeq, snapshot)
    intercept[IndexOutOfBoundsException](kernels.gatherRowsInto(dst, Array(4), MatrixF32.zeros(1, 3)))
    intercept[IllegalArgumentException](kernels.scatterAddRowsInto(dst, Array(0, 1), values)) // count mismatch
    intercept[IllegalArgumentException](kernels.scatterAddRowsInto(dst, Array(0), dst.row(1)))  // overlap
  }

  test("rows: coalesce sorts ids and sums duplicates in occurrence order") {
    val ids = Array(5, 1, 5, 3, 1, 5)
    val values = MatrixF32.wrap(Array.tabulate(12)(i => (i + 1).toFloat), 6, 2)
    val outIds = Array.fill(8)(-7)
    val outValues = MatrixF32.zeros(6, 2)
    val ws = new Workspace(0, kernels.coalesceWorkspaceLongs(ids.length))
    val u = kernels.coalesceRowsInto(ids, values, outIds, outValues, ws)
    assertEquals(u, 3)
    assertEquals(outIds.take(4).toSeq, Seq(1, 3, 5, -7))
    assertEquals(outValues.row(0).toArray.toSeq, Seq(3f + 9f, 4f + 10f))
    assertEquals(outValues.row(1).toArray.toSeq, Seq(7f, 8f))
    assertEquals(outValues.row(2).toArray.toSeq, Seq(1f + 5f + 11f, 2f + 6f + 12f))
    // coalesce + scatter == scatter of the raw rows
    val d1 = MatrixF32.zeros(6, 2); val d2 = MatrixF32.zeros(6, 2)
    kernels.scatterAddRowsInto(d1, ids, values)
    kernels.scatterAddRowsInto(d2, outIds, 0, u, outValues.rowRange(0, u))
    assertEquals(d1.toArray.toSeq, d2.toArray.toSeq)
    // capacity and workspace errors leave the outputs alone
    val small = MatrixF32.wrap(Array.fill(4)(-1f), 2, 2)
    intercept[IllegalArgumentException](kernels.coalesceRowsInto(ids, values, outIds, small, ws))
    assertEquals(small.toArray.toSeq, Seq.fill(4)(-1f))
    intercept[IllegalStateException](kernels.coalesceRowsInto(ids, values, outIds, outValues, new Workspace()))
    intercept[IndexOutOfBoundsException](kernels.coalesceRowsInto(Array(-1), MatrixF32.zeros(1, 2), outIds, outValues, ws))
    assertEquals(kernels.coalesceRowsInto(Array.empty[Int], MatrixF32.zeros(0, 2), outIds, outValues, ws), 0)
  }

  // ------------------------------------------------------------------ K3 elementwise / reductions

  test("elementwise: copy, fill, axpy, mul incl. exact aliasing and padded rows") {
    val rnd = new Random(8)
    val x = randomMatrix(rnd, 3, 37, 1, 2)
    val y = randomMatrix(rnd, 3, 37, 0, 5)
    val y0 = y.toArray
    kernels.axpyInto(0.5f, x, y)
    for i <- 0 until 3; j <- 0 until 37 do assertEqualsFloat(y(i, j), y0(i * 37 + j) + 0.5f * x(i, j), 1e-6f)
    kernels.axpyInto(1f, y, y) // exact alias: y = 2y
    for i <- 0 until 3; j <- 0 until 37 do assertEqualsFloat(y(i, j), 2f * (y0(i * 37 + j) + 0.5f * x(i, j)), 1e-5f)
    val out = MatrixF32.zeros(3, 37)
    kernels.mulInto(x, y, out)
    for i <- 0 until 3; j <- 0 until 37 do assertEquals(out(i, j), x(i, j) * y(i, j))
    kernels.mulInto(out, out, out)
    kernels.copyInto(x, out)
    assertEquals(out.toArray.toSeq, x.toArray.toSeq)
    kernels.fillInto(out, 3f)
    assert(out.toArray.forall(_ == 3f))
    assert(y.data(y.offset + 37).isNaN, "padding written")
    val buf = new Array[Float](10)
    intercept[IllegalArgumentException](kernels.copyInto(MatrixF32.vector(buf, 0, 5), MatrixF32.vector(buf, 2, 5)))
    intercept[IllegalArgumentException](kernels.axpyInto(1f, MatrixF32.zeros(2, 3), MatrixF32.zeros(3, 2)))
    kernels.axpyInto(1f, MatrixF32.zeros(0, 3), MatrixF32.zeros(0, 3))
  }

  test("elementwise: sigmoid and SiLU are overflow-free and handle non-finite values") {
    val xs = Array(-1000f, -88.8f, -20f, -1f, -0f, 0f, 1f, 20f, 88.8f, 1000f, Float.NegativeInfinity, Float.PositiveInfinity, Float.NaN)
    val x = MatrixF32.vector(xs)
    val s = MatrixF32.zeros(1, xs.length)
    val l = MatrixF32.zeros(1, xs.length)
    kernels.sigmoidInto(x, s)
    kernels.siluInto(x, l)
    for j <- xs.indices.dropRight(1) do
      val v = xs(j).toDouble
      val ref = if v.isNegInfinity then 0.0 else if v.isPosInfinity then 1.0 else 1 / (1 + math.exp(-v))
      assertEqualsDouble(s(0, j).toDouble, ref, 1e-7 + 1e-6 * ref, s"sigmoid(${xs(j)})")
      val lref = if v.isNegInfinity then 0.0 else if v.isPosInfinity then Double.PositiveInfinity else v * ref
      if lref.isInfinite then assertEquals(l(0, j), Float.PositiveInfinity)
      else assertEqualsDouble(l(0, j).toDouble, lref, 1e-7 + 1e-6 * math.abs(lref), s"silu(${xs(j)})")
    assert(s(0, xs.length - 1).isNaN && l(0, xs.length - 1).isNaN)
    kernels.sigmoidInto(x, x) // in place
    assertEquals(x.toArray.toSeq.dropRight(1), s.toArray.toSeq.dropRight(1))
  }

  test("reductions: row log-sum-exp is stable and follows the special-value rules") {
    val ninf = Float.NegativeInfinity; val pinf = Float.PositiveInfinity
    val rows = Seq(
      Array(1000f, 1000f, 999f), Array(-1000f, -1001f, -1000f), Array(0.5f, -2f, 3f),
      Array(ninf, ninf, ninf), Array(ninf, 1f, ninf), Array(1f, pinf, ninf), Array(1f, Float.NaN, pinf)
    )
    val x = MatrixF32(new Array[Float](7 * 4), 0, 7, 3, 4)
    for (r, i) <- rows.zipWithIndex; j <- 0 until 3 do x(i, j) = r(j)
    val out = new Array[Float](9)
    kernels.rowLogSumExpInto(x, out, 1)
    def lse(r: Array[Float]) = { val m = r.max.toDouble; m + math.log(r.map(v => math.exp(v - m)).sum) }
    assertEqualsDouble(out(1).toDouble, lse(rows(0)), 1e-3)
    assertEqualsDouble(out(2).toDouble, lse(rows(1)), 1e-3)
    assertEqualsDouble(out(3).toDouble, lse(rows(2)), 1e-5)
    assertEquals(out(4), ninf)
    assertEqualsDouble(out(5).toDouble, 1.0, 1e-6)
    assertEquals(out(6), pinf)
    assert(out(7).isNaN)
    assertEquals(out(0), 0f); assertEquals(out(8), 0f)
    val empty = new Array[Float](2)
    kernels.rowLogSumExpInto(MatrixF32(new Array[Float](0), 0, 2, 0, 1), empty, 0)
    assertEquals(empty.toSeq, Seq(ninf, ninf))
    intercept[IllegalArgumentException](kernels.rowLogSumExpInto(x, new Array[Float](6), 0))
    intercept[IllegalArgumentException](kernels.rowLogSumExpInto(x, x.data, 2))
  }

  test("reductions: row sum of squares") {
    val rnd = new Random(9)
    val x = randomMatrix(rnd, 5, 385, 3, 1)
    val out = new Array[Float](5)
    kernels.rowSumSquaresInto(x, out, 0)
    for i <- 0 until 5 do
      val ref = (0 until 385).map(j => x(i, j).toDouble * x(i, j)).sum
      assertEqualsDouble(out(i).toDouble, ref, 1e-5 * ref)
  }

  // ------------------------------------------------------------------ K4 affine scan

  test("scan: matches the sequential Double reference; a = 0 resets, a = 1, b = 0 holds") {
    val rnd = new Random(10)
    val steps = 6; val bsz = 3; val d = 19
    val a = randomMatrix(rnd, steps * bsz, d, 1, 2)
    val b = randomMatrix(rnd, steps * bsz, d, 0, 1)
    val init = randomMatrix(rnd, bsz, d, 2, 0)
    // boundary at t = 2 for batch 0 (a = 0), padding at t = 4 for batch 1 (a = 1, b = 0)
    for f <- 0 until d do
      a(2 * bsz + 0, f) = 0f
      a(4 * bsz + 1, f) = 1f; b(4 * bsz + 1, f) = 0f
    val out = randomMatrix(rnd, steps * bsz, d, 0, 3)
    kernels.affineScanInto(a, b, init, out)
    for j <- 0 until bsz; f <- 0 until d do
      var s = init(j, f).toDouble
      for t <- 0 until steps do
        s = a(t * bsz + j, f) * s + b(t * bsz + j, f)
        assertEqualsDouble(out(t * bsz + j, f).toDouble, s, 1e-5, s"t=$t j=$j f=$f")
    for f <- 0 until d do
      assertEquals(out(2 * bsz, f), b(2 * bsz, f))
      assertEquals(out(4 * bsz + 1, f), out(3 * bsz + 1, f))
    // out may alias b exactly
    val bCopy = MatrixF32.wrap(b.toArray, steps * bsz, d)
    kernels.affineScanInto(a, bCopy, init, bCopy)
    for r <- 0 until steps * bsz; f <- 0 until d do assertEqualsFloat(bCopy(r, f), out(r, f), 0f)
    // out must not overlap the initial state, rows must be a multiple of the batch
    val shared = MatrixF32.zeros(4, 2)
    intercept[IllegalArgumentException](
      kernels.affineScanInto(MatrixF32.zeros(4, 2), MatrixF32.zeros(4, 2), shared.rowRange(0, 2), shared)
    )
    intercept[IllegalArgumentException](kernels.affineScanInto(MatrixF32.zeros(5, 2), MatrixF32.zeros(5, 2), MatrixF32.zeros(2, 2), MatrixF32.zeros(5, 2)))
    kernels.affineScanInto(MatrixF32.zeros(0, 2), MatrixF32.zeros(0, 2), MatrixF32.zeros(0, 2), MatrixF32.zeros(0, 2))
  }

  test("scan: combine composes earlier then later step") {
    val (a, b) = AffineScan.combine(2f, 3f, 5f, 7f)
    // s -> 2s + 3 -> 5(2s + 3) + 7
    assertEquals((a, b), (10f, 22f))
  }

class ScalarF32KernelsSuite extends F32KernelContract(ScalarF32Kernels)
