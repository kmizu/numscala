package com.github.kmizu.numscala.cpu.vector25

import com.github.kmizu.numscala.cpu.*
import com.github.kmizu.numscala.cpu.F32Fixtures.randomMatrix
import java.util.Random

/** Run-to-run reproducibility of the vector backend for a fixed shape (independent of JIT tier). */
class DeterminismSuite extends munit.FunSuite:
  private val vec = new VectorF32Kernels(0L)
  private val L = Species.L

  /** Bit-exact emulation of the NT kernel: per-lane FMA partial sums, then lanes summed in order 0..L-1. */
  private def emulateNT(a: MatrixF32, b: MatrixF32, i: Int, j: Int, k: Int, alpha: Float): Float =
    val lanes = new Array[Float](L)
    var p = 0
    while p < k do
      var l = 0
      while l < L && p + l < k do
        lanes(l) = Math.fma(a(i, p + l), b(j, p + l), lanes(l))
        l += 1
      p += L
    var s = lanes(0)
    var l = 1
    while l < L do
      s += lanes(l)
      l += 1
    0f + alpha * s

  test("NT GEMM equals its fixed-order emulation bit for bit (any JIT state)") {
    val rnd = new Random(11)
    for (m, n, k) <- Seq((7, 24, 16), (3, 9, 37), (64, 40, 384), (2, 5, 3)) do
      val a = randomMatrix(rnd, m, k, 1, 2); val b = randomMatrix(rnd, n, k, 0, 1)
      val c = MatrixF32.zeros(m, n)
      vec.gemmInto(a, Transpose.No, b, Transpose.Yes, c, 0.5f, 0f, new Workspace())
      for i <- 0 until m; j <- 0 until n do
        val e = emulateNT(a, b, i, j, k, 0.5f)
        assertEquals(java.lang.Float.floatToRawIntBits(c(i, j)), java.lang.Float.floatToRawIntBits(e), s"$m x $n x $k at ($i,$j)")
  }

  test("repeated calls stay bit-identical across JIT compilation (repro: A 7x16, B 24x16, NT, 20000 calls)") {
    val rnd = new Random(12)
    val a = randomMatrix(rnd, 7, 16); val b = randomMatrix(rnd, 24, 16)
    val kernels = new VectorF32Kernels(0L) // fresh instance, but the methods may already be compiled
    val ws = new Workspace()
    val c = MatrixF32.zeros(7, 24)
    kernels.gemmInto(a, Transpose.No, b, Transpose.Yes, c, 1f, 0f, ws)
    val first = c.toArray
    val x = randomMatrix(rnd, 4, 300); val sq0 = new Array[Float](4); val sq = new Array[Float](4)
    kernels.rowSumSquaresInto(x, sq0, 0)
    var it = 0
    while it < 20000 do
      kernels.gemmInto(a, Transpose.No, b, Transpose.Yes, c, 1f, 0f, ws)
      kernels.rowSumSquaresInto(x, sq, 0)
      if !java.util.Arrays.equals(c.toArray, first) then fail(s"NT GEMM changed at call $it")
      if !java.util.Arrays.equals(sq, sq0) then fail(s"rowSumSquares changed at call $it")
      it += 1
  }

  test("all GEMM layouts give identical bits on concurrent workers with their own workspaces") {
    val rnd = new Random(13)
    val a = randomMatrix(rnd, 64, 384); val b = randomMatrix(rnd, 384, 384)
    for (ta, tb) <- Seq(Transpose.No -> Transpose.No, Transpose.No -> Transpose.Yes, Transpose.Yes -> Transpose.No, Transpose.Yes -> Transpose.Yes) do
      val (aa, bb) = (if ta == Transpose.No then a else b.rowRange(0, 384), b)
      val m = aa.logicalRows(ta); val n = bb.logicalCols(tb)
      def run(): Array[Float] =
        val c = MatrixF32.zeros(m, n)
        VectorF32Kernels.gemmInto(aa, ta, bb, tb, c, new Workspace(VectorF32Kernels.gemmWorkspaceFloats(m, n, 384, ta, tb)))
        c.toArray
      val ref = run()
      val results = new Array[Array[Float]](8)
      val threads = (0 until 8).map(t => new Thread(() => { var r = run(); var q = 0; while q < 20 do { r = run(); q += 1 }; results(t) = r }))
      threads.foreach(_.start()); threads.foreach(_.join())
      results.foreach(r => assert(java.util.Arrays.equals(r, ref), s"$ta$tb differs between workers"))
  }

/** Accuracy and reproducibility of the vectorised exp-based kernels (sigmoid, SiLU, log-sum-exp). */
class ExpKernelsSuite extends munit.FunSuite:
  private val vec = VectorF32Kernels
  private val L = Species.L

  test("sigmoid and SiLU stay within a few ulp of the Double reference over [-110, 110]") {
    val n = 220001
    val xs = Array.tabulate(n)(i => (-110.0 + i * 0.001).toFloat)
    val s = new Array[Float](n); val l = new Array[Float](n)
    vec.sigmoidInto(MatrixF32.vector(xs), MatrixF32.vector(s))
    vec.siluInto(MatrixF32.vector(xs), MatrixF32.vector(l))
    var worstS = 0.0; var worstL = 0.0; var worstAt = 0f
    for i <- 0 until n do
      val x = xs(i).toDouble
      val ref = 1 / (1 + math.exp(-x))
      val lref = x * ref
      if ref > 1e-37 then worstS = math.max(worstS, math.abs(s(i) - ref) / ref)
      // below about -87 sigmoid(x) is subnormal in Float32, so SiLU's relative precision degrades there
      if ref > java.lang.Float.MIN_NORMAL && math.abs(lref) > 1e-37 then
        val e = math.abs(l(i) - lref) / math.abs(lref)
        if e > worstL then { worstL = e; worstAt = xs(i) }
    assert(worstS < 5e-7, s"sigmoid max rel err $worstS")
    assert(worstL < 5e-7, s"silu max rel err $worstL at x = $worstAt")
  }

  test("row log-sum-exp matches the Double reference") {
    val rnd = new java.util.Random(30)
    val x = MatrixF32.wrap(Array.fill(16 * 1001)(rnd.nextFloat() * 40f - 20f), 16, 1001)
    val out = new Array[Float](16)
    vec.rowLogSumExpInto(x, out, 0)
    for i <- 0 until 16 do
      val row = (0 until 1001).map(j => x(i, j).toDouble)
      val m = row.max
      val ref = m + math.log(row.map(v => math.exp(v - m)).sum)
      assertEqualsDouble(out(i).toDouble, ref, 2e-6 * math.abs(ref) + 1e-6)
  }

  test("exp-based kernels: lane and tail positions give the same bits, and bits are stable across 20000 calls") {
    val rnd = new java.util.Random(31)
    val base = Array.fill(L)(rnd.nextFloat() * 30f - 15f)
    val row = base ++ base.take(L - 1) // the same values once in a full vector and once in the scalar tail
    val x = MatrixF32.vector(row)
    val s = MatrixF32.zeros(1, row.length); val l = MatrixF32.zeros(1, row.length)
    vec.sigmoidInto(x, s); vec.siluInto(x, l)
    for j <- 0 until L - 1 do
      assertEquals(java.lang.Float.floatToRawIntBits(s(0, L + j)), java.lang.Float.floatToRawIntBits(s(0, j)))
      assertEquals(java.lang.Float.floatToRawIntBits(l(0, L + j)), java.lang.Float.floatToRawIntBits(l(0, j)))
    val big = MatrixF32.wrap(Array.fill(3 * 77)(rnd.nextFloat() * 20f - 10f), 3, 77)
    val s0 = MatrixF32.zeros(3, 77); val l0 = MatrixF32.zeros(3, 77); val e0 = new Array[Float](3)
    vec.sigmoidInto(big, s0); vec.siluInto(big, l0); vec.rowLogSumExpInto(big, e0, 0)
    val s1 = MatrixF32.zeros(3, 77); val l1 = MatrixF32.zeros(3, 77); val e1 = new Array[Float](3)
    var it = 0
    while it < 20000 do
      vec.sigmoidInto(big, s1); vec.siluInto(big, l1); vec.rowLogSumExpInto(big, e1, 0)
      if !java.util.Arrays.equals(s0.toArray, s1.toArray) || !java.util.Arrays.equals(l0.toArray, l1.toArray) ||
        !java.util.Arrays.equals(e0, e1)
      then fail(s"exp-based kernel output changed at call $it")
      it += 1
  }
