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
