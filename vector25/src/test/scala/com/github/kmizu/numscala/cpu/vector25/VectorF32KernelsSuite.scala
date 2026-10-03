package com.github.kmizu.numscala.cpu.vector25

import com.github.kmizu.numscala.*
import com.github.kmizu.numscala.cpu.*
import java.util.Random

/** The shared contract on the default vector backend. */
class VectorF32KernelsSuite extends F32KernelContract(VectorF32Kernels)

/** The shared contract with every GEMM forced onto the SIMD loops (no small-shape fallback). */
class VectorF32KernelsNoFallbackSuite extends F32KernelContract(new VectorF32Kernels(0L))

/** Scalar vs vector on identical inputs, plus backend selection on a JVM that has the module. */
class BackendDiffSuite extends munit.FunSuite:
  import F32Fixtures.*

  private val vec = new VectorF32Kernels(0L)

  test("gemm: vector agrees with scalar (and the Double bound) on representative shapes") {
    val rnd = new Random(42)
    val T = Transpose
    val shapes = Seq((1, 384, 384), (8, 768, 384), (32, 384, 384), (3, 385, 383), (17, 129, 65), (5, 7, 1))
    for (m, n, k) <- shapes; (ta, tb) <- Seq(T.No -> T.No, T.No -> T.Yes, T.Yes -> T.No, T.Yes -> T.Yes) do
      val a = if ta == T.No then randomMatrix(rnd, m, k, 1, 1) else randomMatrix(rnd, k, m, 1, 1)
      val b = if tb == T.No then randomMatrix(rnd, k, n, 2, 0) else randomMatrix(rnd, n, k, 2, 0)
      val c1 = randomMatrix(rnd, m, n, 0, 2)
      val c2 = MatrixF32(c1.data.clone(), c1.offset, m, n, c1.rowStride)
      val c0 = c1.toArray
      val ws = new Workspace(k * n)
      ScalarF32Kernels.gemmInto(a, ta, b, tb, c1, 0.75f, 0.5f, ws)
      vec.gemmInto(a, ta, b, tb, c2, 0.75f, 0.5f, ws)
      checkGemm(a, ta, b, tb, c2, c0, 0.75f, 0.5f, s"vector $m x $n x $k $ta$tb")
      var maxDiff = 0f
      for i <- 0 until m; j <- 0 until n do maxDiff = math.max(maxDiff, math.abs(c1(i, j) - c2(i, j)))
      assert(maxDiff < 1e-4f, s"$m x $n x $k $ta$tb: max |scalar - vector| = $maxDiff")
  }

  test("elementwise, reductions and scan: vector agrees with scalar") {
    val rnd = new Random(43)
    val x = randomMatrix(rnd, 6, 383, 3, 2); val y = randomMatrix(rnd, 6, 383, 0, 1)
    val y1 = MatrixF32.wrap(y.toArray, 6, 383); val y2 = MatrixF32.wrap(y.toArray, 6, 383)
    ScalarF32Kernels.axpyInto(-0.3f, x, y1); vec.axpyInto(-0.3f, x, y2)
    for i <- 0 until 6; j <- 0 until 383 do assertEqualsFloat(y1(i, j), y2(i, j), 1e-6f)
    ScalarF32Kernels.mulInto(x, y, y1); vec.mulInto(x, y, y2)
    assertEquals(y1.toArray.toSeq, y2.toArray.toSeq)
    val s1 = new Array[Float](6); val s2 = new Array[Float](6)
    ScalarF32Kernels.rowSumSquaresInto(x, s1, 0); vec.rowSumSquaresInto(x, s2, 0)
    for i <- 0 until 6 do assertEqualsDouble(s1(i).toDouble, s2(i).toDouble, 1e-5 * s1(i))
    val a = randomMatrix(rnd, 12, 385); val b = randomMatrix(rnd, 12, 385); val init = randomMatrix(rnd, 3, 385)
    val o1 = MatrixF32.zeros(12, 385); val o2 = MatrixF32.zeros(12, 385)
    ScalarF32Kernels.affineScanInto(a, b, init, o1); vec.affineScanInto(a, b, init, o2)
    for i <- 0 until 12; j <- 0 until 385 do assertEqualsFloat(o1(i, j), o2(i, j), 1e-5f)
    val d1 = MatrixF32.zeros(4, 383); val d2 = MatrixF32.zeros(4, 383)
    ScalarF32Kernels.scatterAddRowsInto(d1, Array(3, 1, 3, 0, 3, 2), x)
    vec.scatterAddRowsInto(d2, Array(3, 1, 3, 0, 3, 2), x)
    assertEquals(d1.toArray.toSeq, d2.toArray.toSeq)
  }

  test("np.matmul through the adapter with the vector backend") {
    val rnd = new Random(44)
    val a = NDArray.fromArray(Array.fill(5 * 64 * 96)(rnd.nextFloat() - 0.5f), Array(5, 64, 96))
    val w = NDArray.fromArray(Array.fill(96 * 80)(rnd.nextFloat() - 0.5f), Array(80, 96)).T
    val ws = new Workspace()
    val r1 = NDArrayF32Adapter.matmul(a, w, ws, ScalarF32Kernels)
    val r2 = NDArrayF32Adapter.matmul(a, w, ws, VectorF32Kernels)
    assertEquals(ws.bytesPacked, 0L)
    val d = r1.toArray.zip(r2.toArray).map((p, q) => math.abs(p - q)).max
    assert(d < 1e-4f, s"max diff $d")
  }

  test("backend selection finds vector25 when the module is present") {
    val caps = F32Backend.capabilities
    assert(caps.vectorModulePresent && caps.vectorBackendOnClasspath, caps.toString)
    val sel = F32Backend.select("vector25")
    assertEquals(sel.kernels, VectorF32Kernels)
    assertEquals(F32Backend.select("auto").chosen, "vector25")
    assert(KernelDiagnostics.report(sel).contains("lanes="))
  }
