package com.github.kmizu.numscala.cpu

import com.github.kmizu.numscala.*

/** `np.matmul` on Float32 through [[NDArrayF32Adapter]]: views, copies and dtype rules. */
class AdapterSuite extends munit.FunSuite:

  private def f32(shape: Int*)(seed: Int): NDArray[Float] =
    val rnd = new java.util.Random(seed)
    NDArray.fromArray(Array.fill(shape.product)(rnd.nextFloat() * 2f - 1f), shape.toArray)

  /** Compares a Float32 matmul result with the Float64 product of the same values. */
  private def assertCloseToF64(got: NDArray[Float], a: NDArray[Float], b: NDArray[Float]): Unit =
    val ref = np.matmul(a.astype[Double], b.astype[Double])
    assertEquals(got.shape, ref.shape)
    assertEquals(got.dtype, DType.Float32)
    val g = got.toArray; val r = ref.toArray
    val k = a.shape.last
    for i <- g.indices do
      assert(math.abs(g(i) - r(i)) <= 1e-6 + 8 * k * 6e-8 * (1 + math.abs(r(i))), s"elem $i: ${g(i)} vs ${r(i)}")

  test("contiguous, transposed, offset and gapped views go through without packing") {
    val a = f32(7, 9)(1); val b = f32(9, 5)(2)
    val big = f32(12, 20)(3)
    val cases = Seq(
      "plain" -> (a, b),
      "a.T" -> (f32(9, 7)(4).T, b),
      "b.T" -> (a, f32(5, 9)(5).T),
      "gapped rows with offset" -> (big("2:9", "3:12"), big("1:10", "10:15")),
      "transposed gapped" -> (big("3:12", "1:8").T, big("0:9", "13:18"))
    )
    for (name, (x, y)) <- cases do
      val ws = new Workspace()
      val r = NDArrayF32Adapter.matmul(x, y, ws)
      assertCloseToF64(r, x, y)
      assertEquals(ws.bytesPacked, 0L, name)
      assertEquals(ws.bytesConverted, 0L, name)
      assertEquals(ws.allocatedBytes, 0L, name)
      // and np.matmul / @@ use the same path
      assertCloseToF64(np.matmul(x, y), x, y)
      assertCloseToF64(x @@ y, x, y)
    // A^T B^T is the one layout the GEMM packs: op(B) only, counted in bytesPacked
    val (x, y) = (f32(9, 7)(6).T, f32(5, 9)(7).T)
    val ws = new Workspace()
    assertCloseToF64(NDArrayF32Adapter.matmul(x, y, ws), x, y)
    assertEquals(ws.bytesPacked, 9L * 5 * 4)
  }

  test("negative and non-unit column strides are packed and counted") {
    val a = f32(6, 8)(8); val b = f32(8, 4)(9)
    val rev = a("::-1", ::)
    val stepped = f32(8, 8)(10)(::, "::2")
    val ws = new Workspace()
    assertCloseToF64(NDArrayF32Adapter.matmul(rev, b, ws), rev, b)
    assertEquals(ws.bytesPacked, 6L * 8 * 4)
    ws.resetCounters()
    assertCloseToF64(NDArrayF32Adapter.matmul(a, stepped, ws), a, stepped)
    assertEquals(ws.bytesPacked, 8L * 4 * 4)
  }

  test("batch broadcast is computed per batch without materialising the broadcast operand") {
    val w = f32(16, 12)(11)
    val xs = f32(5, 3, 16)(12)
    val ws = new Workspace()
    val r = NDArrayF32Adapter.matmul(xs, w, ws)
    assertEquals(r.shape, Seq(5, 3, 12))
    assertCloseToF64(r, xs, w)
    assertEquals(ws.bytesPacked, 0L)
    assertEquals(ws.allocatedBytes, 0L)
    val both = NDArrayF32Adapter.matmul(f32(2, 1, 4, 6)(13), f32(3, 6, 2)(14), ws)
    assertEquals(both.shape, Seq(2, 3, 4, 2))
    assertCloseToF64(both, f32(2, 1, 4, 6)(13), f32(3, 6, 2)(14))
    // stacked transposed views
    val st = f32(4, 6, 5)(15).transpose(0, 2, 1)
    assertCloseToF64(NDArrayF32Adapter.matmul(st, f32(6, 3)(16), ws), st, f32(6, 3)(16))
    assertEquals(ws.bytesPacked, 0L)
  }

  test("1-D operands, empty shapes and errors follow np.matmul") {
    val v = f32(9)(17); val m = f32(9, 4)(18)
    assertEquals(np.matmul(v, m).shape, Seq(4))
    assertEquals(np.matmul(m.T, v).shape, Seq(4))
    assertEquals(np.matmul(v, v).shape, Seq())
    assertCloseToF64(np.matmul(v, m), v, m)
    assertEquals(np.matmul(f32(0, 3)(0), f32(3, 2)(0)).shape, Seq(0, 2))
    val z = np.matmul(f32(2, 0)(0), f32(0, 3)(0))
    assertEquals(z.shape, Seq(2, 3))
    assert(z.toArray.forall(_ == 0f))
    intercept[IllegalArgumentException](np.matmul(f32(2, 3)(0), f32(2, 3)(0)))
    intercept[IllegalArgumentException](np.matmul(NDArray.scalar(1f), f32(2)(0)))
  }

  test("dtype promotion is unchanged; converted bytes are counted") {
    val a = f32(3, 4)(19)
    val i8 = np.array(Seq(Seq(1, 2), Seq(3, 4), Seq(5, 6), Seq(7, 8))).astype[Byte]
    val r = np.matmul(a, i8)
    assertEquals(r.dtype, DType.Float32)
    assertCloseToF64(r, a, i8.astype[Float])
    assertEquals(np.matmul(a, i8.astype[Double]).dtype, DType.Float64)
    val ws = new Workspace()
    NDArrayF32Adapter.matmul(a, i8, ws)
    assertEquals(ws.bytesConverted, 8L * 4)
  }

  test("0 * inf propagates NaN on the Float32 path as well") {
    val a = NDArray.fromArray(Array(0f, 1f), Array(1, 2))
    val b = NDArray.fromArray(Array(Float.PositiveInfinity, 1f), Array(2, 1))
    assert(np.matmul(a, b)(0, 0).isNaN)
  }

  test("existing float32 results stay exact for small integer-valued inputs") {
    val a = np.arange(12.0).reshape(3, 4).astype[Float]
    val b = np.arange(8.0).reshape(4, 2).astype[Float]
    assertEquals(np.matmul(a, b).toArray.toSeq, Seq(28f, 34f, 76f, 98f, 124f, 162f))
  }

  test("backend selection: scalar always, vector25 explicit fails loudly without the module, auto falls back") {
    assertEquals(F32Backend.select("scalar").kernels, ScalarF32Kernels)
    val caps = F32Backend.capabilities
    if !caps.vectorModulePresent || !caps.vectorBackendOnClasspath then
      val e = intercept[IllegalStateException](F32Backend.select("vector25"))
      assert(e.getMessage.contains("unavailable"))
      val auto = F32Backend.select("auto")
      assertEquals(auto.chosen, "scalar")
      assert(auto.reason.contains("fell back"))
    intercept[IllegalArgumentException](F32Backend.select("gpu"))
    val report = KernelDiagnostics.report(F32Backend.select("scalar"), new Workspace(4))
    assert(report.contains("backend=scalar") && report.contains("allocatedBytes=16"), report)
  }
