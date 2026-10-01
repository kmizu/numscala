package numscala

/** Every code sample in README.md, kept compiling and correct. */
class ReadmeSuite extends munit.FunSuite:
  test("quick tour") {
    val a = np.arange(12.0).reshape(3, 4)
    assertEquals(a.toString, "[[ 0.  1.  2.  3.]\n [ 4.  5.  6.  7.]\n [ 8.  9. 10. 11.]]")
    assertEquals(a(1, ::).toList, List(4.0, 5.0, 6.0, 7.0))
    assertEquals(a(::, "1:3").shape, Seq(3, 2))
    assertEquals(a("::-1", 0).toList, List(8.0, 4.0, 0.0))
    assertEquals(a(a > 5.0).size, 6)
    assertEquals(a(np.array(0, 2), ::).shape, Seq(2, 4))
    assertEquals(a(---, None).shape, Seq(3, 4, 1))
    assertEquals((a * 2.0 + 1.0).sum(), 144.0)
    assertEquals(a.mean(0).repr, "array([4., 5., 6., 7.])")
    assertEquals((a @@ a.T).shape, Seq(3, 3))
    assertEquals(np.sqrt(np.arange(4)).repr, "array([0.        , 1.        , 1.41421356, 1.73205081])")
    val b = a.copy()
    b(0, ::) = 0.0
    b(b > 9.0) = -1.0
    assertEquals(b.sum(), 4.0 + 5 + 6 + 7 + 8 + 9 - 2)
  }

  test("typed dtypes") {
    val i = np.array(1, 2, 3)
    val f = np.array(1.0f, 2.0f, 3.0f)
    val r: NDArray[Double] = i + f
    val q: NDArray[Double] = i / i
    val s: Long = i.sum()
    val m: Double = i.mean()
    val u = np.array(250, 10).astype[UInt8] + np.array(10, 10).astype[UInt8]
    assertEquals(u.toString, "[ 4 20]")
    assertEquals((r.dtype, q.dtype, s, m), (DType.Float64, DType.Float64, 6L, 2.0))
  }

  test("modules") {
    // linalg
    val A = np.array(Seq(Seq(3.0, 1.0), Seq(1.0, 2.0)))
    val x = np.linalg.solve(A, np.array(9.0, 8.0))
    assertEquals(x.toList, List(2.0, 3.0))
    val eh = np.linalg.eigh(A)
    assertEquals(eh.eigenvalues.size, 2)
    val sv = np.linalg.svd(A)
    np.testing.assert_allclose(sv.U @@ np.diag(sv.S) @@ sv.Vh, A, atol = 1e-12)
    assertEqualsDouble(np.linalg.det(A).item, 5.0, 1e-12)
    // fft
    val spec = np.fft.fft(np.array(0.0, 1.0, 0.0, -1.0))
    np.testing.assert_allclose(spec.imag, np.array(0.0, -2.0, 0.0, 2.0), atol = 1e-12)
    assertEquals(np.fft.fftfreq(4).toList, List(0.0, 0.25, -0.5, -0.25))
    // random (bit-compatible with NumPy)
    val rng = np.random.default_rng(42)
    assertEquals(rng.random(3).toList, List(0.7739560485559633, 0.4388784397520523, 0.8585979199113825))
    val z = rng.normal(0.0, 1.0, 1000)
    assert(math.abs(z.mean()) < 0.2)
    np.random.seed(0)
    assertEquals(np.random.rand(1).item, 0.5488135039273248)
    // polynomials
    val xs = np.linspace(0.0, 1.0, 20)
    val p = np.polynomial.Polynomial.fit(xs, xs * xs * 3.0 + 1.0, 2)
    np.testing.assert_allclose(p.convert().coef, np.array(1.0, 0.0, 3.0), atol = 1e-10)
    assertEquals(np.polyval(np.array(1.0, 0.0, -1.0), np.array(2.0)).toList, List(3.0))
    // masked arrays
    val ma = np.ma.masked_less(np.array(1.0, -2.0, 3.0), 0.0)
    assertEquals(ma.toString, "[1.0 -- 3.0]")
    assertEquals(ma.mean(), 2.0)
    // strings
    assertEquals(np.strings.upper(np.array("ab", "cd")).toList, List("AB", "CD"))
    // ufunc methods
    assertEquals(np.add.reduce(np.arange(5)), 10L)
    assertEquals(np.multiply.outer(np.array(1, 2), np.array(1, 2, 3)).shape, Seq(2, 3))
    // statistics
    assertEquals(np.median(np.array(3.0, 1.0, 2.0)), 2.0)
    assertEquals(np.percentile(np.array(1.0, 2.0, 3.0, 4.0), 50.0), 2.5)
    // I/O
    val tmp = java.nio.file.Files.createTempFile("readme", ".npy")
    np.save(tmp, A)
    assertEquals(np.load(tmp, DType.Float64), A)
    java.nio.file.Files.delete(tmp)
  }

class ReadmeSuite2 extends munit.FunSuite:
  test("remaining README calls") {
    val A = np.array(Seq(Seq(3.0, 1.0), Seq(1.0, 2.0)))
    assertEquals(np.einsum("ij,jk->ik", A, A), A @@ A)
    val signal = np.sin(np.linspace(0.0, 6.0, 16))
    assertEquals(np.fft.rfft(signal).size, 9)
    val x = np.array(3.0, 1.0, 2.0, 5.0)
    val (hist, edges) = np.histogram(x, 2)
    assertEquals(hist.toList.sum, 4L)
    assertEquals(edges.size, 3)
    assertEquals(np.cov(A).shape, Seq(2, 2))
    assertEquals(np.unique(x).toList, List(1.0, 2.0, 3.0, 5.0))
    assertEquals(np.argsort(x).toList, List(1, 2, 0, 3))
    assertEquals(np.searchsorted(np.array(1.0, 2.0, 3.0), 2.5), 2)
    assertEquals(np.where(x > 2.0, x, np.zeros(4)).toList, List(3.0, 0.0, 0.0, 5.0))
    val tmp = java.nio.file.Files.createTempFile("readme", ".npz")
    np.savez(tmp.toString, "a" -> A)
    assertEquals(np.load_npz(tmp.toString)("a"), A)
    java.nio.file.Files.delete(tmp)
  }
