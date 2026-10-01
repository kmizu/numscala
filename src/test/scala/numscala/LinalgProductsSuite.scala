package numscala

class LinalgProductsSuite extends LinalgTestUtil:

  test("norm: vectors and matrices, all orders") {
    assertEqualsDouble(np.linalg.norm(B), 17.435595774162696, 1e-13)
    val ordsM: Seq[Linalg.NormOrd] =
      Seq(None, "fro", "nuc", Double.PositiveInfinity, Double.NegativeInfinity, 1, -1, 2, -2)
    val expM = Seq(17.435595774162696, 17.435595774162696, 18.48453303803646, 25.0, 6.0, 19.0, 12.0, 17.412505166808597,
      0.19686652111743008)
    ordsM.zip(expM).foreach((o, e) => assertEqualsDouble(np.linalg.norm(B, o), e, 1e-12 * e, s"ord $o"))
    val v = np.array(3.0, -4.0, 0.0, 1.0)
    val ordsV: Seq[Linalg.NormOrd] =
      Seq(None, Double.PositiveInfinity, Double.NegativeInfinity, 0, 1, 2, 3, -1, 0.5)
    val expV = Seq(5.0990195135927845, 4.0, 0.0, 3.0, 8.0, 5.0990195135927845, 4.514357435474001, 0.0, 22.392304845413257)
    ordsV.zip(expV).foreach((o, e) => assertEqualsDouble(np.linalg.norm(v, o), e, 1e-13 * math.max(1, e), s"ord $o"))
    val t = np.arange(24.0).reshape(2, 3, 4)
    close(np.linalg.norm(t, axis = 1), Seq(8.94427190999916, 10.344080432788601, 11.832159566199232, 13.379088160259652,
      28.284271247461902, 29.9833287011299, 31.68595903550972, 33.391615714128), 1e-13)
    close(np.linalg.norm(t, ord = 2, axis = Seq(0, 2)), Seq(27.278740934924464, 36.795010111317204, 47.10309908823585), 1e-12)
    assertEquals(np.linalg.norm(t, ord = 1, axis = Seq(2, 1), keepdims = true).shape, Seq(2, 1, 1))
    close(np.linalg.norm(t, ord = 1, axis = Seq(2, 1)), Seq(38.0, 86.0))
    assertEquals(np.linalg.norm(t, keepdims = true).shape, Seq(1, 1, 1))
    // complex
    val zo: Seq[Linalg.NormOrd] = Seq(None, "fro", "nuc", 1, Double.PositiveInfinity, 2, -2)
    val ze = Seq(5.937171043518958, 5.937171043518958, 6.344288770224759, 6.267196851649064, 7.031128874149275,
      5.92214438511238, 0.42214438511237984)
    zo.zip(ze).foreach((o, e) => assertEqualsDouble(np.linalg.norm(Z, o), e, 1e-12 * e, s"complex ord $o"))
    assertEqualsDouble(np.linalg.norm(np.array(Seq(Complex(3, 4), Complex(0, 1))), 1), 6.0, 1e-14)
    // integer input
    assertEqualsDouble(np.linalg.norm(np.array(3, 4)), 5.0, 0)
    // errors
    intercept[IllegalArgumentException](np.linalg.norm(v, "fro"))
    intercept[IllegalArgumentException](np.linalg.norm(B, 3))
    intercept[IllegalArgumentException](np.linalg.norm(t, 2))
    intercept[IllegalArgumentException](np.linalg.norm(t, axis = Seq(1, 1)))
    // NaN propagates in max
    assert(np.linalg.norm(np.array(1.0, Double.NaN), Double.PositiveInfinity).isNaN)
  }

  test("vector_norm and matrix_norm") {
    val t = np.arange(24.0).reshape(2, 3, 4)
    assertEqualsDouble(np.linalg.vector_norm(t).item, 65.75712889109438, 1e-12)
    close(np.linalg.vector_norm(t, axis = Seq(0, 2), ord = 1), Seq(60.0, 92.0, 124.0))
    assertEquals(np.linalg.vector_norm(t, ord = Double.PositiveInfinity, keepdims = true).shape, Seq(1, 1, 1))
    close(np.linalg.matrix_norm(t), Seq(22.494443758403985, 61.78996682310163), 1e-13)
    close(np.linalg.matrix_norm(t, ord = "nuc"), Seq(24.364638499284712, 62.49508468032368), 1e-12)
    assertEquals(np.linalg.matrix_norm(t, keepdims = true).shape, Seq(2, 1, 1))
  }

  test("cond") {
    val exp = Seq(88.4482799206987, 133.0, 28.000000000000007, 158.33333333333334, 90.78423749626242, 116.07658777239588,
      88.4482799206987, 0.011306042366189414)
    val ords: Seq[Linalg.NormOrd] = Seq(None, 1, -1, Double.PositiveInfinity, "fro", "nuc", 2, -2)
    ords.zip(exp).foreach((o, e) => assertEqualsDouble(np.linalg.cond(B, o).item, e, 1e-10 * e, s"cond $o"))
    close(np.linalg.cond(np.array(Seq(Seq(Seq(1.0, 2), Seq(3.0, 4)), Seq(Seq(2.0, 0), Seq(0.0, 1))))),
      Seq(14.933034373659265, 2.0), 1e-12)
    assertEquals(np.linalg.cond(m(Seq(1.0, 2), Seq(2.0, 4)), 1).item, Double.PositiveInfinity)
    assert(np.linalg.cond(m(Seq(1.0, 2), Seq(2.0, 4))).item > 1e15)
  }

  test("multi_dot, tensorsolve, tensorinv, linalg helpers") {
    val r = np.linalg.multi_dot(Seq(np.arange(3.0), np.arange(6.0).reshape(3, 2), np.arange(8.0).reshape(2, 4), np.arange(4.0)))
    assertEquals(r.ndim, 0)
    assertEqualsDouble(r.item, 634.0, 0)
    val a1 = rand(1, 10, 30)
    val a2 = rand(2, 30, 5)
    val a3 = rand(3, 5, 60)
    val a4 = rand(4, 60, 2)
    val md = np.linalg.multi_dot(Seq(a1, a2, a3, a4))
    assert(maxAbsDiff(md, a1 @@ a2 @@ a3 @@ a4) < 1e-12)
    assertEquals(np.linalg.multi_dot(Seq(a1, a2)).shape, Seq(10, 5))
    intercept[IllegalArgumentException](np.linalg.multi_dot(Seq(a1)))
    val ts = np.linalg.tensorsolve(np.eye(6).reshape(2, 3, 2, 3) * 2.0, np.arange(6.0).reshape(2, 3))
    assertEquals(ts.shape, Seq(2, 3))
    close(ts, Seq(0.0, 0.5, 1.0, 1.5, 2.0, 2.5), 1e-15)
    val ta = rand(8, 2, 3, 6)
    val tb = rand(9, 2, 3)
    val tx = np.linalg.tensorsolve(ta, tb)
    assertEquals(tx.shape, Seq(6))
    assert(maxAbsDiff(np.tensordot(ta, tx, 1), tb) < 1e-12)
    val ti = np.linalg.tensorinv(np.eye(24).reshape(4, 6, 8, 3) * 2.0, ind = 2)
    assertEquals(ti.shape, Seq(8, 3, 4, 6))
    intercept[LinAlgError](np.linalg.tensorinv(np.ones(2, 3), ind = 1))
    val st = np.arange(12.0).reshape(2, 2, 3)
    assertEquals(np.linalg.matrix_transpose(st).shape, Seq(2, 3, 2))
    close(np.linalg.diagonal(st), Seq(0.0, 4, 6, 10))
    close(np.linalg.trace(st), Seq(4.0, 16.0))
    close(np.linalg.trace(st, offset = 1), Seq(6.0, 18.0))
    assertEquals(np.linalg.outer(np.array(1, 2), np.array(3, 4, 5)).toList, List(3, 4, 5, 6, 8, 10))
    intercept[IllegalArgumentException](np.linalg.outer(np.ones(2, 2), np.ones(2)))
    assertEquals(np.linalg.cross(np.array(1, 2, 3), np.array(4, 5, 6)).toList, List(-3, 6, -3))
    intercept[IllegalArgumentException](np.linalg.cross(np.array(1, 2), np.array(4, 5)))
    assertEquals(np.linalg.matmul(np.array(Seq(Seq(1, 2))), np.array(Seq(Seq(3), Seq(4)))).toList, List(11))
    val vd = np.linalg.vecdot(np.array(Seq(Complex(1, 1), Complex(2, 0))), np.array(Seq(Complex(1, 0), Complex(0, 1))))
    assertEquals(vd.item, Complex(1, 1))
  }

  test("dot, vdot, inner, outer, matmul") {
    val d3 = np.dot(np.arange(24).reshape(2, 3, 4), np.arange(8).reshape(4, 2))
    assertEquals(d3.shape, Seq(2, 3, 2))
    assertEquals(d3.toList, List(28, 34, 76, 98, 124, 162, 172, 226, 220, 290, 268, 354))
    assertEquals(np.dot(np.array(1.0, 2.0), np.array(3.0, 4.0)).item, 11.0)
    val vdz = np.vdot(np.array(Seq(Complex(1, 2), Complex(3, -1))), np.array(Seq(Complex(2, -1), Complex(0, 1))))
    assertEquals(vdz, Complex(-1, -2))
    val vdi: Int = np.vdot(np.arange(4).reshape(2, 2), np.arange(4).reshape(2, 2))
    assertEquals(vdi, 14)
    assertEquals(np.inner(np.arange(6).reshape(2, 3), np.arange(9).reshape(3, 3)).toList, List(5, 14, 23, 14, 50, 86))
    assertEquals(np.inner(np.array(1, 2, 3), np.array(0, 1, 0)).item, 2)
    intercept[IllegalArgumentException](np.inner(np.ones(2), np.ones(3)))
    assertEquals(np.outer(np.array(Seq(Seq(1, 2), Seq(3, 4))), np.array(1, 10)).toList, List(1, 10, 2, 20, 3, 30, 4, 40))
    val mm: NDArray[Double] = np.matmul(np.array(Seq(Seq(1, 2))), np.array(Seq(Seq(0.5), Seq(1.5))))
    assertEquals(mm.toList, List(3.5))
  }

  test("tensordot") {
    val a = np.arange(60.0).reshape(3, 4, 5)
    val b = np.arange(24.0).reshape(4, 3, 2)
    val r = np.tensordot(a, b, (Seq(1, 0), Seq(0, 1)))
    assertEquals(r.shape, Seq(5, 2))
    close(r, Seq(4400.0, 4730.0, 4532.0, 4874.0, 4664.0, 5018.0, 4796.0, 5162.0, 4928.0, 5306.0))
    assertEquals(np.tensordot(np.arange(6).reshape(2, 3), np.arange(12).reshape(3, 4), 1).toList,
      List(20, 23, 26, 29, 56, 68, 80, 92))
    assertEquals(np.tensordot(np.array(1, 2), np.array(3, 4, 5), 0).toList, List(3, 4, 5, 6, 8, 10))
    assertEquals(np.tensordot(np.ones(2, 3), np.ones(2, 3)).item, 6.0)
    intercept[IllegalArgumentException](np.tensordot(np.ones(2, 3), np.ones(3, 2), (Seq(0), Seq(0))))
  }

  test("einsum") {
    val x = np.arange(6).reshape(2, 3)
    val y = np.arange(12).reshape(3, 4)
    assertEquals(np.einsum("ij,jk", x, y).toList, List(20, 23, 26, 29, 56, 68, 80, 92))
    assertEquals(np.einsum("ij,jk->ik", x, y).toList, List(20, 23, 26, 29, 56, 68, 80, 92))
    val sq = np.arange(9).reshape(3, 3)
    assertEquals(np.einsum("ii", sq).item, 12)
    assertEquals(np.einsum("ii->i", sq).toList, List(0, 4, 8))
    assertEquals(np.einsum("ji", x).shape, Seq(3, 2))
    assertEquals(np.einsum("ba", x).shape, Seq(3, 2))
    assertEquals(np.einsum("ij->ji", x).toList, List(0, 3, 1, 4, 2, 5))
    val t = np.arange(24).reshape(2, 3, 4)
    val u = np.arange(40).reshape(2, 4, 5)
    assertEquals(np.einsum("bij,bjk->bik", t, u).toSeq.take(6).toList, List(70, 76, 82, 88, 94, 190))
    assertEquals(np.einsum("...ij,...jk", t, u).toSeq.take(6).toList, List(70, 76, 82, 88, 94, 190))
    assertEquals(np.einsum("ijk->", t).item, 276)
    assertEquals(np.einsum("ijk->kj", t).toList, List(12, 20, 28, 14, 22, 30, 16, 24, 32, 18, 26, 34))
    assertEquals(np.einsum("i,j", np.array(1, 2), np.array(3, 4, 5)).toList, List(3, 4, 5, 6, 8, 10))
    assertEquals(
      np.einsum("ij,jk,kl->il", np.arange(4).reshape(2, 2), np.arange(6).reshape(2, 3), np.arange(6).reshape(3, 2)).toList,
      List(28, 40, 104, 146)
    )
    assertEquals(np.einsum("Ab,bc", np.ones(2, 3), np.ones(3, 4)).shape, Seq(2, 4))
    close(np.einsum("...i,...i", np.arange(6.0).reshape(2, 3), np.arange(3.0)), Seq(5.0, 14.0))
    assertEquals(np.einsum("iji->j", np.arange(27).reshape(3, 3, 3)).toList, List(30, 39, 48))
    assertEquals(np.einsum("i,i", np.array(1, 2, 3), np.array(4, 5, 6)).item, 32)
    // mixed dtypes promote
    val mixed: NDArray[Double] = np.einsum("i,i->i", np.array(1, 2), np.array(0.5, 0.25))
    close(mixed, Seq(0.5, 0.5))
    val zc = np.einsum("ij,j", Z, np.array(Seq(Complex(1, 0), Complex(0, 1))))
    assert(maxAbsDiffC(zc, Z @@ np.array(Seq(Complex(1, 0), Complex(0, 1)))) < 1e-15)
    // larger, checked against matmul
    val p = rand(1, 7, 9)
    val q = rand(2, 9, 4)
    assert(maxAbsDiff(np.einsum("ij,jk->ik", p, q), p @@ q) < 1e-14)
    assert(maxAbsDiff(np.einsum("ij,jk->ki", p, q), (p @@ q).T) < 1e-14)
    // errors
    intercept[IllegalArgumentException](np.einsum("ij,jk", x))
    intercept[IllegalArgumentException](np.einsum("ijk", x))
    intercept[IllegalArgumentException](np.einsum("ij->ii", x))
    intercept[IllegalArgumentException](np.einsum("ij->k", x))
    intercept[IllegalArgumentException](np.einsum("ij,jk", x, np.ones[Int](4, 4)))
    val (path, report) = np.einsum_path("ij,jk,kl->il", np.ones(2, 2), np.ones(2, 3), np.ones(3, 2))
    assertEquals(path, Seq(Seq(0, 1), Seq(0, 1)))
    assert(report.contains("ij,jk,kl->il"))
  }

  test("kron, trace, cross, vecdot") {
    assertEquals(np.kron(np.array(Seq(Seq(1, 2), Seq(3, 4))), np.array(Seq(Seq(0, 5), Seq(6, 7)))).toList,
      List(0, 5, 0, 10, 6, 7, 12, 14, 0, 15, 0, 20, 18, 21, 24, 28))
    assertEquals(np.kron(np.array(1, 10, 100), np.array(5, 6, 7)).toList, List(5, 6, 7, 50, 60, 70, 500, 600, 700))
    assertEquals(np.kron(np.eye(2), np.ones(2, 2)).shape, Seq(4, 4))
    val km = np.kron(np.arange(4).reshape(2, 2), np.array(1, 2))
    assertEquals(km.shape, Seq(2, 4))
    assertEquals(km.toList, List(0, 0, 1, 2, 2, 4, 3, 6))
    val tr: Long = np.trace(np.arange(9).reshape(3, 3))
    assertEquals(tr, 12L)
    assertEquals(np.trace(np.arange(9).reshape(3, 3), 1).item, 6L)
    assertEquals(np.trace(np.arange(24).reshape(2, 3, 4), 0).toList, List(16L, 18L, 20L, 22L))
    assertEquals(np.trace(np.arange(24).reshape(2, 3, 4), 0, 1, 2).toList, List(15L, 51L))
    intercept[IllegalArgumentException](np.trace(np.arange(24).reshape(2, 3, 4)))
    assertEquals(np.cross(np.array(1, 2, 3), np.array(4, 5, 6)).toList, List(-3, 6, -3))
    assertEquals(np.cross(np.array(1, 2), np.array(3, 4)).item, -2)
    assertEquals(np.cross(np.array(1, 2), np.array(4, 5, 6)).toList, List(12, -6, -3))
    val c1 = np.cross(np.array(Seq(Seq(1, 2, 3), Seq(4, 5, 6))), np.array(Seq(Seq(7, 8, 9), Seq(1, 0, 2))), axis = 1)
    assertEquals(c1.toList, List(-6, 12, -6, 10, -2, -5))
    val c0 = np.cross(np.array(Seq(Seq(1, 4), Seq(2, 5), Seq(3, 6))), np.array(Seq(Seq(7, 1), Seq(8, 0), Seq(9, 2))),
      axisa = 0, axisb = 0, axisc = 0)
    assertEquals(c0.shape, Seq(3, 2))
    assertEquals(c0.toList, List(-6, 10, 12, -2, -6, -5))
    intercept[IllegalArgumentException](np.cross(np.array(1, 2, 3, 4), np.array(1, 2, 3, 4)))
    val vd = np.vecdot(np.array(Seq(Seq(Complex(1, 1), Complex(2, 0)), Seq(Complex(3, 0), Complex(0, 4)))),
      np.array(Seq(Seq(Complex(1, 0), Complex(0, 1)), Seq(Complex(2, 0), Complex(2, 0)))))
    assertEquals(vd.toList, List(Complex(1, 1), Complex(6, -8)))
    assertEquals(np.vecdot(np.arange(6).reshape(2, 3), np.array(1, 1, 1)).toList, List(3, 12))
    assertEquals(np.vecdot(np.arange(6).reshape(2, 3), np.array(1, 1), axis = 0).toList, List(3, 5, 7))
  }
