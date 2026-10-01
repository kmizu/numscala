package com.github.kmizu.numscala

import munit.Location

/** Shared helpers for the linear algebra suites. */
trait LinalgTestUtil extends munit.FunSuite:
  def close(a: NDArray[Double], exp: Seq[Double], tol: Double = 1e-10)(using Location): Unit =
    val got = a.toSeq
    assertEquals(got.length, exp.length, s"length of ${got} vs ${exp}")
    got.zip(exp).zipWithIndex.foreach { case ((g, e), i) =>
      if e.isNaN then assert(g.isNaN, s"index $i: expected NaN, got $g")
      else if e.isInfinite then assertEquals(g, e, s"index $i")
      else assert(math.abs(g - e) <= tol * math.max(1.0, math.abs(e)), s"index $i: $g != $e (all: $got)")
    }

  def closeC(a: NDArray[Complex], exp: Seq[Complex], tol: Double = 1e-10)(using Location): Unit =
    val got = a.toSeq
    assertEquals(got.length, exp.length)
    got.zip(exp).zipWithIndex.foreach { case ((g, e), i) =>
      assert((g - e).abs <= tol * math.max(1.0, e.abs), s"index $i: $g != $e (all: $got)")
    }

  def m(rows: Seq[Double]*): NDArray[Double] = np.array(rows.toSeq)
  def cm(rows: Seq[Complex]*): NDArray[Complex] = np.array(rows.toSeq)

  /** Deterministic pseudo-random matrix in [-1, 1). */
  def rand(seed: Long, shape: Int*): NDArray[Double] =
    var s = seed
    NDArray.tabulate(shape*) { _ =>
      s = s * 6364136223846793005L + 1442695040888963407L
      ((s >>> 11).toDouble / (1L << 53).toDouble) * 2 - 1
    }

  def randC(seed: Long, shape: Int*): NDArray[Complex] =
    val r = rand(seed, shape*).toArray
    val i = rand(seed + 7, shape*).toArray
    NDArray.fromArray(r.indices.map(k => Complex(r(k), i(k))).toArray, shape.toArray)

  def maxAbsDiff(a: NDArray[Double], b: NDArray[Double]): Double =
    a.toSeq.zip(b.toSeq).map((x, y) => math.abs(x - y)).maxOption.getOrElse(0.0)

  def maxAbsDiffC(a: NDArray[Complex], b: NDArray[Complex]): Double =
    a.toSeq.zip(b.toSeq).map((x, y) => (x - y).abs).maxOption.getOrElse(0.0)

  def adjoint(a: NDArray[Complex]): NDArray[Complex] = a.conj.swapaxes(-1, -2)

  val B: NDArray[Double] = m(Seq(1.0, 2, 3), Seq(4.0, 5, 6), Seq(7.0, 8, 10))
  val A: NDArray[Double] = m(Seq(4.0, 1, 2), Seq(1.0, 3, 0), Seq(2.0, 0, 5))
  val C: NDArray[Double] = m(Seq(1.0, 2), Seq(3.0, 4), Seq(5.0, 6))
  val Z: NDArray[Complex] = cm(Seq(Complex(1, 2), Complex(2, -1)), Seq(Complex(0, 3), Complex(4, 0.5)))

class LinalgSuite extends LinalgTestUtil:

  test("det, slogdet: real, integer, complex, stacked, singular") {
    assertEqualsDouble(np.linalg.det(B).item, -2.9999999999999996, 1e-13)
    val d12: NDArray[Double] = np.linalg.det(np.array(Seq(Seq(1, 2), Seq(3, 4))))
    assertEqualsDouble(d12.item, -2.0000000000000004, 1e-14)
    assertEquals(np.linalg.det(B).ndim, 0)
    val dz = np.linalg.det(Z).item
    assert((dz - Complex(0, 2.5)).abs < 1e-14)
    val stack = np.array(Seq(Seq(Seq(1.0, 2), Seq(3.0, 4)), Seq(Seq(2.0, 0), Seq(0.0, 3))))
    close(np.linalg.det(stack), Seq(-2.0, 6.0), 1e-13)
    assertEquals(np.linalg.det(m(Seq(1.0, 2), Seq(2.0, 4))).item, 0.0)
    val sl = np.linalg.slogdet(B)
    assertEqualsDouble(sl.sign.item, -1.0, 0.0)
    assertEqualsDouble(sl.logabsdet.item, 1.0986122886681096, 1e-13)
    val sz = np.linalg.slogdet(Z)
    assert((sz.sign.item - Complex(0, 1)).abs < 1e-14)
    assertEqualsDouble(sz.logabsdet.item, 0.9162907318741548, 1e-13)
    val ss = np.linalg.slogdet(m(Seq(1.0, 2), Seq(2.0, 4)))
    assertEquals(ss.sign.item, 0.0)
    assertEquals(ss.logabsdet.item, Double.NegativeInfinity)
    assertEquals(np.linalg.det(np.zeros(0, 0)).item, 1.0)
    intercept[LinAlgError](np.linalg.det(C))
    intercept[LinAlgError](np.linalg.det(np.ones(3)))
  }

  test("inv: values, singular, complex, stacked") {
    close(
      np.linalg.inv(B),
      Seq(-0.6666666666666661, -1.333333333333333, 0.9999999999999997, -0.6666666666666676, 3.6666666666666665,
        -1.9999999999999996, 1.0000000000000004, -2.0, 0.9999999999999999),
      1e-13
    )
    val e = intercept[LinAlgError](np.linalg.inv(m(Seq(1.0, 2), Seq(2.0, 4))))
    assertEquals(e.getMessage, "Singular matrix")
    closeC(
      np.linalg.inv(Z),
      Seq(Complex(0.2, -1.6), Complex(0.4, 0.8), Complex(-1.2, 0), Complex(0.8, -0.4)),
      1e-13
    )
    val r = rand(1, 4, 5, 5)
    val ri = np.linalg.inv(r)
    assertEquals(ri.shape, Seq(4, 5, 5))
    val prod = r @@ ri
    val eye = np.eye(5)
    for b <- 0 until 4 do assert(maxAbsDiff(prod(b, ::, ::), eye) < 1e-11)
    val ii: NDArray[Double] = np.linalg.inv(np.array(Seq(Seq(2, 0), Seq(0, 4))))
    close(ii, Seq(0.5, 0, 0, 0.25))
    intercept[LinAlgError](np.linalg.inv(C))
  }

  test("solve: vector, matrix, broadcast, complex, errors") {
    val b = np.array(1.0, 2.0, 3.0)
    val x = np.linalg.solve(B, b)
    assertEquals(x.shape, Seq(3))
    assert(maxAbsDiff(B @@ x, b) < 1e-13)
    val bm = rand(2, 3, 4)
    val xm = np.linalg.solve(B, bm)
    assertEquals(xm.shape, Seq(3, 4))
    assert(maxAbsDiff(B @@ xm, bm) < 1e-12)
    // stacked a with shared vector b
    val as = rand(3, 2, 3, 3)
    val xs = np.linalg.solve(as, b)
    assertEquals(xs.shape, Seq(2, 3))
    for k <- 0 until 2 do assert(maxAbsDiff(as(k, ::, ::) @@ xs(k, ::), b) < 1e-11)
    // broadcasting batch dims
    val bb = rand(4, 5, 1, 3, 2)
    assertEquals(np.linalg.solve(as, bb).shape, Seq(5, 2, 3, 2))
    // integer + complex
    val xi = np.linalg.solve(np.array(Seq(Seq(2, 1), Seq(1, 3))), np.array(3, 5))
    close(xi, Seq(0.8, 1.4), 1e-14)
    val bz = np.array(Seq(Complex(1, 0), Complex(0, 1)))
    val xz = np.linalg.solve(Z, bz)
    assert(maxAbsDiffC(Z @@ xz, bz) < 1e-13)
    val mixed: NDArray[Complex] = np.linalg.solve(Z, np.array(1.0, 2.0))
    assert(maxAbsDiffC(Z @@ mixed, np.array(Seq(Complex(1, 0), Complex(2, 0)))) < 1e-13)
    intercept[LinAlgError](np.linalg.solve(m(Seq(1.0, 2), Seq(2.0, 4)), np.array(1.0, 2.0)))
    intercept[IllegalArgumentException](np.linalg.solve(B, np.array(1.0, 2.0)))
  }

  test("lstsq") {
    val x = np.array(0.0, 1.0, 2.0, 3.0)
    val am = np.array(Seq(Seq(0.0, 1), Seq(1.0, 1), Seq(2.0, 1), Seq(3.0, 1)))
    val y = np.array(-1.0, 0.2, 0.9, 2.1)
    val r = np.linalg.lstsq(am, y)
    close(r.x, Seq(1.0, -0.95), 1e-13)
    close(r.residuals, Seq(0.05), 1e-12)
    assertEquals(r.rank, 2)
    close(r.s, Seq(4.10003044816824, 1.090756766696107), 1e-13)
    val r2 = np.linalg.lstsq(C, m(Seq(1.0, 2), Seq(3.0, 5), Seq(4.0, 4)))
    close(r2.x, Seq(0.3333333333333333, -1.6666666666666643, 0.41666666666666674, 2.1666666666666647), 1e-12)
    close(r2.residuals, Seq(0.16666666666666652, 2.6666666666666696), 1e-12)
    // rank deficient: residuals empty, minimum norm solution
    val r3 = np.linalg.lstsq(m(Seq(1.0, 1), Seq(1.0, 1)), np.array(2.0, 2.0))
    assertEquals(r3.rank, 1)
    assertEquals(r3.residuals.shape, Seq(0))
    close(r3.x, Seq(1.0, 1.0), 1e-13)
    // underdetermined complex
    val zr = np.linalg.lstsq(cm(Seq(Complex(1, 1), Complex(0, 2), Complex(3, 0))), np.array(Seq(Complex(1, 0))))
    val back = cm(Seq(Complex(1, 1), Complex(0, 2), Complex(3, 0))) @@ zr.x
    assert((back.item - Complex(1, 0)).abs < 1e-13)
  }

  test("pinv") {
    close(
      np.linalg.pinv(C),
      Seq(-1.3333333333333337, -0.3333333333333329, 0.6666666666666666, 1.083333333333334, 0.33333333333333304, -0.4166666666666667),
      1e-12
    )
    closeC(
      np.linalg.pinv(Z),
      Seq(Complex(0.2, -1.6), Complex(0.4, 0.8), Complex(-1.2, 0), Complex(0.8, -0.4)),
      1e-12
    )
    val r = rand(5, 4, 6)
    val p = np.linalg.pinv(r)
    assertEquals(p.shape, Seq(6, 4))
    assert(maxAbsDiff(r @@ p @@ r, r) < 1e-12)
    val sym = m(Seq(2.0, 1), Seq(1.0, 2))
    assert(maxAbsDiff(np.linalg.pinv(sym, hermitian = true), np.linalg.inv(sym)) < 1e-13)
    assertEquals(np.linalg.pinv(np.zeros(0, 3)).shape, Seq(3, 0))
    val st = np.linalg.pinv(rand(9, 2, 3, 2))
    assertEquals(st.shape, Seq(2, 2, 3))
  }

  test("matrix_rank and matrix_power") {
    assertEquals(np.linalg.matrix_rank(np.eye(4)).item, 4)
    val d = np.eye(4)
    d(3, 3) = 0.0
    assertEquals(np.linalg.matrix_rank(d).item, 3)
    assertEquals(np.linalg.matrix_rank(np.ones(4)).item, 1)
    assertEquals(np.linalg.matrix_rank(np.zeros(4)).item, 0)
    assertEquals(np.linalg.matrix_rank(m(Seq(1.0, 2), Seq(2.0, 4.0000001)), tol = 1e-3).item, 1)
    assertEquals(np.linalg.matrix_rank(m(Seq(1.0, 2), Seq(2.0, 4.0000001))).item, 2)
    assertEquals(np.linalg.matrix_rank(m(Seq(1.0, 2), Seq(2.0, 4)), hermitian = true).item, 1)
    assertEquals(np.linalg.matrix_rank(np.ones(2, 3, 3)).toList, List(1, 1))
    val i = np.array(Seq(Seq(0, 1), Seq(-1, 0)))
    assertEquals(np.linalg.matrix_power(i, 3).toList, List(0, -1, 1, 0))
    assertEquals(np.linalg.matrix_power(i, 0).toList, List(1, 0, 0, 1))
    assertEquals(np.linalg.matrix_power(i, 6).toList, List(-1, 0, 0, -1))
    assertEquals(np.linalg.matrix_power(i, 1).dtype.name, "int32")
    val f = m(Seq(2.0, 1), Seq(1.0, 1))
    close(np.linalg.matrix_power(f, -2), np.linalg.inv(f @@ f).toSeq, 1e-12)
    intercept[IllegalArgumentException](np.linalg.matrix_power(i, -1))
    assertEquals(np.linalg.matrix_power(rand(3, 2, 3, 3), 5).shape, Seq(2, 3, 3))
    val z5 = np.linalg.matrix_power(Z, 2)
    assert(maxAbsDiffC(z5, Z @@ Z) < 1e-14)
  }

  test("cholesky") {
    close(
      np.linalg.cholesky(A),
      Seq(2.0, 0.0, 0.0, 0.5, 1.6583123951777, 0.0, 1.0, -0.30151134457776363, 1.9771421064483223),
      1e-14
    )
    val c2 = np.linalg.cholesky(m(Seq(4.0, 12, -16), Seq(12.0, 37, -43), Seq(-16.0, -43, 98)))
    close(c2, Seq(2.0, 0, 0, 6, 1, 0, -8, 5, 3), 1e-14)
    val u = np.linalg.cholesky(A, upper = true)
    assert(maxAbsDiff(u.T @@ u, A) < 1e-14)
    assertEquals(u(1, 0), 0.0)
    val e = intercept[LinAlgError](np.linalg.cholesky(B))
    assertEquals(e.getMessage, "Matrix is not positive definite")
    val h = cm(Seq(Complex(1, 0), Complex(0, -2)), Seq(Complex(0, 2), Complex(5, 0)))
    closeC(np.linalg.cholesky(h), Seq(Complex(1, 0), Complex(0, 0), Complex(0, 2), Complex(1, 0)), 1e-14)
    val hu = np.linalg.cholesky(h, upper = true)
    assert(maxAbsDiffC(adjoint(hu) @@ hu, h) < 1e-14)
    val st = np.linalg.cholesky(np.array(Seq(Seq(Seq(4.0, 2), Seq(2.0, 3)), Seq(Seq(1.0, 0), Seq(0.0, 9)))))
    close(st, Seq(2.0, 0, 1, math.sqrt(2), 1, 0, 0, 3), 1e-14)
    // only the lower triangle is read
    close(np.linalg.cholesky(m(Seq(4.0, 999), Seq(2.0, 3))), Seq(2.0, 0, 1, math.sqrt(2)), 1e-14)
  }

  test("qr: LAPACK-compatible values and modes") {
    val r = np.linalg.qr(B)
    close(r.Q, Seq(-0.12309149097933281, 0.9045340337332914, 0.4082482904638621, -0.492365963917331, 0.30151134457776285,
      -0.8164965809277264, -0.8616404368553292, -0.3015113445777631, 0.4082482904638634), 1e-13)
    close(r.R, Seq(-8.124038404635959, -9.601136296387955, -11.939874624995277, 0.0, 0.9045340337332927, 1.5075567228888205,
      0.0, 0.0, 0.4082482904638626), 1e-13)
    val raw = np.linalg.qr(B, mode = "raw")
    close(raw.Q, Seq(-8.124038404635959, 0.438402363362213, 0.7672041358838727, -9.601136296387955, 0.9045340337332927,
      0.909076332919289, -11.939874624995277, 1.5075567228888205, 0.4082482904638626), 1e-13)
    close(raw.R, Seq(1.1230914909793328, 1.0950385135524678, 0.0), 1e-13)
    val c = np.linalg.qr(C, mode = "complete")
    assertEquals(c.Q.shape, Seq(3, 3))
    assertEquals(c.R.shape, Seq(3, 2))
    close(c.Q, Seq(-0.16903085094570325, 0.8970852271450607, 0.40824829046386274, -0.50709255283711, 0.27602622373694136,
      -0.8164965809277261, -0.8451542547285166, -0.345032779671177, 0.40824829046386313), 1e-13)
    close(c.R, Seq(-5.916079783099616, -7.437357441610946, 0.0, 0.8280786712108248, 0.0, 0.0), 1e-13)
    val red = np.linalg.qr(C)
    assertEquals(red.Q.shape, Seq(3, 2))
    assertEquals(red.R.shape, Seq(2, 2))
    val craw = np.linalg.qr(C, mode = "raw")
    assertEquals(craw.Q.shape, Seq(2, 3))
    close(craw.R, Seq(1.1690308509457032, 1.1131040011646904), 1e-13)
    assertEquals(np.linalg.qr(C, mode = "r").R.shape, Seq(2, 2))
    val zq = np.linalg.qr(Z)
    closeC(zq.Q, Seq(Complex(-0.2672612419124245, -0.5345224838248488), Complex(0.8017837257372734, 0),
      Complex(0, -0.8017837257372732), Complex(-0.5345224838248488, -0.2672612419124245)), 1e-13)
    closeC(zq.R, Seq(Complex(-3.741657386773941, 0), Complex(-0.4008918628686371, 4.543441112511215), Complex(0, 0),
      Complex(-0.6681531047810605, 0)), 1e-13)
    // wide and stacked
    val w = rand(7, 2, 3, 5)
    val wq = np.linalg.qr(w)
    assertEquals(wq.Q.shape, Seq(2, 3, 3))
    assertEquals(wq.R.shape, Seq(2, 3, 5))
    assert(maxAbsDiff(wq.Q @@ wq.R, w) < 1e-13)
    intercept[IllegalArgumentException](np.linalg.qr(B, mode = "bogus"))
  }
