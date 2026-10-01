package com.github.kmizu.numscala

class StatsSuite extends munit.FunSuite:

  def assertClose(actual: Seq[Double], expected: Seq[Double], tol: Double = 1e-12)(using munit.Location): Unit =
    assertEquals(actual.length, expected.length, s"length: $actual vs $expected")
    actual.zip(expected).foreach { (x, y) =>
      if y.isNaN then assert(x.isNaN, s"expected NaN, got $x in $actual")
      else assert(math.abs(x - y) <= tol * math.max(1.0, math.abs(y)), s"$actual != $expected")
    }

  def assertCloseC(actual: Seq[Complex], expected: Seq[Complex], tol: Double = 1e-12)(using munit.Location): Unit =
    assertEquals(actual.length, expected.length)
    actual.zip(expected).foreach((x, y) => assert((x - y).abs <= tol * math.max(1.0, y.abs), s"$actual != $expected"))

  test("diff") {
    val x = np.array(1.0, 2.0, 4.0, 7.0, 0.0, -3.0)
    assertEquals(np.diff(x).toList, List(1.0, 2.0, 3.0, -7.0, -3.0))
    assertEquals(np.diff(x, n = 2).toList, List(1.0, 1.0, -10.0, 4.0))
    val m = np.array(Seq(Seq(1, 3, 6), Seq(2, 8, 9)))
    val d0 = np.diff(m, axis = 0)
    assertEquals(d0.shape, Seq(1, 3))
    assertEquals(d0.toList, List(1, 5, 3))
    assertEquals(np.diff(m).toList, List(2, 3, 6, 1))
    assertEquals(np.diff(np.array(true, false, false, true)).toList, List(true, false, true))
    assertEquals(
      np.diff(x, prepend = NDArray.scalar(0.0), append = np.array(5.0)).toList,
      List(1.0, 1.0, 2.0, 3.0, -7.0, -3.0, 8.0)
    )
    assertEquals(np.diff(np.array(1, 2), n = 3).shape, Seq(0))
    assert(np.diff(x, n = 0) eq x)
    intercept[IllegalArgumentException](np.diff(x, n = -1))
    intercept[IllegalArgumentException](np.diff(NDArray.scalar(1.0)))
    assertEquals(np.diff(m, prepend = np.array(Seq(Seq(0), Seq(0)))).toList, List(1, 2, 3, 2, 6, 1))
  }

  test("ediff1d") {
    val r = np.ediff1d(np.array(1, 2, 4, 7), to_end = np.array(88, 99), to_begin = np.array(-99))
    assertEquals(r.toList, List(-99, 1, 2, 3, 88, 99))
    assertEquals(np.ediff1d(np.array(Seq(Seq(1.0, 2.0), Seq(4.0, 7.0)))).toList, List(1.0, 2.0, 3.0))
    assertEquals(np.ediff1d(np.array(5.0)).shape, Seq(0))
  }

  test("gradient") {
    val f = np.array(Seq(Seq(1.0, 2.0, 6.0), Seq(3.0, 4.0, 5.0)))
    val g = np.gradient(f)
    assertEquals(g.length, 2)
    assertEquals(g(0).toList, List(2.0, 2.0, -1.0, 2.0, 2.0, -1.0))
    assertEquals(g(1).toList, List(1.0, 2.5, 4.0, 1.0, 1.0, 1.0))
    val g2 = np.gradient(f, Seq(2.0, np.array(0.0, 1.0, 3.0)))
    assertClose(g2(0).toSeq, Seq(1.0, 1.0, -0.5, 1.0, 1.0, -0.5))
    assertClose(g2(1).toSeq, Seq(1.0, 1.3333333333333335, 2.0, 1.0, 0.8333333333333333, 0.5))
    assertEquals(np.gradient(np.array(1, 4, 9, 16, 25), edge_order = 2).head.toList, List(2.0, 4.0, 6.0, 8.0, 10.0))
    assertEquals(np.gradient(np.array(1.0, 2.0, 4.0, 7.0, 11.0)).head.toList, List(1.0, 1.5, 2.5, 3.5, 4.0))
    assertClose(
      np.gradient(np.array(1.0, 2.0, 4.0, 7.0, 11.0), Seq(np.array(0.0, 1.0, 1.5, 3.5, 4.0)), edge_order = 2).head.toSeq,
      Seq(-1.0, 3.0, 3.5, 6.7, 9.3)
    )
    assertEquals(np.gradient(f, axis = 1, edge_order = 2).head.toList, List(-0.5, 2.5, 5.5, 1.0, 1.0, 1.0))
    val gc = np.gradient(np.array(Complex(1, 1), Complex(2, 0), Complex(0, 4))).head
    assertCloseC(gc.toSeq, Seq(Complex(1, -1), Complex(-0.5, 1.5), Complex(-2, 4)))
    assertEquals(np.gradient(f, Seq(2.0)).map(_.toList).head, List(1.0, 1.0, -0.5, 1.0, 1.0, -0.5))
    val gf: Seq[NDArray[Float]] = np.gradient(np.array(1f, 2f, 4f))
    assertEquals(gf.head.toList, List(1f, 1.5f, 2f))
    intercept[IllegalArgumentException](np.gradient(np.array(1.0, 2.0), edge_order = 2))
    intercept[IllegalArgumentException](np.gradient(f, Seq(1.0, 2.0, 3.0)))
    intercept[IllegalArgumentException](np.gradient(f, Seq(1.0, np.array(1.0, 2.0))))
    intercept[IllegalArgumentException](np.gradient(f, edge_order = 3))
  }

  test("trapezoid / trapz") {
    assertEquals(np.trapezoid(np.array(1, 2, 3), np.array(0, 1, 3)).item, 6.5)
    assertEquals(np.trapezoid(np.array(1, 2, 3), dx = 0.5).item, 2.0)
    val t0 = np.trapezoid(np.array(1.0, 2.0, 3.0))
    assertEquals(t0.ndim, 0)
    assertEquals(t0.item, 4.0)
    val f = np.array(Seq(Seq(1.0, 2.0, 6.0), Seq(3.0, 4.0, 5.0)))
    assertEquals(np.trapezoid(f).toList, List(5.5, 8.0))
    assertEquals(np.trapezoid(f, axis = 1).toList, List(5.5, 8.0))
    assertEquals(np.trapezoid(f, axis = 0).toList, List(2.0, 3.0, 5.5))
    assertEquals(np.trapezoid(f, x = np.array(Seq(Seq(0.0, 1.0, 2.0), Seq(0.0, 2.0, 4.0)))).toList, List(5.5, 16.0))
    assertEquals(np.trapezoid(f, x = np.array(0.0, 1.0, 3.0)).toList, List(9.5, 12.5))
    assertEquals(np.trapezoid(f, dx = 2.0).toList, List(11.0, 16.0))
    assertEquals(np.trapezoid(np.array(Complex(0, 1), Complex(2, 0))).item, Complex(1, 0.5))
    assertEquals(np.trapz(np.array(1.0, 2.0, 3.0)).item, 4.0)
    assertEquals(np.trapz(np.array(1.0, 2.0, 3.0), dx = 2.0).item, 8.0)
    assertEquals(np.trapz(np.array(1.0, 2.0, 3.0), np.array(0.0, 1.0, 3.0)).item, 6.5)
    assertEquals(np.trapz(f, axis = 0).toList, List(2.0, 3.0, 5.5))
    assertEquals(np.trapezoid(np.array(5.0, 6.0), axis = 0, dx = 0.0).item, 0.0)
    intercept[IndexOutOfBoundsException](np.trapezoid(NDArray.scalar(1.0)))
    intercept[IllegalArgumentException](np.trapezoid(np.array(1.0, 2.0), np.array(1.0, 2.0, 3.0)))
  }

  test("cov") {
    val m = np.array(Seq(Seq(0.0, 1.0, 2.0), Seq(2.0, 1.0, 0.5)))
    assertClose(np.cov(m).toSeq, Seq(1.0, -0.75, -0.75, 0.5833333333333334))
    assertClose(np.cov(m, bias = true).toSeq, Seq(0.6666666666666666, -0.5, -0.5, 0.3888888888888889))
    assertClose(np.cov(m.T, rowvar = false).toSeq, Seq(1.0, -0.75, -0.75, 0.5833333333333334))
    val c1 = np.cov(np.array(1.0, 2.0, 4.0))
    assertEquals(c1.ndim, 0)
    assertEqualsDouble(c1.item, 2.3333333333333335, 1e-12)
    assertClose(
      np.cov(m, fweights = np.array(1, 2, 1)).toSeq,
      Seq(0.6666666666666666, -0.5, -0.5, 0.3958333333333333)
    )
    assertClose(
      np.cov(m, aweights = np.array(0.5, 1.0, 2.0)).toSeq,
      Seq(0.9285714285714285, -0.6428571428571429, -0.6428571428571429, 0.46428571428571425)
    )
    assertClose(
      np.cov(m, fweights = np.array(1, 2, 1), aweights = np.array(0.5, 1.0, 2.0), ddof = 0).toSeq,
      Seq(0.4444444444444444, -0.2962962962962963, -0.2962962962962963, 0.20987654320987653)
    )
    assertClose(np.cov(np.array(1.0, 2.0, 3.0), np.array(1.0, 3.0, 2.0)).toSeq, Seq(1.0, 0.5, 0.5, 1.0))
    val ci: NDArray[Double] = np.cov(np.array(1, 2, 4))
    assertEqualsDouble(ci.item, 2.3333333333333335, 1e-12)
    val cc = np.cov(np.array(Complex(1, 1), Complex(2, 0), Complex(3, -1)), np.array(Complex(0, 0), Complex(0, 1), Complex(1, 0)))
    assertCloseC(cc.toSeq, Seq(Complex(2, 0), Complex(0.5, -0.5), Complex(0.5, 0.5), Complex(0.6666666666666667, 0)))
    intercept[IllegalArgumentException](np.cov(m, fweights = np.array(1.5, 2.0, 1.0)))
    intercept[IllegalArgumentException](np.cov(m, aweights = np.array(-1.0, 2.0, 1.0)))
    intercept[IllegalStateException](np.cov(m, fweights = np.array(1, 2)))
    intercept[IllegalArgumentException](np.cov(np.zeros(2, 2, 2)))
    assertEquals(np.cov(np.zeros(0, 3)).shape, Seq(0, 0))
  }

  test("corrcoef") {
    val m = np.array(Seq(Seq(0.0, 1.0, 2.0), Seq(2.0, 1.0, 0.5)))
    assertClose(np.corrcoef(m).toSeq, Seq(1.0, -0.9819805060619656, -0.9819805060619656, 1.0))
    assertClose(np.corrcoef(np.array(1.0, 2.0, 3.0), np.array(3.0, 1.0, 2.0)).toSeq, Seq(1.0, -0.5, -0.5, 1.0))
    assertEquals(np.corrcoef(np.array(1.0, 2.0, 3.0)).item, 1.0)
    val r = np.corrcoef(np.array(Seq(Seq(1.0, 2.0, 3.0), Seq(2.0, 4.0, 6.0))))
    assert(r.toSeq.forall(v => v <= 1.0 && v >= -1.0))
  }

  test("correlate / convolve") {
    val a = np.array(1.0, 2.0, 3.0, 4.0, 5.0)
    val v = np.array(1.0, 0.0, 2.0)
    assertEquals(np.correlate(a, v, "full").toList, List(2.0, 4.0, 7.0, 10.0, 13.0, 4.0, 5.0))
    assertEquals(np.correlate(a, v, "same").toList, List(4.0, 7.0, 10.0, 13.0, 4.0))
    assertEquals(np.correlate(a, v).toList, List(7.0, 10.0, 13.0))
    assertEquals(np.correlate(v, a, "full").toList, List(5.0, 4.0, 13.0, 10.0, 7.0, 4.0, 2.0))
    assertEquals(np.correlate(v, a, "same").toList, List(4.0, 13.0, 10.0, 7.0, 4.0))
    assertEquals(np.correlate(v, a, "valid").toList, List(13.0, 10.0, 7.0))
    assertEquals(np.convolve(a, v).toList, List(1.0, 2.0, 5.0, 8.0, 11.0, 8.0, 10.0))
    assertEquals(np.convolve(v, a, "same").toList, List(2.0, 5.0, 8.0, 11.0, 8.0))
    assertEquals(np.convolve(a, v, "valid").toList, List(5.0, 8.0, 11.0))
    assertEquals(np.correlate(np.array(1, 2, 3), np.array(1, 1), "same").toList, List(1, 3, 5))
    assertEquals(np.correlate(np.array(1, 1), np.array(1, 2, 3), "same").toList, List(5, 3, 1))
    assertEquals(np.correlate(np.array(1, 2, 3, 4), np.array(1, 2), "same").toList, List(2, 5, 8, 11))
    assertEquals(np.correlate(np.array(1, 2), np.array(1, 2, 3, 4), "same").toList, List(11, 8, 5, 2))
    assertEquals(np.convolve(np.array(1.0, 2.0, 3.0), np.array(0.0, 1.0, 0.5), "same").toList, List(1.0, 2.5, 4.0))
    val c = np.correlate(np.array(Complex(1, 1), Complex(2, 0)), np.array(Complex(0, 1), Complex(2, 0), Complex(3, 0)), "full")
    assertEquals(c.toList, List(Complex(3, 3), Complex(8, 2), Complex(5, -1), Complex(0, -2)))
    val c2 = np.correlate(np.array(Complex(0, 1), Complex(2, 0), Complex(3, 0)), np.array(Complex(1, 1), Complex(2, 0)), "full")
    assertEquals(c2.toList, List(Complex(0, 2), Complex(5, 1), Complex(8, -2), Complex(3, -3)))
    val mixed: NDArray[Double] = np.convolve(np.array(1, 2), np.array(0.5, 0.5))
    assertEquals(mixed.toList, List(0.5, 1.5, 1.0))
    intercept[IllegalArgumentException](np.correlate(np.zeros(0), v))
    intercept[IllegalArgumentException](np.convolve(a, np.zeros(0)))
    intercept[IllegalArgumentException](np.convolve(a, v, "bogus"))
    intercept[IllegalArgumentException](np.convolve(np.zeros(2, 2), v))
  }

  test("interp") {
    val r = np.interp(np.array(0.0, 1.0, 1.5, 2.72, 3.14, 5.0), np.array(1.0, 2.0, 3.0), np.array(3.0, 2.0, 0.0))
    assertClose(r.toSeq, Seq(3.0, 3.0, 2.5, 0.56, 0.0, 0.0))
    assertEquals(np.interp(2.5, np.array(1.0, 2.0, 3.0), np.array(3.0, 2.0, 0.0)), 1.0)
    assertEquals(
      np.interp(np.array(0.0, 4.0), np.array(1, 2, 3), np.array(3.0, 2.0, 0.0), left = -1.0, right = 99.0).toList,
      List(-1.0, 99.0)
    )
    val p = np.interp(
      np.array(-180.0, -170.0, -185.0, 185.0, -10.0, -5.0, 0.0, 365.0),
      np.array(190.0, -190.0, 350.0, -350.0),
      np.array(5.0, 10.0, 3.0, 4.0),
      period = 360.0
    )
    assertClose(p.toSeq, Seq(7.5, 5.0, 8.75, 6.25, 3.0, 3.25, 3.5, 3.75))
    val c = np.interp(np.array(1.5, 2.5), np.array(1.0, 2.0, 3.0), np.array(Complex(0, 1), Complex(2, 0), Complex(3, 3)))
    assertEquals(c.toList, List(Complex(1, 0.5), Complex(2.5, 1.5)))
    val inf = Double.PositiveInfinity
    val ri = np.interp(np.array(0.0, 1.0, 1.5, 2.5, 3.0, 4.0), np.array(1.0, 2.0, 3.0), np.array(4.0, inf, 6.0))
    assertEquals(ri.toList, List(4.0, 4.0, inf, inf, 6.0, 6.0))
    assert(np.interp(Double.NaN, np.array(1.0, 2.0, 3.0), np.array(4.0, 5.0, 6.0)).isNaN)
    assertEquals(np.interp(Double.NaN, np.full(Seq(1), 1.0), np.full(Seq(1), 5.0)), 5.0)
    assertEquals(np.interp(np.array(Seq(Seq(1.5))), np.array(1.0, 2.0), np.array(0, 10)).shape, Seq(1, 1))
    intercept[IllegalArgumentException](np.interp(1.0, np.array(1.0, 2.0), np.array(1.0)))
    intercept[IllegalArgumentException](np.interp(1.0, np.zeros(0), np.zeros(0)))
    intercept[IllegalArgumentException](
      np.interp(np.array(1.0), np.array(1.0, 2.0), np.array(1.0, 2.0), period = 0.0)
    )
  }

  val hx = np.array(2.0, 3.0, 5.0, 7.0, 1.0, 1.0, 8.0, 4.0, 9.0, 10.0, 3.0, 3.0, 3.0, 6.0)

  test("histogram basics") {
    val (h, e) = np.histogram(hx)
    val hh: NDArray[Long] = h
    assertEquals(hh.toList, List(2L, 1L, 4L, 1L, 1L, 1L, 1L, 1L, 1L, 1L))
    assertClose(e.toSeq, Seq(1.0, 1.9, 2.8, 3.7, 4.6, 5.5, 6.4, 7.3, 8.2, 9.1, 10.0))
    val (h2, e2) = np.histogram(np.array(1, 2, 1), np.array(0, 1, 2, 3))
    assertEquals(h2.toList, List(0L, 2L, 1L))
    assertEquals(e2.toList, List(0.0, 1.0, 2.0, 3.0))
    val (h3, e3) = np.histogram(hx, 3, (2.0, 8.0))
    assertEquals(h3.toList, List(5L, 2L, 3L))
    assertEquals(e3.toList, List(2.0, 4.0, 6.0, 8.0))
    val (h4, _) = np.histogram(hx, bins = Seq(1.0, 2.0, 5.0, 10.0), density = true)
    assertClose(h4.toSeq, Seq(0.14285714285714285, 0.14285714285714285, 0.08571428571428572))
    val (h5, e5) = np.histogram(hx, bins = 4, weights = np.arange(14.0))
    assertEquals(h5.toList, List(43.0, 9.0, 16.0, 23.0))
    assertEquals(e5.toList, List(1.0, 3.25, 5.5, 7.75, 10.0))
    val (h6, _) = np.histogram(hx, bins = Seq(1.0, 2.0, 5.0, 10.0), weights = np.arange(14.0) * 0.1)
    assertClose(h6.toSeq, Seq(0.9, 4.1, 4.1))
    val (h7, e7) = np.histogram(np.zeros(0), 3)
    assertEquals(h7.toList, List(0L, 0L, 0L))
    assertClose(e7.toSeq, Seq(0.0, 1.0 / 3, 2.0 / 3, 1.0))
    val (h8, e8) = np.histogram(np.array(5.0, 5.0), 2)
    assertEquals(h8.toList, List(0L, 2L))
    assertEquals(e8.toList, List(4.5, 5.0, 5.5))
    val (h9, _) = np.histogram(np.array(1.0, Double.NaN, 3.0), 2, (0.0, 4.0))
    assertEquals(h9.toList, List(1L, 1L))
    intercept[IllegalArgumentException](np.histogram(np.array(1.0, Double.NaN)))
    intercept[IllegalArgumentException](np.histogram(hx, 0))
    intercept[IllegalArgumentException](np.histogram(hx, Seq(3.0, 1.0)))
    intercept[IllegalArgumentException](np.histogram(hx, 2, (3.0, 1.0)))
    intercept[IllegalArgumentException](np.histogram(hx, "bogus"))
    intercept[IllegalArgumentException](np.histogram(hx, bins = "auto", weights = np.ones(14)))
  }

  test("histogram bin estimators") {
    val cases = Seq(
      "auto" -> (Seq(3L, 5L, 2L, 2L, 2L), Seq(1.0, 2.8, 4.6, 6.4, 8.2, 10.0)),
      "fd" -> (Seq(7L, 3L, 4L), Seq(1.0, 4.0, 7.0, 10.0)),
      "doane" -> (Seq(3L, 4L, 2L, 1L, 2L, 2L), Seq(1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0)),
      "scott" -> (Seq(7L, 3L, 4L), Seq(1.0, 4.0, 7.0, 10.0)),
      "rice" -> (Seq(3L, 5L, 2L, 2L, 2L), Seq(1.0, 2.8, 4.6, 6.4, 8.2, 10.0)),
      "sturges" -> (Seq(3L, 5L, 2L, 2L, 2L), Seq(1.0, 2.8, 4.6, 6.4, 8.2, 10.0)),
      "sqrt" -> (Seq(7L, 2L, 2L, 3L), Seq(1.0, 3.25, 5.5, 7.75, 10.0))
    )
    for (name, (cnt, edges)) <- cases do
      val (h, e) = np.histogram(hx, name)
      assertEquals(h.toSeq, cnt, name)
      assertClose(e.toSeq, edges)
    val (hs, es) = np.histogram(hx, "stone")
    assertEquals(hs.size, 100)
    assertEquals(hs.sum(), 14L)
    assertEquals(hs(0), 2L)
    assertEquals(hs(22), 4L)
    assertEqualsDouble(es(1), 1.09, 1e-12)
    val (ha, ea) = np.histogram(np.arange(5), "auto")
    assertEquals(ha.toList, List(1L, 1L, 1L, 2L))
    assertEquals(ea.toList, List(0.0, 1.0, 2.0, 3.0, 4.0))
    assertClose(np.histogram_bin_edges(hx, "sturges").toSeq, Seq(1.0, 2.8, 4.6, 6.4, 8.2, 10.0))
    assertEquals(np.histogram_bin_edges(hx, 2, (0.0, 1.0)).toList, List(0.0, 0.5, 1.0))
  }

  test("histogram2d / histogramdd") {
    val x = np.array(0.1, 0.5, 0.9, 0.5)
    val y = np.array(0.2, 0.2, 0.8, 0.9)
    val (h, xe, ye) = np.histogram2d(x, y, bins = 2)
    assertEquals(h.shape, Seq(2, 2))
    assertEquals(h.toList, List(1.0, 0.0, 1.0, 2.0))
    assertClose(xe.toSeq, Seq(0.1, 0.5, 0.9))
    assertClose(ye.toSeq, Seq(0.2, 0.55, 0.9))
    val (hd, _, ye2) = np.histogram2d(x, y, bins = (2, 3), range = ((0.0, 1.0), (0.0, 1.0)), density = true)
    assertClose(hd.toSeq, Seq(1.5, 0.0, 0.0, 1.5, 0.0, 3.0))
    assertEquals(ye2.size, 4)
    val (hb, _, _) = np.histogram2d(x, y, bins = np.array(0.0, 0.5, 1.0))
    assertEquals(hb.toList, List(1.0, 0.0, 1.0, 2.0))
    val (hw, ed) = np.histogramdd(
      np.array(Seq(Seq(0, 0), Seq(1, 1), Seq(2, 2), Seq(2, 0))),
      bins = Seq(2, np.array(0, 1, 3)),
      weights = np.array(1.0, 2.0, 3.0, 4.0)
    )
    assertEquals(hw.toList, List(1.0, 0.0, 4.0, 5.0))
    assertEquals(ed(0).toList, List(0.0, 1.0, 2.0))
    assertEquals(ed(1).toList, List(0.0, 1.0, 3.0))
    val (h1, e1) = np.histogramdd(np.array(1.0, 2.0, 2.0, 3.0), bins = 2)
    assertEquals(h1.toList, List(1.0, 3.0))
    assertEquals(e1.head.toList, List(1.0, 2.0, 3.0))
    intercept[IllegalArgumentException](np.histogramdd(np.zeros(3, 2), bins = Seq(2)))
    intercept[IllegalArgumentException](np.histogramdd(np.zeros(3, 2), bins = Seq(0, 2)))
  }

  test("bincount") {
    assertEquals(np.bincount(np.array(0, 1, 1, 3, 2, 1, 7)).toList, List(1L, 3L, 1L, 1L, 0L, 0L, 0L, 1L))
    assertEquals(
      np.bincount(np.array(0, 1, 1, 2), np.array(0.3, 0.5, 0.2, 0.7), 5).toList,
      List(0.3, 0.7, 0.7, 0.0, 0.0)
    )
    assertEquals(np.bincount(np.array(0, 2), np.array(1.0, 2.0)).toList, List(1.0, 0.0, 2.0))
    assertEquals(np.bincount(np.zeros[Int](0), minlength = 3).toList, List(0L, 0L, 0L))
    assertEquals(np.bincount(np.array(true, false, true)).toList, List(1L, 2L))
    intercept[IllegalArgumentException](np.bincount(np.array(-1, 2)))
    intercept[IllegalArgumentException](np.bincount(np.array(1.0, 2.0)))
    intercept[IllegalArgumentException](np.bincount(np.zeros[Int](2, 2)))
    intercept[IllegalArgumentException](np.bincount(np.array(1, 2), minlength = -1))
    intercept[IllegalArgumentException](np.bincount(np.array(1, 2), np.array(1.0)))
  }

  test("digitize") {
    val x = np.array(0.2, 6.4, 3.0, 1.6, -1.0, 10.0, 4.0)
    val inc = np.array(0.0, 1.0, 2.5, 4.0, 10.0)
    assertEquals(np.digitize(x, inc).toList, List(1, 4, 3, 2, 0, 5, 4))
    assertEquals(np.digitize(x, inc, right = true).toList, List(1, 4, 3, 2, 0, 4, 3))
    val dec = np.array(10.0, 4.0, 2.5, 1.0, 0.0)
    val x2 = np.array(0.2, 6.4, 3.0, 1.6, 4.0, 10.0)
    assertEquals(np.digitize(x2, dec).toList, List(4, 1, 2, 3, 1, 0))
    assertEquals(np.digitize(x2, dec, right = true).toList, List(4, 1, 2, 3, 2, 1))
    assertEquals(np.digitize(3.0, Seq(0.0, 1.0, 2.5, 4.0)), 3)
    assertEquals(np.digitize(4.0, Seq(0.0, 1.0, 2.5, 4.0), true), 3)
    assertEquals(np.digitize(np.array(1, 5), np.array(0, 3)).toList, List(1, 2))
    intercept[IllegalArgumentException](np.digitize(x, np.array(1.0, 3.0, 2.0)))
    intercept[IllegalArgumentException](np.digitize(np.array(Complex(1, 1)), inc))
  }

  test("histogram literal ranges and density with bins count") {
    val (h, e) = np.histogram(np.array(1, 2, 2, 3), 2, (0.0, 4.0))
    assertEquals(h.toList, List(1L, 3L))
    assertEquals(e.toList, List(0.0, 2.0, 4.0))
    val (d, _) = np.histogram(np.array(1, 2, 2, 3), bins = 2, range = (0.0, 4.0), density = true)
    assertEquals(d.toList, List(0.125, 0.375))
    val (w, _) = np.histogram(np.array(1, 2, 2, 3), bins = 2, weights = np.array(1, 1, 1, 2))
    assertEquals(w.toList, List(1.0, 4.0))
    assertEquals(np.histogram_bin_edges(np.array(1.0, 2.0, 3.0), Seq(0.0, 5.0)).toList, List(0.0, 5.0))
    val (h2, xe, ye) =
      np.histogram2d(np.array(1.0, 2.0), np.array(1.0, 2.0), bins = (np.array(0.0, 1.5, 3.0), np.array(0.0, 3.0)))
    assertEquals(h2.toList, List(1.0, 1.0))
    assertEquals(xe.toList, List(0.0, 1.5, 3.0))
    assertEquals(ye.toList, List(0.0, 3.0))
    val (hd, _) = np.histogramdd(
      np.array(Seq(Seq(0.5, 0.5), Seq(1.5, 0.5))),
      bins = 2,
      range = Seq((0.0, 2.0), (0.0, 1.0)),
      density = true
    )
    assertEquals(hd.toList, List(0.0, 1.0, 0.0, 1.0))
  }
