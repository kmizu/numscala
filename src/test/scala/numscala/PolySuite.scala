package numscala

class PolySuite extends munit.FunSuite:

  private def assertD(a: NDArray[Double], e: Seq[Double], tol: Double = 1e-12)(using munit.Location): Unit =
    val av = a.toSeq
    assertEquals(av.length, e.length, s"got $av")
    av.zip(e).zipWithIndex.foreach { case ((x, y), i) =>
      assert(math.abs(x - y) <= tol * math.max(1.0, math.abs(y)), s"index $i: $x != $y (got $av)")
    }

  private def sortedC(a: NDArray[Complex]): Seq[Complex] = a.toSeq.sorted(using Complex.ordering)

  private def assertRoots(a: NDArray[Complex], e: Seq[Complex], tol: Double = 1e-10)(using munit.Location): Unit =
    val av = sortedC(a)
    val ev = e.sorted(using Complex.ordering)
    assertEquals(av.length, ev.length)
    av.zip(ev).foreach((x, y) => assert((x - y).abs <= tol * math.max(1.0, y.abs), s"$x != $y (got $av)"))

  test("poly from roots and from a matrix") {
    val p = np.poly(np.array(1.0, 2.0, 3.0))
    assertD(p, Seq(1.0, -6.0, 11.0, -6.0))
    val pi: NDArray[Double] = np.poly(np.array(1, 2, 3))
    assertD(pi, Seq(1.0, -6.0, 11.0, -6.0))
    assertD(np.poly(np.array(Seq(Seq(1.0, 2.0), Seq(3.0, 4.0)))), Seq(1.0, -5.0, -2.0), 1e-12)
    val pc = np.poly(np.array(Complex(0, 1), Complex(0, -1)))
    assertEquals(pc.toSeq.map(_.re), Seq(1.0, 0.0, 1.0))
    assertD(np.poly(NDArray.fromArray(Array.emptyDoubleArray)), Seq(1.0))
    intercept[IllegalArgumentException](np.poly(np.zeros(2, 3)))
  }

  test("roots") {
    assertRoots(np.roots(np.array(1, -6, 11, -6)), Seq(Complex(1), Complex(2), Complex(3)))
    assertRoots(np.roots(np.array(1.0, 0.0, 1.0)), Seq(Complex(0, 1), Complex(0, -1)))
    // conjugate pairs come out exactly conjugate for real input
    val r = np.roots(np.array(1.0, 0.0, 1.0)).toSeq
    assertEquals(r(0).re, r(1).re)
    assertEquals(r(0).im, -r(1).im)
    assertRoots(np.roots(np.array(0.0, 1.0, -3.0, 0.0, 0.0)), Seq(Complex(3), Complex(0), Complex(0)))
    assertRoots(np.roots(np.array(Complex(2), Complex(0, 1), Complex(3))), Seq(Complex(0, -1.5), Complex(0, 1)))
    assertEquals(np.roots(np.array(0.0, 0.0)).size, 0)
    assertEquals(np.roots(np.array(5.0)).size, 0)
    // a degree-20 Wilkinson-like polynomial with well-separated roots
    val rts = (1 to 10).map(_.toDouble)
    val coeffs = np.poly(NDArray.fromArray(rts.toArray))
    assertRoots(np.roots(coeffs), rts.map(Complex(_)), 1e-8)
    // random complex polynomial: p(root) ~ 0
    val cp = np.array(Complex(1, 2), Complex(-3, 0.5), Complex(0.25, -1), Complex(4, 4), Complex(-2, 0))
    np.roots(cp).toSeq.foreach { z =>
      val v = np.polyval(cp, z)
      assert(v.abs < 1e-12 * 10, s"p($z) = $v")
    }
  }

  test("polyval") {
    assertEquals(np.polyval(np.array(3, 0, 1), 5), 76)
    assertEquals(np.polyval(np.array(1.0, 2.0, 3.0), 0.5), 4.25)
    assertD(np.polyval(np.array(1, 2, 3), np.array(0.5, 1.5)), Seq(4.25, 8.25))
    val cv = np.polyval(np.array(1.0, 0.0, 1.0), Complex(0, 1))
    assertEquals(cv, Complex(0, 0))
    val m = np.polyval(np.array(1.0, -1.0), np.arange(6.0).reshape(2, 3))
    assertEquals(m.shape, Seq(2, 3))
    assertEquals(m.toList, List(-1.0, 0.0, 1.0, 2.0, 3.0, 4.0))
  }

  test("polyadd / polysub / polymul keep integer dtypes") {
    val a = np.polyadd(np.array(1, 2), np.array(9, 5, 4))
    assertEquals(a.toList, List(9, 6, 6))
    assertEquals(a.dtype.name, "int32")
    assertEquals(np.polysub(np.array(1, 2), np.array(9, 5, 4)).toList, List(-9, -4, -2))
    assertEquals(np.polymul(np.array(1, 2, 3), np.array(9, 5, 1)).toList, List(9, 23, 38, 17, 3))
    assertEquals(np.polyadd(np.array(1.0, 2.0), np.array(1, 1)).toList, List(2.0, 3.0))
  }

  test("polydiv") {
    val (q, r) = np.polydiv(np.array(3.0, 5.0, 2.0), np.array(2.0, 1.0))
    assertD(q, Seq(1.5, 1.75))
    assertD(r, Seq(0.25))
    val (q2, r2) = np.polydiv(np.array(1, 0, 0, 0, 1), np.array(1, 1))
    assertD(q2, Seq(1.0, -1.0, 1.0, -1.0))
    assertD(r2, Seq(2.0))
    val (q3, r3) = np.polydiv(np.array(1.0, 2.0), np.array(1.0, 2.0, 3.0))
    assertD(q3, Seq(0.0))
    assertD(r3, Seq(1.0, 2.0))
  }

  test("polyder / polyint") {
    assertEquals(np.polyder(np.array(1, 1, 1, 1)).toList, List(3, 2, 1))
    assertEquals(np.polyder(np.array(1, 1, 1, 1), 2).toList, List(6, 2))
    assertEquals(np.polyder(np.array(1, 1, 1, 1), 0).toList, List(1, 1, 1, 1))
    assertEquals(np.polyder(np.array(1.0, 1.0), 3).size, 0)
    assertD(np.polyint(np.array(1, 1, 1)), Seq(1.0 / 3, 0.5, 1.0, 0.0))
    assertD(np.polyint(np.array(1, 1, 1), 2, Seq(3.0, 4.0)), Seq(1.0 / 12, 1.0 / 6, 0.5, 3.0, 4.0))
    assertD(np.polyint(np.array(1.0), 3, Seq(2.0)), Seq(1.0 / 6, 1.0, 2.0, 2.0))
    intercept[IllegalArgumentException](np.polyint(np.array(1.0), 3, Seq(1.0, 2.0)))
    intercept[IllegalArgumentException](np.polyder(np.array(1.0), -1))
  }

  private val fx = np.array(0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
  private val fy = np.array(0.0, 0.8, 0.9, 0.1, -0.8, -1.0)

  test("polyfit basic, weighted and 2-D") {
    assertD(
      np.polyfit(fx, fy, 3),
      Seq(0.08703703703703684, -0.8134920634920618, 1.693121693121688, -0.03968253968253495),
      1e-10
    )
    assertD(np.polyfit(fx, fy, 2, w = np.array(1.0, 2.0, 1.0, 2.0, 1.0, 2.0)), Seq(-0.11602523659305994, 0.2646687697160883, 0.47652996845425816), 1e-10)
    val y2 = np.array(Seq(0.0, 1.0), Seq(0.8, 2.6), Seq(0.9, 2.8), Seq(0.1, 1.2), Seq(-0.8, -0.6), Seq(-1.0, -1.0))
    val c2 = np.polyfit(fx, y2, 1)
    assertEquals(c2.shape, Seq(2, 2))
    assertD(c2, Seq(-0.3028571428571428, -0.6057142857142855, 0.7571428571428571, 2.5142857142857133), 1e-10)
    // exact fit
    val xs = np.array(-1.0, 0.0, 1.0, 2.0)
    val ys = np.polyval(np.array(2.0, -1.0, 0.5), xs)
    assertD(np.polyfit(xs, ys, 2), Seq(2.0, -1.0, 0.5), 1e-12)
    intercept[IllegalArgumentException](np.polyfit(fx, fy, -1))
    intercept[IllegalArgumentException](np.polyfit(fx, np.array(1.0, 2.0), 1))
  }

  test("polyfit full and cov") {
    val r = np.polyfit(fx, fy, 3, full = true)
    val res: PolyfitResult = r
    assertD(res.coef, Seq(0.08703703703703684, -0.8134920634920618, 1.693121693121688, -0.03968253968253495), 1e-10)
    assertD(res.residuals, Seq(0.03968253968254001), 1e-9)
    assertEquals(res.rank, 4)
    assertD(res.singularValues, Seq(1.8828796636036094, 0.6471092002746208, 0.18783542690800328, 0.02705009623248137), 1e-10)
    assertEqualsDouble(res.rcond, 1.3322676295501878e-15, 1e-30)
    val (c, cv) = np.polyfit(fx, fy, 2, cov = true)
    assertEquals(c.size, 3)
    assertD(
      cv,
      Seq(0.00473724489795921, -0.02368622448979603, 0.0157908163265307, -0.02368622448979603, 0.1285372448979598,
        -0.10421938775510255, 0.01579081632653071, -0.10421938775510259, 0.14527551020408216),
      1e-9
    )
    val (_, cu) = np.polyfit(fx, fy, 2, cov = "unscaled")
    assertD(
      cu,
      Seq(0.02678571428571438, -0.13392857142857192, 0.0892857142857147, -0.13392857142857192, 0.726785714285717,
        -0.5892857142857165, 0.08928571428571473, -0.5892857142857167, 0.8214285714285735),
      1e-9
    )
  }

  test("poly1d arithmetic and evaluation") {
    val p = np.poly1d(np.array(1, 2, 3))
    assertEquals(p(2), 11)
    val pd = np.poly1d(np.array(1.0, 2.0, 3.0))
    assertEquals(pd(0.5), 4.25)
    assertEquals(pd(np.array(0.5, 1.5)).toList, List(4.25, 8.25))
    assertEquals(p.order, 2)
    assertEquals((p * p).coeffs.toList, List(1, 4, 10, 12, 9))
    assertEquals((p + np.poly1d(np.array(1, 1))).coeffs.toList, List(1, 3, 4))
    assertEquals((p - p).coeffs.toList, List(0))
    assertEquals((-p).coeffs.toList, List(-1, -2, -3))
    assertEquals((p ** 3).coeffs.toList, List(1, 6, 21, 44, 63, 54, 27))
    assertEquals(p.deriv().coeffs.toList, List(2, 2))
    assertD(p.integ(k = Seq(1.0)).coeffs, Seq(1.0 / 3, 1.0, 3.0, 1.0))
    val (q, r) = p / np.poly1d(np.array(1, 1))
    assertD(q.coeffs, Seq(1.0, 1.0))
    assertD(r.coeffs, Seq(2.0))
    assertEquals(p.get(0), 3)
    assertEquals(p.get(2), 1)
    assertEquals(p.get(5), 0)
    assertD(np.poly1d(np.array(1.0, 2.0), r = true).coeffs, Seq(1.0, -3.0, 2.0))
    assertRoots(pd.roots, Seq(Complex(-1, math.sqrt(2)), Complex(-1, -math.sqrt(2))))
    assertEquals(np.poly1d(np.array(0, 0, 1, 2)).coeffs.toList, List(1, 2))
    assertEquals(np.poly1d(np.array(1.0, 1.0)), np.poly1d(np.array(0.0, 1.0, 1.0)))
    // composition p(q)
    val comp = pd(np.poly1d(np.array(1.0, 1.0)))
    assertD(comp.coeffs, Seq(1.0, 4.0, 6.0))
    assertD(np.polyval(pd, np.poly1d(np.array(2.0, 0.0))).coeffs, Seq(4.0, 4.0, 3.0))
    assertD((pd / 2.0).coeffs, Seq(0.5, 1.0, 1.5))
    assertEquals(np.polyadd(p, p).coeffs.toList, List(2, 4, 6))
    assertEquals(np.polymul(p, p).coeffs.toList, List(1, 4, 10, 12, 9))
    assertEquals(np.polysub(p, p).coeffs.toList, List(0))
    assertD(np.polydiv(pd, pd)._1.coeffs, Seq(1.0))
  }

  test("poly1d printing") {
    val p = np.poly1d(np.array(1, 2, 3))
    assertEquals(p.toString, "   2\n1 x + 2 x + 3")
    assertEquals(p.repr, "poly1d([1, 2, 3], dtype=int32)")
    assertEquals(np.poly1d(np.array(1.0, 2.0, 3.0)).repr, "poly1d([1., 2., 3.])")
    assertEquals((p * p).toString, "   4     3      2\n1 x + 4 x + 10 x + 12 x + 9")
    assertEquals(np.poly1d(np.array(1.5, -2.0, 0.0, 0.25)).toString, "     3     2\n1.5 x - 2 x + 0.25")
    assertEquals(np.poly1d(np.array(0.0)).toString, " \n0")
    assertEquals(np.poly1d(np.array(3.0)).toString, " \n3")
    assertEquals(np.poly1d(np.array(1.0, 0.0)).toString, " \n1 x")
    assertEquals(np.poly1d(np.array(Complex(1), Complex(0, 2))).toString, " \n1 x + 2j")
    assertEquals(np.poly1d(np.array(1e-5, 123456.7, 1e10)).toString, "       2\n1e-05 x + 1.235e+05 x + 1e+10")
    val long = np.poly1d(np.arange(1.0, 30.0)).toString
    val expected =
      "   28     27     26     25     24     23     22     21     20      19\n" +
        "1 x  + 2 x  + 3 x  + 4 x  + 5 x  + 6 x  + 7 x  + 8 x  + 9 x  + 10 x \n" +
        "       18      17      16      15      14      13      12      11\n" +
        " + 11 x  + 12 x  + 13 x  + 14 x  + 15 x  + 16 x  + 17 x  + 18 x \n" +
        "       10      9      8      7      6      5      4      3      2\n" +
        " + 19 x  + 20 x + 21 x + 22 x + 23 x + 24 x + 25 x + 26 x + 27 x + 28 x + 29"
    assertEquals(long, expected)
  }

  test("python %g formatting helper") {
    assertEquals(PolyLegacy.pyG(1.0), "1")
    assertEquals(PolyLegacy.pyG(0.25), "0.25")
    assertEquals(PolyLegacy.pyG(123456.7), "1.235e+05")
    assertEquals(PolyLegacy.pyG(0.0001234), "0.0001234")
    assertEquals(PolyLegacy.pyG(0.00001234), "1.234e-05")
    assertEquals(PolyLegacy.pyG(-2.5), "-2.5")
    assertEquals(PolyLegacy.pyG(1234.0), "1234")
    assertEquals(PolyLegacy.pyG(12345.0), "1.234e+04")
  }

  test("eigenvalue paths: real Hessenberg, complex QR, rank-deficient least squares") {
    assertD(np.poly(np.array(Seq(Seq(2.0, 1.0, 0.0), Seq(1.0, 3.0, 1.0), Seq(0.0, 1.0, 4.0)))), Seq(1.0, -9.0, 24.0, -18.0))
    val pc = np.poly(np.array(Seq(Seq(Complex(1), Complex(0, 2)), Seq(Complex(3, -1), Complex(4)))))
    val e = Seq(Complex(1), Complex(-5, 0), Complex(2, -6))
    pc.toSeq.zip(e).foreach((x, y) => assert((x - y).abs < 1e-12, s"$x != $y"))
    // rotation blocks -> purely imaginary eigenvalues
    val rot = np.array(Seq(Seq(0.0, -1.0, 0.0, 0.0), Seq(1.0, 0.0, 0.0, 0.0), Seq(0.0, 0.0, 0.0, -2.0), Seq(0.0, 0.0, 2.0, 0.0)))
    assertD(np.poly(rot), Seq(1.0, 0.0, 5.0, 0.0, 4.0))
    // the order of np.roots follows LAPACK's for this classic example
    assertEquals(np.roots(np.array(1.0, -6.0, 11.0, -6.0)).toSeq.map(z => math.rint(z.re)), Seq(3.0, 2.0, 1.0))
    // under-determined fit: minimum-norm solution
    assertD(
      np.polyfit(np.array(0.0, 1.0, 2.0), np.array(1.0, 2.0, 0.0), 4),
      Seq(-0.14027700960497305, -0.18069112182658073, 0.02401243271455386, 1.296955698716999, 1.0000000000000007),
      1e-10
    )
    val r = np.polyfit(np.array(0.0, 0.0, 1.0, 1.0), np.array(1.0, 2.0, 3.0, 4.0), 2, full = true)
    assertEquals(r.rank, 2)
    assertEquals(r.residuals.size, 0)
    assertD(r.coef, Seq(1.0, 1.0, 1.5), 1e-10)
    assertD(r.singularValues(sliceTo(2)), Seq(1.6180339887498951, 0.61803398874989457), 1e-12)
  }

  test("the eigenvalue kernels directly") {
    val tri = PolyLinAlg.eigvalshTridiag(Array(2.0, 2.0, 2.0), Array(-1.0, -1.0))
    val ex = Seq(2 - math.sqrt(2.0), 2.0, 2 + math.sqrt(2.0))
    tri.toSeq.zip(ex).foreach((x, y) => assertEqualsDouble(x, y, 1e-14))
    val (u, sv, v) = PolyLinAlg.svd(Array(Array(3.0, 0.0), Array(4.0, 5.0)), 2, 2)
    assertEqualsDouble(sv(0), 3 * math.sqrt(5.0), 1e-13)
    assertEqualsDouble(sv(1), math.sqrt(5.0), 1e-13)
    // wide matrix goes through the transposed branch
    val (_, sw, _) = PolyLinAlg.svd(Array(Array(3.0, 4.0)), 1, 2)
    assertEqualsDouble(sw(0), 5.0, 1e-14)
    val inv = PolyLinAlg.inv(Array(Array(4.0, 7.0), Array(2.0, 6.0)))
    assertEqualsDouble(inv(0)(0), 0.6, 1e-15)
    assertEqualsDouble(inv(1)(0), -0.2, 1e-15)
    intercept[ArithmeticException](PolyLinAlg.inv(Array(Array(1.0, 2.0), Array(2.0, 4.0))))
    val big = PolyLinAlg.eigvalsComplex(Array.tabulate(6, 6)((i, j) => Complex(i + 2.0 * j, (i * j) % 3 - 1.0)))
    // trace is preserved
    val tr = (0 until 6).map(i => Complex(i + 2.0 * i, (i * i) % 3 - 1.0)).reduce(_ + _)
    assert((big.reduce(_ + _) - tr).abs < 1e-10)
    intercept[ArithmeticException](PolyLinAlg.eigvalsReal(Array(Array(Double.NaN))))
  }

