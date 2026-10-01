package com.github.kmizu.numscala

import PolyFunctional as F

class PolySeriesSuite extends munit.FunSuite:

  private def assertD(a: NDArray[Double], e: Seq[Any], tol: Double = 1e-10)(using munit.Location): Unit =
    val av = a.toSeq
    val ev = e.map(_.asInstanceOf[Double])
    assertEquals(av.length, ev.length, s"got $av expected $ev")
    av.zip(ev).zipWithIndex.foreach { case ((x, y), i) =>
      assert(math.abs(x - y) <= tol * math.max(1.0, math.abs(y)), s"index $i: $x != $y (got $av)")
    }

  private def assertC(a: NDArray[Complex], e: Seq[Any], tol: Double = 1e-10)(using munit.Location): Unit =
    val av = a.toSeq
    val ev = e.map {
      case z: Complex => z
      case d: Double => Complex(d, 0.0)
      case other => fail(s"bad reference $other")
    }
    assertEquals(av.length, ev.length, s"got $av")
    av.zip(ev).foreach((x, y) => assert((x - y).abs <= tol * math.max(1.0, y.abs), s"$x != $y (got $av)"))

  private val c = np.array(1.0, 2.0, 3.0, 4.0)
  private val c2 = np.array(0.5, -1.0, 2.0)
  private val fitX = np.linspace(-1.0, 1.0, 9)
  private val fitY = fitX.map(x => math.cos(2 * x) + 0.1 * x)

  for b <- PolyBasis.all do
    val ref = PolyRef.ref(b.prefix)
    test(s"${b.prefix}: functional module matches NumPy") {
      assertD(F.valArr(b, np.array(-0.5, 0.3, 1.7), c, true), ref("val"), 1e-12)
      assertD(F.mul(b, c, c2), ref("mul"), 1e-12)
      val (q, r) = F.div(b, c, c2)
      assertD(q, ref("divq"), 1e-12)
      assertD(r, ref("divr"), 1e-11)
      assertD(F.pow(b, c2, 3, 16), ref("pow"), 1e-12)
      assertD(F.mulx(b, c), ref("mulx"), 1e-12)
      assertD(F.der(b, c, 1, 1.0, 0), ref("der"), 1e-12)
      assertD(F.der(b, c, 2, 0.5, 0), ref("der2"), 1e-12)
      assertD(F.integ(b, c, 1, Nil, 0.0, 1.0, 0), ref("int"), 1e-12)
      assertD(F.integ(b, c, 2, Seq(1.0, 2.0), 0.5, 2.0, 0), ref("int2"), 1e-12)
      assertC(F.roots(b, c), ref("roots"), 1e-10)
      assertD(F.fromroots(b, np.array(-1.0, 0.5, 2.0)), ref("fromroots"), 1e-12)
      assertD(F.vander(b, np.array(0.5, -2.0), 3), ref("vander"), 1e-12)
      assertEquals(F.vander(b, np.array(0.5, -2.0), 3).shape, Seq(2, 4))
      assertD(F.companion(b, c), ref("companion"), 1e-12)
      assertD(F.fit(b, fitX, fitY, 3, -1.0, null)._1, ref("fit"), 1e-9)
      assertD(F.fit(b, fitX, fitY, Seq(0, 2, 3), -1.0, null)._1, ref("fitdegs"), 1e-9)
      assertD(F.line(b, 3.0, 2.0), ref("line"), 1e-12)
      assertD(F.add(b, c, c2), ref("add"), 1e-12)
      assertD(F.sub(b, c2, c), ref("sub"), 1e-12)
      val xs = np.array(0.5, -0.25)
      val ys = np.array(1.5, 0.75)
      val c6 = np.arange(6.0).reshape(2, 3)
      assertD(F.valNd(b, c6, xs, ys), ref("val2d"), 1e-12)
      val g = F.gridNd(b, c6, xs, np.array(1.5, 0.75, 2.0))
      assertEquals(g.shape, Seq(2, 3))
      assertD(g, ref("grid2d"), 1e-12)
      val v2 = F.vanderNd(b, Seq(xs, ys), Seq(1, 2))
      assertEquals(v2.shape, Seq(2, 6))
      assertD(v2, ref("vander2d"), 1e-12)
      val vt = F.valArr(b, np.array(0.5, -0.25, 2.0), np.arange(6.0).reshape(3, 2), true)
      assertEquals(vt.shape, Seq(2, 3))
      assertD(vt, ref("valtensor"), 1e-12)
      val da = F.der(b, np.arange(12.0).reshape(3, 4), 1, 1.0, 1)
      assertEquals(da.shape, Seq(3, 3))
      assertD(da, ref("dera1"), 1e-12)
      val ia = F.integ(b, np.arange(6.0).reshape(3, 2), 1, Seq(2.0), 0.0, 1.0, 0)
      assertEquals(ia.shape, Seq(4, 2))
      assertD(ia, ref("inta0"), 1e-12)
      if b ne PolyBasis.Power then
        assertD(F.toPower(b, c), ref("topoly"), 1e-12)
        assertD(F.fromPower(b, c), ref("frompoly"), 1e-12)
        val (gx, gw) = F.gauss(b, 4)
        assertD(gx, ref("gaussx"), 1e-12)
        assertD(gw, ref("gaussw"), 1e-12)
        val (gx7, gw7) = F.gauss(b, 7)
        assertD(gx7, ref("gauss7x"), 1e-12)
        assertD(gw7, ref("gauss7w"), 1e-12)
        assertD(F.weight(b, np.array(0.3, 0.5)), ref("weight"), 1e-14)
        // round trips
        assertD(F.fromPower(b, F.toPower(b, c)), c.toSeq, 1e-12)
    }

  test("generated module forwarders") {
    val cheb = np.polynomial.chebyshev
    assertEquals(cheb.chebval(0.5, np.array(1.0, 2.0, 3.0)), 0.5)
    assertD(cheb.chebval(np.array(0.5, 1.0), np.array(1.0, 2.0, 3.0)), Seq(0.5, 6.0))
    val cf: NDArray[Double] = cheb.chebfit(fitX, fitY, 3)
    assertD(cf, PolyRef.ref("cheb")("fit"), 1e-9)
    val (cfull, info) = cheb.chebfit(fitX, fitY, 3, full = true)
    assertEquals(info.rank, 4)
    assertD(cfull, cf.toSeq)
    assertD(cheb.chebpts1(3), Seq(-0.8660254037844386, 0.0, 0.8660254037844386), 1e-15)
    assertD(cheb.chebpts2(3), Seq(-1.0, 0.0, 1.0), 1e-15)
    assertD(cheb.chebinterpolate(x => x * x, 2), Seq(0.5, 0.0, 0.5), 1e-14)
    assertD(cheb.chebx, Seq(0.0, 1.0))
    assertD(np.polynomial.hermite.hermx, Seq(0.0, 0.5))
    assertD(np.polynomial.laguerre.lagx, Seq(1.0, -1.0))
    assertD(np.polynomial.laguerre.lagdomain, Seq(0.0, 1.0))
    assertD(np.polynomial.polynomial.polyval(np.array(2.0), np.array(1.0, 2.0, 3.0)), Seq(17.0))
    assertD(np.polynomial.polynomial.polyvalfromroots(np.array(0.0, 3.0), np.array(1.0, 2.0)), Seq(2.0, 2.0))
    assertD(np.polynomial.legendre.legtrim(np.array(1.0, 2.0, 1e-12, 0.0), tol = 1e-10), Seq(1.0, 2.0))
    assertD(np.polynomial.legendre.leg2poly(np.array(0.0, 0.0, 1.0)), Seq(-0.5, 0.0, 1.5))
    assertD(np.polynomial.hermite_e.hermeval(np.array(2.0), np.array(0.0, 0.0, 1.0)), Seq(3.0))
    val (lq, lr) = np.polynomial.laguerre.lagdiv(np.array(1.0, 2.0, 3.0), np.array(1.0, 1.0))
    assertD(np.polynomial.laguerre.lagadd(np.polynomial.laguerre.lagmul(lq, np.array(1.0, 1.0)), lr), Seq(1.0, 2.0, 3.0), 1e-12)
    assertD(np.polynomial.polynomial.polyval3d(np.array(1.0), np.array(2.0), np.array(3.0), np.ones(2, 2, 2)), Seq(24.0))
    assertEquals(np.polynomial.polynomial.polygrid3d(np.array(1.0, 2.0), np.array(2.0), np.array(3.0), np.ones(2, 2, 2)).shape, Seq(2))
    assertEquals(np.polynomial.polynomial.polyvander3d(np.array(1.0), np.array(2.0), np.array(3.0), Seq(1, 1, 1)).shape, Seq(8))
    intercept[IllegalArgumentException](np.polynomial.polynomial.polyval2d(np.array(1.0, 2.0), np.array(1.0), np.ones(2, 2)))
    intercept[IllegalArgumentException](np.polynomial.chebyshev.chebint(np.array(1.0), 1, Seq(1.0, 2.0)))
    intercept[IllegalArgumentException](np.polynomial.chebyshev.chebpow(np.array(1.0, 1.0), 20))
    intercept[ArithmeticException](np.polynomial.chebyshev.chebdiv(np.array(1.0), np.array(0.0)))
    intercept[IllegalArgumentException](np.polynomial.chebyshev.chebcompanion(np.array(1.0)))
    assertEquals(np.polynomial.chebyshev.chebroots(np.array(1.0)).size, 0)
    assertD(np.polynomial.chebyshev.chebder(np.array(1.0, 2.0), 3), Seq(0.0))
  }

  test("polyutils") {
    val pu = np.polynomial.polyutils
    assertD(pu.trimseq(np.array(1.0, 0.0, 0.0)), Seq(1.0))
    assertD(pu.trimseq(np.array(0.0, 0.0)), Seq(0.0))
    assertD(pu.trimcoef(np.array(1.0, -1e-3, 1e-4), 1e-3), Seq(1.0))
    assertD(pu.getdomain(np.array(3.0, -1.0, 2.0)), Seq(-1.0, 3.0))
    assertEquals(pu.mapparms(Seq(0.0, 4.0), Seq(-1.0, 1.0)), (-1.0, 0.5))
    assertD(pu.mapdomain(np.array(0.0, 2.0, 4.0), Seq(0.0, 4.0), Seq(-1.0, 1.0)), Seq(-1.0, 0.0, 1.0))
    assertEquals(pu.as_series(Seq(np.array(1.0, 0.0), np.array(2, 3))).map(_.toList), Seq(List(1.0), List(2.0, 3.0)))
  }

  test("series class printing") {
    val P = np.polynomial.Polynomial
    assertEquals(P(Seq(1.0, 2.0, 3.0)).toString, "1.0 + 2.0·x + 3.0·x²")
    assertEquals(P(Seq(1.0, 2.0, 3.0)).repr, "Polynomial([1., 2., 3.], domain=[-1.,  1.], window=[-1.,  1.], symbol='x')")
    assertEquals(np.polynomial.Chebyshev(Seq(1.0, 2.0, 3.0)).toString, "1.0 + 2.0·T₁(x) + 3.0·T₂(x)")
    assertEquals(np.polynomial.HermiteE(Seq(1.0, -2.5, 3e-9, 1e9)).toString, "1.0 - 2.5·He₁(x) + (3.0e-09)·He₂(x) + (1.0e+09)·He₃(x)")
    assertEquals(P(Seq(1.0, 2.0), domain = Seq(0.0, 2.0)).toString, "1.0 + 2.0·(-1.0 + x)")
    assertEquals(np.polynomial.Legendre(Seq(1.0, 2.0), domain = Seq(0.0, 4.0)).toString, "1.0 + 2.0·P₁((-1.0 + 0.5x))")
    assertEquals(np.polynomial.Laguerre(Seq(1.0, 2.0)).repr, "Laguerre([1., 2.], domain=[0., 1.], window=[0., 1.], symbol='x')")
    val long = P(np.arange(30.0)).toString
    assertEquals(
      long,
      "0.0 + 1.0·x + 2.0·x² + 3.0·x³ + 4.0·x⁴ + 5.0·x⁵ + 6.0·x⁶ + 7.0·x⁷ +\n" +
        "8.0·x⁸ + 9.0·x⁹ + 10.0·x¹⁰ + 11.0·x¹¹ + 12.0·x¹² + 13.0·x¹³ + 14.0·x¹⁴ +\n" +
        "15.0·x¹⁵ + 16.0·x¹⁶ + 17.0·x¹⁷ + 18.0·x¹⁸ + 19.0·x¹⁹ + 20.0·x²⁰ +\n" +
        "21.0·x²¹ + 22.0·x²² + 23.0·x²³ + 24.0·x²⁴ + 25.0·x²⁵ + 26.0·x²⁶ +\n" +
        "27.0·x²⁷ + 28.0·x²⁸ + 29.0·x²⁹"
    )
    np.polynomial.set_default_printstyle("ascii")
    try
      assertEquals(P(Seq(1.0, 2.0, 3.0)).toString, "1.0 + 2.0 x + 3.0 x**2")
      assertEquals(np.polynomial.Chebyshev(Seq(1.0, 2.0, 3.0)).toString, "1.0 + 2.0 T_1(x) + 3.0 T_2(x)")
    finally np.polynomial.set_default_printstyle("unicode")
    intercept[IllegalArgumentException](np.polynomial.set_default_printstyle("latex"))
  }

  private val cx = np.linspace(0.0, 10.0, 11)
  private val cy = cx.map(x => x * x - 3 * x + 1 + math.sin(x))

  test("Chebyshev.fit, evaluation, conversion and calculus") {
    val f = np.polynomial.Chebyshev.fit(cx, cy, 3)
    assertD(f.coef, Seq(23.66205771186779, 34.84638708233201, 12.668839299694472, -0.20671987852997273), 1e-10)
    assertD(f.domain, Seq(0.0, 10.0))
    assertD(f(np.array(2.5, 7.0)), Seq(-0.3022753576754251, 29.180945386340653), 1e-10)
    assertEqualsDouble(f(2.5), -0.3022753576754251, 1e-10)
    val pf = f.convert(np.polynomial.Polynomial)
    assertD(pf.coef, Seq(1.6912298077602248, -3.537889804643127, 1.1127326856699447, -0.00661503611295913), 1e-10)
    assertD(f.convert().coef, Seq(2.2475961505951947, -3.542851081727846, 0.55636634283497233, -0.0016537590282397819), 1e-10)
    assertD(f.deriv().coef, Seq(6.845245489348418, 10.135071439755578, -0.24806385423596727), 1e-10)
    assertD(f.integ().coef, Seq(43.94558362515871, 86.63819031010277, 43.81638370107748, 10.557366083078726, -0.12919992408123296), 1e-10)
    assertD(f.integ(k = Seq(1.0), lbnd = 2.0).coef, Seq(55.26086559261637, 86.63819031010277, 43.81638370107748, 10.557366083078726, -0.12919992408123296), 1e-10)
    assertC(f.roots(), Seq(0.5854659893670746, 2.6468949220858597, 164.980278697582), 1e-9)
    val (r, info) = np.polynomial.Polynomial.fit(cx, cy, 2, full = true)
    assertD(r.coef, Seq(10.993218412173306, 34.877808503868565, 25.33767859938899), 1e-10)
    assertD(r.convert().coef, Seq(1.4530885076937317, -3.1595097389818836, 1.0135071439755599), 1e-10)
    assertD(info.residuals, Seq(4.598158093573061), 1e-9)
    assertEquals(info.rank, 3)
    assertD(info.singularValues, Seq(1.3226986387669195, 0.9999999999999998, 0.5004680918941173), 1e-10)
    assertEqualsDouble(info.rcond, 2.4424906541753444e-15, 1e-25)
    assertD(np.polynomial.HermiteE.fit(cx, cy, Seq(1, 3)).coef, Seq(0.0, 32.985908175562244, 0.0, -0.8268795141198954), 1e-9)
  }

  test("class construction helpers") {
    assertD(np.polynomial.Polynomial.fromroots(Seq(1.0, 2.0, 3.0)).coef, Seq(-6.0, 11.0, -6.0, 1.0))
    val q2 = np.polynomial.Chebyshev.fromroots(Seq(1.0, 2.0, 3.0), domain = null)
    assertD(q2.coef, Seq(0.0, -0.25, 0.0, 0.25), 1e-14)
    assertD(q2.domain, Seq(1.0, 3.0))
    assertD(np.polynomial.Legendre.basis(3).coef, Seq(0.0, 0.0, 0.0, 1.0))
    assertD(np.polynomial.Hermite.identity().coef, Seq(0.0, 0.5))
    assertD(np.polynomial.Laguerre.identity(domain = Seq(0.0, 2.0)).coef, Seq(2.0, -2.0))
    assertD(np.polynomial.Legendre.cast(np.polynomial.Polynomial(Seq(1.0, 2.0, 3.0))).coef, Seq(2.0, 2.0, 2.0), 1e-14)
    assertD(np.polynomial.Polynomial(Seq(0.0, 1.0)).convert(np.polynomial.Chebyshev).coef, Seq(0.0, 1.0))
    assertD(np.polynomial.Polynomial(np.array(1, 2)).coef, Seq(1.0, 2.0))
    assertD(np.polynomial.Chebyshev.domain, Seq(-1.0, 1.0))
    assertD(np.polynomial.Laguerre.window, Seq(0.0, 1.0))
    val ci = np.polynomial.Chebyshev.interpolate(x => x * x * x, 3, domain = Seq(0.0, 2.0))
    assertEqualsDouble(ci(1.5), 3.375, 1e-12)
    intercept[IllegalArgumentException](np.polynomial.Polynomial(Seq(1.0), domain = Seq(1.0)))
    intercept[IllegalArgumentException](np.polynomial.Polynomial(Seq(1.0), symbol = "1x"))
    intercept[IllegalArgumentException](np.polynomial.Polynomial(Seq.empty[Double]))
  }

  test("class arithmetic") {
    val C = np.polynomial.Chebyshev
    val a = C(Seq(1.0, 2.0, 3.0, 4.0))
    val b = C(Seq(1.0, 1.0))
    val (q, r) = a.divmod(b)
    assertD(q.coef, Seq(8.0, -10.0, 8.0), 1e-12)
    assertD(r.coef, Seq(-2.0), 1e-12)
    assertD(a.floorDiv(b).coef, Seq(8.0, -10.0, 8.0), 1e-12)
    assertD((a % b).coef, Seq(-2.0), 1e-12)
    assertD((a ** 2).coef, Seq(15.5, 22.0, 16.0, 14.0, 12.5, 12.0, 8.0), 1e-12)
    assertD((a * 2.0).coef, Seq(2.0, 4.0, 6.0, 8.0))
    assertD((a / 2.0).coef, Seq(0.5, 1.0, 1.5, 2.0))
    assertD((a - 1.0).coef, Seq(0.0, 2.0, 3.0, 4.0))
    assertD((a + b).coef, Seq(2.0, 3.0, 3.0, 4.0))
    assertD((a - a).coef, Seq(0.0))
    assertD((-b).coef, Seq(-1.0, -1.0))
    assertEquals(a + b, C(Seq(2.0, 3.0, 3.0, 4.0)))
    assertNotEquals(a, C(Seq(1.0, 2.0, 3.0, 4.0), domain = Seq(0.0, 1.0)))
    intercept[IllegalArgumentException](a + C(Seq(1.0), domain = Seq(0.0, 1.0)))
    intercept[IllegalArgumentException](a + C(Seq(1.0), window = Seq(0.0, 1.0)))
    intercept[IllegalArgumentException](a + C(Seq(1.0), symbol = "t"))
    intercept[IllegalArgumentException](a ** 101)
    val P = np.polynomial.Polynomial
    assertD(P(Seq(1.0, 2.0, 3.0)).compose(P(Seq(0.0, 1.0, 1.0))).coef, Seq(1.0, 2.0, 5.0, 6.0, 3.0))
    assertD(C(Seq(1.0, 2.0, 3.0)).compose(C(Seq(0.0, 1.0))).coef, Seq(1.0, 2.0, 3.0), 1e-14)
  }

  test("class misc") {
    val P = np.polynomial.Polynomial
    assertD(P(Seq(1.0, 2.0, 3.0)).linspace(5)._2, Seq(2.0, 0.75, 1.0, 2.75, 6.0))
    assertD(P(Seq(1.0, 2.0, 3.0), domain = Seq(0.0, 1.0)).linspace(3)._1, Seq(0.0, 0.5, 1.0))
    assertD(P(Seq(1.0, 2.0, 0.0, 0.0)).trim().coef, Seq(1.0, 2.0))
    assertD(P(Seq(1.0, 2.0, 3.0, 4.0)).truncate(2).coef, Seq(1.0, 2.0))
    assertD(P(Seq(1.0, 2.0, 3.0, 4.0)).cutdeg(2).coef, Seq(1.0, 2.0, 3.0))
    assertD(P(Seq(1.0, 1e-12, 3e-14)).trim(1e-10).coef, Seq(1.0))
    assertEquals(P(Seq(1.0, 2.0, 3.0)).degree, 2)
    assertC(np.polynomial.Legendre(Seq(1.0, 2.0, 3.0), domain = Seq(0.0, 2.0)).roots(), Seq(Complex(0.37716096939289), Complex(1.1783945861626657)), 1e-12)
    val p = P(Seq(1.0, 2.0, 3.0))
    assertEquals(p.mapparms, (0.0, 1.0))
    assert(p.has_samecoef(p.copy()))
    assert(p.has_sametype(P(Seq(1.0))))
    assert(!p.has_samedomain(P(Seq(1.0), domain = Seq(0.0, 1.0))))
    // derivative / integral are inverse operations on the domain variable
    val q = P(Seq(1.0, 2.0, 3.0), domain = Seq(0.0, 4.0))
    assertD(q.integ().deriv().coef, q.coef.toSeq, 1e-12)
    assertEqualsDouble(q.integ(lbnd = 1.0)(1.0), 0.0, 1e-12)
  }
