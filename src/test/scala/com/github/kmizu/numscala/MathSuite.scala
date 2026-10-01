package com.github.kmizu.numscala

class MathSuite extends munit.FunSuite with MathTestUtil:
  private val nan = Double.NaN
  private val inf = Double.PositiveInfinity

  test("sqrt / cbrt with dtypes and scalars") {
    val r: NDArray[Double] = np.sqrt(np.array(1, 4, 9))
    assertEquals(r.toList, List(1.0, 2.0, 3.0))
    close(np.sqrt(np.array(-1.0, 2.0)).toSeq, Seq(nan, math.sqrt(2.0)))
    val f: NDArray[Float] = np.sqrt(np.array(4f))
    assertEquals(f.toList, List(2f))
    val c: NDArray[Complex] = np.sqrt(np.array(Complex(-4, 0), Complex(3, 4)))
    closeC(c.toList(0), Complex(0, 2))
    closeC(c.toList(1), Complex(2, 1))
    assertEquals(np.sqrt(2.0), math.sqrt(2.0))
    assertEquals(np.sqrt(4), 2.0)
    closeC(np.sqrt(Complex(-1, 0)), Complex(0, 1))
    val b: NDArray[Double] = np.sqrt(np.array(true, false))
    assertEquals(b.toList, List(1.0, 0.0))
    assertEquals(np.cbrt(np.array(-27.0, 8.0)).toList, List(-3.0, 2.0))
  }

  test("exponentials and logarithms") {
    close(np.exp(np.array(0.0, 1.0)).toSeq, Seq(1.0, math.E))
    assertEquals(np.exp2(np.array(3.0, -1.0)).toList, List(8.0, 0.5))
    close(np.expm1(np.array(1e-10)).toSeq, Seq(1.00000000005e-10), 1e-15)
    assertEquals(np.log2(np.array(8.0, 1024.0, 0.5, 1.0)).toList, List(3.0, 10.0, -1.0, 0.0))
    close(np.log2(np.array(3.0, 10.0, 5e-324)).toSeq, Seq(1.584962500721156, 3.321928094887362, -1074.0), 1e-15)
    close(np.log2(np.array(0.0, -1.0, inf)).toSeq, Seq(Double.NegativeInfinity, nan, inf))
    assertEquals(np.log10(np.array(1000.0, 0.01)).toList, List(3.0, -2.0))
    close(np.log(np.array(1, 0, -1)).toSeq, Seq(0.0, Double.NegativeInfinity, nan))
    close(np.log1p(np.array(1e-10)).toSeq, Seq(9.9999999995e-11), 1e-15)
    closeC(np.log(Complex(-1, 0)), Complex(0, math.Pi))
    closeC(np.exp(Complex(0, math.Pi)), Complex(-1, 0))
    closeC(np.log(np.array(Complex(0, 1))).item, Complex(0, math.Pi / 2))
    closeC(np.log2(np.array(Complex(-8, 0))).item, Complex(3, math.Pi / math.log(2)))
    closeC(np.log10(np.array(Complex(-100, 0))).item, Complex(2, 1.3643763538418414))
    closeC(np.log1p(np.array(Complex(0, 1))).item, Complex(0.34657359027997264, 0.7853981633974483))
    closeC(np.expm1(np.array(Complex(1, 1))).item, Complex(0.46869393991588515, 2.2873552871788423))
    closeC(np.exp2(np.array(Complex(0, 1))).item, Complex(0.7692389013639721, 0.6389612763136348))
    val f32: NDArray[Float] = np.log(np.array(1f))
    assertEquals(f32.toList, List(0f))
  }

  test("trigonometric and hyperbolic (real)") {
    close(np.sin(np.array(0.0, math.Pi / 2)).toSeq, Seq(0.0, 1.0))
    close(np.cos(np.array(0.0, math.Pi)).toSeq, Seq(1.0, -1.0))
    close(np.tan(np.array(math.Pi / 4)).toSeq, Seq(1.0))
    close(np.arcsin(np.array(1.0, 2.0)).toSeq, Seq(math.Pi / 2, nan))
    close(np.arccos(np.array(-1.0)).toSeq, Seq(math.Pi))
    close(np.arctan(np.array(1.0, inf)).toSeq, Seq(math.Pi / 4, math.Pi / 2))
    close(np.sinh(np.array(1.0)).toSeq, Seq(1.1752011936438014))
    close(np.cosh(np.array(1.0)).toSeq, Seq(1.5430806348152437))
    close(np.tanh(np.array(0.5, 1000.0)).toSeq, Seq(0.46211715726000974, 1.0))
    close(np.arcsinh(np.array(1.0, -1e300, 1e-20)).toSeq, Seq(0.881373587019543, -691.4686750787736, 1e-20))
    close(np.arccosh(np.array(2.0, 1.0, 0.5, 1e300)).toSeq, Seq(1.3169578969248166, 0.0, nan, 691.4686750787736))
    close(np.arctanh(np.array(0.5, 1.0, -1.0, 2.0, 1e-3)).toSeq,
      Seq(0.5493061443340549, inf, Double.NegativeInfinity, nan, 0.0010000003333335333))
    assertEquals(np.asin(np.array(1.0)).toList, np.arcsin(np.array(1.0)).toList)
    assertEquals(np.acos(1.0), math.acos(1.0))
    assertEquals(np.atan(np.array(1.0)).toList, List(math.atan(1.0)))
    assertEquals(np.asinh(np.array(1.0)).toList, List(np.arcsinh(1.0)))
    assertEquals(np.acosh(np.array(2.0)).toList, List(np.arccosh(2.0)))
    assertEquals(np.atanh(np.array(0.5)).toList, List(np.arctanh(0.5)))
    close(Seq(np.degrees(math.Pi), np.radians(180.0), np.rad2deg(math.Pi / 2), np.deg2rad(90.0)),
      Seq(180.0, math.Pi, 90.0, math.Pi / 2))
    val d: NDArray[Double] = np.degrees(np.array(0, 1))
    close(d.toSeq, Seq(0.0, 57.29577951308232))
    close(np.radians(np.array(180.0)).toSeq, Seq(math.Pi))
    close(np.deg2rad(np.array(360f)).toSeq.map(_.toDouble), Seq(2 * math.Pi), 1e-7)
    close(np.rad2deg(np.array(math.Pi)).toSeq, Seq(180.0))
  }

  test("trigonometric and hyperbolic (complex)") {
    val z = Complex(1, 2)
    closeC(np.sin(z), Complex(3.165778513216168, 1.9596010414216063))
    closeC(np.cos(z), Complex(2.0327230070196656, -3.0518977991518))
    closeC(np.tan(z), Complex(0.0338128260798967, 1.0147936161466335))
    closeC(np.sinh(z), Complex(-0.4890562590412937, 1.4031192506220405))
    closeC(np.cosh(z), Complex(-0.64214812471552, 1.0686074213827783))
    closeC(np.tanh(z), Complex(1.16673625724092, -0.24345820118572525))
    closeC(np.arcsin(z), Complex(0.4270785863924761, 1.5285709194809982))
    closeC(np.arccos(z), Complex(1.1437177404024204, -1.5285709194809982))
    closeC(np.arctan(z), Complex(1.3389725222944935, 0.40235947810852507))
    closeC(np.arcsinh(z), Complex(1.4693517443681852, 1.0634400235777521))
    closeC(np.arccosh(z), Complex(1.5285709194809982, 1.1437177404024204))
    closeC(np.arctanh(z), Complex(0.17328679513998632, 1.1780972450961724))
    // branch cuts with +0 imaginary part
    closeC(np.arcsin(Complex(2, 0)), Complex(math.Pi / 2, 1.3169578969248166))
    closeC(np.arccos(Complex(2, 0)), Complex(0, -1.3169578969248166))
    closeC(np.arctanh(Complex(2, 0)), Complex(0.5493061443340549, math.Pi / 2))
    closeC(np.arccosh(Complex(-2, 0)), Complex(1.3169578969248166, math.Pi))
    closeC(np.tanh(Complex(1000, 0)), Complex(1, 0))
    closeC(np.sin(Complex(0, 1)), Complex(0, math.sinh(1)))
    val arr: NDArray[Complex] = np.cos(np.array(Complex(0, 0), Complex(math.Pi, 0)))
    closeC(arr.toList(0), Complex(1, 0))
    closeC(arr.toList(1), Complex(-1, 0))
  }

  test("power-ish unary: square, reciprocal, negative, positive, sign, abs") {
    assertEquals(np.square(np.array(-3, 4)).toList, List(9, 16))
    closeC(np.square(np.array(Complex(1, 1))).item, Complex(0, 2))
    assertEquals(np.reciprocal(np.array(1, 2, -1, 0)).toList, List(1, 0, -1, Int.MinValue))
    assertEquals(np.reciprocal(np.array(0L)).toList, List(Long.MinValue))
    assertEquals(np.reciprocal(np.array(Seq(0.toByte))).toList, List(0.toByte))
    close(np.reciprocal(np.array(4.0, 0.0)).toSeq, Seq(0.25, inf))
    assertEquals(np.negative(np.array(1, -2)).toList, List(-1, 2))
    assertEquals(np.negative(3.0), -3.0)
    val p = np.array(1.0, 2.0)
    val pp = np.positive(p)
    assertEquals(pp.toList, List(1.0, 2.0))
    assert(!(pp eq p))
    close(np.sign(np.array(-2.0, -0.0, 3.0, nan)).toSeq, Seq(-1.0, 0.0, 1.0, nan))
    same(np.sign(-0.0), 0.0)
    assertEquals(np.sign(np.array(-5, 0, 7)).toList, List(-1, 0, 1))
    closeC(np.sign(np.array(Complex(3, 4))).item, Complex(0.6, 0.8))
    assertEquals(np.sign(np.array(Complex(inf, 0), Complex(0, -inf))).toList, List(Complex(1, 0), Complex(0, -1)))
    assert(np.sign(np.array(Complex(nan, 1))).item.isNaN)
    val a: NDArray[Double] = np.abs(np.array(Complex(3, 4)))
    assertEquals(a.toList, List(5.0))
    assertEquals(np.absolute(np.array(-1, Int.MinValue)).toList, List(1, Int.MinValue))
    same(np.abs(np.array(-0.0)).item, 0.0)
    assertEquals(np.abs(Complex(3, 4)), 5.0)
    assertEquals(np.abs(-2.5), 2.5)
    assertEquals(np.abs(np.array(true, false)).toList, List(true, false))
    val fa: NDArray[Double] = np.fabs(np.array(-1, 2))
    assertEquals(fa.toList, List(1.0, 2.0))
  }

  test("rounding") {
    assertEquals(np.floor(np.array(-1.5, 1.5)).toList, List(-2.0, 1.0))
    assertEquals(np.ceil(np.array(-1.5, 1.5)).toList, List(-1.0, 2.0))
    assertEquals(np.trunc(np.array(-1.7, 1.7)).toList, List(-1.0, 1.0))
    assertEquals(np.fix(np.array(-2.5, 2.5)).toList, List(-2.0, 2.0))
    same(np.fix(-0.5), -0.0)
    val fi: NDArray[Int] = np.floor(np.array(1, -2))
    assertEquals(fi.toList, List(1, -2))
    assertEquals(np.rint(np.array(0.5, 1.5, 2.5, -0.5)).toList, List(0.0, 2.0, 2.0, -0.0))
    val ri: NDArray[Double] = np.rint(np.array(3))
    assertEquals(ri.toList, List(3.0))
    closeC(np.rint(np.array(Complex(1.5, -2.5))).item, Complex(2, -2))
    close(np.round(np.array(1.25, 2.675, 0.5), 2).toSeq, Seq(1.25, 2.68, 0.5)) // NumPy's x*10**d, rint, /10**d algorithm
    close(np.round(np.array(1.25, 1.35), 1).toSeq, Seq(1.2, 1.4))
    assertEquals(np.round(np.array(1234, 1250), -2).toList, List(1200, 1200))
    assertEquals(np.around(np.array(2.5, 3.5)).toList, List(2.0, 4.0))
    assertEquals(np.round(2.5), 2.0)
    assertEquals(np.round(2.675, 2), 2.68)
    assertEquals(np.floor(2.7), 2.0)
    assertEquals(np.ceil(2.1), 3.0)
    assertEquals(np.trunc(-2.7), -2.0)
    assertEquals(np.rint(2.5), 2.0)
  }

  test("predicates") {
    val x = np.array(1.0, nan, inf, Double.NegativeInfinity, -0.0)
    assertEquals(np.isnan(x).toList, List(false, true, false, false, false))
    assertEquals(np.isinf(x).toList, List(false, false, true, true, false))
    assertEquals(np.isfinite(x).toList, List(true, false, false, false, true))
    assertEquals(np.isposinf(x).toList, List(false, false, true, false, false))
    assertEquals(np.isneginf(x).toList, List(false, false, false, true, false))
    assertEquals(np.signbit(x).toList, List(false, false, false, true, true))
    assertEquals(np.isnan(np.array(1, 2)).toList, List(false, false))
    assertEquals(np.isfinite(np.array(1, 2)).toList, List(true, true))
    assertEquals(np.signbit(np.array(-1, 0)).toList, List(true, false))
    assertEquals(np.isnan(np.array(Complex(nan, 0), Complex(1, 0))).toList, List(true, false))
    assertEquals(np.isinf(np.array(Complex(0, inf))).toList, List(true))
    intercept[IllegalArgumentException](np.isposinf(np.array(Complex(1, 0))))
    intercept[IllegalArgumentException](np.isnan(np.array("a")))
    assert(np.isnan(nan) && np.isinf(inf) && !np.isfinite(inf) && np.signbit(-0.0))
    assert(np.isposinf(inf) && np.isneginf(Double.NegativeInfinity))
    assertEquals(np.logical_not(np.array(0.0, 2.0, nan)).toList, List(true, false, false))
    assertEquals(np.logical_not(np.array(true, false)).toList, List(false, true))
    assertEquals(np.logical_not(np.array(Complex(0, 0), Complex(0, 1))).toList, List(true, false))
    assert(np.logical_not(false))
  }

  test("bitwise unary") {
    assertEquals(np.bitwise_not(np.array(0, 5, -1)).toList, List(-1, -6, 0))
    assertEquals(np.invert(np.array(true, false)).toList, List(false, true))
    assertEquals(np.bitwise_invert(np.array(1L)).toList, List(-2L))
    assertEquals(np.bitwise_count(np.array(3, -1, 0, 255)).toList, List(2, 1, 0, 8).map(_.toByte))
    assertEquals(np.bitwise_count(np.array(Long.MinValue)).toList, List(1.toByte))
  }

  test("clip") {
    assertEquals(np.clip(np.array(1.0, 5.0, 10.0), 2.0, 8.0).toList, List(2.0, 5.0, 8.0))
    close(np.clip(np.array(nan, 1.0), 0.0, 0.5).toSeq, Seq(nan, 0.5))
    assertEquals(np.clip(np.array(1, 5, 10), 2, 8).toList, List(2, 5, 8))
    assertEquals(np.clip(np.array(1, 5), 4, 2).toList, List(2, 2))
    assertEquals(np.clip(np.array(1.0, 5.0), 4.0, 2.0).toList, List(2.0, 2.0))
    assertEquals(np.clip(np.arange(5), np.array(1, 1, 1, 1, 1), np.array(3)).toList, List(1, 1, 2, 3, 3))
    assertEquals(np.clip(np.arange(5.0), None, Some(2.0)).toList, List(0.0, 1.0, 2.0, 2.0, 2.0))
    assertEquals(np.clip(np.arange(5.0), Some(3.0), None).toList, List(3.0, 3.0, 3.0, 3.0, 4.0))
    assertEquals(np.clip(np.arange(3.0), None, None).toList, List(0.0, 1.0, 2.0))
    assertEquals(np.clip(np.array(1.5f, 9f), 2f, 3f).toList, List(2f, 3f))
    assertEquals(np.clip(5.0, 0.0, 1.0), 1.0)
  }

  test("nan_to_num") {
    val x = np.array(nan, inf, Double.NegativeInfinity, 1.5)
    assertEquals(np.nan_to_num(x).toList, List(0.0, Double.MaxValue, -Double.MaxValue, 1.5))
    assertEquals(np.nan_to_num(x, nan = -1.0, posinf = Some(9.0), neginf = Some(-9.0)).toList, List(-1.0, 9.0, -9.0, 1.5))
    assertEquals(np.nan_to_num(np.array(Float.NaN, Float.PositiveInfinity)).toList, List(0f, Float.MaxValue))
    assertEquals(np.nan_to_num(np.array(Complex(nan, inf))).toList, List(Complex(0, Double.MaxValue)))
    assertEquals(np.nan_to_num(np.array(1, 2)).toList, List(1, 2))
    assertEquals(np.nan_to_num(nan), 0.0)
    assertEquals(np.nan_to_num(Double.NegativeInfinity), -Double.MaxValue)
  }

  test("complex helpers: real, imag, conj, angle, real_if_close, isreal...") {
    val z = np.array(Complex(1, 2), Complex(-3, 0))
    val re: NDArray[Double] = np.real(z)
    assertEquals(re.toList, List(1.0, -3.0))
    assertEquals(np.imag(z).toList, List(2.0, 0.0))
    val ri: NDArray[Int] = np.real(np.array(1, 2))
    assertEquals(ri.toList, List(1, 2))
    val ii: NDArray[Int] = np.imag(np.array(1, 2))
    assertEquals(ii.toList, List(0, 0))
    val base = np.array(1.0, 2.0)
    np.real(base)(0) = 5.0
    assertEquals(base.toList, List(5.0, 2.0))
    assertEquals(np.conj(z).toList, List(Complex(1, -2), Complex(-3, -0.0)))
    assertEquals(np.conjugate(np.array(1.0)).toList, List(1.0))
    assertEquals(np.conj(Complex(1, 1)), Complex(1, -1))
    close(np.angle(z).toSeq, Seq(math.atan2(2, 1), math.Pi))
    close(np.angle(z, deg = true).toSeq, Seq(63.43494882292201, 180.0))
    val ar: NDArray[Double] = np.angle(np.array(-1, 1))
    close(ar.toSeq, Seq(math.Pi, 0.0))
    assertEquals(np.angle(Complex(0, 1)), math.Pi / 2)
    assertEquals(np.real(Complex(1, 2)), 1.0)
    assertEquals(np.imag(Complex(1, 2)), 2.0)
    val c1 = np.real_if_close(np.array(Complex(2.1, 4e-14), Complex(5.2, 3e-15)), tol = 1000)
    assertEquals(c1.dtype.name, "float64")
    assertEquals(c1.toList: List[Any], List(2.1, 5.2))
    val c2 = np.real_if_close(np.array(Complex(2.1, 4e-13)), tol = 1000)
    assertEquals(c2.dtype.name, "complex128")
    assertEquals(np.real_if_close(np.array(Complex(1, 0.5)), tol = 0.6).dtype.name, "float64")
    assertEquals(np.real_if_close(np.array(1, 2)).dtype.name, "int32")
    assertEquals(np.isreal(np.array(Complex(1, 1), Complex(1, 0))).toList, List(false, true))
    assertEquals(np.iscomplex(np.array(Complex(1, 1), Complex(1, 0))).toList, List(true, false))
    assertEquals(np.isreal(np.array(1.0)).toList, List(true))
    assertEquals(np.iscomplex(np.array(1.0)).toList, List(false))
    assert(np.iscomplexobj(z) && !np.isrealobj(z) && np.isrealobj(base) && !np.iscomplexobj(base))
  }

  test("frexp, modf, ldexp, spacing") {
    val (m, e) = np.frexp(np.array(0.0, 1.0, -3.0, 8.0, inf))
    close(m.toSeq, Seq(0.0, 0.5, -0.75, 0.5, inf))
    assertEquals(e.toList, List(0, 1, 2, 4, 0))
    assertEquals(np.frexp(5e-324), (0.5, -1073))
    assertEquals(np.frexp(1e-310)._2, -1029)
    assertEqualsDouble(np.frexp(1e-310)._1 * math.pow(2, -1029), 1e-310, 1e-320)
    val (mf, ef) = np.frexp(np.array(12f))
    assertEquals((mf.toList, ef.toList), (List(0.75f), List(4)))
    val (fr, ip) = np.modf(np.array(-2.5, 3.25, -0.0))
    assertEquals(fr.toList, List(-0.5, 0.25, -0.0))
    assertEquals(ip.toList, List(-2.0, 3.0, -0.0))
    same(fr.toList(2), -0.0)
    assertEquals(np.modf(inf), (0.0, inf))
    val ld: NDArray[Double] = np.ldexp(np.array(1.0, 3.0, 5.0), np.array(2, -1, 0))
    assertEquals(ld.toList, List(4.0, 1.5, 5.0))
    assertEquals(np.ldexp(np.array(1, 2), 3).toList, List(8.0, 16.0))
    assertEquals(np.ldexp(1.0, 1024), inf)
    assertEquals(np.spacing(np.array(1.0, -1.0, 0.0, -0.0, -5e-324)).toList,
      List(math.ulp(1.0), -math.ulp(1.0), 5e-324, 5e-324, -5e-324))
    assertEquals(np.spacing(np.array(-0.0f)).toList, List(java.lang.Float.MIN_VALUE))
    assertEquals(np.spacing(np.array(1f)).toList, List(math.ulp(1f)))
    assert(np.spacing(inf).isNaN)
    assertEquals(np.spacing(1e10), math.ulp(1e10))
  }

  test("sinc and i0") {
    close(np.sinc(np.array(0.0, 0.5, 1.0, -0.5)).toSeq, Seq(1.0, 0.6366197723675814, 3.8981718325193755e-17, 0.6366197723675814))
    assertEquals(np.sinc(0.0), 1.0)
    closeC(np.sinc(np.array(Complex(0, 0))).item, Complex(1, 0))
    close(np.i0(np.array(0.0, 1.0, -1.0, 10.0, 0.5)).toSeq,
      Seq(1.0, 1.2660658777520082, 1.2660658777520082, 2815.716628466254, 1.0634833707413234), 1e-15)
    val ii: NDArray[Double] = np.i0(np.array(2))
    close(ii.toSeq, Seq(2.279585302336067), 1e-15)
    assertEquals(np.i0(0.0), 1.0)
  }

  test("unwrap") {
    val phase = np.linspace(0, math.Pi, 5)
    phase("3:") = np.add(phase("3:"), math.Pi)
    close(np.unwrap(phase).toSeq, Seq(0.0, 0.7853981633974483, 1.5707963267948966, -0.7853981633974483, 0.0))
    val p6: NDArray[Double] = np.unwrap(np.arange(10), period = 6.0)
    close(p6.toSeq, (0 until 10).map(_.toDouble))
    close(np.unwrap(np.array(2, 3, 4, 5, 2, 3, 4, 5), period = 4.0).toSeq, Seq(2.0, 3, 4, 5, 6, 7, 8, 9))
    val deg = np.subtract(np.remainder(np.linspace(0, 720, 19), 360.0), 180.0)
    close(np.unwrap(deg, period = 360.0).toSeq, (0 until 19).map(k => -180.0 + 40 * k))
    val m = np.array(Seq(Seq(0.0, 6.0), Seq(0.0, 0.5)))
    close(np.unwrap(m, axis = 0).toSeq, Seq(0.0, 6.0, 0.0, 0.5 + 2 * math.Pi))
    close(np.unwrap(m).toSeq, Seq(0.0, 6.0 - 2 * math.Pi, 0.0, 0.5))
    close(np.unwrap(np.array(0.0, 3.5), discont = 4.0).toSeq, Seq(0.0, 3.5))
    close(np.unwrap(np.array(0.0, 3.5)).toSeq, Seq(0.0, 3.5 - 2 * math.Pi))
  }

  test("emath") {
    assertEquals(np.emath.sqrt(4.0), 2.0)
    assertEquals(np.emath.sqrt(-1.0), Complex(0, 1))
    np.emath.log(-1.0) match
      case c: Complex => closeC(c, Complex(0, math.Pi))
      case other => fail(s"expected complex, got $other")
    assertEquals(np.emath.log2(8.0), 3.0)
    np.emath.log10(-100.0) match
      case c: Complex => closeC(c, Complex(2, 1.3643763538418414))
      case other => fail(s"expected complex, got $other")
    np.emath.logn(2.0, 8.0) match
      case d: Double => assertEqualsDouble(d, 3.0, 1e-15)
      case other => fail(s"expected real, got $other")
    np.emath.logn(2.0, -8.0) match
      case c: Complex => closeC(c, Complex(3, math.Pi / math.log(2)))
      case other => fail(s"expected complex, got $other")
    np.emath.power(-2.0, 2.0) match
      case c: Complex => closeC(c, Complex(4, 0))
      case other => fail(s"expected complex, got $other")
    assertEquals(np.emath.power(2.0, 3.0), 8.0)
    np.emath.arccos(2.0) match
      case c: Complex => closeC(c, Complex(0, -1.3169578969248166))
      case other => fail(s"expected complex, got $other")
    np.emath.arcsin(2.0) match
      case c: Complex => closeC(c, Complex(math.Pi / 2, 1.3169578969248166))
      case other => fail(s"expected complex, got $other")
    np.emath.arctanh(2.0) match
      case c: Complex => closeC(c, Complex(0.5493061443340549, math.Pi / 2))
      case other => fail(s"expected complex, got $other")
    assertEquals(np.emath.arctanh(0.5), np.arctanh(0.5))
    assertEquals(np.emath.arccos(1.0), 0.0)
    val s1 = np.emath.sqrt(np.array(4.0, -1.0))
    assertEquals(s1.dtype.name, "complex128")
    s1 match
      case a: NDArray[?] => assertEquals(a.toList: List[Any], List(Complex(2, 0), Complex(0, 1)))
    val s2 = np.emath.sqrt(np.array(4, 9))
    assertEquals(s2.dtype.name, "float64")
    assertEquals(s2.asInstanceOf[NDArray[Double]].toList, List(2.0, 3.0))
    assertEquals(np.emath.log(np.array(1.0, -1.0)).dtype.name, "complex128")
    assertEquals(np.emath.log2(np.array(8.0)).asInstanceOf[NDArray[Double]].toList, List(3.0))
    assertEquals(np.emath.log10(np.array(-10.0)).dtype.name, "complex128")
    assertEquals(np.emath.logn(-2.0, np.array(4.0)).dtype.name, "complex128")
    close(np.emath.logn(2.0, np.array(8.0, 4.0)).asInstanceOf[NDArray[Double]].toSeq, Seq(3.0, 2.0))
    assertEquals(np.emath.logn(2.0, np.array(-8.0)).dtype.name, "complex128")
    assertEquals(np.emath.power(np.array(-2.0, 3.0), 2.0).dtype.name, "complex128")
    assertEquals(np.emath.power(np.array(2.0, 3.0), 2.0).asInstanceOf[NDArray[Double]].toList, List(4.0, 9.0))
    assertEquals(np.emath.power(np.array(2.0, 3.0), np.array(2.0, 0.0)).asInstanceOf[NDArray[Double]].toList, List(4.0, 1.0))
    assertEquals(np.emath.power(np.array(-2.0), np.array(2.0)).dtype.name, "complex128")
    assertEquals(np.emath.arccos(np.array(0.5, 2.0)).dtype.name, "complex128")
    assertEquals(np.emath.arcsin(np.array(0.5)).dtype.name, "float64")
    assertEquals(np.emath.arctanh(np.array(Complex(0, 1))).dtype.name, "complex128")
    closeC(np.emath.sqrt(Complex(-4, 0)), Complex(0, 2))
    closeC(np.emath.log(Complex(-1, 0)), Complex(0, math.Pi))
    closeC(np.emath.log2(Complex(4, 0)), Complex(2, 0))
    closeC(np.emath.log10(Complex(10, 0)), Complex(1, 0))
    closeC(np.emath.arccos(Complex(1, 0)), Complex(0, 0))
    closeC(np.emath.arcsin(Complex(0, 0)), Complex(0, 0))
    closeC(np.emath.arctanh(Complex(0, 0)), Complex(0, 0))
  }

  test("float32 and mixed dtype paths") {
    val f = np.array(1.5f, -2.5f)
    val fl: NDArray[Float] = np.floor(f)
    assertEquals(fl.toList, List(1f, -3f))
    val s: NDArray[Float] = np.sin(np.array(0f))
    assertEquals(s.toList, List(0f))
    assertEquals(np.isnan(np.array(Float.NaN, 1f)).toList, List(true, false))
    val at2: NDArray[Float] = np.arctan2(np.array(1f), np.array(1f))
    assertEqualsDouble(at2.item.toDouble, math.Pi / 4, 1e-7)
    val hv: NDArray[Double] = np.heaviside(np.array(-1, 0, 1), np.array(0.5))
    assertEquals(hv.toList, List(0.0, 0.5, 1.0))
    assertEquals(np.clip(np.arange(5.0), 1, 3).toList, List(1.0, 1.0, 2.0, 3.0, 3.0))
    val a = np.zeros(3)
    np.add.at(a, Seq(1, 1), 1)
    assertEquals(a.toList, List(0.0, 2.0, 0.0))
    assertEquals(np.logical_not(np.array(0, 3)).toList, List(true, false))
    assertEquals(np.negative(np.array(1.5f)).toList, List(-1.5f))
    assertEquals(np.square(np.array(3f)).toList, List(9f))
    assertEquals(np.sign(np.array(-2f)).toList, List(-1f))
    assertEquals(np.absolute(np.array(-2f, 3f)).toList, List(2f, 3f))
    assertEquals(np.fabs(np.array(-2f)).toList, List(2f))
    assertEquals(np.reciprocal(np.array(4f)).toList, List(0.25f))
    assertEquals(np.conj(np.array(1f)).toList, List(1f))
    val sb: NDArray[Boolean] = np.signbit(np.array(-1f))
    assertEquals(sb.toList, List(true))
    val rt: NDArray[Float] = np.rint(np.array(2.5f))
    assertEquals(rt.toList, List(2f))
    assertEquals(np.nan_to_num(np.array(Float.NegativeInfinity)).toList, List(-Float.MaxValue))
    assertEquals(np.remainder(np.array(-7f), np.array(2f)).toList, List(1f))
    assertEquals(np.floor_divide(np.array(-7f), np.array(2f)).toList, List(-4f))
    assertEquals(np.copysign(np.array(2f), np.array(-1f)).toList, List(-2f))
    assertEquals(np.maximum(np.array(1f, Float.NaN), np.array(2f, 0f)).toList.map(_.isNaN), List(false, true))
    val u: NDArray[Float] = np.unwrap(np.array(0f, 3.5f))
    assertEqualsDouble(u.toList(1).toDouble, 3.5 - 2 * math.Pi, 1e-6)
    val (m, e) = np.frexp(np.array(3))
    assertEquals((m.toList, e.toList), (List(0.75), List(2)))
    val (fr, ip) = np.modf(np.array(1.5f))
    assertEquals((fr.toList, ip.toList), (List(0.5f), List(1f)))
    assertEquals(np.ldexp(np.array(1f), np.array(3L)).toList, List(8f))
    val ang: NDArray[Float] = np.angle(np.array(-1f))
    assertEqualsDouble(ang.item.toDouble, math.Pi, 1e-6)
  }

  test("ufunc errors and edge cases") {
    intercept[IllegalArgumentException](np.add.reduceat(np.array(1.0).reshape(), Seq(0)))
    intercept[IllegalArgumentException](np.add(np.array(1.0, 2.0), np.array(1.0, 2.0, 3.0)))
    assertEquals(np.add.reduce(np.array(5.0).reshape()), 5.0)
    assertEquals(np.add(np.zeros(0), 1.0).shape, Seq(0))
    assertEquals(np.divmod(np.array(7), 2)._1.toList, List(3))
    intercept[IllegalArgumentException](np.frexp(np.array(Complex(1, 0)).asInstanceOf[NDArray[Double]]))
    intercept[IllegalArgumentException](np.emath.sqrt(np.array("a")))
  }

  test("scalar overloads") {
    val x = 0.3
    close(
      Seq(np.exp(x), np.exp2(x), np.expm1(x), np.log(x), np.log2(x), np.log10(x), np.log1p(x), np.sin(x), np.cos(x),
        np.tan(x), np.arcsin(x), np.arccos(x), np.arctan(x), np.sinh(x), np.cosh(x), np.tanh(x), np.arcsinh(x),
        np.arccosh(1 + x), np.arctanh(x), np.cbrt(x), np.square(x), np.reciprocal(x), np.positive(x), np.fabs(-x),
        np.absolute(-x), np.fix(-x), np.sinc(x), np.i0(x), np.real(x), np.imag(x), np.angle(-x), np.asinh(x),
        np.acosh(1 + x), np.atanh(x), np.asin(x), np.atan(x)),
      Seq(math.exp(x), math.pow(2, x), math.expm1(x), math.log(x), math.log(x) / math.log(2), math.log10(x),
        math.log1p(x), math.sin(x), math.cos(x), math.tan(x), math.asin(x), math.acos(x), math.atan(x), math.sinh(x),
        math.cosh(x), math.tanh(x), 0.29567304756342244, 0.7564329108569596, 0.30951960420311175, math.cbrt(x),
        x * x, 1 / x, x, x, x, -0.0, math.sin(math.Pi * x) / (math.Pi * x), 1.0226268793515974, x, 0.0, math.Pi,
        0.29567304756342244, 0.7564329108569596, 0.30951960420311175, math.asin(x), math.atan(x)),
      1e-14
    )
    val z = Complex(0.3, 0.4)
    for (o, e) <- Seq(
        np.exp(z) -> CMath.exp(z), np.exp2(z) -> CMath.exp2(z), np.expm1(z) -> CMath.expm1(z), np.log(z) -> CMath.log(z),
        np.log2(z) -> CMath.log2(z), np.log10(z) -> CMath.log10(z), np.log1p(z) -> CMath.log1p(z),
        np.sin(z) -> CMath.sin(z), np.cos(z) -> CMath.cos(z), np.tan(z) -> CMath.tan(z), np.arcsin(z) -> CMath.asin(z),
        np.arccos(z) -> CMath.acos(z), np.arctan(z) -> CMath.atan(z), np.sinh(z) -> CMath.sinh(z),
        np.cosh(z) -> CMath.cosh(z), np.tanh(z) -> CMath.tanh(z), np.arcsinh(z) -> CMath.asinh(z),
        np.arccosh(z) -> CMath.acosh(z), np.arctanh(z) -> CMath.atanh(z), np.sqrt(z) -> CMath.sqrt(z),
        np.conjugate(z) -> z.conj
      )
    do closeC(o, e)
    // reference values (Python cmath) for the complex kernels
    closeC(CMath.log1p(z), Complex(0.30759281954511675, 0.2984989315861793))
    closeC(CMath.asinh(z), Complex(0.3189624333048183, 0.3903162045220237))
    closeC(CMath.acosh(z), Complex(0.4051123371780309, 1.2901667645030908))
    closeC(CMath.atanh(z), Complex(0.2614921387956719, 0.4088225229163512))
    closeC(CMath.tan(z), Complex(0.261073681841663, 0.41063347084255636))
    closeC(CMath.cosh(Complex(0, 1)), Complex(math.cos(1), 0))
    closeC(CMath.sinh(Complex(0, 1)), Complex(0, math.sin(1)))
    closeC(CMath.cos(Complex(0, 1)), Complex(math.cosh(1), 0))
    closeC(CMath.pow(Complex(1, 1), Complex(3, 0)), Complex(-2, 2))
    closeC(CMath.pow(Complex(1, 1), Complex(-2, 0)), Complex(0, -0.5))
    closeC(CMath.pow(Complex(1, 1), Complex(1, 0)), Complex(1, 1))
    closeC(CMath.pow(Complex(1, 1), Complex(0.5, 0)), Complex(1.09868411346781, 0.45508986056222733))
    closeC(CMath.pow(Complex(0, 0), Complex(2, 0)), Complex(0, 0))
    assert(CMath.pow(Complex(0, 0), Complex(-1, 0)).isNaN)
    assertEquals(CMath.pow(Complex(5, 5), Complex(0, 0)), Complex(1, 0))
    assert(np.isnan(Complex(nan, 0)) && np.isinf(Complex(inf, 0)) && np.isfinite(Complex(1, 1)))
    assertEquals(np.negative(2.0), -2.0)
    assertEquals(np.sign(-3.0), -1.0)
  }
