package com.github.kmizu.numscala

/** Shared assertions for the math test suites. */
trait MathTestUtil:
  self: munit.FunSuite =>
  def close(obtained: Seq[Double], expected: Seq[Double], tol: Double = 1e-12)(using munit.Location): Unit =
    assertEquals(obtained.length, expected.length, s"lengths differ: $obtained vs $expected")
    obtained.zip(expected).foreach { (o, e) =>
      if e.isNaN then assert(o.isNaN, s"expected NaN but got $o in $obtained")
      else if e.isInfinite then assertEquals(o, e)
      else assertEqualsDouble(o, e, tol * math.max(1.0, math.abs(e)), s"in $obtained vs $expected")
    }
  def closeC(o: Complex, e: Complex, tol: Double = 1e-12)(using munit.Location): Unit =
    close(Seq(o.re, o.im), Seq(e.re, e.im), tol)
  def same(o: Double, e: Double)(using munit.Location): Unit =
    assertEquals(java.lang.Double.doubleToRawLongBits(o), java.lang.Double.doubleToRawLongBits(e), s"$o vs $e")

class UfuncSuite extends munit.FunSuite with MathTestUtil:

  test("add: broadcasting, dtypes, scalar forms") {
    val a = np.array(1.0, 2.0, 3.0)
    val b = np.array(Seq(Seq(10.0), Seq(20.0)))
    val r = np.add(a, b)
    assertEquals(r.shape, Seq(2, 3))
    assertEquals(r.toList, List(11.0, 12.0, 13.0, 21.0, 22.0, 23.0))
    val mixed: NDArray[Double] = np.add(np.array(1, 2), np.array(0.5, 0.25))
    assertEquals(mixed.toList, List(1.5, 2.25))
    val ii: NDArray[Int] = np.add(np.array(1, 2), 3)
    assertEquals(ii.toList, List(4, 5))
    assertEquals(ii.dtype.name, "int32")
    // a Scala Long is a weak Python int (NEP 50): int32 + 3L stays int32
    val ll: NDArray[Int] = np.add(np.array(1, 2), 3L)
    assertEquals(ll.toList, List(4, 5))
    val left: NDArray[Int] = np.subtract(10, np.array(1, 2, 3))
    assertEquals(left.toList, List(9, 8, 7))
    val s: Double = np.add(1.0, 2.0)
    assertEquals(s, 3.0)
    val c: NDArray[Complex] = np.add(np.array(Complex(1, 1)), 1.0)
    assertEquals(c.toList, List(Complex(2, 1)))
    val f32: NDArray[Float] = np.add(np.array(1.5f), np.array(2f))
    assertEquals(f32.toList, List(3.5f))
  }

  test("bool and string add/multiply") {
    val t = np.array(true, false, false)
    val u = np.array(true, true, false)
    val o: NDArray[Boolean] = np.add(t, u)
    assertEquals(o.toList, List(true, true, false))
    assertEquals(np.multiply(t, u).toList, List(true, false, false))
    val s = np.add(np.array("a", "b"), np.array("x", "y"))
    assertEquals(s.toList, List("ax", "by"))
    intercept[IllegalArgumentException](np.multiply(np.array("a"), np.array("b")))
  }

  test("int fast paths and wrap-around") {
    assertEquals(np.multiply(np.array(Int.MaxValue), 2).toList, List(-2))
    assertEquals(np.subtract(np.array(5L, 7L), np.array(1L, 10L)).toList, List(4L, -3L))
    val b = np.array(Seq(1.toByte, 100.toByte))
    assertEquals(np.add(b, b).toList, List(2.toByte, -56.toByte))
  }

  test("divide and true_divide") {
    val r: NDArray[Double] = np.divide(np.array(1, 2, 0, -1), np.array(2, 0, 0, 0))
    close(r.toSeq, Seq(0.5, Double.PositiveInfinity, Double.NaN, Double.NegativeInfinity))
    assert(np.true_divide eq np.divide)
    val f: NDArray[Float] = np.divide(np.array(1f), np.array(4f))
    assertEquals(f.toList, List(0.25f))
    val c = np.divide(np.array(Complex(1, 1)), np.array(Complex(0, 1)))
    closeC(c.item, Complex(1, -1))
  }

  test("floor_divide, remainder, fmod, divmod") {
    assertEquals(np.floor_divide(np.array(-7, 7, 5), np.array(2, 2, 0)).toList, List(-4, 3, 0))
    close(np.floor_divide(np.array(1.0, 1.0, -1.0, 0.0, -7.5), np.array(0.1, 0.0, 0.0, 0.0, 2.0)).toSeq,
      Seq(9.0, Double.PositiveInfinity, Double.NegativeInfinity, Double.NaN, -4.0))
    assertEquals(np.remainder(np.array(-7, 7, 5), np.array(3, -3, 0)).toList, List(2, -2, 0))
    close(np.remainder(np.array(-7.5, 7.0, 1.0), np.array(2.0, -2.0, 0.0)).toSeq, Seq(0.5, -1.0, Double.NaN))
    same(np.remainder(np.array(-4.0), 2.0).item, 0.0)
    same(np.remainder(np.array(4.0), -2.0).item, -0.0)
    assert(np.mod eq np.remainder)
    assertEquals(np.fmod(np.array(-7, 7, 3), np.array(3, -3, 0)).toList, List(-1, 1, 0))
    close(np.fmod(np.array(-7.5), 2.0).toSeq, Seq(-1.5))
    val (q, m) = np.divmod(np.array(7, -7), np.array(2, 2))
    assertEquals(q.toList, List(3, -4))
    assertEquals(m.toList, List(1, 1))
    val (q2, m2) = np.divmod(np.array(5.5), 2.0)
    assertEquals((q2.item, m2.item), (2.0, 1.5))
    assertEquals(np.divmod(-7.0, 2.0), (-4.0, 1.0))
    assertEquals(np.divmod(-7L, 2L), (-4L, 1L))
    intercept[IllegalArgumentException](np.floor_divide(np.array(Complex(1, 0)), np.array(Complex(1, 0))))
  }

  test("power and float_power") {
    assertEquals(np.power(np.array(2, 3, -2), np.array(3, 2, 3)).toList, List(8, 9, -8))
    intercept[ArithmeticException](np.power(np.array(2), -1))
    close(np.power(np.array(1.0, -1.0, 2.0, -8.0), np.array(Double.NaN, Double.PositiveInfinity, 0.5, 1.0 / 3)).toSeq,
      Seq(1.0, 1.0, math.sqrt(2.0), Double.NaN))
    closeC(np.power(np.array(Complex(0, 1)), 2.0).item, Complex(-1, 0))
    val fp: NDArray[Double] = np.float_power(np.array(2, 3), np.array(2, -1))
    close(fp.toSeq, Seq(4.0, 1.0 / 3))
    val fp32: NDArray[Double] = np.float_power(np.array(2f), np.array(0.5f))
    close(fp32.toSeq, Seq(math.sqrt(2.0)))
    val fc: NDArray[Complex] = np.float_power(np.array(Complex(0, 1)), np.array(2.0))
    closeC(fc.item, Complex(-1, 0))
    assert(np.pow eq np.power)
  }

  test("maximum / minimum / fmax / fmin") {
    val a = np.array(1.0, Double.NaN, 3.0, Double.NaN)
    val b = np.array(2.0, 2.0, Double.NaN, Double.NaN)
    close(np.maximum(a, b).toSeq, Seq(2.0, Double.NaN, Double.NaN, Double.NaN))
    close(np.minimum(a, b).toSeq, Seq(1.0, Double.NaN, Double.NaN, Double.NaN))
    close(np.fmax(a, b).toSeq, Seq(2.0, 2.0, 3.0, Double.NaN))
    close(np.fmin(a, b).toSeq, Seq(1.0, 2.0, 3.0, Double.NaN))
    assertEquals(np.maximum(np.array(1, 5), np.array(3, 2)).toList, List(3, 5))
    val mixed: NDArray[Double] = np.maximum(np.array(1, 5), 2.5)
    assertEquals(mixed.toList, List(2.5, 5.0))
    val c = np.maximum(np.array(Complex(1, 2), Complex(1, 3)), np.array(Complex(1, 3), Complex(0, 9)))
    assertEquals(c.toList, List(Complex(1, 3), Complex(1, 3)))
    assertEquals(np.maximum(np.array(true, false), np.array(false, false)).toList, List(true, false))
    // signed zeros: NumPy 2 returns the second operand on ties for maximum/minimum, the first for fmax/fmin
    val nz = np.array(-0.0, 0.0)
    val pz = np.array(0.0, -0.0)
    assertEquals(np.maximum(nz, pz).toList.map(MathK.signbit), List(false, true))
    assertEquals(np.minimum(nz, pz).toList.map(MathK.signbit), List(false, true))
    assertEquals(np.fmax(nz, pz).toList.map(MathK.signbit), List(true, false))
    assertEquals(np.fmin(nz, pz).toList.map(MathK.signbit), List(true, false))
    assert(!MathK.signbit(np.maximum.reduce(np.array(-0.0, 0.0))))
  }

  test("bitwise ops and shifts") {
    assertEquals(np.bitwise_and(np.array(12, 12), np.array(10, 7)).toList, List(8, 4))
    assertEquals(np.bitwise_or(np.array(12), 10).toList, List(14))
    assertEquals(np.bitwise_xor(np.array(12), 10).toList, List(6))
    assertEquals(np.bitwise_and(np.array(true, true), np.array(true, false)).toList, List(true, false))
    assertEquals(np.left_shift(np.array(1, 1, 1), np.array(3, 31, 32)).toList, List(8, Int.MinValue, 0))
    assertEquals(np.right_shift(np.array(-8L, 8L, -1L), np.array(1L, 64L, 70L)).toList, List(-4L, 0L, -1L))
    assert(np.bitwise_left_shift eq np.left_shift)
    assert(np.bitwise_right_shift eq np.right_shift)
    intercept[IllegalArgumentException](np.bitwise_and(np.array(1.0), np.array(2.0)))
  }

  test("gcd / lcm") {
    assertEquals(np.gcd(np.array(12, -18, 0, 7), np.array(8, 12, 0, 0)).toList, List(4, 6, 0, 7))
    assertEquals(np.lcm(np.array(4, 6, -4, 0), np.array(6, 4, 6, 5)).toList, List(12, 12, 12, 0))
    val g: Int = np.gcd.reduce(np.array(12, 18, 24))
    assertEquals(g, 6)
    assertEquals(np.lcm.reduce(np.array(2, 3, 4)), 12)
    intercept[IllegalArgumentException](np.gcd(np.array(1.0), np.array(2.0)))
  }

  test("comparisons and logical ufuncs") {
    val a = np.array(1.0, 2.0, Double.NaN)
    val b = np.array(2.0, 2.0, Double.NaN)
    assertEquals(np.less(a, b).toList, List(true, false, false))
    assertEquals(np.less_equal(a, b).toList, List(true, true, false))
    assertEquals(np.greater(a, b).toList, List(false, false, false))
    assertEquals(np.greater_equal(a, b).toList, List(false, true, false))
    assertEquals(np.equal(a, b).toList, List(false, true, false))
    assertEquals(np.not_equal(a, b).toList, List(true, false, true))
    assertEquals(np.equal(np.array(1, 2), 2.0).toList, List(false, true))
    assertEquals(np.less(np.array(Complex(1, 2)), np.array(Complex(1, 3))).toList, List(true))
    assertEquals(np.equal(np.array("a", "b"), np.array("a", "c")).toList, List(true, false))
    val b1: Boolean = np.less(1, 2)
    assert(b1)
    assertEquals(np.logical_and(np.array(1.0, 0.0, Double.NaN), np.array(2, 3, 1)).toList, List(true, false, true))
    assertEquals(np.logical_or(np.array(0, 0), np.array(0, 5)).toList, List(false, true))
    assertEquals(np.logical_xor(np.array(true, true), np.array(true, false)).toList, List(false, true))
    assertEquals(np.logical_and.reduce(np.array(1.0, 2.0, 3.0)), true)
    assertEquals(np.logical_or.reduce(np.zeros(0)), false)
    assertEquals(np.logical_and.reduce(np.zeros(0)), true)
    assertEquals(np.logical_or.accumulate(np.array(false, true, false)).toList, List(false, true, true))
    assertEquals(np.equal.outer(np.array(1, 2), np.array(2, 1)).toList, List(false, true, true, false))
  }

  test("float binary functions") {
    close(np.arctan2(np.array(1.0, -1.0, 0.0), np.array(-1.0, -1.0, -1.0)).toSeq,
      Seq(3 * math.Pi / 4, -3 * math.Pi / 4, math.Pi))
    assert(np.atan2 eq np.arctan2)
    val h: NDArray[Double] = np.hypot(np.array(3, 5), np.array(4, 12))
    assertEquals(h.toList, List(5.0, 13.0))
    close(np.logaddexp(np.array(1.0, Double.NegativeInfinity, 1000.0), np.array(2.0, Double.NegativeInfinity, 1000.0)).toSeq,
      Seq(2.3132616875182226, Double.NegativeInfinity, 1000.6931471805599))
    close(np.logaddexp2(np.array(1.0, 3.0), np.array(2.0, 3.0)).toSeq, Seq(2.584962500721156, 4.0))
    close(np.copysign(np.array(1.0, 2.0, 3.0), np.array(-0.0, 1.0, -5.0)).toSeq, Seq(-1.0, 2.0, -3.0))
    assertEquals(np.nextafter(np.array(1.0), 2.0).item, 1.0 + math.ulp(1.0))
    assertEquals(np.nextafter(np.array(1.0f), np.array(2.0f)).item, 1.0f + math.ulp(1.0f))
    assertEquals(np.nextafter(np.array(0.0), -1.0).item, -java.lang.Double.MIN_VALUE)
    close(np.heaviside(np.array(-1.5, 0.0, 2.0, Double.NaN), 0.5).toSeq, Seq(0.0, 0.5, 1.0, Double.NaN))
    intercept[IllegalArgumentException](np.arctan2(np.array(Complex(1, 0)), np.array(Complex(1, 0))))
  }

  test("reduce: full and along axes") {
    val l: Long = np.add.reduce(np.array(1, 2, 3))
    assertEquals(l, 6L)
    assertEquals(np.add.reduce(np.array(true, true, false)), 2L)
    assertEquals(np.multiply.reduce(np.array(1.0, 2.0, 3.0, 4.0)), 24.0)
    assertEquals(np.subtract.reduce(np.array(10, 1, 2)), 7)
    assertEquals(np.divide.reduce(np.array(8, 2, 2)), 2.0)
    val m = np.arange(6.0).reshape(2, 3)
    assertEquals(np.add.reduce(m, 0).toList, List(3.0, 5.0, 7.0))
    assertEquals(np.add.reduce(m, -1).toList, List(3.0, 12.0))
    val kd = np.add.reduce(m, 1, keepdims = true)
    assertEquals(kd.shape, Seq(2, 1))
    assertEquals(np.add.reduce(m, Seq(0, 1)).item, 15.0)
    assertEquals(np.maximum.reduce(m, 0).toList, List(3.0, 4.0, 5.0))
    assertEquals(np.subtract.reduce(m, 1).toList, List(-3.0, -6.0))
    intercept[IllegalArgumentException](np.subtract.reduce(m, Seq(0, 1)))
    intercept[IllegalArgumentException](np.subtract.reduce(m))
    assertEquals(np.add.reduce(np.zeros(0)), 0.0)
    assertEquals(np.multiply.reduce(np.zeros(0)), 1.0)
    assertEquals(np.bitwise_and.reduce(np.zeros[Int](0)), -1)
    assertEquals(np.bitwise_or.reduce(np.array(1, 2, 4)), 7)
    val e = intercept[IllegalArgumentException](np.maximum.reduce(np.zeros(0)))
    assert(e.getMessage.contains("zero-size array to reduction operation maximum which has no identity"))
    assertEquals(np.maximum.reduce(np.zeros(0, 3), 1).shape, Seq(0))
    intercept[IllegalArgumentException](np.maximum.reduce(np.zeros(0, 3), 0))
    assertEquals(np.maximum.reduce(np.array(1.0, Double.NaN, 3.0)).isNaN, true)
    assertEquals(np.fmax.reduce(np.array(1.0, Double.NaN, 3.0)), 3.0)
    assertEquals(np.hypot.reduce(np.array(3.0, 4.0)), 5.0)
    assertEquals(np.logaddexp.reduce(np.zeros(0)), Double.NegativeInfinity)
    // pairwise summation like np.sum
    val big = np.full(Seq(1000001), 0.1)
    assertEquals(np.add.reduce(big), big.sum())
  }

  test("accumulate") {
    val c: NDArray[Long] = np.add.accumulate(np.array(1, 2, 3))
    assertEquals(c.toList, List(1L, 3L, 6L))
    assertEquals(np.maximum.accumulate(np.array(1, 3, 2, 5, 4)).toList, List(1, 3, 3, 5, 5))
    assertEquals(np.multiply.accumulate(np.array(1.0, 2.0, 3.0)).toList, List(1.0, 2.0, 6.0))
    val m = np.arange(6).reshape(2, 3)
    assertEquals(np.add.accumulate(m).toList, List(0L, 1L, 2L, 3L, 5L, 7L))
    assertEquals(np.add.accumulate(m, axis = 1).toList, List(0L, 1L, 3L, 3L, 7L, 12L))
    assertEquals(np.subtract.accumulate(np.array(10, 1, 2)).toList, List(10, 9, 7))
    intercept[IllegalArgumentException](np.add.accumulate(np.array(1.0).reshape()))
  }

  test("outer") {
    val o = np.multiply.outer(np.array(1, 2, 3), np.array(1, 2))
    assertEquals(o.shape, Seq(3, 2))
    assertEquals(o.toList, List(1, 2, 2, 4, 3, 6))
    val s = np.subtract.outer(np.arange(4.0).reshape(2, 2), np.array(Seq(1.0)))
    assertEquals(s.shape, Seq(2, 2, 1))
    assertEquals(s.toList, List(-1.0, 0.0, 1.0, 2.0))
  }

  test("at: unbuffered in-place") {
    val a = np.array(1, 2, 3, 4)
    np.add.at(a, Seq(0, 0, 2), 1)
    assertEquals(a.toList, List(3, 2, 4, 4))
    val b = np.array(1.0, 2.0, 3.0)
    np.multiply.at(b, np.array(1, 1), np.array(3.0, 4.0))
    assertEquals(b.toList, List(1.0, 24.0, 3.0))
    val z = np.zeros(2, 2)
    np.add.at(z, (Seq(0, 0, 1), Seq(0, 0, 1)), 1.0)
    assertEquals(z.toList, List(2.0, 0.0, 0.0, 1.0))
    val r = np.zeros(2, 3)
    np.add.at(r, 1, np.array(1.0, 2.0, 3.0))
    assertEquals(r.toList, List(0.0, 0.0, 0.0, 1.0, 2.0, 3.0))
    val mx = np.array(1, 5, 2)
    np.maximum.at(mx, Seq(0, 2, 0), np.array(4, 1, 3))
    assertEquals(mx.toList, List(4, 5, 2))
    val s = np.arange(5.0)
    np.subtract.at(s, "1:3", 1.0)
    assertEquals(s.toList, List(0.0, 0.0, 1.0, 3.0, 4.0))
  }

  test("reduceat") {
    val x = np.arange(8)
    val r: NDArray[Long] = np.add.reduceat(x, Seq(0, 4, 1, 5, 2, 6, 3, 7))
    assertEquals(r.toList, List(6L, 4L, 10L, 5L, 14L, 6L, 18L, 7L))
    val m = np.arange(16.0).reshape(4, 4)
    val rm = np.add.reduceat(m, np.array(0, 3), axis = 1)
    assertEquals(rm.shape, Seq(4, 2))
    assertEquals(rm.toList, List(3.0, 3.0, 15.0, 7.0, 27.0, 11.0, 39.0, 15.0))
    assertEquals(np.multiply.reduceat(np.array(1.0, 2.0, 3.0, 4.0), Seq(0, 2)).toList, List(2.0, 12.0))
    intercept[IndexOutOfBoundsException](np.add.reduceat(x, Seq(0, 8)))
  }

  test("ufunc attributes") {
    assertEquals(np.add.name, "add")
    assertEquals(np.add.toString, "<ufunc 'add'>")
    assertEquals(np.add.nin, 2)
    assertEquals(np.add.nout, 1)
    assertEquals(np.add.identity, Some(0L))
    assertEquals(np.maximum.identity, None)
    assert(np.add.reorderable)
    assert(!np.subtract.reorderable)
  }
