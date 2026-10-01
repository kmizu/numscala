package numscala

class ReduceSuite extends munit.FunSuite:
  val nan = Double.NaN

  def assertClose(actual: Seq[Double], expected: Seq[Double], tol: Double = 1e-12)(using munit.Location): Unit =
    assertEquals(actual.length, expected.length, s"length: $actual vs $expected")
    actual.zip(expected).foreach { (x, y) =>
      if y.isNaN then assert(x.isNaN, s"expected NaN, got $x in $actual")
      else if y.isInfinite then assertEquals(x, y)
      else assert(math.abs(x - y) <= tol * math.max(1.0, math.abs(y)), s"$actual != $expected")
    }

  val b = np.array(Seq(Seq(1, 5, 2), Seq(7, 3, 9)))

  test("sum / prod dtypes and axes") {
    val s: Long = np.sum(b)
    assertEquals(s, 27L)
    val sa: NDArray[Long] = np.sum(b, axis = 0)
    assertEquals(sa.toList, List(8L, 8L, 11L))
    assertEquals(np.sum(b, axis = -1).toList, List(8L, 19L))
    assertEquals(np.sum(b, axis = Seq(0, 1)).shape, Seq())
    assertEquals(np.sum(b, axis = 1, keepdims = true).shape, Seq(2, 1))
    assertEquals(np.sum(b, keepdims = true).shape, Seq(1, 1))
    assertEquals(np.sum(b, keepdims = true).toList, List(27L))
    assertEquals(np.prod(b), 1890L)
    assertEquals(np.prod(b, axis = 1).toList, List(10L, 189L))
    val bools = np.array(true, false, true)
    assertEquals(np.sum(bools), 2L)
    assertEquals(np.sum(np.array(1.5f, 2.5f)), 4.0f)
    assertEquals(np.sum(np.zeros(0)), 0.0)
    assertEquals(np.prod(np.zeros(0)), 1.0)
    assertEquals(np.sum(np.array(Complex(1, 2), Complex(3, -1))), Complex(4, 1))
    intercept[IndexOutOfBoundsException](np.sum(b, axis = 2))
    intercept[IllegalArgumentException](np.sum(b, axis = Seq(0, 0)))
  }

  test("mean / var / std") {
    val m: Double = np.mean(b)
    assertEqualsDouble(m, 4.5, 1e-12)
    assertClose(np.mean(b, axis = 1).toSeq, Seq(2.6666666666666665, 6.333333333333333))
    assertEqualsDouble(np.std(b), 2.8136571693556887, 1e-12)
    assertClose(np.`var`(b, axis = 0, ddof = 1).toSeq, Seq(18.0, 2.0, 24.5))
    assertClose(np.std(b, axis = 0, ddof = 1).toSeq, Seq(18.0, 2.0, 24.5).map(math.sqrt))
    assertEqualsDouble(np.`var`(np.array(1.0, 2.0, 3.0, 4.0)), 1.25, 1e-12)
    assert(np.mean(np.zeros(0)).isNaN)
    val f: Float = np.mean(np.array(1f, 2f))
    assertEquals(f, 1.5f)
    val c = np.array(Complex(1, 1), Complex(3, -1))
    assertEquals(np.mean(c), Complex(2, 0))
    val cv: Double = np.`var`(c)
    assertEqualsDouble(cv, 2.0, 1e-12)
    assert(np.`var`(np.ones(1), axis = 0, ddof = 1).item.isNaN)
    assertEquals(np.mean(b, axis = 0, keepdims = true).shape, Seq(1, 3))
  }

  test("max / min / amax / amin / ptp") {
    assertEquals(np.max(b), 9)
    assertEquals(np.min(b), 1)
    assertEquals(np.max(b, axis = 0).toList, List(7, 5, 9))
    assertEquals(np.amin(b, axis = 1).toList, List(1, 3))
    assertEquals(np.amax(b), 9)
    assertEquals(np.amin(b), 1)
    assertEquals(np.amax(b, axis = 1, keepdims = true).shape, Seq(2, 1))
    assert(np.max(np.array(1.0, nan, 3.0)).isNaN)
    assertEquals(np.ptp(b), 8)
    assertEquals(np.ptp(b, axis = 1).toList, List(4, 6))
    intercept[IllegalArgumentException](np.max(np.zeros(0)))
    assertEquals(np.max(np.zeros(0, 3), axis = 1).shape, Seq(0))
    intercept[IllegalArgumentException](np.max(np.zeros(0, 3), axis = 0))
  }

  test("argmax / argmin") {
    assertEquals(np.argmax(b), 5)
    assertEquals(np.argmin(b), 0)
    assertEquals(np.argmax(b, axis = 0).toList, List(1, 0, 1))
    assertEquals(np.argmin(b, axis = 1).toList, List(0, 1))
    assertEquals(np.argmax(b, axis = 1, keepdims = true).shape, Seq(2, 1))
    assertEquals(np.argmax(b, keepdims = true).shape, Seq(1, 1))
    assertEquals(np.argmax(np.array(1.0, nan, 3.0)), 1)
    assertEquals(np.argmin(np.array(2.0, 1.0, 1.0)), 1)
    intercept[IllegalArgumentException](np.argmax(np.zeros(0)))
  }

  test("all / any / count_nonzero") {
    val m = np.array(Seq(Seq(true, false), Seq(true, true)))
    assertEquals(np.all(m), false)
    assertEquals(np.any(m), true)
    assertEquals(np.all(m, axis = 0).toList, List(true, false))
    assertEquals(np.any(m, axis = 1).toList, List(true, true))
    assertEquals(np.all(np.zeros(0)), true)
    assertEquals(np.any(np.zeros(0)), false)
    assertEquals(np.all(np.array(1.0, nan)), true)
    assertEquals(np.count_nonzero(b - np.array(1, 3, 2)), 3L)
    assertEquals(np.count_nonzero(m, axis = 0).toList, List(2L, 1L))
    assertEquals(np.count_nonzero(m, axis = 1, keepdims = true).shape, Seq(2, 1))
  }

  test("cumsum / cumprod / cumulative_*") {
    assertEquals(np.cumsum(b).toList, List(1L, 6L, 8L, 15L, 18L, 27L))
    assertEquals(np.cumprod(b, axis = 1).toList, List(1L, 5L, 10L, 7L, 21L, 189L))
    assertEquals(np.cumsum(b, axis = 0).toList, List(1L, 5L, 2L, 8L, 8L, 11L))
    assertEquals(np.cumsum(np.array(1.0, 2.0)).toList, List(1.0, 3.0))
    assertEquals(np.cumulative_sum(np.array(1, 2, 3), include_initial = true).toList, List(0L, 1L, 3L, 6L))
    val c2 = np.cumulative_sum(np.ones(2, 2), axis = 0, include_initial = true)
    assertEquals(c2.shape, Seq(3, 2))
    assertEquals(c2.toList, List(0.0, 0.0, 1.0, 1.0, 2.0, 2.0))
    assertEquals(np.cumulative_prod(np.array(2, 3, 4)).toList, List(2L, 6L, 24L))
    assertEquals(np.cumulative_prod(np.array(2.0, 3.0), include_initial = true).toList, List(1.0, 2.0, 6.0))
    intercept[IllegalArgumentException](np.cumulative_sum(np.ones(2, 2)))
    assertEquals(np.cumulative_sum(np.array(3.0)).toList, List(3.0))
    assertEquals(np.cumsum(np.zeros(0)).shape, Seq(0))
  }

  test("average with weights and returned") {
    assertClose(np.average(b, axis = 1, weights = np.array(1, 2, 3)).toSeq, Seq(2.8333333333333335, 6.666666666666667))
    assertEqualsDouble(np.average(b, weights = np.array(Seq(Seq(1, 2, 3), Seq(4, 5, 6)))).item, 5.428571428571429, 1e-12)
    val (avg, scl) = np.average(b, axis = 0, weights = np.array(1.0, 3.0), returned = true)
    assertClose(avg.toSeq, Seq(5.5, 3.5, 7.25))
    assertEquals(scl.toList, List(4.0, 4.0, 4.0))
    val (avg2, scl2) = np.average(b, axis = 1, weights = null, returned = true)
    assertClose(avg2.toSeq, Seq(8.0 / 3, 19.0 / 3))
    assertEquals(scl2.toList, List(3.0, 3.0))
    assertEqualsDouble(np.average(b), 4.5, 1e-12)
    intercept[ArithmeticException](np.average(b, axis = 1, weights = np.array(1.0, -1.0, 0.0)))
    intercept[IllegalArgumentException](np.average(b, weights = np.array(1.0, 2.0, 3.0)))
    intercept[IllegalArgumentException](np.average(b, axis = 1, weights = np.array(1.0, 2.0)))
    assertEquals(np.average(b, axis = 1, weights = np.array(1, 2, 3), keepdims = true).shape, Seq(2, 1))
  }

  test("nan reductions") {
    val a = np.arange(12.0).reshape(3, 4)
    a(1, 2) = nan
    assertEquals(np.nansum(a), 60.0)
    assertEquals(np.nansum(a, axis = 0).toList, List(12.0, 15.0, 12.0, 21.0))
    assertClose(np.nanmean(a, axis = 1).toSeq, Seq(1.5, 5.333333333333333, 9.5))
    assertClose(np.nanvar(a, axis = 1, ddof = 1).toSeq, Seq(1.6666666666666667, 2.3333333333333335, 1.6666666666666667))
    assertEqualsDouble(np.nanstd(a), 3.6021114102107186, 1e-12)
    assertEquals(np.nanmax(a, axis = 0).toList, List(8.0, 9.0, 10.0, 11.0))
    assertEquals(np.nanargmin(a, axis = 1).toList, List(0, 0, 0))
    assertEquals(np.nancumsum(a, axis = 1).toList, List(0.0, 1, 3, 6, 4, 9, 9, 16, 8, 17, 27, 38))
    assertEquals(np.nancumprod(np.array(1.0, nan, 2.0, 3.0)).toList, List(1.0, 1.0, 2.0, 6.0))
    assertEquals(np.nanprod(np.array(2.0, nan, 3.0)), 6.0)
    assertEquals(np.nanprod(np.array(Seq(Seq(2.0, nan), Seq(3.0, 4.0))), axis = 1).toList, List(2.0, 12.0))
    assertEquals(np.nanmin(np.array(nan, 2.0, 1.0)), 1.0)
    assertEquals(np.nanmax(np.array(nan, 2.0, 1.0)), 2.0)
    assert(np.nanmax(np.array(nan, nan)).isNaN)
    assert(np.nanmean(np.array(nan, nan)).isNaN)
    assertEquals(np.nansum(np.array(nan, nan)), 0.0)
    assertEquals(np.nanargmax(np.array(nan, 2.0, 5.0)), 2)
    assertEquals(np.nanargmin(np.array(nan, 2.0, 5.0)), 1)
    assertEquals(np.nanargmax(np.array(nan, 2.0, 5.0), keepdims = true).shape, Seq(1))
    intercept[IllegalArgumentException](np.nanargmax(np.array(nan, nan)))
    intercept[IllegalArgumentException](np.nanmin(np.zeros(0)))
    assertEquals(np.nansum(b), 27L)
    assertEqualsDouble(np.nanstd(np.array(1.0, nan, 3.0), axis = 0).item, 1.0, 1e-12)
    assertEqualsDouble(np.nanvar(np.array(1.0, nan, 3.0)), 1.0, 1e-12)
    assertEquals(np.nanmean(np.array(1.0, nan, 3.0)), 2.0)
    assertEquals(np.nanmin(a, axis = 1).toList, List(0.0, 4.0, 8.0))
    assertEquals(np.nancumsum(np.array(1.0, nan, 2.0)).toList, List(1.0, 1.0, 3.0))
  }

  test("quantile: all 13 methods") {
    val x = np.array(3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0)
    val qs = Seq(0.0, 0.1, 0.25, 0.4, 0.5, 0.75, 0.9, 1.0)
    val expected = Map(
      "inverted_cdf" -> Seq(1.0, 1.0, 1.0, 3.0, 3.0, 5.0, 9.0, 9.0),
      "averaged_inverted_cdf" -> Seq(1.0, 1.0, 1.5, 3.0, 3.5, 5.5, 9.0, 9.0),
      "closest_observation" -> Seq(1.0, 1.0, 1.0, 2.0, 3.0, 5.0, 6.0, 9.0),
      "interpolated_inverted_cdf" -> Seq(1.0, 1.0, 1.0, 2.2, 3.0, 5.0, 6.6000000000000005, 9.0),
      "hazen" -> Seq(1.0, 1.0, 1.5, 2.7, 3.5, 5.5, 8.100000000000001, 9.0),
      "weibull" -> Seq(1.0, 1.0, 1.25, 2.6, 3.5, 5.75, 9.0, 9.0),
      "linear" -> Seq(1.0, 1.0, 1.75, 2.8000000000000003, 3.5, 5.25, 6.8999999999999995, 9.0),
      "median_unbiased" -> Seq(1.0, 1.0, 1.4166666666666665, 2.666666666666667, 3.5, 5.583333333333333, 8.500000000000002, 9.0),
      "normal_unbiased" -> Seq(1.0, 1.0, 1.4375, 2.6750000000000003, 3.5, 5.5625, 8.399999999999999, 9.0),
      "lower" -> Seq(1.0, 1.0, 1.0, 2.0, 3.0, 5.0, 6.0, 9.0),
      "higher" -> Seq(1.0, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 9.0),
      "nearest" -> Seq(1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0),
      "midpoint" -> Seq(1.0, 1.0, 1.5, 2.5, 3.5, 5.5, 7.5, 9.0)
    )
    for (m, e) <- expected do
      assertClose(np.quantile(x, qs, method = m).toSeq, e, 1e-15)
      assertEqualsDouble(np.quantile(x, 0.4, m), e(3), 1e-15)
      assertClose(np.percentile(x, qs.map(_ * 100), method = m).toSeq, e, 1e-12)
    intercept[IllegalArgumentException](np.quantile(x, 0.5, "bogus"))
    intercept[IllegalArgumentException](np.quantile(x, 1.5))
    intercept[IllegalArgumentException](np.percentile(x, 101.0))
  }

  test("quantile / percentile / median shapes") {
    val p = np.percentile(b, Seq(25.0, 50.0), axis = 1)
    assertEquals(p.shape, Seq(2, 2))
    assertEquals(p.toList, List(1.5, 5.0, 2.0, 7.0))
    assertEquals(np.percentile(b, 50), 4.0)
    assertEquals(np.median(b, axis = 0).toList, List(4.0, 4.0, 5.5))
    assertEquals(np.median(np.array(3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0)), 3.5)
    assertEquals(np.median(np.array(3, 1, 2)), 2.0)
    assert(np.median(np.array(1.0, nan, 3.0)).isNaN)
    assert(np.median(np.zeros(0)).isNaN)
    assertEquals(np.median(b, axis = 1, keepdims = true).shape, Seq(2, 1))
    val q = np.quantile(np.arange(24.0).reshape(2, 3, 4), np.array(0.5, 1.0), axis = Seq(0, 2), keepdims = true)
    assertEquals(q.shape, Seq(2, 1, 3, 1))
    assertEquals(q.toList, List(7.5, 11.5, 15.5, 15.0, 19.0, 23.0))
    assertEquals(np.quantile(np.array(1.0, 2.0), 0.5, axis = 0).shape, Seq())
    val f: Float = np.median(np.array(1f, 2f, 4f, 8f))
    assertEquals(f, 3f)
    assert(np.quantile(np.array(1.0, Double.PositiveInfinity), 1.0).isNaN) // NumPy's lerp quirk
    assertEquals(np.median(np.array(1.0, Double.PositiveInfinity)), Double.PositiveInfinity)
    intercept[IndexOutOfBoundsException](np.quantile(np.zeros(0), 0.5))
  }

  test("nanmedian / nanquantile / nanpercentile") {
    val a = np.arange(12.0).reshape(3, 4)
    a(1, 2) = nan
    assertEquals(np.nanmedian(a, axis = 1).toList, List(1.5, 5.0, 9.5))
    assertEquals(np.nanmedian(np.array(nan, 1.0, 3.0)), 2.0)
    assert(np.nanmedian(np.array(nan, nan)).isNaN)
    assertEquals(np.nanquantile(np.array(nan, 1.0, 2.0, 3.0), 0.5), 2.0)
    assertEquals(np.nanquantile(np.array(nan, 1.0, 2.0, 3.0), 0.5, "lower"), 2.0)
    assertEquals(np.nanpercentile(np.array(nan, 1.0, 2.0, 3.0), 100.0), 3.0)
    assertEquals(np.nanpercentile(np.array(nan, 1.0, 2.0, 3.0), 25.0, "higher"), 2.0)
    assertEquals(np.nanquantile(a, Seq(0.0, 1.0), axis = 1).toList, List(0.0, 4.0, 8.0, 3.0, 7.0, 11.0))
    assertEquals(np.nanpercentile(a, 50.0, axis = 0).toList, List(4.0, 5.0, 6.0, 7.0))
    assert(np.nanquantile(np.array(nan, nan), 0.5).isNaN)
  }

  test("dtype rules and misc edge cases") {
    val i8 = np.arrayOf(Seq(100, 100, 100), DType.Int8)
    val s8: Long = np.sum(i8)
    assertEquals(s8, 300L)
    assertEquals(np.cumsum(i8).dtype.name, "int64")
    assertEquals(np.max(i8), 100.toByte)
    val f32 = np.array(1f, 2f, 3f, 4f)
    val q32: NDArray[Float] = np.quantile(f32, Seq(0.5))
    assertEquals(q32.toList, List(2.5f))
    val v32: Float = np.`var`(f32)
    assertEquals(v32, 1.25f)
    assertEquals(np.nanmax(b, axis = 0).toList, List(7, 5, 9))
    assertEquals(np.nanmin(b), 1)
    assertEquals(np.nanargmax(b, axis = 1).toList, List(1, 2))
    assertEquals(np.nanargmin(b), 0)
    assertEquals(np.nanmean(b, axis = 0, keepdims = true).shape, Seq(1, 3))
    assertEquals(np.nancumprod(b, axis = 0).toList, List(1L, 5L, 2L, 7L, 15L, 18L))
    assertEquals(np.nancumsum(b).toList, List(1L, 6L, 8L, 15L, 18L, 27L))
    assertEquals(np.max(np.array(true, false)), true)
    assertEquals(np.max(np.array(Complex(1, 5), Complex(2, 0))), Complex(2, 0))
    assertEquals(np.median(np.array(1.0, 2.0), axis = null).ndim, 0)
    assertEquals(np.argmin(np.array(Seq(Seq(3.0, 1.0), Seq(0.0, 2.0)))), 2)
    assertEquals(np.prod(b, axis = Seq(0, 1), keepdims = true).toList, List(1890L))
    assertEquals(np.std(b, 1).toSeq.length, 2)
    assertEquals(np.ptp(np.array(2.5, -1.0)), 3.5)
    assertEquals(np.amax(np.arange(6).reshape(2, 3), axis = Seq(0, 1)).item, 5)
  }
