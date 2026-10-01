package numscala

class TestingSuite extends munit.FunSuite:
  import np.testing.*
  test("assert_allclose") {
    assert_allclose(np.array(1.0, 2.0), np.array(1.0, 2.0 + 1e-9))
    val e = intercept[AssertionError](assert_allclose(np.array(1.0, 2.0), np.array(1.0, 2.1)))
    assert(e.getMessage.contains("Mismatched elements: 1 / 2 (50%)"), e.getMessage)
    assert_allclose(np.array(Double.NaN), np.array(Double.NaN))
    intercept[AssertionError](assert_allclose(np.array(Double.NaN), np.array(Double.NaN), equal_nan = false))
    assert_allclose(np.array(1, 2), np.array(1.0, 2.0))
    assert_allclose(np.array(Complex(1, 1)), np.array(Complex(1, 1 + 1e-12)))
    intercept[AssertionError](assert_allclose(np.zeros(2), np.zeros(3)))
  }
  test("equality helpers") {
    assert_array_equal(np.array(1, 2), np.array(1L, 2L))
    intercept[AssertionError](assert_array_equal(np.array(1, 2), np.array(1, 3)))
    assert_array_equal(np.zeros(3), np.array(0.0))
    assert_equal(3, 3)
    intercept[AssertionError](assert_equal("a", "b"))
    assert_array_almost_equal(np.array(1.0), np.array(1.0000001))
    assert_almost_equal(1.0, 1.00000001)
    intercept[AssertionError](assert_almost_equal(1.0, 1.001))
    assert_approx_equal(1234567.0, 1234568.0, significant = 6)
    assert_array_less(np.array(1.0), np.array(2.0))
    intercept[AssertionError](assert_array_less(np.array(2.0), np.array(2.0)))
    assert_array_max_ulp(np.array(1.0), np.array(1.0 + math.ulp(1.0)))
    assert_raises[IllegalArgumentException](np.zeros(2).reshape(3))
    assert_string_equal("x", "x")
  }
