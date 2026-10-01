package com.github.kmizu.numscala

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

class StrideTricksSuite extends munit.FunSuite:
  import np.lib.stride_tricks.*
  test("sliding_window_view") {
    val x = np.arange(6)
    val v = sliding_window_view(x, 3)
    assertEquals(v.shape, Seq(4, 3))
    assertEquals(v(1, ::).toList, List(1, 2, 3))
    assertEquals(v.sum(1).toList, List(3L, 6L, 9L, 12L))
    val m = np.arange(12).reshape(3, 4)
    val w = sliding_window_view(m, Seq(2, 2))
    assertEquals(w.shape, Seq(2, 3, 2, 2))
    assertEquals(w(1, 2, ::, ::).toList, List(6, 7, 10, 11))
    assertEquals(sliding_window_view(m, 2, 1).shape, Seq(3, 3, 2))
    intercept[IllegalArgumentException](sliding_window_view(x, 7))
  }
  test("as_strided") {
    val x = np.arange(5.0)
    val r = as_strided(x, Seq(3, 3), Seq(8, 8))
    assertEquals(r.toList, List(0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0))
  }
