package com.github.kmizu.numscala

/** Regression: matrix products must not skip `0 * x` terms (NumPy: `0 * inf` and `0 * nan` are NaN). */
class MatmulNonFiniteSuite extends munit.FunSuite:

  test("float64 matmul propagates 0 * inf and 0 * nan") {
    val a = np.array(Seq(Seq(0.0, 1.0)))
    val inf = np.array(Seq(Seq(Double.PositiveInfinity), Seq(1.0)))
    val nan = np.array(Seq(Seq(Double.NaN), Seq(1.0)))
    assert(np.matmul(a, inf)(0, 0).isNaN)
    assert(np.matmul(a, nan)(0, 0).isNaN)
    assert(np.dot(a, inf)(0, 0).isNaN)
  }

  test("float64 matmul with a zero row of A against inf column") {
    val a = np.array(Seq(Seq(0.0, 0.0), Seq(1.0, 2.0)))
    val b = np.array(Seq(Seq(Double.PositiveInfinity, 1.0), Seq(1.0, 1.0)))
    val r = np.matmul(a, b)
    assert(r(0, 0).isNaN)
    assertEquals(r(0, 1), 0.0)
    assertEquals(r(1, 0), Double.PositiveInfinity)
    assertEquals(r(1, 1), 3.0)
  }

  test("LinalgReal.gemm propagates 0 * inf") {
    val r = LinalgReal.gemm(Array(0.0, 1.0), Array(Double.PositiveInfinity, 1.0), 1, 2, 1)
    assert(r(0).isNaN)
  }
