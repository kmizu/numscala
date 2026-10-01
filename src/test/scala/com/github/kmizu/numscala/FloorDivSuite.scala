package com.github.kmizu.numscala

class FloorDivSuite extends munit.FunSuite:
  test("a `//` b on arrays and scalars (NumPy floor division)") {
    val i = np.array(7, -7, 8)
    assertEquals((i `//` 2).toList, List(3, -4, 4))
    assertEquals((i `//` np.array(2, 2, -3)).toList, List(3, -4, -3))
    assertEquals((np.array(7.0, -7.0) `//` 2.0).toList, List(3.0, -4.0))
    assertEquals((1.0 `//` np.array(0.1)).toList, List(9.0)) // NumPy: 1.0 // 0.1 == 9.0
    assertEquals((7 `//` np.array(2, -2)).toList, List(3, -4))
    assertEquals((7.0 `//` np.array(2.0, -2.0)).toList, List(3.0, -4.0))
    assertEquals((i `//` 0).toList, List(0, 0, 0)) // integer division by zero gives 0 like NumPy
  }

  test("`//` binds like * and /, left-associative") {
    val a = np.array(7, 9)
    assertEquals((np.array(1, 1) + a `//` 2).toList, List(4, 5))
    assertEquals((a `//` 2 * 3).toList, List(9, 12))
    assertEquals((a * 3 `//` 4).toList, List(5, 6))
  }

  test("in-place `//=`") {
    val a = np.array(7.0, -7.0)
    a `//=` 2.0
    assertEquals(a.toList, List(3.0, -4.0))
    val b = np.array(9, 10)
    b `//=` np.array(2, 3)
    assertEquals(b.toList, List(4, 3))
  }

  test("masked arrays and polynomials") {
    val m = np.ma.masked_less(np.array(7, -1, 9), 0)
    assertEquals((m `//` 2).toString, "[3 -- 4]")
    val p = np.polynomial.Polynomial(Seq(-1.0, 0.0, 1.0)) // x^2 - 1
    val q = np.polynomial.Polynomial(Seq(1.0, 1.0)) //       x + 1
    assertEquals((p `//` q).coef.toList, List(-1.0, 1.0)) // x - 1
  }
