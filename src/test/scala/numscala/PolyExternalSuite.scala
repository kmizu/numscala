package numscalaexternal

import numscala.*

/** Checks that the inline APIs (`polyfit`, `<kind>fit`, `Series.fit`) expand outside the library package. */
class PolyExternalSuite extends munit.FunSuite:
  test("inline fit APIs are usable from user code") {
    val x = np.array(0.0, 1.0, 2.0, 3.0)
    val y = np.array(1.0, 3.0, 5.0, 7.0)
    val c: NDArray[Double] = np.polyfit(x, y, 1)
    assertEqualsDouble(c(0), 2.0, 1e-12)
    val full: PolyfitResult = np.polyfit(x, y, 1, full = true)
    assertEquals(full.rank, 2)
    val (cc, cov) = np.polyfit(x, y, 1, cov = "unscaled")
    assertEquals(cov.shape, Seq(2, 2))
    val s: Polynomial = np.polynomial.Polynomial.fit(x, y, 1)
    assertEqualsDouble(s(10.0), 21.0, 1e-12)
    val (s2, info) = np.polynomial.Legendre.fit(x, y, 1, full = true)
    assertEquals(info.rank, 2)
    val lc: NDArray[Double] = np.polynomial.laguerre.lagfit(x, y, 1)
    assertEqualsDouble(np.polynomial.laguerre.lagval(3.0, lc), 7.0, 1e-12)
    val p: np.poly1d[Double] = np.poly1d(np.array(1.0, 0.0))
    assertEquals(p(3.0), 3.0)
  }
