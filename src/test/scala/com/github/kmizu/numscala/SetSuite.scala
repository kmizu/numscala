package com.github.kmizu.numscala

class SetSuite extends munit.FunSuite:
  private def ia(xs: Int*): NDArray[Int] = NDArray.fromArray(xs.toArray)
  private def da(xs: Double*): NDArray[Double] = NDArray.fromArray(xs.toArray)
  private def i2(rows: Seq[Int]*): NDArray[Int] = np.array(rows.toSeq)
  private def str(a: NDArray[Double]): List[String] = a.toList.map(_.toString)

  test("unique basic") {
    assertEquals(np.unique(ia(1, 1, 2, 2, 3, 3)).toList, List(1, 2, 3))
    val u = np.unique(i2(Seq(1, 1), Seq(2, 3)))
    assertEquals(u.shape, Seq(3))
    assertEquals(u.toList, List(1, 2, 3))
    assertEquals(np.unique(ia()).toList, Nil)
    assertEquals(np.unique(np.array("b", "a", "b")).toList, List("a", "b"))
  }

  test("unique with return_index / return_inverse / return_counts") {
    val s = np.unique(np.array("a", "b", "b", "c", "a"), return_index = true)
    assertEquals(s.values.toList, List("a", "b", "c"))
    assertEquals(s.indices.get.toList, List(0, 1, 3))
    assertEquals(s.inverse, None)
    val a = ia(1, 2, 6, 4, 2, 3, 2)
    val r = np.unique(a, return_inverse = true, return_counts = true)
    assertEquals(r.values.toList, List(1, 2, 3, 4, 6))
    assertEquals(r.inverse.get.toList, List(0, 1, 4, 3, 1, 2, 1))
    assertEquals(r.counts.get.toList, List(1, 3, 1, 1, 1))
    assertEquals(np.take(r.values, r.inverse.get).toList, a.toList)
    // inverse has the input's shape when axis=None
    val m = np.unique(i2(Seq(1, 2), Seq(2, 1)), return_inverse = true)
    assertEquals(m.inverse.get.shape, Seq(2, 2))
    assertEquals(m.inverse.get.toList, List(0, 1, 1, 0))
  }

  test("unique along an axis") {
    val a = i2(Seq(1, 0, 0), Seq(1, 0, 0), Seq(2, 3, 4))
    val r = np.unique(a, return_index = true, return_inverse = true, return_counts = true, axis = 0)
    assertEquals(r.values.shape, Seq(2, 3))
    assertEquals(r.values.toList, List(1, 0, 0, 2, 3, 4))
    assertEquals(r.indices.get.toList, List(0, 2))
    assertEquals(r.inverse.get.toList, List(0, 0, 1))
    assertEquals(r.counts.get.toList, List(2, 1))
    val b = i2(Seq(1, 0, 1), Seq(2, 0, 2))
    val c = np.unique(b, return_inverse = true, axis = 1)
    assertEquals(c.values.shape, Seq(2, 2))
    assertEquals(c.values.toList, List(0, 1, 0, 2))
    assertEquals(c.inverse.get.toList, List(1, 0, 1))
    assertEquals(np.unique(b, axis = -1).values.toList, List(0, 1, 0, 2))
  }

  test("unique NaN handling") {
    val a = da(1.0, Double.NaN, Double.NaN, 2.0)
    assertEquals(str(np.unique(a)), List("1.0", "2.0", "NaN"))
    val r = np.unique(a, return_counts = true)
    assertEquals(r.counts.get.toList, List(1, 1, 2))
    assertEquals(str(np.unique(a, equal_nan = false).values), List("1.0", "2.0", "NaN", "NaN"))
    assertEquals(str(np.unique_values(da(Double.NaN, 1.0, Double.NaN))), List("1.0", "NaN", "NaN"))
    // -0.0 and 0.0 are the same value
    assertEquals(np.unique(da(0.0, -0.0)).size, 1)
    // complex NaNs collapse
    val c = NDArray.fromArray(Array(Complex(Double.NaN, 0), Complex(1, 0), Complex(0, Double.NaN), Complex(1, 0)))
    val uc = np.unique(c)
    assertEquals(uc.size, 2)
    assertEquals(uc(0), Complex(1, 0))
    assert(uc(1).isNaN)
  }

  test("unique_counts / unique_inverse / unique_all") {
    val uc = np.unique_counts(ia(1, 1, 2))
    assertEquals(uc.values.toList, List(1, 2))
    assertEquals(uc.counts.toList, List(2, 1))
    val x = i2(Seq(3, 1), Seq(1, 3))
    val ui = np.unique_inverse(x)
    assertEquals(ui.values.toList, List(1, 3))
    assertEquals(ui.inverse_indices.shape, Seq(2, 2))
    assertEquals(ui.inverse_indices.toList, List(1, 0, 0, 1))
    val ua = np.unique_all(x)
    assertEquals(ua.values.toList, List(1, 3))
    assertEquals(ua.indices.toList, List(1, 0))
    assertEquals(ua.inverse_indices.toList, List(1, 0, 0, 1))
    assertEquals(ua.counts.toList, List(2, 2))
  }

  test("isin / in1d") {
    val element = np.arange(4).reshape(2, 2) * 2
    val test = ia(1, 2, 4, 8)
    val m = np.isin(element, test)
    assertEquals(m.shape, Seq(2, 2))
    assertEquals(m.toList, List(false, true, true, false))
    assertEquals(np.isin(element, test, invert = true).toList, List(true, false, false, true))
    assertEquals(np.isin(da(Double.NaN, 1.0), da(Double.NaN)).toList, List(false, false))
    assertEquals(np.isin(ia(1, 2), da(1.0, 2.5)).toList, List(true, false))
    assertEquals(np.isin(ia(1, 2), ia()).toList, List(false, false))
    assertEquals(np.in1d(ia(0, 1, 2, 5, 0), ia(0, 2)).toList, List(true, false, true, false, true))
    assertEquals(np.in1d(i2(Seq(0, 1), Seq(2, 5)), ia(0, 2), invert = true).toList, List(false, true, false, true))
  }

  test("intersect1d") {
    assertEquals(np.intersect1d(ia(1, 3, 4, 3), ia(3, 1, 2, 1)).toList, List(1, 3))
    val (xy, xi, yi) = np.intersect1d(ia(1, 1, 2, 3, 4), ia(2, 1, 4, 6), false, true)
    assertEquals(xy.toList, List(1, 2, 4))
    assertEquals(xi.toList, List(0, 2, 4))
    assertEquals(yi.toList, List(1, 0, 2))
    val (u, ui, uj) = np.intersect1d(ia(4, 1, 2), ia(2, 4), true, true)
    assertEquals(u.toList, List(2, 4))
    assertEquals(ui.toList, List(2, 0))
    assertEquals(uj.toList, List(0, 1))
    assertEquals(np.intersect1d(da(Double.NaN, 1.0), da(Double.NaN, 1.0)).toList, List(1.0))
    val p = np.intersect1d(ia(1, 2), da(2.0, 3.0))
    assertEquals(p.dtype.name, "float64")
    assertEquals(p.toList, List(2.0))
  }

  test("union1d, setdiff1d, setxor1d") {
    assertEquals(np.union1d(ia(-1, 0, 1), ia(-2, 0, 2)).toList, List(-2, -1, 0, 1, 2))
    assertEquals(np.setdiff1d(ia(1, 2, 3, 2, 4, 1), ia(3, 4, 5, 6)).toList, List(1, 2))
    assertEquals(np.setdiff1d(ia(5, 1), ia(1), assume_unique = true).toList, List(5))
    assertEquals(np.setxor1d(ia(1, 2, 3, 2, 4), ia(2, 3, 5, 7, 5)).toList, List(1, 4, 5, 7))
    assertEquals(np.setxor1d(ia(1, 2), ia(2, 3), assume_unique = true).toList, List(1, 3))
    assertEquals(np.union1d(ia(1), da(0.5)).toList, List(0.5, 1.0))
  }
