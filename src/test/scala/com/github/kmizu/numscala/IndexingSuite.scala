package com.github.kmizu.numscala

class IndexingSuite extends munit.FunSuite:
  private def ia(xs: Int*): NDArray[Int] = NDArray.fromArray(xs.toArray)
  private def i2(rows: Seq[Int]*): NDArray[Int] = np.array(rows.toSeq)

  test("diag_indices / diag_indices_from") {
    val di = np.diag_indices(3)
    assertEquals(di.length, 2)
    assertEquals(di.map(_.toList), Seq(List(0, 1, 2), List(0, 1, 2)))
    val a = np.arange(16).reshape(4, 4)
    val d = np.diag_indices_from(a)
    assertEquals(a(d(0), d(1)).toList, List(0, 5, 10, 15))
    assertEquals(np.diag_indices(2, 3).length, 3)
    intercept[IllegalArgumentException](np.diag_indices_from(np.zeros(2, 3)))
    intercept[IllegalArgumentException](np.diag_indices_from(np.zeros(2)))
  }

  test("tril_indices / triu_indices and *_from") {
    val (r, c) = np.tril_indices(4)
    assertEquals(r.toList, List(0, 1, 1, 2, 2, 2, 3, 3, 3, 3))
    assertEquals(c.toList, List(0, 0, 1, 0, 1, 2, 0, 1, 2, 3))
    assertEquals(np.tril_indices(4, 2)._1.size, 15)
    val (ur, uc) = np.triu_indices(4, 1)
    assertEquals(ur.toList, List(0, 0, 0, 1, 1, 2))
    assertEquals(uc.toList, List(1, 2, 3, 2, 3, 3))
    assertEquals(np.triu_indices(3, 0, 4)._1.size, 9)
    val (fr, fc) = np.tril_indices_from(np.arange(12).reshape(3, 4))
    assertEquals(fr.toList, List(0, 1, 1, 2, 2, 2))
    assertEquals(fc.toList, List(0, 0, 1, 0, 1, 2))
    val (tr, tc) = np.triu_indices_from(np.arange(4).reshape(2, 2), -1)
    assertEquals(tr.toList, List(0, 0, 1, 1))
    assertEquals(tc.toList, List(0, 1, 0, 1))
    intercept[IllegalArgumentException](np.tril_indices_from(np.arange(3)))
  }

  test("mask_indices") {
    val (r, c) = np.mask_indices(3, (m, k) => np.triu(m, k))
    assertEquals(r.toList, List(0, 0, 0, 1, 1, 2))
    assertEquals(c.toList, List(0, 1, 2, 1, 2, 2))
    val (r1, c1) = np.mask_indices(3, (m, k) => np.triu(m, k), 1)
    assertEquals(r1.toList, List(0, 0, 1))
    assertEquals(c1.toList, List(1, 2, 2))
  }

  test("ravel_multi_index") {
    val idx = Seq(ia(3, 6, 6), ia(4, 5, 1))
    assertEquals(np.ravel_multi_index(idx, Seq(7, 6)).toList, List(22, 41, 37))
    assertEquals(np.ravel_multi_index(idx, Seq(7, 6), order = 'F').toList, List(31, 41, 13))
    assertEquals(np.ravel_multi_index(idx, Seq(4, 6), mode = "clip").toList, List(22, 23, 19))
    assertEquals(np.ravel_multi_index(idx, Seq(4, 4), mode = Seq("clip", "wrap")).toList, List(12, 13, 13))
    assertEquals(np.ravel_multi_index(Seq(3, 1, 4, 1), Seq(6, 7, 8, 9)), 1621)
    intercept[IllegalArgumentException](np.ravel_multi_index(idx, Seq(4, 6)))
    intercept[IllegalArgumentException](np.ravel_multi_index(Seq(ia(-1)), Seq(4)))
    intercept[IllegalArgumentException](np.ravel_multi_index(idx, Seq(7)))
    val b = np.ravel_multi_index(Seq(i2(Seq(0, 1)), ia(2)), Seq(2, 3))
    assertEquals(b.shape, Seq(1, 2))
    assertEquals(b.toList, List(2, 5))
  }

  test("unravel_index") {
    val u = np.unravel_index(ia(22, 41, 37), Seq(7, 6))
    assertEquals(u.map(_.toList), Seq(List(3, 6, 6), List(4, 5, 1)))
    val f = np.unravel_index(ia(31, 41, 13), Seq(7, 6), 'F')
    assertEquals(f.map(_.toList), Seq(List(3, 6, 6), List(4, 5, 1)))
    assertEquals(np.unravel_index(1621, Seq(6, 7, 8, 9)), Seq(3, 1, 4, 1))
    intercept[IllegalArgumentException](np.unravel_index(42, Seq(6, 7)))
    intercept[IllegalArgumentException](np.unravel_index(-1, Seq(6, 7)))
    val s = np.unravel_index(i2(Seq(1, 2)), Seq(2, 2))
    assertEquals(s(0).shape, Seq(1, 2))
  }

  test("ix_") {
    val a = np.arange(10).reshape(2, 5)
    val g = np.ix_(ia(0, 1), ia(2, 4))
    assertEquals(g(0).shape, Seq(2, 1))
    assertEquals(g(1).shape, Seq(1, 2))
    assertEquals(a(g(0), g(1)).toList, List(2, 4, 7, 9))
    val gb = np.ix_(np.array(true, false), np.array(false, true, false, false, true))
    assertEquals(a(gb(0), gb(1)).toList, List(1, 4))
    intercept[IllegalArgumentException](np.ix_(np.arange(4).reshape(2, 2)))
  }

  test("ndindex / ndenumerate") {
    assertEquals(
      np.ndindex(3, 2, 1).toList,
      List(Seq(0, 0, 0), Seq(0, 1, 0), Seq(1, 0, 0), Seq(1, 1, 0), Seq(2, 0, 0), Seq(2, 1, 0))
    )
    assertEquals(np.ndindex().toList, List(Seq()))
    assertEquals(np.ndindex(2, 0).toList, Nil)
    val e = np.ndenumerate(i2(Seq(1, 2), Seq(3, 4))).toList
    assertEquals(e, List((Seq(0, 0), 1), (Seq(0, 1), 2), (Seq(1, 0), 3), (Seq(1, 1), 4)))
  }

  test("apply_along_axis") {
    val b = np.array(Seq(Seq(1.0, 2.0, 3.0), Seq(4.0, 5.0, 6.0), Seq(7.0, 8.0, 9.0)))
    val f = (a: NDArray[Double]) => (a(0) + a(-1)) * 0.5
    assertEquals(np.apply_along_axis(f, 0, b).toList, List(4.0, 5.0, 6.0))
    assertEquals(np.apply_along_axis(f, 1, b).toList, List(2.0, 5.0, 8.0))
    val c = i2(Seq(8, 1, 7), Seq(4, 3, 9), Seq(5, 2, 6))
    val s = np.apply_along_axis((r: NDArray[Int]) => np.sort(r), 1, c)
    assertEquals(s.toList, List(1, 7, 8, 3, 4, 9, 2, 5, 6))
    val dg = np.apply_along_axis((r: NDArray[Double]) => np.diag(r), -1, b)
    assertEquals(dg.shape, Seq(3, 3, 3))
    assertEquals(dg(1, ::, ::).toList, List(4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0))
    val m = np.arange(6).reshape(2, 3)
    val two = np.apply_along_axis((r: NDArray[Int]) => ia(r(0) + r(1), math.max(r(0), r(1))), 0, m)
    assertEquals(two.shape, Seq(2, 3))
    assertEquals(two.toList, List(3, 5, 7, 3, 4, 5))
    val t3 = np.arange(24).reshape(2, 3, 4)
    val sums = np.apply_along_axis((r: NDArray[Int]) => r.toList.sum, 1, t3)
    assertEquals(sums.shape, Seq(2, 4))
    assertEquals(sums.toList, List(12, 15, 18, 21, 48, 51, 54, 57))
    intercept[IllegalArgumentException](np.apply_along_axis(f, 0, np.zeros(3, 0)))
  }

  test("apply_over_axes") {
    val a = np.arange(24).reshape(2, 3, 4)
    val r = np.apply_over_axes((x: NDArray[Int], ax: Int) => x.sum(ax).astype[Int], a, Seq(0, 2))
    assertEquals(r.shape, Seq(1, 3, 1))
    assertEquals(r.toList, List(60, 92, 124))
    val k = np.apply_over_axes((x: NDArray[Int], ax: Int) => x.sum(ax, keepdims = true).astype[Int], a, -1)
    assertEquals(k.shape, Seq(2, 3, 1))
  }

  test("vectorize") {
    val f = np.vectorize((a: Int, b: Int) => if a > b then a - b else a + b)
    assertEquals(f(ia(1, 2, 3, 4), NDArray.scalar(2)).toList, List(3, 4, 1, 2))
    val g = np.vectorize((x: Int) => x.toDouble / 2)
    assertEquals(g(ia(1, 2)).toList, List(0.5, 1.0))
    val h = np.vectorize((a: Int, b: Int, c: Int) => a * b + c)
    assertEquals(h(ia(1, 2), ia(3), ia(10, 20)).toList, List(13, 26))
  }

  test("piecewise") {
    val x = np.linspace(-2.5, 2.5, 6)
    val sgn = np.piecewise(x, Seq(x < 0.0, x >= 0.0), Seq[(Double => Double) | Double](-1.0, 1.0))
    assertEquals(sgn.toList, List(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0))
    val abs = np.piecewise(x, Seq(x < 0.0, x >= 0.0), Seq[(Double => Double) | Double]((v: Double) => -v, (v: Double) => v))
    assertEquals(abs.toList, List(2.5, 1.5, 0.5, 0.5, 1.5, 2.5))
    val other = np.piecewise(x, Seq(x < 0.0), Seq[(Double => Double) | Double]((v: Double) => v * 2, 9.0))
    assertEquals(other.toList, List(-5.0, -3.0, -1.0, 9.0, 9.0, 9.0))
    val none = np.piecewise(x, Seq(x > 2.0), Seq[(Double => Double) | Double](1.0))
    assertEquals(none.toList, List(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    intercept[IllegalArgumentException](np.piecewise(x, Seq(x > 2.0), Seq[(Double => Double) | Double](1.0, 2.0, 3.0)))
  }

  test("s_ / index_exp") {
    val ix = np.s_("1:3, ::2")
    assertEquals(ix, Seq(Index.Slice(Some(1), Some(3), 1), Index.Slice(None, None, 2)))
    val a = np.arange(20).reshape(4, 5)
    assertEquals(a.index(ix).toList, List(5, 7, 9, 10, 12, 14))
    assertEquals(np.index_exp(1, ::, "...").length, 3)
    assertEquals(np.s_(0, "None"), Seq(Index.At(0), Index.NewAxis))
  }
