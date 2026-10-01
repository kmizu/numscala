package numscala

class SortSuite extends munit.FunSuite:
  private def ia(xs: Int*): NDArray[Int] = NDArray.fromArray(xs.toArray)
  private def da(xs: Double*): NDArray[Double] = NDArray.fromArray(xs.toArray)
  private def i2(rows: Seq[Int]*): NDArray[Int] = np.array(rows.toSeq)
  private def dd(xs: Seq[Double], ys: Seq[Double]): Unit =
    assertEquals(xs.length, ys.length)
    xs.zip(ys).foreach((x, y) => if !(x.isNaN && y.isNaN) then assertEqualsDouble(x, y, 1e-12))

  // ------------------------------------------------------------------ sort / argsort

  test("sort 1-D, 2-D along axes, axis=None") {
    assertEquals(np.sort(ia(3, 1, 2)).toList, List(1, 2, 3))
    val m = i2(Seq(3, 1), Seq(2, 4))
    assertEquals(np.sort(m).toList, List(1, 3, 2, 4))
    assertEquals(np.sort(m, 0).toList, List(2, 1, 3, 4))
    assertEquals(np.sort(m, -2).toList, List(2, 1, 3, 4))
    val f = np.sort(m, None)
    assertEquals(f.shape, Seq(4))
    assertEquals(f.toList, List(1, 2, 3, 4))
    assertEquals(np.sort(m, 1, "stable").toList, List(1, 3, 2, 4))
    intercept[IllegalArgumentException](np.sort(m, 0, "bogus"))
    intercept[IndexOutOfBoundsException](np.sort(m, 2))
  }

  test("sort puts NaN last") {
    val s = np.sort(da(Double.NaN, 1.0, Double.NegativeInfinity, 0.5))
    dd(s.toSeq, Seq(Double.NegativeInfinity, 0.5, 1.0, Double.NaN))
    assertEquals(np.argsort(da(Double.NaN, 1.0, 0.0)).toList, List(2, 1, 0))
  }

  test("argsort is stable and supports axis") {
    assertEquals(np.argsort(ia(3, 1, 2)).toList, List(1, 2, 0))
    assertEquals(np.argsort(ia(1, 1, 0)).toList, List(2, 0, 1))
    val m = i2(Seq(3, 1), Seq(2, 4))
    assertEquals(np.argsort(m, None).toList, List(1, 2, 0, 3))
    assertEquals(np.argsort(m, 0).toList, List(1, 0, 0, 1))
    assertEquals(np.argsort(m).toList, List(1, 0, 0, 1))
  }

  test("lexsort: last key is primary") {
    val a = ia(1, 5, 1, 4, 3, 4, 4)
    val b = ia(9, 4, 0, 4, 0, 2, 1)
    assertEquals(np.lexsort(Seq(b, a)).toList, List(2, 0, 4, 6, 5, 3, 1))
    assertEquals(np.lexsort(np.array(Seq(b, a))).toList, List(2, 0, 4, 6, 5, 3, 1))
    // mixed dtypes
    val names = np.array("b", "a", "b", "a")
    val ages = da(30.0, 20.0, 10.0, 20.0)
    assertEquals(np.lexsort(Seq[NDArray[?]](ages, names)).toList, List(1, 3, 2, 0))
    val x = i2(Seq(3, 1), Seq(1, 2))
    assertEquals(np.lexsort(Seq(x), axis = 0).toList, List(1, 0, 0, 1))
    assertEquals(np.lexsort(Seq(x)).toList, List(1, 0, 0, 1))
    intercept[IllegalArgumentException](np.lexsort(Seq.empty[NDArray[Int]]))
  }

  test("sort_complex") {
    val s = np.sort_complex(ia(5, 3, 6, 2, 1))
    assertEquals(s.toList, List(1, 2, 3, 5, 6).map(x => Complex(x.toDouble, 0.0)))
    val c = NDArray.fromArray(Array(Complex(1, 2), Complex(2, -1), Complex(3, -2), Complex(3, -3), Complex(3, 5)))
    assertEquals(
      np.sort_complex(c).toList,
      List(Complex(1, 2), Complex(2, -1), Complex(3, -3), Complex(3, -2), Complex(3, 5))
    )
  }

  // ------------------------------------------------------------------ partition

  private def checkPartition(orig: Array[Double], res: Array[Double], ks: Seq[Int]): Unit =
    val sorted = orig.sorted(using DType.Float64.ordering)
    val cmp = DType.Float64.ordering
    for k <- ks do
      assert(cmp.compare(res(k), sorted(k)) == 0, s"kth $k: ${res(k)} != ${sorted(k)}")
      for i <- 0 until k do assert(cmp.compare(res(i), res(k)) <= 0)
      for i <- k + 1 until res.length do assert(cmp.compare(res(i), res(k)) >= 0)
    assertEquals(res.sorted(using cmp).toSeq.map(_.toString), sorted.toSeq.map(_.toString))

  test("partition satisfies the partition property (random, NaN, duplicates)") {
    val rnd = new scala.util.Random(1234)
    for trial <- 0 until 200 do
      val n = 1 + rnd.nextInt(150)
      val arr = Array.fill(n) {
        val r = rnd.nextInt(20)
        if r == 0 then Double.NaN else (rnd.nextInt(30) - 10).toDouble
      }
      val ks = Seq.fill(1 + rnd.nextInt(3))(rnd.nextInt(n))
      val p = np.partition(NDArray.fromArray(arr), ks)
      checkPartition(arr, p.toArray, ks)
      val ap = np.argpartition(NDArray.fromArray(arr), ks)
      checkPartition(arr, ap.toArray.map(arr(_)), ks)
      assertEquals(ap.toArray.sorted.toSeq, (0 until n).toSeq)
      val _ = trial
  }

  test("partition on adversarial input (sorted, reversed, constant, large)") {
    for arr <- Seq(
        Array.tabulate(5000)(_.toDouble),
        Array.tabulate(5000)(i => (5000 - i).toDouble),
        Array.fill(5000)(7.0),
        Array.tabulate(5000)(i => (i % 3).toDouble)
      )
    do
      val ks = Seq(0, 2500, 4999)
      checkPartition(arr, np.partition(NDArray.fromArray(arr), ks).toArray, ks)
  }

  test("partition examples, axis and errors") {
    val a = ia(3, 4, 2, 1)
    val p = np.partition(a, 3)
    assertEquals(p(3), 4)
    assertEquals(np.partition(a, Seq(1, 3)).toList, List(1, 2, 3, 4))
    assertEquals(np.partition(a, -1)(3), 4)
    val m = i2(Seq(9, 1, 5), Seq(3, 8, 2))
    val p0 = np.partition(m, 0, axis = 0)
    assertEquals(p0.toList, List(3, 1, 2, 9, 8, 5))
    val p1 = np.partition(m, 1, axis = 1)
    assertEquals(p1(0, 1), 5)
    assertEquals(p1(1, 1), 3)
    val pn = np.partition(m, 2, None)
    assertEquals(pn.shape, Seq(6))
    assertEquals(pn(2), 3)
    intercept[IllegalArgumentException](np.partition(a, 4))
    val ap = np.argpartition(m, 0, axis = 0)
    assertEquals(ap.toList, List(1, 0, 1, 0, 1, 0))
    assertEquals(np.argpartition(a, 0, None)(0), 3)
  }

  // ------------------------------------------------------------------ searchsorted

  test("searchsorted") {
    val a = ia(1, 2, 3, 4, 5)
    assertEquals(np.searchsorted(a, 3), 2)
    assertEquals(np.searchsorted(a, 3, "right"), 3)
    assertEquals(np.searchsorted(a, ia(-10, 10, 2, 3)).toList, List(0, 5, 1, 2))
    assertEquals(np.searchsorted(a, ia(-10, 10, 2, 3), "right").toList, List(0, 5, 2, 3))
    val u = np.searchsorted(a, np.array(Seq(Seq(1, 6))))
    assertEquals(u.shape, Seq(1, 2))
    // sorter
    val b = ia(3, 1, 2)
    assertEquals(np.searchsorted(b, 2, "left", ia(1, 2, 0)), 1)
    assertEquals(np.searchsorted(b, ia(4, 0), "left", ia(1, 2, 0)).toList, List(3, 0))
    // NaN at the end, promotion
    val f = da(1.0, 2.0, Double.NaN)
    assertEquals(np.searchsorted(f, Double.NaN), 2)
    assertEquals(np.searchsorted(f, 5.0), 2)
    assertEquals(np.searchsorted(a, da(2.5, 0.5)).toList, List(2, 0))
    intercept[IllegalArgumentException](np.searchsorted(a, 3, "middle"))
    intercept[IllegalArgumentException](np.searchsorted(i2(Seq(1, 2)), 3))
  }

  // ------------------------------------------------------------------ where & friends

  test("where with arrays, scalars, broadcasting and promotion") {
    val c = np.array(true, false, true)
    assertEquals(np.where(c, ia(1, 2, 3), ia(10, 20, 30)).toList, List(1, 20, 3))
    val wi = np.where(c, ia(1, 2, 3), 0)
    assertEquals(wi.dtype.name, "int32")
    assertEquals(wi.toList, List(1, 0, 3))
    val a = np.arange(10)
    assertEquals(np.where(a < 5, a, a * 10).toList, List(0, 1, 2, 3, 4, 50, 60, 70, 80, 90))
    val x = np.arange(3).reshape(3, 1)
    val y = np.arange(4).reshape(1, 4)
    val w = np.where(x < y, x, y + 10)
    assertEquals(w.shape, Seq(3, 4))
    assertEquals(w.toList, List(10, 0, 0, 0, 10, 11, 1, 1, 10, 11, 12, 2))
    val wd = np.where(c, ia(1, 2, 3), da(0.5, 0.5, 0.5))
    assertEquals(wd.dtype.name, "float64")
    assertEquals(wd.toList, List(1.0, 0.5, 3.0))
    val ws = np.where(c, 1, 2.5)
    assertEquals(ws.toList, List(1.0, 2.5, 1.0))
    assertEquals(np.where(c, 7, ia(1, 2, 3)).toList, List(7, 2, 7))
    // non-boolean condition uses truthiness
    assertEquals(np.where(ia(0, 3, 0), 1, 2).toList, List(2, 1, 2))
  }

  test("nonzero, where(cond), argwhere, flatnonzero") {
    val m = i2(Seq(1, 0, 0), Seq(0, 2, 0), Seq(1, 1, 0))
    val nz = np.nonzero(m)
    assertEquals(nz.map(_.toList), Seq(List(0, 1, 2, 2), List(0, 1, 0, 1)))
    assertEquals(np.where(m > 0).map(_.toList), nz.map(_.toList))
    intercept[IllegalArgumentException](np.nonzero(NDArray.scalar(1)))
    val aw = np.argwhere(np.arange(6).reshape(2, 3) > 1)
    assertEquals(aw.shape, Seq(4, 2))
    assertEquals(aw.toList, List(0, 2, 1, 0, 1, 1, 1, 2))
    assertEquals(np.argwhere(NDArray.scalar(3)).shape, Seq(1, 0))
    assertEquals(np.argwhere(ia()).shape, Seq(0, 1))
    assertEquals(np.flatnonzero(np.arange(-2, 3)).toList, List(0, 1, 3, 4))
  }

  test("extract and select") {
    val arr = np.arange(12).reshape(3, 4)
    assertEquals(np.extract(arr % 3 === 0, arr).toList, List(0, 3, 6, 9))
    assertEquals(np.extract(ia(0, 1, 2), da(1.0, 2.0, 3.0)).toList, List(2.0, 3.0))
    val x = np.arange(6)
    assertEquals(np.select(Seq(x < 3, x > 3), Seq(x, x * x), 42).toList, List(0, 1, 2, 42, 16, 25))
    assertEquals(np.select(Seq(x < 3, x > 1), Seq(x, x * x)).toList, List(0, 1, 2, 9, 16, 25))
    intercept[IllegalArgumentException](np.select(Seq(x < 3), Seq(x, x)))
  }

  // ------------------------------------------------------------------ take / put / choose

  test("choose with modes and broadcasting") {
    val choices = Seq(ia(0, 1, 2, 3), ia(10, 11, 12, 13), ia(20, 21, 22, 23), ia(30, 31, 32, 33))
    assertEquals(np.choose(ia(2, 3, 1, 0), choices).toList, List(20, 31, 12, 3))
    assertEquals(np.choose(ia(2, 4, 1, 0), choices, "clip").toList, List(20, 31, 12, 3))
    assertEquals(np.choose(ia(2, 4, 1, 0), choices, "wrap").toList, List(20, 1, 12, 3))
    intercept[IllegalArgumentException](np.choose(ia(2, 4, 1, 0), choices))
    intercept[IllegalArgumentException](np.choose(ia(-1), choices))
    val a = i2(Seq(1, 0, 1), Seq(0, 1, 0), Seq(1, 0, 1))
    val r = np.choose(a, Seq(NDArray.scalar(-10), NDArray.scalar(10)))
    assertEquals(r.toList, List(10, -10, 10, -10, 10, -10, 10, -10, 10))
    intercept[IllegalArgumentException](np.choose(ia(0), choices, "bogus"))
  }

  test("take") {
    val a = ia(4, 3, 5, 7, 6, 8)
    assertEquals(np.take(a, ia(0, 1, 4)).toList, List(4, 3, 6))
    val t2 = np.take(a, i2(Seq(0, 1), Seq(2, 3)))
    assertEquals(t2.shape, Seq(2, 2))
    assertEquals(t2.toList, List(4, 3, 5, 7))
    val m = np.arange(6).reshape(2, 3)
    val tm = np.take(m, ia(2, 0), axis = 1)
    assertEquals(tm.shape, Seq(2, 2))
    assertEquals(tm.toList, List(2, 0, 5, 3))
    assertEquals(np.take(m, ia(1), axis = 0).toList, List(3, 4, 5))
    assertEquals(np.take(m, ia(-1)).toList, List(5))
    assertEquals(np.take(a, ia(7), mode = "wrap").toList, List(3))
    assertEquals(np.take(a, ia(-1, 10), mode = "clip").toList, List(4, 8))
    intercept[IndexOutOfBoundsException](np.take(a, ia(6)))
    assertEquals(np.take(a, 2), 5)
    assertEquals(np.take(m, i2(Seq(0, 1)), axis = -1).shape, Seq(2, 1, 2))
  }

  test("put") {
    val a = np.arange(5)
    np.put(a, ia(0, 2), ia(-44, -55))
    assertEquals(a.toList, List(-44, 1, -55, 3, 4))
    val b = np.arange(5)
    np.put(b, ia(22), -5, "clip")
    assertEquals(b.toList, List(0, 1, 2, 3, -5))
    val c = np.arange(5)
    np.put(c, ia(0, 1, 2), 9)
    assertEquals(c.toList, List(9, 9, 9, 3, 4))
    np.put(c, ia(-1, 6), ia(7, 8), "wrap")
    assertEquals(c.toList, List(9, 8, 9, 3, 7))
    intercept[IndexOutOfBoundsException](np.put(c, ia(5), 0))
    // cyclic values, through a non-contiguous view
    val m = np.zeros[Int](3, 4)
    val v = m(::, "::2")
    np.put(v, ia(0, 1, 2, 3, 4, 5), ia(1, 2))
    assertEquals(m.toList, List(1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0))
  }

  test("take_along_axis / put_along_axis") {
    val a = i2(Seq(10, 30, 20), Seq(60, 40, 50))
    val ai = np.argsort(a, 1)
    assertEquals(ai.toList, List(0, 2, 1, 1, 2, 0))
    assertEquals(np.take_along_axis(a, ai, 1).toList, List(10, 20, 30, 40, 50, 60))
    val mx = i2(Seq(1), Seq(0))
    val t = np.take_along_axis(a, mx, 1)
    assertEquals(t.shape, Seq(2, 1))
    assertEquals(t.toList, List(30, 60))
    val t0 = np.take_along_axis(a, i2(Seq(1, 0, 1)), 0)
    assertEquals(t0.toList, List(60, 30, 50))
    assertEquals(np.take_along_axis(a, ia(5, 0), None).toList, List(50, 10))
    // broadcasting of arr along a non-axis dimension
    val b = i2(Seq(1, 2, 3))
    val tb = np.take_along_axis(b, i2(Seq(2, 0, 1), Seq(0, 0, 0)), 1)
    assertEquals(tb.toList, List(3, 1, 2, 1, 1, 1))
    intercept[IllegalArgumentException](np.take_along_axis(a, ia(0), 1))
    val c = a.copy()
    np.put_along_axis(c, mx, 99, 1)
    assertEquals(c.toList, List(10, 99, 20, 99, 40, 50))
    val d = a.copy()
    np.put_along_axis(d, i2(Seq(0, 1, 0)), i2(Seq(-1, -2, -3)), 0)
    assertEquals(d.toList, List(-1, 30, -3, 60, -2, 50))
    val e = a.copy()
    np.put_along_axis(e, ia(0, 5), ia(1, 2), None)
    assertEquals(e.toList, List(1, 30, 20, 60, 40, 2))
  }

  test("compress, place, putmask") {
    val a = i2(Seq(1, 2), Seq(3, 4), Seq(5, 6))
    assertEquals(np.compress(np.array(false, true), a, 0).toList, List(3, 4))
    assertEquals(np.compress(np.array(false, true, true), a, 0).shape, Seq(2, 2))
    val c1 = np.compress(np.array(false, true), a, 1)
    assertEquals(c1.shape, Seq(3, 1))
    assertEquals(c1.toList, List(2, 4, 6))
    assertEquals(np.compress(np.array(false, true), a).toList, List(2))
    intercept[IndexOutOfBoundsException](np.compress(np.array(false, false, false, true), a, 0))

    val arr = np.arange(6).reshape(2, 3)
    np.place(arr, arr > 2, ia(44, 55))
    assertEquals(arr.toList, List(0, 1, 2, 44, 55, 44))
    np.place(arr, arr > 50, 0)
    assertEquals(arr.toList, List(0, 1, 2, 44, 0, 44))
    intercept[IllegalArgumentException](np.place(arr, arr > 2, ia()))

    val x = np.arange(6).reshape(2, 3)
    np.putmask(x, x > 2, x * x)
    assertEquals(x.toList, List(0, 1, 2, 9, 16, 25))
    val y = np.arange(5)
    np.putmask(y, y > 1, ia(-33, -44))
    assertEquals(y.toList, List(0, 1, -33, -44, -33))
    np.putmask(y, y < 0, 7)
    assertEquals(y.toList, List(0, 1, 7, 7, 7))
    intercept[IllegalArgumentException](np.putmask(y, np.array(true), 1))
  }

  test("fill_diagonal and diagonal") {
    val a = np.zeros[Int](3, 3)
    np.fill_diagonal(a, 5)
    assertEquals(a.toList, List(5, 0, 0, 0, 5, 0, 0, 0, 5))
    val tall = np.zeros[Int](5, 3)
    np.fill_diagonal(tall, 4)
    assertEquals(tall.toList, List(4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0))
    val wrapped = np.zeros[Int](5, 3)
    np.fill_diagonal(wrapped, 4, wrap = true)
    assertEquals(wrapped.toList, List(4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0))
    val wide = np.zeros[Int](3, 5)
    np.fill_diagonal(wide, ia(1, 2))
    assertEquals(wide.toList, List(1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0))
    val cube = np.zeros[Int](2, 2, 2)
    np.fill_diagonal(cube, 4)
    assertEquals(cube.toList, List(4, 0, 0, 0, 0, 0, 0, 4))
    intercept[IllegalArgumentException](np.fill_diagonal(np.zeros[Int](2, 3, 2), 1))
    intercept[IllegalArgumentException](np.fill_diagonal(np.zeros[Int](3), 1))

    val m = np.arange(4).reshape(2, 2)
    assertEquals(np.diagonal(m).toList, List(0, 3))
    assertEquals(np.diagonal(m, 1).toList, List(1))
    val t = np.arange(8).reshape(2, 2, 2)
    val dg = np.diagonal(t, 0, 0, 1)
    assertEquals(dg.shape, Seq(2, 2))
    assertEquals(dg.toList, List(0, 6, 1, 7))
  }
