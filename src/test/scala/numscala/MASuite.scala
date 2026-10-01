package numscala

class MASuite extends munit.FunSuite:
  private val ma = np.ma
  private val F = false
  private val T = true

  private def lmask(a: MaskedArray[?]): List[Boolean] = a.maskArray.toList
  private def opt[X](a: MaskedArray[X]): List[Option[X]] = a.toOptionList
  private def assertD(x: Any, y: Double, eps: Double = 1e-12)(using loc: munit.Location): Unit = x match
    case d: Double => assertEqualsDouble(d, y, eps)
    case other => fail(s"expected a Double, got $other")

  // ------------------------------------------------------------------ construction & printing

  test("str and repr of 1-D int arrays") {
    val a = ma.array(Seq(1L, 2L, 3L), Seq(F, T, F))
    assertEquals(a.toString, "[1 -- 3]")
    assertEquals(
      a.repr,
      "masked_array(data=[1, --, 3],\n             mask=[False,  True, False],\n       fill_value=999999)"
    )
    val c = ma.array(Seq(1L, 2L, 3L))
    assertEquals(c.repr, "masked_array(data=[1, 2, 3],\n             mask=False,\n       fill_value=999999)")
    assert(c.mask eq ma.nomask)
    assert(c.hasNoMask)
  }

  test("repr of 2-D float arrays, dtype and fill value variants") {
    val b = ma.array(Seq(Seq(1.5, 20.0, 3.0), Seq(4.0, 5.0, 6.0)), Seq(Seq(F, T, F), Seq(F, F, F)))
    assertEquals(b.toString, "[[1.5 -- 3.0]\n [4.0 5.0 6.0]]")
    assertEquals(
      b.repr,
      "masked_array(\n  data=[[1.5, --, 3.0],\n        [4.0, 5.0, 6.0]],\n  mask=[[False,  True, False],\n        [False, False, False]],\n  fill_value=1e+20)"
    )
    assertEquals(
      ma.array(Seq(1.0, 2.0), Seq(T, T)).repr,
      "masked_array(data=[--, --],\n             mask=[ True,  True],\n       fill_value=1e+20,\n            dtype=float64)"
    )
    assertEquals(
      ma.array(Seq(1, 2), Seq(F, T)).repr,
      "masked_array(data=[1, --],\n             mask=[False,  True],\n       fill_value=np.int64(999999),\n            dtype=int32)"
    )
    val i32 = ma.array(Seq(1, 2), Seq(F, T))
    i32.fill_value = 5
    assertEquals(i32.repr, "masked_array(data=[1, --],\n             mask=[False,  True],\n       fill_value=5,\n            dtype=int32)")
    val f = ma.array(Seq(1.0, 2.0), Seq(F, T))
    f.fill_value = 5.0
    assertEquals(f.repr, "masked_array(data=[1.0, --],\n             mask=[False,  True],\n       fill_value=5.0)")
    assertEquals(
      ma.array(Seq(1.5f, 2.25f), Seq(F, T)).repr,
      "masked_array(data=[1.5, --],\n             mask=[False,  True],\n       fill_value=np.float64(1e+20),\n            dtype=float32)"
    )
    assertEquals(ma.array(Seq(1.5f, 2.25f), Seq(F, T)).toString, "[1.5 --]")
    assertEquals(
      ma.array(Seq(Seq(1L, 2L), Seq(100L, 2L)), Seq(Seq(F, T), Seq(F, F))).toString,
      "[[1 --]\n [100 2]]"
    )
    assertEquals(
      ma.array(Seq(Seq(1L, 2L)), Seq(Seq(F, T))).repr,
      "masked_array(data=[[1, --]],\n             mask=[[False,  True]],\n       fill_value=999999)"
    )
  }

  test("repr of other dtypes, 0-d and empty arrays") {
    assertEquals(
      ma.array(Seq("a", "bb"), Seq(F, T)).repr,
      "masked_array(data=['a', --],\n             mask=[False,  True],\n       fill_value='N/A',\n            dtype='<U2')"
    )
    assertEquals(
      ma.array(Seq(Complex(1, 2), Complex(3, 0)), Seq(F, T)).repr,
      "masked_array(data=[(1+2j), --],\n             mask=[False,  True],\n       fill_value=(1e+20+0j))"
    )
    assertEquals(
      ma.array(Seq(true, false), Seq(F, T)).repr,
      "masked_array(data=[True, --],\n             mask=[False,  True],\n       fill_value=True)"
    )
    assertEquals(ma.array(Seq(0.1, 1e-5, 3.0), Seq(F, F, T)).toString, "[0.1 1e-05 --]")
    assertEquals(ma.array(5.0).repr, "masked_array(data=5.,\n             mask=False,\n       fill_value=1e+20)")
    assertEquals(
      ma.array(5.0, true).repr,
      "masked_array(data=--,\n             mask=True,\n       fill_value=1e+20,\n            dtype=float64)"
    )
    assertEquals(
      ma.array(Seq.empty[Double]).repr,
      "masked_array(data=[],\n             mask=False,\n       fill_value=1e+20,\n            dtype=float64)"
    )
    assertEquals(ma.array(Seq(1e10, 2.5), Seq(F, T)).repr.linesIterator.next(), "masked_array(data=[10000000000.0, --],")
    assertEquals(ma.masked.toString, "--")
    assertEquals(ma.masked.repr, "masked")
  }

  test("summarized printing of large arrays") {
    val a = ma.masked_where(np.arange(2000).map(_ % 3 == 0), np.arange(2000.0))
    assertEquals(a.toString, "[-- 1.0 2.0 ... 1997.0 -- 1999.0]")
  }

  test("mask construction: scalar, nested, fitting and errors") {
    val a = ma.array(Seq(1.0, 2.0, 3.0), true)
    assertEquals(lmask(a), List(T, T, T))
    val b = ma.array(Seq(Seq(1, 2), Seq(3, 4)), Seq(F, T, F, F))
    assertEquals(lmask(b), List(F, T, F, F))
    intercept[IllegalArgumentException](ma.array(Seq(1, 2, 3), Seq(F, T)))
    val c = ma.array(Seq(1, 2, 3), Seq(F, T, F), fill_value = 7)
    assertEquals(c.fill_value, 7)
    assertEquals(c.filled().toList, List(1, 7, 3))
    assert(ma.array(Seq(1, 2), Seq(F, F), hard_mask = true).hardmask)
    val nd = np.array(1.0, 2.0)
    val wrapped = ma.asarray(nd)
    wrapped(0) = 9.0
    assertEquals(nd(0), 9.0)
  }

  test("attributes, filled, compressed, count, tolist") {
    val m = ma.array(Seq(Seq(1.0, 2.0, 3.0), Seq(4.0, 5.0, 6.0)), Seq(Seq(F, T, F), Seq(T, F, F)))
    assertEquals(m.shape, Seq(2, 3))
    assertEquals(m.ndim, 2)
    assertEquals(m.size, 6)
    assertEquals(m.dtype.name, "float64")
    assertEquals(m.compressed().toList, List(1.0, 3.0, 5.0, 6.0))
    assertEquals(m.filled(0.0).toList, List(1.0, 0.0, 3.0, 0.0, 5.0, 6.0))
    assertEquals(m.filled().toList, List(1.0, 1e20, 3.0, 1e20, 5.0, 6.0))
    assertEquals(m.data.toList, List(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    assertEquals(m.count(), 4)
    assertEquals(m.count(0).toList, List(1, 1, 2))
    assertEquals(m.count(1).toList, List(2, 2))
    assertEquals(ma.count_masked(m), 2)
    assertEquals(ma.count_masked(m, 0).toList, List(1, 1, 0))
    assertEquals(m.tolist, List(List(1.0, null, 3.0), List(null, 5.0, 6.0)))
    assertEquals(ma.array(Seq(1, 2, 3), Seq(F, T, F)).tolist, List(1, null, 3))
    assertEquals(ma.array(Seq(1, 2, 3)).count(0).toList, List(3))
  }

  test("default fill values") {
    assertEquals(ma.default_fill_value(DType.Int64), 999999L)
    assertEquals(ma.default_fill_value(DType.Int32), 999999)
    assertEquals(ma.default_fill_value(DType.Float64), 1e20)
    assertEquals(ma.default_fill_value(DType.Bool), true)
    assertEquals(ma.default_fill_value(DType.Str), "N/A")
    assertEquals(ma.default_fill_value(DType.Complex128), Complex(1e20, 0))
    assertEquals(ma.minimum_fill_value(DType.Float64), Double.PositiveInfinity)
    assertEquals(ma.maximum_fill_value(DType.Int32), Int.MinValue)
    val a = ma.array(Seq(1.0, 2.0), Seq(F, T))
    ma.set_fill_value(a, -1.0)
    assertEquals(a.filled().toList, List(1.0, -1.0))
    assertEquals(ma.default_fill_value(a), 1e20)
  }

  // ------------------------------------------------------------------ indexing

  test("indexing returns elements, masked constant, and views") {
    val m = ma.array(Seq(Seq(1.0, 2.0, 3.0), Seq(4.0, 5.0, 6.0)), Seq(Seq(F, T, F), Seq(T, F, F)))
    assertEquals(m(0, 0), 1.0)
    assert(m(0, 1) == ma.masked)
    val row = m(1, ::)
    assertEquals(opt(row), List(None, Some(5.0), Some(6.0)))
    val col = m(::, 1)
    assertEquals(opt(col), List(None, Some(5.0)))
    assertEquals(opt(m.subArray(0)), List(Some(1.0), None, Some(3.0)))
    // views share data and mask
    row(0) = 40.0
    assertEquals(m(1, 0), 40.0)
    col(1) = ma.masked
    assert(m(1, 1) == ma.masked)
    // fancy / boolean indexing gives copies
    val sel = m(np.array(true, false), ::)
    assertEquals(sel.shape, Seq(1, 3))
    val gt = m(m > 2.5)
    assertEquals(opt(gt), List(Some(3.0), Some(40.0), Some(6.0)))
    intercept[IndexOutOfBoundsException](m(5, 0))
  }

  test("setting elements: masked, soft and hard masks") {
    val a = ma.array(Seq(1L, 2L, 3L, 4L), Seq(F, T, F, F))
    a.harden_mask()
    a(1) = 10L
    a(0) = 20L
    assertEquals(opt(a), List(Some(20L), None, Some(3L), Some(4L)))
    assertEquals(a.data.toList, List(20L, 2L, 3L, 4L))
    a.soften_mask()
    a(1) = 10L
    assertEquals(opt(a), List(Some(20L), Some(10L), Some(3L), Some(4L)))
    assertEquals(lmask(a), List(F, F, F, F))
    a(::) = ma.masked
    assertEquals(a.count(), 0)
    // hard mask ignores unmasking via array assignment
    val h = ma.array(Seq(1.0, 2.0, 3.0), Seq(F, T, F), hard_mask = true)
    h(::) = np.array(7.0, 8.0, 9.0)
    assertEquals(opt(h), List(Some(7.0), None, Some(9.0)))
    // assign a masked array
    val s = ma.array(Seq(1.0, 2.0, 3.0))
    s("0:2") = ma.array(Seq(5.0, 6.0), Seq(T, F))
    assertEquals(opt(s), List(None, Some(6.0), Some(3.0)))
    assertEquals(s.data.toList, List(5.0, 6.0, 3.0))
  }

  test("views without mask do not propagate a new mask; views with mask do") {
    val b = ma.array(Seq(1.0, 2.0, 3.0))
    val v = b("1:")
    v(0) = ma.masked
    assert(b.hasNoMask)
    assertEquals(lmask(v), List(T, F))
    val c = ma.array(Seq(1.0, 2.0, 3.0), Seq(F, F, T))
    val w = c(":2")
    w(0) = ma.masked
    assertEquals(lmask(c), List(T, F, T))
    c(2) = 7.0
    assertEquals(opt(c), List(None, Some(2.0), Some(7.0)))
  }

  test("mask setter and boolean masked index assignment") {
    val d = ma.array(np.arange(6).reshape(2, 3))
    d(d > 3) = ma.masked
    assertEquals(lmask(d), List(F, F, F, F, T, T))
    d.mask = false
    assertEquals(lmask(d), List(F, F, F, F, F, F))
    val e = ma.array(Seq(1, 2, 3))
    e.mask = np.array(true, false, true)
    assertEquals(opt(e), List(None, Some(2), None))
    val h = ma.array(Seq(1, 2, 3), Seq(T, F, F), hard_mask = true)
    h.mask = false
    assertEquals(lmask(h), List(T, F, F))
    e.unshare_mask()
    assertEquals(lmask(e.shrink_mask()), List(T, F, T))
    assert(ma.array(Seq(1, 2), Seq(F, F)).shrink_mask().hasNoMask)
  }

  // ------------------------------------------------------------------ shape manipulation

  test("reshape, transpose, ravel, flatten, squeeze, astype, copy") {
    val m = ma.array(Seq(Seq(1.0, 2.0, 3.0), Seq(4.0, 5.0, 6.0)), Seq(Seq(F, T, F), Seq(T, F, F)))
    assertEquals(opt(m.T), List(Some(1.0), None, None, Some(5.0), Some(3.0), Some(6.0)))
    assertEquals(m.T.shape, Seq(3, 2))
    assertEquals(opt(m.reshape(3, 2)), List(Some(1.0), None, Some(3.0), None, Some(5.0), Some(6.0)))
    assertEquals(opt(m.ravel()), List(Some(1.0), None, Some(3.0), None, Some(5.0), Some(6.0)))
    val r = m.ravel()
    r(0) = 100.0
    assertEquals(m(0, 0), 100.0)
    val f = m.flatten()
    f(2) = -1.0
    assertEquals(m(0, 2), 3.0)
    assertEquals(m.expandDims(0).shape, Seq(1, 2, 3))
    assertEquals(m.expandDims(0).squeeze().shape, Seq(2, 3))
    assertEquals(m.swapaxes(0, 1).shape, Seq(3, 2))
    val i = m.astype[Int]
    assertEquals(opt(i), List(Some(100), None, Some(3), None, Some(5), Some(6)))
    val cp = m.copy()
    cp(0, 0) = ma.masked
    assertEquals(m(0, 0), 100.0)
    assertEquals(ma.array(Seq(1, 2, 3), Seq(F, T, F)).astype[Double].repr,
      "masked_array(data=[1.0, --, 3.0],\n             mask=[False,  True, False],\n       fill_value=1e+20)")
  }

  // ------------------------------------------------------------------ arithmetic

  test("arithmetic with arrays and scalars unions masks and keeps masked data") {
    val a = ma.array(Seq(1L, 2L, 3L, 4L), Seq(F, T, F, F))
    assertEquals((a + 1L).repr, "masked_array(data=[2, --, 4, 5],\n             mask=[False,  True, False, False],\n       fill_value=999999)")
    assertEquals(opt(a * a), List(Some(1L), None, Some(9L), Some(16L)))
    assertEquals(opt(a - np.array(1L, 1L, 1L, 1L)), List(Some(0L), None, Some(2L), Some(3L)))
    assertEquals((a + 1L).data.toList, List(2L, 2L, 4L, 5L))
    assertEquals(opt(-a), List(Some(-1L), None, Some(-3L), Some(-4L)))
    val b = ma.array(Seq(1.0, 2.0), Seq(F, F))
    val c = ma.array(Seq(1.0, 2.0), Seq(T, F))
    assertEquals(lmask(b + c), List(T, F))
    val i = ma.array(Seq(1, 2))
    val d: MaskedArray[Double] = i + c
    assertEquals(opt(d), List(None, Some(4.0)))
    assert((ma.array(Seq(1.0)) + ma.array(Seq(2.0))).hasNoMask)
    // broadcasting
    val m2 = ma.array(Seq(Seq(1.0, 2.0), Seq(3.0, 4.0)), Seq(Seq(F, T), Seq(F, F)))
    val r = m2 + ma.array(Seq(10.0, 20.0), Seq(T, F))
    assertEquals(lmask(r), List(T, T, T, F))
  }

  test("division masks zero divisors and non-finite results") {
    val a = ma.array(Seq(1L, 2L, 3L, 4L), Seq(F, T, F, F))
    val z = a / 0L
    assertEquals(z.repr, "masked_array(data=[--, --, --, --],\n             mask=[ True,  True,  True,  True],\n       fill_value=1e+20,\n            dtype=float64)")
    assertEquals(z.data.toList, List(1.0, 2.0, 3.0, 4.0))
    assertEquals((a / 2L).repr, "masked_array(data=[0.5, --, 1.5, 2.0],\n             mask=[False,  True, False, False],\n       fill_value=1e+20)")
    val d = ma.array(Seq(1L, 0L, 3L), Seq(F, F, T))
    val q = ma.array(Seq(4L, 5L, 6L)) / d
    assertEquals(opt(q), List(Some(4.0), None, None))
    assertEquals(q.data.toList, List(4.0, 5.0, 6.0))
    val fd = ma.array(Seq(4L, 5L, 6L)).floorDiv(d)
    assertEquals(opt(fd), List(Some(4L), None, None))
    assertEquals(fd.data.toList, List(4L, 5L, 6L))
    val r = ma.array(Seq(1L, 2L, 3L), Seq(F, T, F)) % ma.array(Seq(2L, 0L, 2L))
    assertEquals(opt(r), List(Some(1L), None, Some(1L)))
    assert(ma.divide(ma.array(Seq(Double.NaN, 1.0)), np.array(1.0, 1.0)).maskArray.toList == List(T, F))
    val f32 = ma.array(Seq(1.0f, 2.0f)) / 2.0f
    assertEquals(f32.dtype.name, "float32")
  }

  test("power masks invalid results and stores the fill value") {
    val x = ma.array(Seq(1.0, 2.0, 3.0), Seq(F, T, F))
    val p = x ** 2.0
    assertEquals(opt(p), List(Some(1.0), None, Some(9.0)))
    assertEquals(p.data.toList, List(1.0, 2.0, 9.0))
    val q = ma.array(Seq(-1.0, 4.0)) ** 0.5
    assertEquals(opt(q), List(None, Some(2.0)))
    assertEquals(q.data.toList, List(1e20, 2.0))
    val r = ma.array(Seq(1.0, 2.0)) ** -1.0
    assert(r.hasNoMask)
    assertEquals(r.repr, "masked_array(data=[1. , 0.5],\n             mask=False,\n       fill_value=1e+20)")
    val s = ma.array(Seq(0.0, 2.0)) ** -1.0
    assertEquals(opt(s), List(None, Some(0.5)))
    assertEquals(ma.power(ma.array(Seq(2, 3)), np.array(2, 2)).toOptionList, List(Some(4), Some(9)))
  }

  test("comparisons") {
    val a = ma.array(Seq(1L, 2L, 3L), Seq(F, T, F))
    assertEquals((a > 1L).repr, "masked_array(data=[False, --, True],\n             mask=[False,  True, False],\n       fill_value=True)")
    assertEquals((a === 2L).repr, "masked_array(data=[False, --, False],\n             mask=[False,  True, False],\n       fill_value=True)")
    val e = a === ma.array(Seq(1L, 5L, 3L), Seq(F, T, T))
    assertEquals(e.repr, "masked_array(data=[True, --, --],\n             mask=[False,  True,  True],\n       fill_value=True)")
    assertEquals(e.data.toList, List(true, true, false))
    val n = ma.array(Seq(1.0, 2.0, 3.0), Seq(F, F, F)) < 2.0
    assert(n.hasNoMask)
    assertEquals(n.repr, "masked_array(data=[ True, False, False],\n             mask=False,\n       fill_value=True)")
    assertEquals(opt(a =!= np.array(1L, 1L, 1L)), List(Some(false), None, Some(true)))
    assertEquals(opt(a >= 3L), List(Some(false), None, Some(true)))
    assertEquals(opt(a <= 1L), List(Some(true), None, Some(false)))
    assertEquals(opt(ma.equal(a, np.array(2L))), List(Some(false), None, Some(false)))
    assertEquals(opt(ma.greater(a, np.array(1L))), List(Some(false), None, Some(true)))
    assertEquals(opt(ma.less_equal(a, np.array(2L))), List(Some(true), None, Some(false)))
  }

  // ------------------------------------------------------------------ elementwise math

  test("domained unary functions mask invalid inputs") {
    val s = ma.sqrt(ma.array(Seq(4L, -1L, 9L), Seq(F, F, T)))
    assertEquals(s.repr, "masked_array(data=[2.0, --, --],\n             mask=[False,  True,  True],\n       fill_value=1e+20)")
    assertEquals(s.data.toList, List(2.0, -1.0, 9.0))
    val l = ma.log(ma.array(Seq(1.0, -1.0, 4.0), Seq(F, F, F)))
    assertEquals(opt(l).map(_.map(x => math.round(x * 1e12) / 1e12)), List(Some(0.0), None, Some(math.round(math.log(4.0) * 1e12) / 1e12)))
    assertEquals(opt(ma.log10(ma.array(Seq(100.0, 0.0, -5.0, 1000.0)))), List(Some(2.0), None, None, Some(3.0)))
    assertEquals(opt(ma.log2(ma.array(Seq(8.0, 0.0)))), List(Some(3.0), None))
    val as = ma.arcsin(ma.array(Seq(0.5, 2.0)))
    assertEquals(lmask(as), List(F, T))
    assertEqualsDouble(as.data(0), 0.5235987755982989, 1e-15)
    assertEquals(lmask(ma.arccos(ma.array(Seq(-1.5, 1.0)))), List(T, F))
    assertEquals(lmask(ma.arccosh(ma.array(Seq(0.5, 1.0)))), List(T, F))
    assertEquals(lmask(ma.arctanh(ma.array(Seq(1.0, 0.5)))), List(T, F))
    val e = ma.exp(ma.array(Seq(1.0, 2.0), Seq(F, T)))
    assertEquals(e.repr, "masked_array(data=[2.718281828459045, --],\n             mask=[False,  True],\n       fill_value=1e+20)")
    assert(ma.exp(ma.array(Seq(1.0))).hasNoMask)
    assertEquals(lmask(ma.sqrt(np.array(Double.NaN, 1.0))), List(T, F))
  }

  test("other elementwise functions") {
    val x = ma.array(Seq(-1.5, 2.5, 3.0), Seq(F, F, T))
    assertEquals(opt(ma.absolute(x)), List(Some(1.5), Some(2.5), None))
    assertEquals(opt(ma.abs(ma.array(Seq(Complex(3, 4))))), List(Some(5.0)))
    assertEquals(opt(ma.floor(x)), List(Some(-2.0), Some(2.0), None))
    assertEquals(opt(ma.ceil(x)), List(Some(-1.0), Some(3.0), None))
    assertEquals(opt(ma.fabs(x)), List(Some(1.5), Some(2.5), None))
    assertEquals(opt(ma.negative(x)), List(Some(1.5), Some(-2.5), None))
    assertEquals(opt(ma.sin(ma.array(Seq(0.0)))), List(Some(0.0)))
    assertEquals(opt(ma.cos(ma.array(Seq(0.0)))), List(Some(1.0)))
    assertEquals(opt(ma.tan(ma.array(Seq(0.0)))), List(Some(0.0)))
    assertEquals(opt(ma.tanh(ma.array(Seq(0.0)))), List(Some(0.0)))
    assertEquals(opt(ma.sinh(ma.array(Seq(0.0)))), List(Some(0.0)))
    assertEquals(opt(ma.cosh(ma.array(Seq(0.0)))), List(Some(1.0)))
    assertEquals(opt(ma.arctan(ma.array(Seq(0.0)))), List(Some(0.0)))
    assertEquals(opt(ma.arcsinh(ma.array(Seq(0.0)))), List(Some(0.0)))
    assertEquals(opt(ma.around(ma.array(Seq(1.26, 2.5), Seq(F, T)), 1)), List(Some(1.3), None))
    assertEquals(opt(ma.conjugate(ma.array(Seq(Complex(1, 2))))), List(Some(Complex(1, -2))))
    assertEquals(opt(ma.logical_not(ma.array(Seq(0, 2), Seq(F, T)))), List(Some(true), None))
    assertEquals(opt(ma.logical_and(ma.array(Seq(true, true)), np.array(true, false))), List(Some(true), Some(false)))
    assertEquals(opt(ma.logical_or(ma.array(Seq(false, true), Seq(F, T)), np.array(false, false))), List(Some(false), None))
    assertEquals(opt(ma.logical_xor(ma.array(Seq(true, true)), np.array(true, false))), List(Some(false), Some(true)))
    assertEquals(opt(ma.maximum(ma.array(Seq(1, 5), Seq(T, F)), np.array(3, 3))), List(None, Some(5)))
    assertEquals(opt(ma.minimum(ma.array(Seq(1, 5)), np.array(3, 3))), List(Some(1), Some(3)))
    assertEquals(opt(ma.hypot(ma.array(Seq(3.0)), np.array(4.0))), List(Some(5.0)))
    assertEquals(opt(ma.arctan2(ma.array(Seq(0.0)), np.array(1.0))), List(Some(0.0)))
    assertEquals(opt(ma.add(ma.array(Seq(1, 2), Seq(F, T)), np.array(1, 1))), List(Some(2), None))
    assertEquals(opt(ma.subtract(np.array(5, 5), ma.array(Seq(1, 2), Seq(F, T)))), List(Some(4), None))
    assertEquals(opt(ma.multiply(np.array(2.0, 2.0), np.array(3.0, 4.0))), List(Some(6.0), Some(8.0)))
    assertEquals(opt(ma.true_divide(np.array(1, 2), np.array(0, 4))), List(None, Some(0.5)))
    assertEquals(opt(ma.floor_divide(np.array(7, 2), np.array(2, 0))), List(Some(3), None))
    assertEquals(opt(ma.mod(np.array(-7, 2), np.array(3, 0))), List(Some(2), None))
    assertEquals(opt(ma.fmod(np.array(-7, 2), np.array(3, 0))), List(Some(-1), None))
  }

  // ------------------------------------------------------------------ reductions

  private def m23 = ma.array(Seq(Seq(1.0, 2.0, 3.0), Seq(4.0, 5.0, 6.0)), Seq(Seq(F, T, F), Seq(T, F, F)))

  test("full reductions ignore masked values") {
    val m = m23
    assertEquals(m.sum(), 15.0)
    assertEquals(m.prod(), 90.0)
    assertEquals(m.mean(), 3.75)
    assertD(m.variance(), 3.6875)
    assertD(m.std(), 1.920286436967152)
    assertD(m.variance(1), 4.916666666666667)
    assertEquals(m.min(), 1.0)
    assertEquals(m.max(), 6.0)
    assertEquals(m.argmin(), 0)
    assertEquals(m.argmax(), 5)
    assertEquals(m.ptp(), 5.0)
    assertEquals(ma.array(Seq(1, 2, 3), Seq(F, T, F)).sum(), 4L)
    assertEquals(ma.array(Seq(1, 2, 3), Seq(F, T, F)).mean(), 2.0)
    assertEquals(ma.array(Seq(1, 2), Seq(F, T)).all(), true)
    assertEquals(ma.array(Seq(0, 2), Seq(F, T)).any(), false)
    // all masked
    val allm = ma.array(Seq(1, 2, 3), Seq(T, T, T))
    assert(allm.sum() == ma.masked)
    assert(allm.mean() == ma.masked)
    assert(allm.min() == ma.masked)
    assert(allm.all() == ma.masked)
    assert(allm.ptp() == ma.masked)
    assert(allm.variance() == ma.masked)
    assert(allm.std() == ma.masked)
    // nomask behaves like ndarray
    assertEquals(ma.array(Seq.empty[Double]).sum(), 0.0)
    assertEquals(ma.sum(np.array(1.0, 2.0)), 3.0)
  }

  test("axis reductions") {
    val m = m23
    assertEquals(m.sum(1).repr, "masked_array(data=[4.0, 11.0],\n             mask=[False, False],\n       fill_value=1e+20)")
    assertEquals(m.mean(0).repr, "masked_array(data=[1.0, 5.0, 4.5],\n             mask=[False, False, False],\n       fill_value=1e+20)")
    val v = m.variance(1, 0, false)
    assertEquals(v.repr, "masked_array(data=[1.  , 0.25],\n             mask=False,\n       fill_value=1e+20)")
    val s = m.std(0, 1, false)
    assertEquals(lmask(s), List(T, T, F))
    assertEqualsDouble(s.data(2), 2.1213203435596424, 1e-12)
    assertEquals(m.ptp(0).repr, "masked_array(data=[0.0, 0.0, 3.0],\n             mask=[False, False, False],\n       fill_value=1e+20)")
    assertEquals(m.argmin(0).toList, List(0, 1, 0))
    assertEquals(m.argmax(1).toList, List(2, 2))
    val b = ma.array(Seq(Seq(1L, 2L), Seq(3L, 4L)), Seq(Seq(F, T), Seq(T, T)))
    assertEquals(b.sum(0).repr, "masked_array(data=[1, --],\n             mask=[False,  True],\n       fill_value=999999)")
    assertEquals(b.sum(0).data.toList, List(1L, 0L))
    assertEquals(b.max(1).data.toList, List(1L, 999999L))
    assertEquals(opt(b.max(1)), List(Some(1L), None))
    assertEquals(b.mean(1).data.toList, List(1.0, 0.0))
    assertEquals(opt(b.mean(1)), List(Some(1.0), None))
    assertEquals(opt(b.variance(0, 0, false)), List(Some(0.0), None))
    val c = ma.array(Seq(Seq(1L, 2L), Seq(3L, 4L)), Seq(Seq(T, T), Seq(F, F)))
    assertEquals(opt(c.sum(1)), List(None, Some(7L)))
    assertEquals(opt(c.min(1)), List(None, Some(3L)))
    assertEquals(opt(c.any(1)), List(None, Some(true)))
    assertEquals(opt(ma.array(Seq(Seq(1, 0), Seq(3, 4)), Seq(Seq(F, T), Seq(F, F))).all(1)), List(Some(true), Some(true)))
    assertEquals(m.sum(Seq(0, 1), keepdims = true).shape, Seq(1, 1))
    assertEquals(m.prod(0).data.toList, List(1.0, 5.0, 18.0))
    assert(ma.array(Seq(Seq(1.0, 2.0))).sum(1).hasNoMask)
  }

  test("cumulative operations") {
    val b = ma.array(Seq(Seq(1L, 2L), Seq(3L, 4L)), Seq(Seq(F, T), Seq(T, T)))
    assertEquals(b.cumsum().repr, "masked_array(data=[1, --, --, --],\n             mask=[False,  True,  True,  True],\n       fill_value=999999)")
    assertEquals(b.cumsum().data.toList, List(1L, 1L, 1L, 1L))
    assertEquals(opt(b.cumsum(0)), List(Some(1L), None, None, None))
    val m = m23
    assertEquals(opt(m.cumprod(1)), List(Some(1.0), None, Some(3.0), None, Some(5.0), Some(30.0)))
    assertEquals(opt(ma.cumsum(ma.array(Seq(1, 2, 3), Seq(F, T, F)))), List(Some(1L), None, Some(4L)))
  }

  test("median and average") {
    assertEquals(ma.median(ma.array(Seq(1, 2, 3, 4), Seq(F, F, F, T))), 2.0)
    assertEquals(ma.median(ma.array(Seq(1, 2, 3, 4))), 2.5)
    val md = ma.median(ma.array(Seq(Seq(1, 2, 7), Seq(3, 4, 5)), Seq(Seq(F, T, F), Seq(F, F, F))), 1)
    assertEquals(opt(md), List(Some(4.0), Some(4.0)))
    val b = ma.array(Seq(Seq(1L, 2L), Seq(3L, 4L)), Seq(Seq(F, T), Seq(T, T)))
    assertEquals(opt(ma.median(b, 0)), List(Some(1.0), None))
    assertEquals(ma.median(np.array(Seq(Seq(1, 2, 3), Seq(4, 5, 6))), 0).data.toList, List(2.5, 3.5, 4.5))
    assert(ma.median(ma.array(Seq(1.0), Seq(T))) == ma.masked)
    assertEquals(ma.average(ma.array(Seq(1.0, 2.0, 3.0, 4.0), Seq(F, F, T, F)), np.array(1.0, 2.0, 3.0, 4.0)), 3.0)
    assertEquals(ma.average(ma.array(Seq(1.0, 2.0, 3.0, 4.0), Seq(F, F, T, F))), 7.0 / 3)
    val av = ma.average(ma.array(Seq(Seq(1.0, 2.0), Seq(3.0, 4.0)), Seq(Seq(F, T), Seq(F, F))), 0)
    assertEquals(av.repr, "masked_array(data=[2.0, 4.0],\n             mask=[False, False],\n       fill_value=1e+20)")
    val aw = ma.average(ma.array(Seq(Seq(1.0, 2.0), Seq(3.0, 4.0)), Seq(Seq(F, T), Seq(F, F))), 1, np.array(1.0, 3.0))
    assertEquals(opt(aw), List(Some(1.0), Some(3.75)))
    assertEquals(ma.mean(ma.array(Seq(1.0, 3.0))), 2.0)
    assertD(ma.`var`(ma.array(Seq(1.0, 3.0, 100.0), Seq(F, F, T))), 1.0)
    assertD(ma.std(ma.array(Seq(1.0, 3.0, 100.0), Seq(F, F, T)), None, 1), math.sqrt(2.0))
    assertD(ma.`var`(ma.array(Seq(1.0, 3.0, 100.0), Seq(F, F, T)), None, 1), 2.0)
    assertEquals(opt(ma.variance(ma.array(Seq(Seq(1.0, 3.0))), 1)), List(Some(1.0)))
    assertEquals(opt(ma.std(ma.array(Seq(Seq(1.0, 3.0))), 1)), List(Some(1.0)))
    assertEquals(opt(ma.anom(ma.array(Seq(1.0, 3.0, 100.0), Seq(F, F, T)))), List(Some(-1.0), Some(1.0), None))
  }

  // ------------------------------------------------------------------ masked_* family

  test("masked_where and comparison-based constructors") {
    val r = ma.masked_where(np.array(1, 2, 3) > 1, np.array(1, 2, 3))
    assertEquals(opt(r), List(Some(1), None, None))
    assert(ma.masked_where(np.array(false, false), np.array(1, 2)).hasNoMask)
    val withMask = ma.masked_where(np.array(false, true, false), ma.array(Seq(1, 2, 3), Seq(T, F, F)))
    assertEquals(lmask(withMask), List(T, T, F))
    assertEquals(lmask(ma.masked_where(np.array(true), np.array(1, 2))), List(T, T))
    intercept[IndexOutOfBoundsException](ma.masked_where(np.array(true, false, true), np.array(1, 2)))
    val src = np.array(1.0, 2.0)
    val nocopy = ma.masked_where(np.array(true, false), src, copy = false)
    nocopy(1) = 5.0
    assertEquals(src(1), 5.0)
    assertEquals(opt(ma.masked_greater(np.array(1, 5, 3), 2)), List(Some(1), None, None))
    assertEquals(opt(ma.masked_greater_equal(np.array(1, 5, 3), 3)), List(Some(1), None, None))
    assertEquals(opt(ma.masked_less(np.array(1, 5, 3), 3)), List(None, Some(5), Some(3)))
    assertEquals(opt(ma.masked_less_equal(np.array(1, 5, 3), 3)), List(None, Some(5), None))
    assertEquals(opt(ma.masked_not_equal(np.array(1, 5, 3), 3)), List(None, None, Some(3)))
    assertEquals(opt(ma.masked_inside(np.array(1.0, 2.0, 3.0, 4.0), 3.0, 2.0)), List(Some(1.0), None, None, Some(4.0)))
    assertEquals(opt(ma.masked_outside(np.array(1.0, 2.0, 3.0, 4.0), 2.0, 3.0)), List(None, Some(2.0), Some(3.0), None))
    val eq = ma.masked_equal(np.array(1L, 2L, 1L), 1L)
    assertEquals(eq.repr, "masked_array(data=[--, 2, --],\n             mask=[ True, False,  True],\n       fill_value=1)")
    assertEquals(opt(ma.masked_object(np.array("a", "b"), "a")), List(None, Some("b")))
  }

  test("masked_invalid, masked_values, fix_invalid") {
    val mi = ma.masked_invalid(np.array(1.0, Double.NaN, Double.PositiveInfinity))
    assertEquals(opt(mi), List(Some(1.0), None, None))
    assertEquals(ma.masked_invalid(np.array(1L, 2L)).repr, "masked_array(data=[1, 2],\n             mask=[False, False],\n       fill_value=999999)")
    val mv = ma.masked_values(np.array(1.0, 1.1, 2.0, 1.0000001), 1.0)
    assertEquals(mv.repr, "masked_array(data=[--, 1.1, 2.0, --],\n             mask=[ True, False, False,  True],\n       fill_value=1.0)")
    assertEquals(ma.masked_values(np.array(1L, 2L, 3L), 2L).repr,
      "masked_array(data=[1, --, 3],\n             mask=[False,  True, False],\n       fill_value=2)")
    assert(ma.masked_values(np.array(1.0, 3.0), 2.0).hasNoMask)
    val fi = ma.fix_invalid(np.array(1.0, Double.NaN, Double.PositiveInfinity))
    assertEquals(fi.repr, "masked_array(data=[1.0, --, --],\n             mask=[False,  True,  True],\n       fill_value=1e+20)")
    assertEquals(fi.data.toList, List(1.0, 1e20, 1e20))
    assertEquals(ma.fix_invalid(np.array(Double.NaN), fill_value = 0.0).data.toList, List(0.0))
  }

  test("mask helpers") {
    assertEquals(ma.make_mask(Seq(0, 1, 0)).toList, List(F, T, F))
    assert(ma.make_mask(Seq(0, 0)) eq ma.nomask)
    assertEquals(ma.make_mask(np.array(0, 0), shrink = false).toList, List(F, F))
    assertEquals(ma.make_mask_none(2, 2).toList, List(F, F, F, F))
    assertEquals(ma.mask_or(np.array(true, false), np.array(false, false)).toList, List(T, F))
    assert(ma.mask_or(ma.nomask, np.array(false, false)) eq ma.nomask)
    assert(ma.mask_or(ma.nomask, ma.nomask) eq ma.nomask)
    val a = ma.array(Seq(1, 2), Seq(F, T))
    assert(ma.is_masked(a))
    assert(!ma.is_masked(ma.array(Seq(1, 2), Seq(F, F))))
    assert(!ma.is_masked(np.array(1, 2)))
    assert(ma.isMaskedArray(a))
    assert(ma.isMA(a))
    assert(!ma.isMA(np.array(1)))
    assert(ma.is_mask(np.array(true)))
    assertEquals(ma.getmask(a).toList, List(F, T))
    assert(ma.getmask(np.array(1, 2)) eq ma.nomask)
    assertEquals(ma.getmaskarray(np.array(1, 2)).toList, List(F, F))
    assertEquals(ma.getdata(a).toList, List(1, 2))
    assertEquals(ma.filled(a, 0).toList, List(1, 0))
    assertEquals(ma.filled(np.array(1, 2)).toList, List(1, 2))
    assertEquals(ma.compressed(a).toList, List(1))
    assertEquals(ma.count(a), 1)
    assertEquals(ma.count(a, 0).toList, List(1))
    assert(ma.harden_mask(a).hardmask)
    assert(!ma.soften_mask(a).hardmask)
    assertEquals(ma.common_fill_value(a, ma.array(Seq(3))), Some(999999))
  }

  // ------------------------------------------------------------------ creation & joining

  test("creation functions") {
    assertEquals(ma.zeros(3).repr, "masked_array(data=[0., 0., 0.],\n             mask=False,\n       fill_value=1e+20)")
    assertEquals(ma.ones[Int](2).toOptionList, List(Some(1), Some(1)))
    assertEquals(ma.empty(2).size, 2)
    assertEquals(ma.arange(3).toOptionList, List(Some(0), Some(1), Some(2)))
    assertEquals(ma.arange(1.0, 2.0, 0.5).toOptionList, List(Some(1.0), Some(1.5)))
    assertEquals(ma.masked_all(2).count(), 0)
    val z = ma.zeros_like(ma.array(Seq(1L, 2L), Seq(F, T)))
    assertEquals(z.repr, "masked_array(data=[0, --],\n             mask=[False,  True],\n       fill_value=999999)")
    assertEquals(opt(ma.ones_like(np.array(2.0))), List(Some(1.0)))
    assertEquals(ma.masked_all_like(np.array(1, 2)).count(), 0)
    assertEquals(ma.masked_array(Seq(1, 2), Seq(T, F)).count(), 1)
    assertEquals(ma.array(ma.array(Seq(1, 2), Seq(T, F))).count(), 1)
  }

  test("concatenate, stack, vstack, hstack, column_stack") {
    assertEquals(ma.concatenate(Seq(ma.array(Seq(1L, 2L), Seq(F, F)), np.array(Seq(3L)))).repr,
      "masked_array(data=[1, 2, 3],\n             mask=False,\n       fill_value=999999)")
    assertEquals(ma.concatenate(Seq(ma.array(Seq(1L, 2L), Seq(F, T)), np.array(Seq(3L)))).repr,
      "masked_array(data=[1, --, 3],\n             mask=[False,  True, False],\n       fill_value=999999)")
    val c2 = ma.concatenate(Seq(ma.array(Seq(Seq(1L, 2L)), Seq(Seq(F, T))), np.array(Seq(Seq(3L, 4L)))), axis = 0)
    assertEquals(c2.shape, Seq(2, 2))
    assertEquals(opt(c2), List(Some(1L), None, Some(3L), Some(4L)))
    assertEquals(opt(ma.concatenate(Seq(np.array(Seq(Seq(1L), Seq(2L))), np.array(Seq(Seq(3L), Seq(4L)))), 1)), List(Some(1L), Some(3L), Some(2L), Some(4L)))
    intercept[IllegalArgumentException](ma.concatenate(Seq(np.array(1, 2), np.array(Seq(Seq(1, 2))))))
    val st = ma.stack(Seq(np.arange(2L), np.arange(2L)))
    assertEquals(st.repr, "masked_array(\n  data=[[0, 1],\n        [0, 1]],\n  mask=[[False, False],\n        [False, False]],\n  fill_value=999999)")
    val s1 = ma.stack(Seq(ma.array(Seq(1L, 2L), Seq(F, T)), np.array(3L, 4L)), axis = 1)
    assertEquals(opt(s1), List(Some(1L), Some(3L), None, Some(4L)))
    val vs = ma.vstack(Seq(ma.array(Seq(1L, 2L), Seq(F, T)), np.array(3L, 4L)))
    assertEquals(vs.shape, Seq(2, 2))
    assertEquals(opt(vs), List(Some(1L), None, Some(3L), Some(4L)))
    val hs = ma.hstack(Seq(ma.array(Seq(1L, 2L), Seq(F, T)), np.array(3L, 4L)))
    assertEquals(opt(hs), List(Some(1L), None, Some(3L), Some(4L)))
    val cs = ma.column_stack(Seq(ma.array(Seq(1L, 2L), Seq(F, T)), np.array(3L, 4L)))
    assertEquals(opt(cs), List(Some(1L), Some(3L), None, Some(4L)))
  }

  // ------------------------------------------------------------------ selection & sorting

  test("where") {
    val w = ma.where(np.array(true, false, true), ma.array(Seq(1L, 2L, 3L), Seq(T, F, F)), ma.masked)
    assertEquals(w.repr, "masked_array(data=[--, --, 3],\n             mask=[ True,  True, False],\n       fill_value=999999)")
    val w2 = ma.where(ma.array(Seq(true, false, true), Seq(F, F, T)), np.array(1L, 2L, 3L), np.array(10L, 20L, 30L))
    assertEquals(w2.repr, "masked_array(data=[1, 20, --],\n             mask=[False, False,  True],\n       fill_value=999999)")
    assertEquals(opt(ma.where(np.array(true, false), ma.masked, np.array(1, 2))), List(None, Some(2)))
    assert(ma.where(np.array(true, false), np.array(1, 2), np.array(3, 4)).hasNoMask)
    assertEquals(ma.where(ma.array(Seq(true, true, false), Seq(F, T, F))).head.toList, List(0))
    assertEquals(ma.nonzero(ma.array(Seq(1, 2, 0), Seq(F, T, F))).head.toList, List(0))
  }

  test("sort, argsort, unique, clip") {
    val s = ma.sort(ma.array(Seq(3L, 1L, 2L, 5L), Seq(F, F, T, F)))
    assertEquals(s.repr, "masked_array(data=[1, 3, 5, --],\n             mask=[False, False, False,  True],\n       fill_value=999999)")
    assertEquals(ma.argsort(ma.array(Seq(3L, 1L, 2L, 5L), Seq(F, F, T, F))).toList, List(1, 0, 3, 2))
    assertEquals(opt(ma.sort(ma.array(Seq(3.0, 1.0, 2.0, 5.0), Seq(F, F, T, F)), endwith = false)), List(None, Some(1.0), Some(3.0), Some(5.0)))
    val m2 = ma.array(Seq(Seq(3, 1), Seq(0, 2)), Seq(Seq(F, F), Seq(T, F)))
    assertEquals(opt(ma.sort(m2, 0)), List(Some(3), Some(1), None, Some(2)))
    assertEquals(opt(ma.sort(m2, 1)), List(Some(1), Some(3), Some(2), None))
    val inPlace = ma.array(Seq(2.0, 1.0, 0.0), Seq(F, F, T))
    inPlace.sort()
    assertEquals(opt(inPlace), List(Some(1.0), Some(2.0), None))
    assertEquals(ma.sort(np.array(3, 1, 2)).toOptionList, List(Some(1), Some(2), Some(3)))
    val u = ma.unique(ma.array(Seq(3L, 1L, 3L, 2L, 5L), Seq(F, F, F, T, T)))
    assertEquals(u.repr, "masked_array(data=[1, 3, --],\n             mask=[False, False,  True],\n       fill_value=999999)")
    assertEquals(opt(ma.unique(np.array(2, 1, 2))), List(Some(1), Some(2)))
    assertEquals(ma.clip(ma.array(Seq(1L, 5L, 9L), Seq(F, T, F)), 2L, 6L).repr,
      "masked_array(data=[2, --, 6],\n             mask=[False,  True, False],\n       fill_value=999999)")
  }

  test("apply_along_axis") {
    val m = ma.array(Seq(Seq(1.0, 2.0, 3.0), Seq(4.0, 5.0, 6.0)), Seq(Seq(F, T, F), Seq(T, T, T)))
    val r = ma.apply_along_axis((row: MaskedArray[Double]) => row.sum(), 1, m)
    assertEquals(opt(r), List(Some(4.0), None))
    val c = ma.apply_along_axis((col: MaskedArray[Double]) => col.count().toDouble, 0, m)
    assertEquals(opt(c), List(Some(1.0), Some(0.0), Some(1.0)))
  }

  // ------------------------------------------------------------------ contiguous runs & 2-D helpers

  test("contiguous runs and edges") {
    val a = ma.array(np.arange(8), Seq(F, F, T, T, F, T, F, F))
    assertEquals(ma.notmasked_contiguous(a).map(_.toString), Seq("0:2", "4:5", "6:8"))
    assertEquals(ma.flatnotmasked_contiguous(a).map(_.toString), Seq("0:2", "4:5", "6:8"))
    assertEquals(ma.clump_masked(a).map(_.toString), Seq("2:4", "5:6"))
    assertEquals(ma.clump_unmasked(a).map(_.toString), Seq("0:2", "4:5", "6:8"))
    assertEquals(ma.flatnotmasked_edges(ma.array(np.arange(8), Seq(T, F, T, T, F, T, F, T))), Some((1, 6)))
    assertEquals(ma.flatnotmasked_edges(ma.array(Seq(1, 2), Seq(T, T))), None)
    assertEquals(ma.flatnotmasked_edges(np.array(1, 2)), Some((0, 1)))
    assertEquals(ma.clump_masked(np.array(1, 2)), Seq.empty)
    val m2 = ma.array(Seq(Seq(1, 2), Seq(3, 4)), Seq(Seq(F, T), Seq(F, F)))
    assertEquals(ma.notmasked_contiguous(m2, 0).map(_.map(_.toString)), Seq(Seq("0:2"), Seq("1:2")))
    assertEquals(ma.notmasked_contiguous(m2, 1).map(_.map(_.toString)), Seq(Seq("0:1"), Seq("0:2")))
  }

  test("mask_rows/cols and compress_rows/cols") {
    val m = ma.array(Seq(Seq(1L, 2L), Seq(3L, 4L)), Seq(Seq(F, T), Seq(F, F)))
    assertEquals(ma.mask_rows(m).repr, "masked_array(\n  data=[[--, --],\n        [3, 4]],\n  mask=[[ True,  True],\n        [False, False]],\n  fill_value=999999)")
    assertEquals(lmask(ma.mask_cols(m)), List(F, T, F, T))
    assertEquals(lmask(ma.mask_rowcols(m)), List(T, T, F, T))
    val c = ma.array(Seq(Seq(1, 2, 3), Seq(3, 4, 5)), Seq(Seq(F, T, F), Seq(F, F, F)))
    assertEquals(ma.compress_rowcols(c).toString, "[[3 5]]")
    assertEquals(ma.compress_cols(c).toString, "[[1 3]\n [3 5]]")
    assertEquals(ma.compress_rows(c).toString, "[[3 4 5]]")
    assertEquals(ma.compress_rows(np.array(Seq(Seq(1, 2)))).toString, "[[1 2]]")
    intercept[IllegalArgumentException](ma.mask_rows(ma.array(Seq(1, 2), Seq(T, F))))
  }

  test("dot, cov, corrcoef") {
    val a = ma.array(Seq(Seq(1L, 2L), Seq(3L, 4L)), Seq(Seq(F, T), Seq(F, F)))
    val ones = ma.array(Seq(Seq(1L, 1L), Seq(1L, 1L)))
    assertEquals(ma.dot(a, ones).repr, "masked_array(\n  data=[[1, 1],\n        [7, 7]],\n  mask=[[False, False],\n        [False, False]],\n  fill_value=999999)")
    assertEquals(ma.dot(a, ones, strict = true).repr, "masked_array(\n  data=[[--, --],\n        [7, 7]],\n  mask=[[ True,  True],\n        [False, False]],\n  fill_value=999999)")
    assertEquals(opt(a.dot(ones)), List(Some(1L), Some(1L), Some(7L), Some(7L)))
    val v = ma.dot(ma.array(Seq(1.0, 2.0), Seq(T, F)), ma.array(Seq(3.0, 4.0), Seq(F, T)))
    assertEquals(v.ndim, 0)
    assert(v.mask.item)
    val x = ma.array(Seq(Seq(1.0, 2.0, 3.0, 4.0), Seq(2.0, 4.0, 5.0, 9.0)), Seq(Seq(F, F, T, F), Seq(F, F, F, F)))
    val cv = ma.cov(x)
    assertEqualsDouble(cv.data(0, 0), 2.3333333333333335, 1e-12)
    assertEquals(cv.data.toList.drop(1), List(5.5, 5.5, 8.666666666666666))
    assertEquals(lmask(cv), List(F, F, F, F))
    val cc = ma.corrcoef(x)
    assertEqualsDouble(cc.data(0, 1), 1.2230613724908168, 1e-12)
    assertEqualsDouble(cc.data(0, 0), 1.0, 1e-12)
    val c1 = ma.cov(np.array(1.0, 2.0, 3.0))
    assertEquals(c1.ndim, 0)
    assertEqualsDouble(c1.data.item, 1.0, 1e-12)
    assertEquals(ma.cov(np.array(Seq(Seq(1.0, 2.0), Seq(2.0, 4.0), Seq(3.0, 6.0))), rowvar = false).data.toList, List(1.0, 2.0, 2.0, 4.0))
    assertEquals(ma.cov(np.array(1.0, 2.0, 3.0), bias = true).data.item, 2.0 / 3)
  }

  test("equality and misc methods") {
    val a = ma.array(Seq(1, 2, 3), Seq(F, T, F))
    val b = ma.array(Seq(1, 9, 3), Seq(F, T, F))
    assertEquals(a, b)
    assertNotEquals(a, ma.array(Seq(1, 2, 3)))
    assertEquals(a.copy().hashCode, a.hashCode)
    assertEquals(opt(a.clip(2, 2)), List(Some(2), None, Some(2)))
    assertEquals(a.nonzero.head.toList, List(0, 2))
    assertEquals(a.iterator.size, 3)
    assertEquals(opt(+a), opt(a))
    assertEquals(opt(ma.array(Seq(1.0, 2.0, 3.0), Seq(F, T, F)).anom(0)), List(Some(-1.0), None, Some(1.0)))
    assertEquals(a.argsort().toList, List(0, 2, 1))
    val bc = ma.array(Seq(1.0, 2.0), Seq(F, T)).broadcastTo(2, 2)
    assertEquals(lmask(bc), List(F, T, F, T))
    assertEquals(a.toOptionList, List(Some(1), None, Some(3)))
    assertEquals(ma.array(Seq(4.0)).item, 4.0)
    assert(ma.array(Seq(4.0), Seq(T)).item == ma.masked)
    intercept[IllegalArgumentException](a.item)
  }
