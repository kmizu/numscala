package com.github.kmizu.numscala

/** Tests of the `np.ma` aliases, shape/selection helpers, set routines and friends
  * (expected values from NumPy 2.4).
  */
class MAExtrasSuite extends munit.FunSuite:
  private val ma = np.ma
  private val F = false
  private val T = true

  private def opt[X](a: MaskedArray[X]): List[Option[X]] = a.toOptionList
  private def lmask(a: MaskedArray[?]): List[Boolean] = a.maskArray.toList

  // a = [1, --, 3, 4], b = [1, 5, --, 9], x = [[1, --], [3, 4]]
  private def a = ma.array(Seq(1L, 2L, 3L, 4L), Seq(F, T, F, F))
  private def b = ma.array(Seq(1L, 5L, 3L, 9L), Seq(F, F, T, F))
  private def x = ma.array(Seq(Seq(1L, 2L), Seq(3L, 4L)), Seq(Seq(F, T), Seq(F, F)))
  private def c = ma.array(Seq(1L, 2L))

  // ------------------------------------------------------------------ aliases & attributes

  test("amax, amin, product, anomalies, round_") {
    assertEquals(ma.amax(a), 4L)
    assertEquals(ma.amin(a), 1L)
    assertEquals(opt(ma.amax(x, 0)), List(Some(3L), Some(4L)))
    assertEquals(opt(ma.amin(x, 1)), List(Some(1L), Some(3L)))
    assertEquals(ma.product(a), 12L)
    assertEquals(opt(ma.product(x, 0)), List(Some(3L), Some(4L)))
    val an = ma.anomalies(a)
    assertEqualsDouble(an.data.flatGet(0), -1.6666666666666665, 1e-12)
    assertEqualsDouble(an.data.flatGet(3), 1.3333333333333335, 1e-12)
    assertEquals(lmask(an), List(F, T, F, F))
    assertEquals(opt(ma.anomalies(x, 0)).map(_.map(math.round)), List(Some(-1L), None, Some(1L), Some(0L)))
    assertEquals(opt(ma.round_(ma.array(Seq(1.25, 2.5), Seq(F, T)), 1)), List(Some(1.2), None))
  }

  test("alltrue and sometrue") {
    assertEquals(ma.alltrue(a), true)
    assertEquals(ma.sometrue(ma.array(Seq(0L, 1L), Seq(F, T))), false)
    assertEquals(ma.alltrue(ma.array(Seq(1L), Seq(T))), MaskedConstant)
    val y = ma.array(Seq(Seq(1L, 0L), Seq(1L, 1L)), Seq(Seq(F, T), Seq(F, F)))
    assertEquals(opt(ma.alltrue(y, 0)), List(Some(true), Some(true)))
    assertEquals(opt(ma.sometrue(y, 1)), List(Some(true), Some(true)))
    intercept[IllegalArgumentException](ma.alltrue(y))
    intercept[IllegalArgumentException](ma.sometrue(y))
  }

  test("angle") {
    val z = ma.angle(ma.array(Seq(Complex(1, 1), Complex(-1, 0)), Seq(F, T)))
    assertEqualsDouble(z.data.flatGet(0), 0.7853981633974483, 1e-15)
    assertEquals(lmask(z), List(F, T))
    val d = ma.angle(ma.array(Seq(1.0, -1.0), Seq(F, T)), deg = true)
    assertEquals(opt(d), List(Some(0.0), None))
  }

  test("ndim, shape, size, ids, flatten_mask, bool_, copy") {
    assertEquals(ma.ndim(x), 2)
    assertEquals(ma.shape(x), Seq(2, 2))
    assertEquals(ma.size(x), 4)
    assertEquals(ma.size(x, 1), 2)
    assertEquals(ma.size(x, -1), 2)
    val y = x
    val (di, mi) = ma.ids(y)
    assertEquals(di, System.identityHashCode(y.data))
    assertEquals(mi, System.identityHashCode(y.mask))
    assertEquals(ma.ids(c)._2, System.identityHashCode(ma.nomask))
    assertEquals(ma.flatten_mask(x.mask).toList, List(F, T, F, F))
    assertEquals(ma.flatten_mask(ma.nomask).toList, List(F))
    assert(ma.bool_ eq DType.Bool)
    val cp = ma.copy(a)
    cp(0) = 9L
    assertEquals(opt(a).head, Some(1L))
    assertEquals(opt(cp), List(Some(9L), None, Some(3L), Some(4L)))
  }

  test("MAError, MaskError and masked_print_option") {
    val e = intercept[MA.MaskError](ma.array(Seq(1, 2, 3), Seq(T, F)))
    assert(e.isInstanceOf[MA.MAError])
    assert(e.isInstanceOf[IllegalArgumentException])
    val err: MA.MAError = new ma.MAError("boom")
    assertEquals(err.getMessage, "boom")
    val opt0 = ma.masked_print_option
    assertEquals(opt0.display(), "--")
    assertEquals(opt0.toString, "--")
    assert(opt0.enabled())
    try
      opt0.set_display("??")
      assertEquals(a.toString, "[1 ?? 3 4]")
      assertEquals(MaskedConstant.toString, "??")
      opt0.enable(false)
      assert(!opt0.enabled())
    finally
      opt0.set_display("--")
      opt0.enable()
    assertEquals(a.toString, "[1 -- 3 4]")
  }

  // ------------------------------------------------------------------ creation

  test("identity, indices, fromfunction, frombuffer") {
    val i = ma.identity[Double](2)
    assertEquals(i.data.toList, List(1.0, 0.0, 0.0, 1.0))
    assert(i.hasNoMask)
    val ix = ma.indices(2, 3)
    assertEquals(ix.shape, Seq(2, 2, 3))
    assertEquals(ix.data.toList, List(0, 0, 0, 1, 1, 1, 0, 1, 2, 0, 1, 2))
    val f = ma.fromfunction(2, 2)(s => s(0) + s(1))
    assertEquals(f.data.toList, List(0, 1, 1, 2))
    val bytes = java.nio.ByteBuffer.allocate(16).order(java.nio.ByteOrder.LITTLE_ENDIAN).putDouble(1.0).putDouble(2.0).array()
    val fb = ma.frombuffer(bytes)
    assertEquals(fb.data.toList, List(1.0, 2.0))
    assert(fb.hasNoMask)
  }

  // ------------------------------------------------------------------ shape manipulation

  test("ravel, reshape, resize, transpose, swapaxes, squeeze, expand_dims") {
    assertEquals(opt(ma.ravel(x)), List(Some(1L), None, Some(3L), Some(4L)))
    val r = ma.reshape(a, 2, 2)
    assertEquals(r.shape, Seq(2, 2))
    assertEquals(lmask(r), List(F, T, F, F))
    assertEquals(lmask(ma.reshape(a, Seq(2, 2), 'F')), List(F, F, T, F))
    val rs = ma.resize(a, 2, 3)
    assertEquals(opt(rs), List(Some(1L), None, Some(3L), Some(4L), Some(1L), None))
    val rs2 = ma.resize(c, 3)
    assertEquals(rs2.data.toList, List(1L, 2L, 1L))
    assert(rs2.hasNoMask)
    assertEquals(opt(ma.transpose(x)), List(Some(1L), Some(3L), None, Some(4L)))
    assertEquals(opt(ma.swapaxes(x, 0, 1)), List(Some(1L), Some(3L), None, Some(4L)))
    val sq = ma.squeeze(ma.array(Seq(Seq(1L, 2L)), Seq(Seq(F, T))))
    assertEquals(sq.shape, Seq(2))
    assertEquals(opt(sq), List(Some(1L), None))
    assertEquals(ma.squeeze(ma.array(Seq(Seq(1L, 2L))), 0).shape, Seq(2))
    val e = ma.expand_dims(a, 0)
    assertEquals(e.shape, Seq(1, 4))
    assertEquals(lmask(e), List(F, T, F, F))
    assert(ma.expand_dims(c, 0).hasNoMask)
  }

  test("atleast_1d/2d/3d always give a full mask") {
    val s = ma.atleast_1d(ma.array(5L))
    assertEquals(s.shape, Seq(1))
    assert(!s.hasNoMask)
    val t = ma.atleast_2d(c)
    assertEquals(t.shape, Seq(1, 2))
    assert(!t.hasNoMask)
    assertEquals(lmask(ma.atleast_2d(a)), List(F, T, F, F))
    val u = ma.atleast_3d(ma.array(Seq(Seq(1L, 2L)), Seq(Seq(F, T))))
    assertEquals(u.shape, Seq(1, 2, 1))
    assertEquals(lmask(u), List(F, T))
    assertEquals(ma.atleast_1d(Seq(a, c)).map(_.shape), Seq(Seq(4), Seq(2)))
    assertEquals(ma.atleast_2d(Seq(a)).head.shape, Seq(1, 4))
    assertEquals(ma.atleast_3d(Seq(c)).head.shape, Seq(1, 2, 1))
  }

  test("dstack, hsplit, append, mr_") {
    val d = ma.dstack(Seq(a, a))
    assertEquals(d.shape, Seq(1, 4, 2))
    assertEquals(lmask(d), List(F, F, T, T, F, F, F, F))
    assert(!ma.dstack(Seq(c, c)).hasNoMask)
    val h = ma.hsplit(x, 2)
    assertEquals(h.length, 2)
    assertEquals(h(0).shape, Seq(2, 1))
    assertEquals(opt(h(1)), List(None, Some(4L)))
    assertEquals(opt(ma.hsplit(c, Seq(1))(1)), List(Some(2L)))
    val ap = ma.append(a, b)
    assertEquals(opt(ap), List(Some(1L), None, Some(3L), Some(4L), Some(1L), Some(5L), None, Some(9L)))
    assertEquals(opt(ma.append(x, x, 0)).length, 8)
    assertEquals(ma.append(x, x, 1).shape, Seq(2, 4))
    val mr = ma.mr_(a, ma.array(Seq(7L, 8L)))
    assertEquals(opt(mr), List(Some(1L), None, Some(3L), Some(4L), Some(7L), Some(8L)))
    assert(ma.mr_(ma.array(Seq(1L, 2L)), ma.array(Seq(7L, 8L))).hasNoMask)
    assertEquals(ma.mr_(ma.array(5L), c).data.toList, List(5L, 1L, 2L))
  }

  test("apply_over_axes") {
    val r = ma.apply_over_axes[Long]((m, ax) => ma.sum(m, ax), x, Seq(0, 1))
    assertEquals(r.shape, Seq(1, 1))
    assertEquals(opt(r), List(Some(8L)))
    val r2 = ma.apply_over_axes[Long]((m, ax) => ma.sum(m, ax, keepdims = true), x, -1)
    assertEquals(opt(r2), List(Some(1L), Some(7L)))
    assertEquals(ma.apply_over_axes[Long]((m, _) => m.ravel(), x, 0).shape, Seq(1, 4))
    intercept[IllegalArgumentException](ma.apply_over_axes[Long]((_, _) => ma.array(5L), x, 0))
  }

  // ------------------------------------------------------------------ diagonals

  test("diag, diagflat, diagonal, trace") {
    assertEquals(opt(ma.diag(x)), List(Some(1L), Some(4L)))
    val dm = ma.diag(a)
    assertEquals(dm.shape, Seq(4, 4))
    assertEquals(dm(1, 1), MaskedConstant)
    assertEquals(dm(0, 1), 0L)
    assertEquals(MA.count_masked(dm), 1)
    assert(ma.diag(c).hasNoMask)
    assertEquals(opt(ma.diag(x, 1)), List(None))
    val df = ma.diagflat(ma.array(Seq(1L, 2L), Seq(F, T)))
    assertEquals(opt(df), List(Some(1L), Some(0L), Some(0L), None))
    assert(!ma.diagflat(c).hasNoMask)
    assertEquals(opt(ma.diagonal(x)), List(Some(1L), Some(4L)))
    assert(ma.diagonal(ma.array(Seq(Seq(1L, 2L), Seq(3L, 4L)))).hasNoMask)
    assertEquals(ma.trace(x), 5.0)
    assertEquals(ma.trace(ma.array(Seq(Seq(1L, 2L), Seq(3L, 4L)), Seq(Seq(T, F), Seq(F, T)))), 0.0)
    assertEquals(ma.trace(ma.array(Seq(Seq(1L, 2L), Seq(3L, 4L)))), 5.0)
    assertEquals(ma.trace(x, 1), 0.0)
    val cube = ma.array(Seq(Seq(Seq(1.0, 2.0), Seq(3.0, 4.0)), Seq(Seq(5.0, 6.0), Seq(7.0, 8.0))))
    assertEquals(ma.trace(cube, 0, 1, 2).toList, List(5.0, 13.0))
    intercept[IllegalArgumentException](ma.trace(cube))
  }

  // ------------------------------------------------------------------ differences

  test("diff") {
    assertEquals(opt(ma.diff(a)), List(None, None, Some(1L)))
    val d2 = ma.diff(b, n = 2)
    assertEquals(lmask(d2), List(T, T))
    assert(ma.diff(c).hasNoMask)
    assertEquals(ma.diff(c).data.toList, List(1L))
    assertEquals(ma.diff(a, 0), a)
    assertEquals(opt(ma.diff(x, axis = 0)), List(Some(2L), None))
    assertEquals(opt(ma.diff(c, prepend = ma.array(0L), append = ma.array(Seq(10L), Seq(T)))), List(Some(1L), Some(1L), None))
    assertEquals(ma.diff(ma.array(Seq(1.0, 4.0, 9.0, 16.0)), n = 2).data.toList, List(2.0, 2.0))
    assertEquals(ma.diff(c, n = 5).size, 0)
    intercept[IllegalArgumentException](ma.diff(a, -1))
    intercept[IllegalArgumentException](ma.diff(ma.array(1L)))
  }

  test("ediff1d") {
    assertEquals(opt(ma.ediff1d(a)), List(None, None, Some(1L)))
    val e = ma.ediff1d(a, to_end = ma.array(Seq(8L)), to_begin = ma.array(Seq(9L)))
    assertEquals(opt(e), List(Some(9L), None, None, Some(1L), Some(8L)))
    assert(ma.ediff1d(c).hasNoMask)
    val e2 = ma.ediff1d(c, to_end = ma.array(Seq(1L)))
    assertEquals(e2.data.toList, List(1L, 1L))
    assert(!e2.hasNoMask)
    assertEquals(ma.ediff1d(ma.array(Seq(5L))).size, 0)
  }

  // ------------------------------------------------------------------ selection

  test("choose") {
    val r = ma.choose(ma.array(Seq(0, 1, 0, 1)), Seq(a, b))
    assertEquals(r.data.toList, List(1L, 5L, 3L, 9L))
    assert(r.hasNoMask)
    val r2 = ma.choose(ma.array(Seq(1, 0, 1, 0), Seq(F, F, F, T)), Seq(a, b))
    assertEquals(opt(r2), List(Some(1L), None, None, None))
    val r3 = ma.choose(ma.array(Seq(2, -1)), Seq(c, c), mode = "wrap")
    assertEquals(r3.data.toList, List(1L, 2L))
  }

  test("compress and compress_nd") {
    val r = ma.compress(ma.array(Seq(T, F, T, T)), a)
    assertEquals(opt(r), List(Some(1L), Some(3L), Some(4L)))
    assert(!r.hasNoMask)
    assert(ma.compress(ma.array(Seq(T, F)), c).hasNoMask)
    val r2 = ma.compress(ma.array(Seq(T, F)), x, 1)
    assertEquals(r2.shape, Seq(2, 1))
    assertEquals(opt(r2), List(Some(1L), Some(3L)))
    assertEquals(ma.compress_nd(x).toList, List(3L))
    assertEquals(ma.compress_nd(x).shape, Seq(1, 1))
    assertEquals(ma.compress_nd(x, 0).toList, List(3L, 4L))
    assertEquals(ma.compress_nd(x, 1).shape, Seq(2, 1))
    assertEquals(ma.compress_nd(x, Seq(0, 1)).toList, List(3L))
    assertEquals(ma.compress_nd(c).toList, List(1L, 2L))
    assertEquals(ma.compress_nd(ma.array(Seq(1L, 2L), Seq(T, T))).size, 0)
    assertEquals(ma.compress_nd(a).toList, List(1L, 3L, 4L))
  }

  test("take") {
    assertEquals(opt(ma.take(a, ma.array(Seq(3, 1)))), List(Some(4L), None))
    val t = ma.take(x, ma.array(Seq(1)), 1)
    assertEquals(t.shape, Seq(2, 1))
    assertEquals(opt(t), List(None, Some(4L)))
    assert(ma.take(c, ma.array(Seq(0))).hasNoMask)
    assertEquals(opt(ma.take(c, ma.array(Seq(0, 1), Seq(F, T)))), List(Some(1L), None))
    assertEquals(opt(ma.take(c, ma.array(Seq(5)), mode = "clip")), List(Some(2L)))
    intercept[IndexOutOfBoundsException](ma.take(c, ma.array(Seq(5))))
  }

  test("repeat") {
    assertEquals(opt(ma.repeat(a, 2)), List(Some(1L), Some(1L), None, None, Some(3L), Some(3L), Some(4L), Some(4L)))
    val r = ma.repeat(x, Seq(1, 2), 0)
    assertEquals(r.shape, Seq(3, 2))
    assertEquals(opt(r), List(Some(1L), None, Some(3L), Some(4L), Some(3L), Some(4L)))
    assert(ma.repeat(c, 2).hasNoMask)
  }

  test("vander") {
    val v = ma.vander(ma.array(Seq(1L, 2L, 3L), Seq(F, T, F)))
    assertEquals(v.toList, List(1L, 1L, 1L, 0L, 0L, 0L, 9L, 3L, 1L))
    assertEquals(ma.vander(ma.array(Seq(1L, 2L, 3L)), 2).toList, List(1L, 1L, 2L, 1L, 3L, 1L))
  }

  // ------------------------------------------------------------------ placement

  test("put") {
    val p1 = a.copy()
    ma.put(p1, np.array(0, 1), ma.array(Seq(9L, 8L)))
    assertEquals(p1.data.toList, List(9L, 8L, 3L, 4L))
    assert(p1.hasNoMask)
    val p2 = a.copy()
    ma.put(p2, np.array(0, 2), ma.array(Seq(9L, 8L), Seq(T, F)))
    assertEquals(opt(p2), List(None, None, Some(8L), Some(4L)))
    val p3 = ma.array(Seq(1L, 2L, 3L), Seq(T, F, F), hard_mask = true)
    ma.put(p3, np.array(0, 1), ma.array(Seq(7L, 8L)))
    assertEquals(opt(p3), List(None, Some(8L), Some(3L)))
    val p4 = ma.array(Seq(1L, 2L, 3L))
    ma.put(p4, np.array(-1, 0), ma.array(Seq(7L), Seq(T)))
    assertEquals(opt(p4), List(None, Some(2L), None))
    val p5 = ma.array(Seq(1L, 2L, 3L))
    ma.put(p5, np.array(4, -4), ma.array(Seq(7L, 8L)), mode = "wrap")
    assertEquals(p5.data.toList, List(1L, 7L, 8L))
    ma.put(p5, np.array(10), ma.array(Seq(0L)), mode = "clip")
    assertEquals(p5.data.toList, List(1L, 7L, 0L))
    val nd = np.array(1L, 2L)
    ma.put(nd, np.array(0), ma.array(Seq(5L)))
    assertEquals(nd.toList, List(5L, 2L))
    intercept[IndexOutOfBoundsException](ma.put(p5, np.array(3), ma.array(Seq(0L))))
    intercept[IllegalArgumentException](ma.put(p5, np.array(0), ma.array(Seq(0L)), mode = "bogus"))
  }

  test("putmask") {
    val p1 = a.copy()
    ma.putmask(p1, np.array(T, T, F, F), ma.array(Seq(7L, 6L, 5L, 4L), Seq(F, F, T, F)))
    assertEquals(opt(p1), List(Some(7L), Some(6L), Some(3L), Some(4L)))
    assert(!p1.hasNoMask)
    val p2 = ma.array(Seq(1L, 2L, 3L))
    ma.putmask(p2, np.array(T, F, T), ma.array(Seq(7L, 6L, 5L), Seq(T, F, F)))
    assertEquals(opt(p2), List(None, Some(2L), Some(5L)))
    val p3 = ma.array(Seq(1L, 2L, 3L), Seq(T, F, F), hard_mask = true)
    ma.putmask(p3, np.array(T, T, F), ma.array(Seq(7L, 8L, 9L), Seq(F, T, F)))
    assertEquals(opt(p3), List(None, None, Some(3L)))
    val p4 = ma.array(Seq(1L, 2L, 3L))
    ma.putmask(p4, np.array(F, T, T), ma.array(0L))
    assertEquals(p4.data.toList, List(1L, 0L, 0L))
    assert(p4.hasNoMask)
    val p5 = ma.array(Seq(1L, 2L, 3L), Seq(T, F, F), hard_mask = true)
    ma.putmask(p5, np.array(T, F, F), ma.array(0L))
    assertEquals(lmask(p5), List(T, F, F))
  }

  // ------------------------------------------------------------------ comparisons

  test("allclose") {
    assert(!ma.allclose(a, b))
    assert(ma.allclose(a, ma.array(Seq(1L, 9L, 3L, 4L))))
    assert(!ma.allclose(a, ma.array(Seq(1L, 9L, 3L, 4L)), masked_equal = false))
    val inf = Double.PositiveInfinity
    assert(ma.allclose(ma.array(Seq(1.0, inf)), ma.array(Seq(1.0, inf))))
    assert(!ma.allclose(ma.array(Seq(1.0, inf)), ma.array(Seq(1.0, -inf))))
    assert(ma.allclose(ma.array(Seq(1.0, inf), Seq(F, T)), ma.array(Seq(1.0, 5.0))))
    assert(!ma.allclose(ma.array(Seq(1.0, 2.0)), ma.array(Seq(1.0, inf))))
    assert(ma.allclose(ma.array(Seq(1e10)), ma.array(Seq(1.00001e10))))
    assert(!ma.allclose(ma.array(Seq(Double.NaN)), ma.array(Seq(Double.NaN))))
    assert(ma.allclose(ma.array(Seq(Seq(1.0, 2.0))), ma.array(Seq(1.0, 2.0 + 1e-9))))
    assert(ma.allclose(ma.array(Seq(1.0)), ma.array(Seq(1.5)), atol = 0.6))
  }

  test("allequal") {
    assert(ma.allequal(a, ma.array(Seq(1L, 9L, 3L, 4L))))
    assert(!ma.allequal(a, ma.array(Seq(1L, 9L, 3L, 4L)), fill_value = false))
    assert(ma.allequal(c, ma.array(Seq(1L, 2L))))
    assert(!ma.allequal(c, ma.array(Seq(1L, 3L))))
    assert(ma.allequal(ma.array(Seq(1L, 2L), Seq(F, F)), c, fill_value = false))
    assert(!ma.allequal(a, b))
  }

  // ------------------------------------------------------------------ bitwise

  test("bitwise ufuncs and shifts") {
    val r = ma.bitwise_and(a, b)
    assertEquals(opt(r), List(Some(1L), None, None, Some(0L)))
    assertEquals(opt(ma.bitwise_or(a, b)), List(Some(1L), None, None, Some(13L)))
    assertEquals(opt(ma.bitwise_xor(a, b)), List(Some(0L), None, None, Some(13L)))
    assertEquals(opt(ma.left_shift(a, 1L)), List(Some(2L), None, Some(6L), Some(8L)))
    assertEquals(opt(ma.right_shift(a, 1L)), List(Some(0L), None, Some(1L), Some(2L)))
    assertEquals(opt(ma.left_shift(a, ma.array(Seq(1L, 1L, 2L, 3L)))), List(Some(2L), None, Some(12L), Some(32L)))
    assertEquals(opt(ma.right_shift(a, ma.array(Seq(1L, 1L, 1L, 2L)))), List(Some(0L), None, Some(1L), Some(1L)))
    assertEquals(opt(ma.bitwise_and(a, 6L)), List(Some(0L), None, Some(2L), Some(4L)))
    assertEquals(opt(ma.bitwise_or(a, 8L)), List(Some(9L), None, Some(11L), Some(12L)))
    assertEquals(opt(ma.bitwise_xor(a, 1L)), List(Some(0L), None, Some(2L), Some(5L)))
    assertEquals(opt(ma.bitwise_and(ma.array(Seq(T, T, F), Seq(F, T, F)), ma.array(Seq(T, F, T)))), List(Some(T), None, Some(F)))
    assert(ma.bitwise_and(c, c).hasNoMask)
  }

  // ------------------------------------------------------------------ products

  test("inner / innerproduct, outer / outerproduct") {
    assertEquals(ma.inner(a, a).item, 26L)
    assertEquals(ma.innerproduct(a, a).item, 26L)
    val ix = ma.inner(x, x)
    assertEquals(ix.data.toList, List(1L, 3L, 3L, 25L))
    assert(ix.hasNoMask)
    assertEquals(ma.inner(ma.array(2L), ma.array(Seq(3L))).data.toList, List(6L))
    intercept[IllegalArgumentException](ma.inner(ma.array(2L), ma.array(Seq(1L, 2L))))
    val o = ma.outer(a, ma.array(Seq(1L, 2L)))
    assertEquals(o.shape, Seq(4, 2))
    assertEquals(opt(o), List(Some(1L), Some(2L), None, None, Some(3L), Some(6L), Some(4L), Some(8L)))
    val op = ma.outerproduct(ma.array(Seq(1L, 2L)), ma.array(Seq(3L, 4L), Seq(T, F)))
    assertEquals(opt(op), List(None, Some(4L), None, Some(8L)))
    assert(ma.outer(c, c).hasNoMask)
    assert(ma.outer(c, ma.array(Seq(1L), Seq(F))).hasNoMask)
  }

  // ------------------------------------------------------------------ convolution

  test("convolve and correlate") {
    val one = ma.array(Seq(1L, 1L))
    assertEquals(opt(ma.convolve(a, one)), List(Some(1L), None, None, Some(7L), Some(4L)))
    val np1 = ma.convolve(a, one, propagate_mask = false)
    assertEquals(opt(np1), List(Some(1L), Some(1L), Some(3L), Some(7L), Some(4L)))
    assert(!np1.hasNoMask)
    assertEquals(opt(ma.correlate(a, one)), List(None, None, Some(7L)))
    assertEquals(opt(ma.correlate(a, one, mode = "full", propagate_mask = false)), List(Some(1L), Some(1L), Some(3L), Some(7L), Some(4L)))
    assertEquals(opt(ma.convolve(c, c)), List(Some(1L), Some(4L), Some(4L)))
    assertEquals(opt(ma.convolve(a, one, mode = "same")), List(Some(1L), None, None, Some(7L)))
    val v = ma.array(Seq(1L, 2L), Seq(F, T))
    assertEquals(opt(ma.convolve(ma.array(Seq(1L, 2L, 3L)), v, mode = "valid")), List(None, None))
    val all = ma.convolve(ma.array(Seq(1L, 2L), Seq(T, F)), ma.array(Seq(1L, 2L), Seq(F, T)), propagate_mask = false)
    assertEquals(opt(all), List(None, Some(2L), None))
  }

  // ------------------------------------------------------------------ set routines

  test("in1d and isin follow NumPy's masked algorithm") {
    assertEquals(opt(ma.in1d(a, ma.array(Seq(2L, 3L)))), List(Some(F), Some(F), Some(T), None))
    assertEquals(opt(ma.isin(a, ma.array(Seq(2L, 3L)))), List(Some(F), Some(F), Some(T), None))
    val r = ma.in1d(c, ma.array(Seq(2L)))
    assertEquals(r.data.toList, List(F, T))
    assert(r.hasNoMask)
    assertEquals(ma.in1d(c, ma.array(Seq(2L)), invert = true).data.toList, List(T, F))
    val i2 = ma.isin(ma.array(Seq(Seq(1L, 5L), Seq(2L, 7L))), ma.array(Seq(7L, 1L)))
    assertEquals(i2.shape, Seq(2, 2))
    assertEquals(i2.data.toList, List(T, F, F, T))
    assertEquals(ma.in1d(ma.array(Seq(1L, 2L, 3L)), ma.array(Seq(3L, 5L)), assume_unique = true).data.toList, List(F, F, T))
  }

  test("intersect1d, union1d, setdiff1d, setxor1d") {
    val p = ma.array(Seq(1L, 3L, 3L, 5L), Seq(F, F, T, T))
    val q = ma.array(Seq(3L, 1L, 1L, 5L), Seq(F, F, F, T))
    assertEquals(opt(ma.intersect1d(p, q)), List(Some(1L), Some(3L), None))
    assertEquals(ma.intersect1d(c, ma.array(Seq(2L))).data.toList, List(2L))
    assert(ma.intersect1d(c, ma.array(Seq(2L))).hasNoMask)
    assertEquals(ma.intersect1d(ma.array(Seq(1L, 2L)), ma.array(Seq(2L, 3L)), assume_unique = true).data.toList, List(2L))
    val u = ma.union1d(p, ma.array(Seq(3L, 1L, 1L, 2L), Seq(F, F, F, T)))
    assertEquals(opt(u), List(Some(1L), Some(3L), None))
    val u2 = ma.union1d(c, ma.array(Seq(5L)))
    assertEquals(u2.data.toList, List(1L, 2L, 5L))
    assert(u2.hasNoMask)
    val s = ma.setdiff1d(ma.array(Seq(1L, 3L, 3L, 5L, 7L), Seq(F, F, T, T, F)), q)
    assertEquals(opt(s), List(Some(7L)))
    assertEquals(ma.setdiff1d(c, ma.array(Seq(2L))).data.toList, List(1L))
    assertEquals(ma.setdiff1d(ma.array(Seq(4L, 1L)), ma.array(Seq(1L)), assume_unique = true).data.toList, List(4L))
    val xr = ma.setxor1d(ma.array(Seq(1L, 3L, 3L, 5L, 7L), Seq(F, F, T, T, F)), ma.array(Seq(3L, 1L, 4L, 2L), Seq(F, F, F, T)))
    assertEquals(opt(xr), List(Some(4L), Some(7L)))
    assertEquals(ma.setxor1d(c, ma.array(Seq(2L, 5L))).data.toList, List(1L, 5L))
    assertEquals(opt(ma.setxor1d(ma.array(Seq(1L, 2L), Seq(F, T)), ma.array(Seq(1L)))), List(None))
    assertEquals(ma.setxor1d(ma.array(Seq.empty[Long]), ma.array(Seq.empty[Long])).size, 0)
  }

  // ------------------------------------------------------------------ polyfit

  test("polyfit excludes masked points") {
    val px = ma.array(Seq(0.0, 1.0, 2.0, 3.0))
    val py = ma.array(Seq(1.0, 3.0, 100.0, 7.0), Seq(F, F, T, F))
    val coef = ma.polyfit(px, py, 1)
    assertEqualsDouble(coef.flatGet(0), 2.0, 1e-12)
    assertEqualsDouble(coef.flatGet(1), 1.0, 1e-12)
    val px2 = ma.array(Seq(0.0, 1.0, 2.0, 3.0), Seq(F, F, F, T))
    val coef2 = ma.polyfit(px2, ma.array(Seq(1.0, 3.0, 5.0, 100.0)), 1)
    assertEqualsDouble(coef2.flatGet(0), 2.0, 1e-12)
    val y2 = ma.array(Seq(Seq(1.0, 0.0), Seq(3.0, 1.0), Seq(5.0, 99.0), Seq(7.0, 3.0)), Seq(Seq(F, F), Seq(F, F), Seq(F, T), Seq(F, F)))
    val c2 = ma.polyfit(px, y2, 1)
    assertEquals(c2.shape, Seq(2, 2))
    assertEqualsDouble(c2.flatGet(0), 2.0, 1e-12)
    assertEqualsDouble(c2.flatGet(1), 1.0, 1e-12)
    val w = ma.array(Seq(1.0, 1.0, 1.0, 1.0), Seq(F, F, T, F))
    val c3 = ma.polyfit(px, ma.array(Seq(1.0, 3.0, 100.0, 7.0)), 1, w = w)
    assertEqualsDouble(c3.flatGet(0), 2.0, 1e-12)
    val c4 = ma.polyfit(px, ma.array(Seq(1.0, 3.0, 5.0, 7.0)), 1)
    assertEqualsDouble(c4.flatGet(1), 1.0, 1e-12)
    intercept[IllegalArgumentException](ma.polyfit(px, ma.zeros[Double](4, 1, 1), 1))
    intercept[IllegalArgumentException](ma.polyfit(px, py, 1, w = ma.array(Seq(1.0))))
  }
