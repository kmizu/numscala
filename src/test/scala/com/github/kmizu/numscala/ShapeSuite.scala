package com.github.kmizu.numscala

class ShapeSuite extends munit.FunSuite:
  def check[T](a: NDArray[T], shape: Seq[Int], vals: Seq[T])(using munit.Location): Unit =
    assertEquals(a.shape, shape)
    assertEquals(a.toList, vals.toList)

  def ar(xs: Int*): NDArray[Int] = np.array(xs)

  test("shape, ndim, size") {
    val a = np.zeros(2, 3, 4)
    assertEquals(np.shape(a), Seq(2, 3, 4))
    assertEquals(np.ndim(a), 3)
    assertEquals(np.size(a), 24)
    assertEquals(np.size(a, 1), 3)
    assertEquals(np.size(a, -1), 4)
    intercept[IndexOutOfBoundsException](np.size(a, 3))
  }

  test("reshape and ravel") {
    val a = np.arange(6)
    check(np.reshape(a, 2, 3), Seq(2, 3), 0 until 6)
    check(np.reshape(a, Seq(3, -1), 'F'), Seq(3, 2), Seq(0, 3, 1, 4, 2, 5))
    check(np.reshape(a, Seq(-1, 3)), Seq(2, 3), 0 until 6)
    val m = np.reshape(a, 2, 3)
    check(np.ravel(m), Seq(6), 0 until 6)
    check(np.ravel(m, 'F'), Seq(6), Seq(0, 3, 1, 4, 2, 5))
    // ravel of a contiguous array is a view
    val r = np.ravel(m)
    r(0) = 42
    assertEquals(m(0, 0), 42)
    intercept[IllegalArgumentException](np.reshape(a, 4, 2))
    intercept[IllegalArgumentException](np.ravel(m, 'X'))
  }

  test("resize") {
    check(np.resize(np.arange(3), 2, 4), Seq(2, 4), Seq(0, 1, 2, 0, 1, 2, 0, 1))
    check(np.resize(np.array(Seq(Seq(0, 1), Seq(2, 3))), 1, 3), Seq(1, 3), Seq(0, 1, 2))
    check(np.resize(np.arange(4), Seq(2, 2)), Seq(2, 2), Seq(0, 1, 2, 3))
    check(np.resize(np.zeros[Int](0), 3), Seq(3), Seq(0, 0, 0))
    check(np.resize(np.arange(4), 0), Seq(0), Seq())
    intercept[IllegalArgumentException](np.resize(np.arange(4), -1))
  }

  test("transpose, permute_dims, matrix_transpose, swapaxes") {
    val a = np.arange(24).reshape(2, 3, 4)
    assertEquals(np.transpose(a).shape, Seq(4, 3, 2))
    assertEquals(np.transpose(a, 1, 0, 2).shape, Seq(3, 2, 4))
    assertEquals(np.transpose(a, Seq(2, 0, 1)).shape, Seq(4, 2, 3))
    assertEquals(np.transpose(a, Seq(-1, 0, 1)).shape, Seq(4, 2, 3))
    assertEquals(np.permute_dims(a, Seq(1, 2, 0)).shape, Seq(3, 4, 2))
    assertEquals(np.permute_dims(a).shape, Seq(4, 3, 2))
    val mt = np.matrix_transpose(a)
    assertEquals(mt.shape, Seq(2, 4, 3))
    assertEquals(mt(1, 2, 0), a(1, 0, 2))
    intercept[IllegalArgumentException](np.matrix_transpose(np.arange(3)))
    val s = np.swapaxes(a, 0, 2)
    assertEquals(s.shape, Seq(4, 3, 2))
    assertEquals(s(3, 1, 0), a(0, 1, 3))
    // transpose is a view
    val m = np.arange(6).reshape(2, 3)
    np.transpose(m)(2, 1) = 99
    assertEquals(m(1, 2), 99)
    intercept[IllegalArgumentException](np.transpose(a, Seq(0, 1)))
  }

  test("moveaxis and rollaxis") {
    val x = np.zeros(3, 4, 5)
    assertEquals(np.moveaxis(x, 0, -1).shape, Seq(4, 5, 3))
    assertEquals(np.moveaxis(x, -1, 0).shape, Seq(5, 3, 4))
    assertEquals(np.moveaxis(x, Seq(0, 1), Seq(-1, -2)).shape, Seq(5, 4, 3))
    assertEquals(np.moveaxis(x, Seq(0, 1, 2), Seq(-1, -2, -3)).shape, Seq(5, 4, 3))
    assertEquals(np.moveaxis(x, Seq(2, 0), Seq(0, 2)).shape, Seq(5, 4, 3))
    intercept[IllegalArgumentException](np.moveaxis(x, Seq(0, 1), Seq(0)))
    intercept[IllegalArgumentException](np.moveaxis(x, Seq(0, 0), Seq(1, 2)))
    intercept[IndexOutOfBoundsException](np.moveaxis(x, 3, 0))
    val a = np.ones(3, 4, 5, 6)
    assertEquals(np.rollaxis(a, 3, 1).shape, Seq(3, 6, 4, 5))
    assertEquals(np.rollaxis(a, 2).shape, Seq(5, 3, 4, 6))
    assertEquals(np.rollaxis(a, 1, 4).shape, Seq(3, 5, 6, 4))
    assertEquals(np.rollaxis(a, 1, 2).shape, Seq(3, 4, 5, 6))
    assertEquals(np.rollaxis(a, -1, -2).shape, Seq(3, 4, 6, 5))
    intercept[IndexOutOfBoundsException](np.rollaxis(a, 0, 5))
  }

  test("expand_dims and squeeze") {
    val x = np.array(1, 2)
    assertEquals(np.expand_dims(x, 0).shape, Seq(1, 2))
    assertEquals(np.expand_dims(x, 1).shape, Seq(2, 1))
    assertEquals(np.expand_dims(x, -1).shape, Seq(2, 1))
    assertEquals(np.expand_dims(x, Seq(0, 1)).shape, Seq(1, 1, 2))
    assertEquals(np.expand_dims(x, Seq(2, 0)).shape, Seq(1, 2, 1))
    assertEquals(np.expand_dims(x, Seq(0, -1)).shape, Seq(1, 2, 1))
    intercept[IllegalArgumentException](np.expand_dims(x, Seq(0, 0)))
    intercept[IndexOutOfBoundsException](np.expand_dims(x, 3))
    val y = np.zeros(1, 3, 1)
    assertEquals(np.squeeze(y).shape, Seq(3))
    assertEquals(np.squeeze(y, 0).shape, Seq(3, 1))
    assertEquals(np.squeeze(y, Seq(0, 2)).shape, Seq(3))
    assertEquals(np.squeeze(np.zeros(1, 1)).shape, Seq())
    intercept[IllegalArgumentException](np.squeeze(y, 1))
  }

  test("atleast_1d / 2d / 3d") {
    val s = NDArray.scalar(1.0)
    assertEquals(np.atleast_1d(s).shape, Seq(1))
    assertEquals(np.atleast_2d(s).shape, Seq(1, 1))
    assertEquals(np.atleast_3d(s).shape, Seq(1, 1, 1))
    val v = np.arange(3.0)
    assertEquals(np.atleast_1d(v).shape, Seq(3))
    assertEquals(np.atleast_2d(v).shape, Seq(1, 3))
    assertEquals(np.atleast_3d(v).shape, Seq(1, 3, 1))
    val m = np.arange(6.0).reshape(2, 3)
    assertEquals(np.atleast_2d(m).shape, Seq(2, 3))
    assertEquals(np.atleast_3d(m).shape, Seq(2, 3, 1))
    assertEquals(np.atleast_3d(np.zeros(2, 3, 4, 5)).shape, Seq(2, 3, 4, 5))
    assertEquals(np.atleast_1d(s, v).map(_.shape), Seq(Seq(1), Seq(3)))
    assertEquals(np.atleast_2d(Seq(s, v, m)).map(_.shape), Seq(Seq(1, 1), Seq(1, 3), Seq(2, 3)))
    assertEquals(np.atleast_3d(v, m).map(_.shape), Seq(Seq(1, 3, 1), Seq(2, 3, 1)))
    // views
    np.atleast_3d(m)(1, 2, 0) = 50.0
    assertEquals(m(1, 2), 50.0)
  }

  test("broadcast_to, broadcast_shapes, broadcast_arrays") {
    val x = np.array(1, 2, 3)
    check(np.broadcast_to(x, 3, 3), Seq(3, 3), Seq(1, 2, 3, 1, 2, 3, 1, 2, 3))
    check(np.broadcast_to(x, Seq(2, 3)), Seq(2, 3), Seq(1, 2, 3, 1, 2, 3))
    check(np.broadcast_to(NDArray.scalar(7), 2), Seq(2), Seq(7, 7))
    val e = intercept[IllegalArgumentException](np.broadcast_to(x, 4))
    assert(e.getMessage.contains("(3,) and requested shape (4,)"), e.getMessage)
    intercept[IllegalArgumentException](np.broadcast_to(np.zeros(2, 3), 3))
    assertEquals(np.broadcast_shapes(Seq(1, 2), Seq(3, 1), Seq(3, 2)), Seq(3, 2))
    assertEquals(np.broadcast_shapes(Seq(6, 7), Seq(5, 6, 1), Seq(7), Seq(5, 1, 7)), Seq(5, 6, 7))
    assertEquals(np.broadcast_shapes(), Seq())
    assertEquals(np.broadcast_shapes(Seq(0), Seq(1)), Seq(0))
    val e2 = intercept[IllegalArgumentException](np.broadcast_shapes(Seq(3), Seq(4)))
    assert(e2.getMessage.contains("arg 0 with shape (3,) and arg 1 with shape (4,)"), e2.getMessage)
    val Seq(p, q) = np.broadcast_arrays(Seq(np.array(Seq(Seq(1, 2, 3))), np.array(Seq(Seq(4), Seq(5)))))
    check(p, Seq(2, 3), Seq(1, 2, 3, 1, 2, 3))
    check(q, Seq(2, 3), Seq(4, 4, 4, 5, 5, 5))
    val (u, w) = np.broadcast_arrays(np.arange(3.0), np.array(Seq(Seq(1), Seq(2))))
    assertEquals(u.shape, Seq(2, 3))
    assertEquals(w.toList, List(1, 1, 1, 2, 2, 2))
    val (_, _, z) = np.broadcast_arrays(np.arange(3.0), np.arange(3), np.zeros(2, 1))
    assertEquals(z.shape, Seq(2, 3))
  }

  test("concatenate") {
    val a = np.array(Seq(Seq(1, 2), Seq(3, 4)))
    val b = np.array(Seq(Seq(5, 6)))
    check(np.concatenate(Seq(a, b)), Seq(3, 2), Seq(1, 2, 3, 4, 5, 6))
    check(np.concatenate(Seq(a, b), axis = 0), Seq(3, 2), Seq(1, 2, 3, 4, 5, 6))
    check(np.concatenate(Seq(a, b.T), axis = 1), Seq(2, 3), Seq(1, 2, 5, 3, 4, 6))
    check(np.concatenate(Seq(a, b.T), axis = -1), Seq(2, 3), Seq(1, 2, 5, 3, 4, 6))
    check(np.concatenate(Seq(a, b), axis = None), Seq(6), Seq(1, 2, 3, 4, 5, 6))
    check(np.concatenate(ar(1, 2), ar(3), ar()), Seq(3), Seq(1, 2, 3))
    // non-contiguous inputs
    check(np.concatenate(Seq(a.T, a("::-1", ::))), Seq(4, 2), Seq(1, 3, 2, 4, 3, 4, 1, 2))
    val c = np.concatenate(Seq(a, a))
    c(0, 0) = 100
    assertEquals(a(0, 0), 1)
    intercept[IllegalArgumentException](np.concatenate(Seq.empty[NDArray[Int]]))
    intercept[IllegalArgumentException](np.concatenate(Seq(NDArray.scalar(1), NDArray.scalar(2))))
    val e = intercept[IllegalArgumentException](np.concatenate(Seq(a, ar(1, 2))))
    assert(e.getMessage.contains("same number of dimensions"))
    val e2 = intercept[IllegalArgumentException](np.concatenate(Seq(a, b), axis = 1))
    assert(e2.getMessage.contains("along dimension 0, the array at index 0 has size 2 and the array at index 1 has size 1"), e2.getMessage)
    intercept[IndexOutOfBoundsException](np.concatenate(Seq(a, b), axis = 2))
    // strings
    val s = np.concatenate(Seq(NDArray.fromArray(Array("a", "b")), NDArray.fromArray(Array("c"))))
    assertEquals(s.toList, List("a", "b", "c"))
  }

  test("stack and friends") {
    val arrs = Seq.fill(10)(np.zeros(3, 4))
    assertEquals(np.stack(arrs).shape, Seq(10, 3, 4))
    assertEquals(np.stack(arrs, axis = 1).shape, Seq(3, 10, 4))
    assertEquals(np.stack(arrs, axis = 2).shape, Seq(3, 4, 10))
    assertEquals(np.stack(arrs, axis = -1).shape, Seq(3, 4, 10))
    val a = ar(1, 2, 3)
    val b = ar(4, 5, 6)
    check(np.stack(a, b), Seq(2, 3), Seq(1, 2, 3, 4, 5, 6))
    check(np.stack(Seq(a, b), axis = -1), Seq(3, 2), Seq(1, 4, 2, 5, 3, 6))
    check(np.stack(Seq(NDArray.scalar(1), NDArray.scalar(2))), Seq(2), Seq(1, 2))
    intercept[IllegalArgumentException](np.stack(Seq(a, ar(1, 2))))
    intercept[IllegalArgumentException](np.stack(Seq.empty[NDArray[Int]]))
    intercept[IndexOutOfBoundsException](np.stack(Seq(a, b), axis = 2))

    check(np.vstack(a, b), Seq(2, 3), Seq(1, 2, 3, 4, 5, 6))
    check(np.vstack(Seq(a.reshape(3, 1), b.reshape(3, 1))), Seq(6, 1), Seq(1, 2, 3, 4, 5, 6))
    check(np.row_stack(a, b), Seq(2, 3), Seq(1, 2, 3, 4, 5, 6))
    check(np.row_stack(Seq(a, b)), Seq(2, 3), Seq(1, 2, 3, 4, 5, 6))
    check(np.hstack(a, b), Seq(6), Seq(1, 2, 3, 4, 5, 6))
    check(np.hstack(Seq(a.reshape(3, 1), b.reshape(3, 1))), Seq(3, 2), Seq(1, 4, 2, 5, 3, 6))
    check(np.hstack(Seq(NDArray.scalar(1), NDArray.scalar(2))), Seq(2), Seq(1, 2))
    check(np.dstack(a, b), Seq(1, 3, 2), Seq(1, 4, 2, 5, 3, 6))
    check(np.dstack(Seq(a.reshape(3, 1), b.reshape(3, 1))), Seq(3, 1, 2), Seq(1, 4, 2, 5, 3, 6))
    check(np.column_stack(a, b), Seq(3, 2), Seq(1, 4, 2, 5, 3, 6))
    check(np.column_stack(Seq(a, np.arange(6).reshape(3, 2))), Seq(3, 3), Seq(1, 0, 1, 2, 2, 3, 3, 4, 5))
    check(np.column_stack(Seq(NDArray.scalar(1), NDArray.scalar(2))), Seq(1, 2), Seq(1, 2))
  }

  test("block") {
    val A = np.eye(2) * 2.0
    val B = np.eye(3) * 3.0
    val m = np.block(Seq(Seq(A, np.zeros(2, 3)), Seq(np.ones(3, 2), B)))
    check(
      m,
      Seq(5, 5),
      Seq[Double](2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 3, 0, 0, 1, 1, 0, 3, 0, 1, 1, 0, 0, 3)
    )
    check(np.block(Seq(ar(1, 2, 3), ar(4, 5, 6))), Seq(6), Seq(1, 2, 3, 4, 5, 6))
    check(np.block(Seq(Seq(ar(1, 2, 3)), Seq(ar(4, 5, 6)))), Seq(2, 3), Seq(1, 2, 3, 4, 5, 6))
    check(np.block(Seq(Seq(ar(1, 2), ar(3)), Seq(ar(4, 5, 6)))), Seq(2, 3), Seq(1, 2, 3, 4, 5, 6))
    // 2-D blocks in a 1-level list concatenate along the last axis
    check(np.block(Seq(np.ones[Int](2, 1), np.zeros[Int](2, 2))), Seq(2, 3), Seq(1, 0, 0, 1, 0, 0))
    val c = np.block(Seq(Seq(Seq(ar(1)), Seq(ar(2))), Seq(Seq(ar(3)), Seq(ar(4)))))
    check(c, Seq(2, 2, 1), Seq(1, 2, 3, 4))
    check(np.block(ar(7, 8)), Seq(2), Seq(7, 8))
    check(np.block(Seq(NDArray.scalar(1), NDArray.scalar(2))), Seq(2), Seq(1, 2))
    intercept[IllegalArgumentException](np.block(Seq(Seq(ar(1)), Seq.empty[NDArray[Int]])))
  }

  test("split family") {
    val x = np.arange(9.0)
    assertEquals(np.split(x, 3).map(_.toList), Seq(List(0.0, 1, 2), List(3.0, 4, 5), List(6.0, 7, 8)))
    val y = np.arange(8.0)
    assertEquals(
      np.split(y, Seq(3, 5, 6, 10)).map(_.toList),
      Seq(List(0.0, 1, 2), List(3.0, 4), List(5.0), List(6.0, 7), List())
    )
    assertEquals(np.array_split(y, 3).map(_.toList), Seq(List(0.0, 1, 2), List(3.0, 4, 5), List(6.0, 7)))
    assertEquals(np.array_split(np.arange(9), 4).map(_.toList), Seq(List(0, 1, 2), List(3, 4), List(5, 6), List(7, 8)))
    assertEquals(np.array_split(np.arange(2), 3).map(_.size), Seq(1, 1, 0))
    assertEquals(np.array_split(y, Seq(-3, 2)).map(_.toList), Seq(List(0.0, 1, 2, 3, 4), List(), List(2.0, 3, 4, 5, 6, 7)))
    intercept[IllegalArgumentException](np.split(y, 3))
    intercept[IllegalArgumentException](np.split(y, 0))
    val m = np.arange(16.0).reshape(4, 4)
    val hs = np.hsplit(m, 2)
    check(hs(0), Seq(4, 2), Seq(0.0, 1, 4, 5, 8, 9, 12, 13))
    check(hs(1), Seq(4, 2), Seq(2.0, 3, 6, 7, 10, 11, 14, 15))
    assertEquals(np.hsplit(m, Seq(3, 6)).map(_.shape), Seq(Seq(4, 3), Seq(4, 1), Seq(4, 0)))
    assertEquals(np.hsplit(np.arange(6), 3).map(_.toList), Seq(List(0, 1), List(2, 3), List(4, 5)))
    val vs = np.vsplit(m, 2)
    check(vs(1), Seq(2, 4), (8 until 16).map(_.toDouble))
    val ds = np.dsplit(np.arange(16.0).reshape(2, 2, 4), Seq(3, 6))
    assertEquals(ds.map(_.shape), Seq(Seq(2, 2, 3), Seq(2, 2, 1), Seq(2, 2, 0)))
    check(ds(1), Seq(2, 2, 1), Seq(3.0, 7, 11, 15))
    assertEquals(np.split(m, 2, axis = 1).map(_.shape), Seq(Seq(4, 2), Seq(4, 2)))
    intercept[IllegalArgumentException](np.hsplit(NDArray.scalar(1), 1))
    intercept[IllegalArgumentException](np.vsplit(np.arange(4), 2))
    intercept[IllegalArgumentException](np.dsplit(m, 2))
    // views
    hs(0)(0, 0) = -1.0
    assertEquals(m(0, 0), -1.0)
  }

  test("unstack") {
    val a = np.arange(6).reshape(2, 3)
    assertEquals(np.unstack(a).map(_.toList), Seq(List(0, 1, 2), List(3, 4, 5)))
    assertEquals(np.unstack(a, axis = 1).map(_.toList), Seq(List(0, 3), List(1, 4), List(2, 5)))
    assertEquals(np.unstack(a, axis = -1).length, 3)
    intercept[IllegalArgumentException](np.unstack(NDArray.scalar(1)))
  }

  test("tile") {
    val a = ar(0, 1, 2)
    check(np.tile(a, 2), Seq(6), Seq(0, 1, 2, 0, 1, 2))
    check(np.tile(a, Seq(2, 2)), Seq(2, 6), Seq(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2))
    check(np.tile(a, Seq(2, 1, 2)), Seq(2, 1, 6), Seq(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2))
    val b = np.array(Seq(Seq(1, 2), Seq(3, 4)))
    check(np.tile(b, 2), Seq(2, 4), Seq(1, 2, 1, 2, 3, 4, 3, 4))
    check(np.tile(b, Seq(2, 1)), Seq(4, 2), Seq(1, 2, 3, 4, 1, 2, 3, 4))
    assertEquals(np.tile(ar(1, 2, 3, 4), Seq(4, 1)).shape, Seq(4, 4))
    check(np.tile(a, 0), Seq(0), Seq())
    check(np.tile(NDArray.scalar(5), 3), Seq(3), Seq(5, 5, 5))
    check(np.tile(b, Seq(1, 1)), Seq(2, 2), Seq(1, 2, 3, 4))
    check(np.tile(b.T, 2), Seq(2, 4), Seq(1, 3, 1, 3, 2, 4, 2, 4))
  }

  test("repeat") {
    check(np.repeat(NDArray.scalar(3), 4), Seq(4), Seq(3, 3, 3, 3))
    val x = np.array(Seq(Seq(1, 2), Seq(3, 4)))
    check(np.repeat(x, 2), Seq(8), Seq(1, 1, 2, 2, 3, 3, 4, 4))
    check(np.repeat(x, 3, axis = 1), Seq(2, 6), Seq(1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4))
    check(np.repeat(x, Seq(1, 2), axis = 0), Seq(3, 2), Seq(1, 2, 3, 4, 3, 4))
    check(np.repeat(x, ar(0, 2), axis = -1), Seq(2, 2), Seq(2, 2, 4, 4))
    check(np.repeat(x, Seq(2), axis = 0), Seq(4, 2), Seq(1, 2, 1, 2, 3, 4, 3, 4))
    check(np.repeat(x, Seq(1, 0, 2, 1)), Seq(4), Seq(1, 3, 3, 4))
    intercept[IllegalArgumentException](np.repeat(x, Seq(1, 2, 3), axis = 0))
    intercept[IllegalArgumentException](np.repeat(x, -1))
    intercept[IndexOutOfBoundsException](np.repeat(x, 2, axis = 2))
  }

  test("flip, fliplr, flipud") {
    val A = np.arange(8).reshape(2, 2, 2)
    check(np.flip(A, 0), Seq(2, 2, 2), Seq(4, 5, 6, 7, 0, 1, 2, 3))
    check(np.flip(A, 1), Seq(2, 2, 2), Seq(2, 3, 0, 1, 6, 7, 4, 5))
    check(np.flip(A), Seq(2, 2, 2), Seq(7, 6, 5, 4, 3, 2, 1, 0))
    check(np.flip(A, Seq(0, 2)), Seq(2, 2, 2), Seq(5, 4, 7, 6, 1, 0, 3, 2))
    check(np.flip(A, -1), Seq(2, 2, 2), Seq(1, 0, 3, 2, 5, 4, 7, 6))
    check(np.flip(NDArray.scalar(3)), Seq(), Seq(3))
    check(np.flip(ar()), Seq(0), Seq())
    // a view with negative strides
    val f = np.flip(A, 0)
    f(0, 0, 0) = 40
    assertEquals(A(1, 0, 0), 40)
    val m = np.arange(6).reshape(2, 3)
    check(np.fliplr(m), Seq(2, 3), Seq(2, 1, 0, 5, 4, 3))
    check(np.flipud(m), Seq(2, 3), Seq(3, 4, 5, 0, 1, 2))
    check(np.flipud(ar(1, 2, 3)), Seq(3), Seq(3, 2, 1))
    intercept[IllegalArgumentException](np.fliplr(ar(1, 2)))
    intercept[IllegalArgumentException](np.flipud(NDArray.scalar(1)))
    // flipping a flipped (sliced) view
    check(np.flip(m("::-1", ::), 0), Seq(2, 3), Seq(0, 1, 2, 3, 4, 5))
    check(np.flip(m(::, "::2")), Seq(2, 2), Seq(5, 3, 2, 0))
  }

  test("roll") {
    val x = np.arange(10)
    check(np.roll(x, 2), Seq(10), Seq(8, 9, 0, 1, 2, 3, 4, 5, 6, 7))
    check(np.roll(x, -2), Seq(10), Seq(2, 3, 4, 5, 6, 7, 8, 9, 0, 1))
    check(np.roll(x, 12), Seq(10), Seq(8, 9, 0, 1, 2, 3, 4, 5, 6, 7))
    val x2 = x.reshape(2, 5)
    check(np.roll(x2, 1), Seq(2, 5), Seq(9, 0, 1, 2, 3, 4, 5, 6, 7, 8))
    check(np.roll(x2, -1), Seq(2, 5), Seq(1, 2, 3, 4, 5, 6, 7, 8, 9, 0))
    check(np.roll(x2, 1, axis = 0), Seq(2, 5), Seq(5, 6, 7, 8, 9, 0, 1, 2, 3, 4))
    check(np.roll(x2, -1, axis = 0), Seq(2, 5), Seq(5, 6, 7, 8, 9, 0, 1, 2, 3, 4))
    check(np.roll(x2, 1, axis = 1), Seq(2, 5), Seq(4, 0, 1, 2, 3, 9, 5, 6, 7, 8))
    check(np.roll(x2, -1, axis = 1), Seq(2, 5), Seq(1, 2, 3, 4, 0, 6, 7, 8, 9, 5))
    check(np.roll(x2, Seq(1, 1), axis = Seq(1, 0)), Seq(2, 5), Seq(9, 5, 6, 7, 8, 4, 0, 1, 2, 3))
    check(np.roll(x2, Seq(2, 1), axis = Seq(1, 1)), Seq(2, 5), Seq(2, 3, 4, 0, 1, 7, 8, 9, 5, 6))
    check(np.roll(x2, 1, axis = Seq(0, 1)), Seq(2, 5), Seq(9, 5, 6, 7, 8, 4, 0, 1, 2, 3))
    check(np.roll(x, Seq(1, 2)), Seq(10), Seq(7, 8, 9, 0, 1, 2, 3, 4, 5, 6))
    check(np.roll(ar(), 3), Seq(0), Seq())
    check(np.roll(np.zeros[Int](0, 3), 1, axis = 1), Seq(0, 3), Seq())
    intercept[IllegalArgumentException](np.roll(x2, Seq(1, 2, 3), axis = Seq(0, 1)))
  }

  test("rot90") {
    val m = np.array(Seq(Seq(1, 2), Seq(3, 4)))
    check(np.rot90(m), Seq(2, 2), Seq(2, 4, 1, 3))
    check(np.rot90(m, 2), Seq(2, 2), Seq(4, 3, 2, 1))
    check(np.rot90(m, 3), Seq(2, 2), Seq(3, 1, 4, 2))
    check(np.rot90(m, -1), Seq(2, 2), Seq(3, 1, 4, 2))
    check(np.rot90(m, 4), Seq(2, 2), Seq(1, 2, 3, 4))
    check(np.rot90(m, 1, (1, 0)), Seq(2, 2), Seq(3, 1, 4, 2))
    val m3 = np.arange(8).reshape(2, 2, 2)
    check(np.rot90(m3, 1, (1, 2)), Seq(2, 2, 2), Seq(1, 3, 0, 2, 5, 7, 4, 6))
    val r = np.arange(6).reshape(2, 3)
    check(np.rot90(r), Seq(3, 2), Seq(2, 5, 1, 4, 0, 3))
    check(np.rot90(r, 3), Seq(3, 2), Seq(3, 0, 4, 1, 5, 2))
    intercept[IllegalArgumentException](np.rot90(m, 1, (0, 0)))
    intercept[IllegalArgumentException](np.rot90(m, 1, (0, 2)))
    intercept[IllegalArgumentException](np.rot90(m, 1, (0, -2)))
    // view
    np.rot90(m)(0, 0) = 20
    assertEquals(m(0, 1), 20)
  }

  test("pad: constant, edge, wrap, empty") {
    val a = ar(1, 2, 3, 4, 5)
    check(np.pad(a, (2, 3), "constant", constant_values = (4, 6)), Seq(10), Seq(4, 4, 1, 2, 3, 4, 5, 6, 6, 6))
    check(np.pad(a, 1), Seq(7), Seq(0, 1, 2, 3, 4, 5, 0))
    check(np.pad(a, (2, 3), "edge"), Seq(10), Seq(1, 1, 1, 2, 3, 4, 5, 5, 5, 5))
    check(np.pad(a, (2, 3), "wrap"), Seq(10), Seq(4, 5, 1, 2, 3, 4, 5, 1, 2, 3))
    check(np.pad(ar(1, 2, 3), 4, "wrap"), Seq(11), Seq(3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1))
    check(np.pad(a, Seq(1, 2), "constant", constant_values = 9), Seq(8), Seq(9, 1, 2, 3, 4, 5, 9, 9))
    val m = np.array(Seq(Seq(1, 2), Seq(3, 4)))
    check(np.pad(m, 1), Seq(4, 4), Seq(0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0))
    check(np.pad(m, Seq((1, 0), (0, 2))), Seq(3, 4), Seq(0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0))
    check(np.pad(m, Seq((1, 0))), Seq(3, 3), Seq(0, 0, 0, 0, 1, 2, 0, 3, 4))
    // corners take the later axis' constants
    check(
      np.pad(ar(5).reshape(1, 1), 1, constant_values = Seq((1, 2), (3, 4))),
      Seq(3, 3),
      Seq(3, 1, 4, 3, 5, 4, 3, 2, 4)
    )
    check(np.pad(m, 1, "edge"), Seq(4, 4), Seq(1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4))
    check(np.pad(m, (0, 1), "wrap"), Seq(3, 3), Seq(1, 2, 1, 3, 4, 3, 1, 2, 1))
    assertEquals(np.pad(m, 2, "empty").shape, Seq(6, 6))
    check(np.pad(np.arange(2.0), (1, 1), constant_values = 1.5), Seq(4), Seq(1.5, 0.0, 1.0, 1.5))
    check(np.pad(NDArray.scalar(3), 2), Seq(), Seq(3))
    // constant may extend empty axes
    check(np.pad(ar(), 2, constant_values = 7), Seq(4), Seq(7, 7, 7, 7))
    intercept[IllegalArgumentException](np.pad(ar(), 2, "edge"))
    check(np.pad(np.zeros[Int](0, 2), Seq((0, 0), (1, 1)), "edge"), Seq(0, 4), Seq())
    intercept[IllegalArgumentException](np.pad(a, 1, "bogus"))
    intercept[IllegalArgumentException](np.pad(a, -1))
    intercept[IllegalArgumentException](np.pad(m, Seq((1, 1), (1, 1), (1, 1))))
  }

  test("pad: reflect and symmetric") {
    val a = ar(1, 2, 3, 4, 5)
    check(np.pad(a, (2, 3), "reflect"), Seq(10), Seq(3, 2, 1, 2, 3, 4, 5, 4, 3, 2))
    check(np.pad(a, (2, 3), "reflect", reflect_type = "odd"), Seq(10), Seq(-1, 0, 1, 2, 3, 4, 5, 6, 7, 8))
    check(np.pad(a, (2, 3), "symmetric"), Seq(10), Seq(2, 1, 1, 2, 3, 4, 5, 5, 4, 3))
    check(np.pad(a, (2, 3), "symmetric", reflect_type = "odd"), Seq(10), Seq(0, 1, 1, 2, 3, 4, 5, 5, 6, 7))
    check(np.pad(ar(1, 2, 3), 5, "reflect"), Seq(13), Seq(2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2))
    check(np.pad(ar(1, 2, 3), 4, "symmetric"), Seq(11), Seq(3, 3, 2, 1, 1, 2, 3, 3, 2, 1, 1))
    check(np.pad(ar(1, 2, 3), 4, "reflect", reflect_type = "odd"), Seq(11), Seq(-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7))
    check(np.pad(ar(5), 2, "reflect"), Seq(5), Seq(5, 5, 5, 5, 5))
    check(np.pad(ar(5), 2, "symmetric"), Seq(5), Seq(5, 5, 5, 5, 5))
    val m = np.array(Seq(Seq(1, 2, 3), Seq(4, 5, 6)))
    check(
      np.pad(m, 1, "reflect"),
      Seq(4, 5),
      Seq(5, 4, 5, 6, 5, 2, 1, 2, 3, 2, 5, 4, 5, 6, 5, 2, 1, 2, 3, 2)
    )
    check(
      np.pad(m, 1, "symmetric"),
      Seq(4, 5),
      Seq(1, 1, 2, 3, 3, 1, 1, 2, 3, 3, 4, 4, 5, 6, 6, 4, 4, 5, 6, 6)
    )
    intercept[IllegalArgumentException](np.pad(a, 1, "reflect", reflect_type = "weird"))
  }

  test("pad: statistics and linear_ramp") {
    val a = ar(1, 2, 3, 4, 5)
    check(np.pad(a, (2, 3), "linear_ramp", end_values = (5, -4)), Seq(10), Seq(5, 3, 1, 2, 3, 4, 5, 2, -1, -4))
    check(np.pad(a, Seq(2), "maximum"), Seq(9), Seq(5, 5, 1, 2, 3, 4, 5, 5, 5))
    check(np.pad(a, Seq(2), "mean"), Seq(9), Seq(3, 3, 1, 2, 3, 4, 5, 3, 3))
    check(np.pad(a, Seq(2), "median"), Seq(9), Seq(3, 3, 1, 2, 3, 4, 5, 3, 3))
    check(np.pad(a, 2, "minimum"), Seq(9), Seq(1, 1, 1, 2, 3, 4, 5, 1, 1))
    check(np.pad(a, 2, "maximum", stat_length = (1, 2)), Seq(9), Seq(1, 1, 1, 2, 3, 4, 5, 5, 5))
    check(np.pad(a, 1, "mean", stat_length = 2), Seq(7), Seq(2, 1, 2, 3, 4, 5, 4))
    check(np.pad(ar(1, 2), 1, "mean"), Seq(4), Seq(2, 1, 2, 2))
    check(np.pad(ar(1, 4), 1, "median"), Seq(4), Seq(2, 1, 4, 2))
    check(np.pad(np.array(1.0, 2.0), 1, "mean"), Seq(4), Seq(1.5, 1.0, 2.0, 1.5))
    check(np.pad(np.array(1.0, Double.NaN), 1, "maximum").map(_.isNaN), Seq(4), Seq(true, false, true, true))
    intercept[IllegalArgumentException](np.pad(a, 1, "maximum", stat_length = 0))
    val m = np.array(Seq(Seq(1, 2), Seq(3, 4)))
    check(
      np.pad(m, Seq((3, 2), (2, 3)), "minimum"),
      Seq(7, 7),
      Seq(1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 3, 3, 3, 4, 3, 3, 3,
        1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1)
    )
    val r = np.pad(np.array(1.0, 2.0), (3, 0), "linear_ramp")
    assertEquals(r.shape, Seq(5))
    assertEqualsDouble(r(1), 1.0 / 3, 1e-15)
    assertEqualsDouble(r(2), 2.0 / 3, 1e-15)
    assertEquals(r(0), 0.0)
    // integer ramps are floored
    check(np.pad(ar(10), (3, 0), "linear_ramp"), Seq(4), Seq(0, 3, 6, 10))
    check(np.pad(ar(0), (0, 3), "linear_ramp", end_values = 10), Seq(4), Seq(0, 3, 6, 10))
    check(np.pad(m, 1, "mean"), Seq(4, 4), Seq(2, 2, 3, 2, 2, 1, 2, 2, 4, 3, 4, 4, 2, 2, 3, 2))
  }

  test("append") {
    check(np.append(ar(1, 2, 3), np.array(Seq(Seq(4, 5, 6), Seq(7, 8, 9)))), Seq(9), 1 to 9)
    val m = np.array(Seq(Seq(1, 2, 3), Seq(4, 5, 6)))
    check(np.append(m, np.array(Seq(Seq(7, 8, 9))), axis = 0), Seq(3, 3), 1 to 9)
    check(np.append(m, 7), Seq(7), 1 to 7)
    check(np.append(m, Seq(7, 8)), Seq(8), 1 to 8)
    val e = intercept[IllegalArgumentException](np.append(m, ar(7, 8, 9), axis = 0))
    assert(e.getMessage.contains("same number of dimensions"))
  }

  test("insert") {
    val a = np.arange(6).reshape(3, 2)
    check(np.insert(a, 1, 6), Seq(7), Seq(0, 6, 1, 2, 3, 4, 5))
    check(np.insert(a, 1, 6, axis = 1), Seq(3, 3), Seq(0, 6, 1, 2, 6, 3, 4, 6, 5))
    check(np.insert(a, Seq(1), np.array(Seq(Seq(7), Seq(8), Seq(9))), axis = 1), Seq(3, 3), Seq(0, 7, 1, 2, 8, 3, 4, 9, 5))
    check(np.insert(a, 1, Seq(7, 8, 9), axis = 1), Seq(3, 3), Seq(0, 7, 1, 2, 8, 3, 4, 9, 5))
    check(np.insert(a, 1, Seq(7, 8), axis = 0), Seq(4, 2), Seq(0, 1, 7, 8, 2, 3, 4, 5))
    val b = a.flatten()
    check(np.insert(b, Seq(2, 2), Seq(6, 7)), Seq(8), Seq(0, 1, 6, 7, 2, 3, 4, 5))
    check(np.insert(b, slice(2, 4), Seq(7, 8)), Seq(8), Seq(0, 1, 7, 2, 8, 3, 4, 5))
    check(np.insert(b, 2 until 4, Seq(7, 8)), Seq(8), Seq(0, 1, 7, 2, 8, 3, 4, 5))
    check(np.insert(b, Seq(4, 1), Seq(40, 10)), Seq(8), Seq(0, 10, 1, 2, 3, 40, 4, 5))
    check(np.insert(b, Seq(6, 0), 9), Seq(8), Seq(9, 0, 1, 2, 3, 4, 5, 9))
    val x = np.arange(8).reshape(2, 4)
    check(np.insert(x, Seq(1, 3), 999, axis = 1), Seq(2, 6), Seq(0, 999, 1, 2, 999, 3, 4, 999, 5, 6, 999, 7))
    check(np.insert(ar(1, 2, 3), Seq(1), Seq(10, 20)), Seq(5), Seq(1, 10, 20, 2, 3))
    check(np.insert(ar(1, 2, 3), -1, 9), Seq(4), Seq(1, 2, 9, 3))
    check(np.insert(ar(1, 2, 3), 3, 9), Seq(4), Seq(1, 2, 3, 9))
    check(np.insert(ar(1, 2, 3), ar(0, 3), ar(8, 9)), Seq(5), Seq(8, 1, 2, 3, 9))
    check(np.insert(ar(1, 2, 3), Seq.empty[Int], 9), Seq(3), Seq(1, 2, 3))
    check(np.insert(np.arange(2.0), 1, 0.5), Seq(3), Seq(0.0, 0.5, 1.0))
    intercept[IndexOutOfBoundsException](np.insert(ar(1, 2, 3), 4, 9))
    intercept[IndexOutOfBoundsException](np.insert(ar(1, 2, 3), Seq(0, 5), 9))
  }

  test("delete") {
    val arr = np.arange(1, 13).reshape(3, 4)
    check(np.delete(arr, 1, axis = 0), Seq(2, 4), Seq(1, 2, 3, 4, 9, 10, 11, 12))
    check(np.delete(arr, sliceFrom(0, 2), axis = 1), Seq(3, 2), Seq(2, 4, 6, 8, 10, 12))
    check(np.delete(arr, 0 until 4 by 2, axis = 1), Seq(3, 2), Seq(2, 4, 6, 8, 10, 12))
    check(np.delete(arr, Seq(1, 3, 5)), Seq(9), Seq(1, 3, 5, 7, 8, 9, 10, 11, 12))
    check(np.delete(arr, -1, axis = -1), Seq(3, 3), Seq(1, 2, 3, 5, 6, 7, 9, 10, 11))
    check(np.delete(ar(1, 2, 3), Seq(true, false, true)), Seq(1), Seq(2))
    check(np.delete(ar(1, 2, 3), np.array(false, true, false)), Seq(2), Seq(1, 3))
    check(np.delete(ar(1, 2, 3), ar(0, 0, -1)), Seq(1), Seq(2))
    check(np.delete(ar(1, 2, 3), Seq.empty[Int]), Seq(3), Seq(1, 2, 3))
    check(np.delete(ar(1, 2, 3), slice(1, 10)), Seq(1), Seq(1))
    intercept[IndexOutOfBoundsException](np.delete(ar(1, 2, 3), 3))
    intercept[IndexOutOfBoundsException](np.delete(ar(1, 2, 3), Seq(0, -4)))
    intercept[IllegalArgumentException](np.delete(ar(1, 2, 3), Seq(true, false)))
  }

  test("trim_zeros") {
    val a = ar(0, 0, 0, 1, 2, 3, 0, 2, 1, 0)
    check(np.trim_zeros(a), Seq(6), Seq(1, 2, 3, 0, 2, 1))
    check(np.trim_zeros(a, "b"), Seq(9), Seq(0, 0, 0, 1, 2, 3, 0, 2, 1))
    check(np.trim_zeros(a, "f"), Seq(7), Seq(1, 2, 3, 0, 2, 1, 0))
    check(np.trim_zeros(a, "BF"), Seq(6), Seq(1, 2, 3, 0, 2, 1))
    check(np.trim_zeros(ar(0, 0)), Seq(0), Seq())
    check(np.trim_zeros(np.array(0.0, Double.NaN, 0.0)).map(_.isNaN), Seq(1), Seq(true))
    val m = np.array(Seq(Seq(0, 0, 2, 3, 0, 0), Seq(0, 1, 0, 3, 0, 0), Seq(0, 0, 0, 0, 0, 0)))
    check(np.trim_zeros(m), Seq(2, 3), Seq(0, 2, 3, 1, 0, 3))
    check(np.trim_zeros(m, axis = -1), Seq(3, 3), Seq(0, 2, 3, 1, 0, 3, 0, 0, 0))
    intercept[IllegalArgumentException](np.trim_zeros(a, "x"))
  }

  test("misc dtypes and edge cases") {
    // complex mean / median padding
    val c = NDArray.fromArray(Array(Complex(1, 1), Complex(3, -1)))
    check(np.pad(c, 1, "mean"), Seq(4), Seq(Complex(2, 0), Complex(1, 1), Complex(3, -1), Complex(2, 0)))
    // string arrays
    val s = NDArray.fromArray(Array("a", "b"))
    check(np.pad(s, 1, "edge"), Seq(4), Seq("a", "a", "b", "b"))
    check(np.tile(s, 2), Seq(4), Seq("a", "b", "a", "b"))
    check(np.flip(s), Seq(2), Seq("b", "a"))
    // float32 and boolean
    val f = np.arange(3.0).astypeOf(DType.Float32)
    check(np.roll(f, 1), Seq(3), Seq(2.0f, 0.0f, 1.0f))
    check(np.concatenate(Seq(np.array(true, false), np.array(Seq(true)))), Seq(3), Seq(true, false, true))
    // linear_ramp on 2-D with per-axis end values
    val m = np.ones[Int](1, 1) * 4
    check(
      np.pad(m, Seq((1, 0), (0, 2)), "linear_ramp", end_values = Seq((0, 0), (0, 8))),
      Seq(2, 3),
      Seq(0, 4, 8, 4, 6, 8)
    )
    // insert a 2-D block along axis 0 with a sequence of indices
    val a = np.arange(4).reshape(2, 2)
    check(np.insert(a, Seq(0, 2), np.array(Seq(Seq(8, 9))), axis = 0), Seq(4, 2), Seq(8, 9, 0, 1, 2, 3, 8, 9))
    check(np.delete(a, np.array(true, false), axis = 1), Seq(2, 1), Seq(1, 3))
    check(np.reshape(a, 4), Seq(4), Seq(0, 1, 2, 3))
    check(np.repeat(np.arange(3), ar(1, 2, 0)), Seq(3), Seq(0, 1, 1))
    check(np.stack(Seq(a, a), axis = 1), Seq(2, 2, 2), Seq(0, 1, 0, 1, 2, 3, 2, 3))
    assertEquals(np.rollaxis(np.zeros(2, 3), 0, 2).shape, Seq(3, 2))
    assertEquals(np.rollaxis(np.zeros(2, 3), 0, 1).shape, Seq(2, 3))
  }

  test("copyto") {
    val dst = np.zeros(2, 3)
    np.copyto(dst, np.array(1.0, 2.0, 3.0))
    check(dst, Seq(2, 3), Seq(1.0, 2, 3, 1, 2, 3))
    val mask = np.array(Seq(Seq(true, false, true), Seq(false, true, false)))
    np.copyto(dst, np.full(Seq(2, 3), 9.0), where = mask)
    check(dst, Seq(2, 3), Seq(9.0, 2, 9, 1, 9, 3))
    np.copyto(dst, np.array(7, 8, 9))
    check(dst, Seq(2, 3), Seq(7.0, 8, 9, 7, 8, 9))
    np.copyto(dst, -1.0)
    assert(dst.toList.forall(_ == -1.0))
    np.copyto(dst, np.array(5.0), where = np.array(true, false, false))
    check(dst, Seq(2, 3), Seq(5.0, -1, -1, 5, -1, -1))
    val idst = np.zeros[Int](3)
    intercept[IllegalArgumentException](np.copyto(idst, np.array(1.5, 2.5, 3.5)))
    np.copyto(idst, np.array(1.5, 2.5, -3.5), casting = "unsafe")
    check(idst, Seq(3), Seq(1, 2, -3))
    intercept[IllegalArgumentException](np.copyto(idst, np.array(1L, 2L, 3L), casting = "safe"))
    np.copyto(idst, np.array(4L, 5L, 6L))
    check(idst, Seq(3), Seq(4, 5, 6))
    intercept[IllegalArgumentException](np.copyto(idst, np.array(1, 2), casting = "no"))
    intercept[IllegalArgumentException](np.copyto(dst, np.array(1.0, 2.0)))
    // overlapping source and destination
    val o = np.arange(5)
    np.copyto(o(1 until 5), o(0 until 4))
    check(o, Seq(5), Seq(0, 0, 1, 2, 3))
  }
