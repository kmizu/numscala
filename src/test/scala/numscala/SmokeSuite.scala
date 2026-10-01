package numscala

class SmokeSuite extends munit.FunSuite:
  test("inference and basics") {
    val z = np.zeros(2, 3)
    val zz: NDArray[Double] = z
    val zi = np.zeros[Int](4)
    assertEquals(zi.dtype.name, "int32")
    val a = np.array(1.0, 2.0, 3.0)
    val b = np.array(1, 2, 3)
    val c = a + b
    val cc: NDArray[Double] = c
    assertEquals(c.toList, List(2.0, 4.0, 6.0))
    val d = b / b
    val dd: NDArray[Double] = d
    val s = a.sum()
    val ss: Double = s
    val si: Long = b.sum()
    assertEquals(si, 6L)
    assertEquals((a * 2).toList, List(2.0, 4.0, 6.0))
    assertEquals((2.0 * a).toList, List(2.0, 4.0, 6.0))
    assertEquals((2 * b).toList, List(2, 4, 6))
    val m = np.arange(12.0).reshape(3, 4)
    assertEquals(m.toString, "[[ 0.  1.  2.  3.]\n [ 4.  5.  6.  7.]\n [ 8.  9. 10. 11.]]")
    assertEquals(m(1, ::).toList, List(4.0, 5.0, 6.0, 7.0))
    assertEquals(m(::, 1).toList, List(1.0, 5.0, 9.0))
    assertEquals(m("::-1", 0).toList, List(8.0, 4.0, 0.0))
    assertEquals(m(1, 2), 6.0)
    assertEquals(m(m > 8.0).toList, List(9.0, 10.0, 11.0))
    m(0, ::) = 0.0
    assertEquals(m(0, 3), 0.0)
    m(m > 9.0) = -1.0
    assertEquals(m(2, ::).toList, List(8.0, 9.0, -1.0, -1.0))
    assertEquals(np.array(1, 2, 3).repr, "array([1, 2, 3], dtype=int32)")
    assertEquals(np.array(0.1, 2.5, 1e-5).toString, "[1.0e-01 2.5e+00 1.0e-05]")
    assertEquals(np.array(1.5, 2.25).repr, "array([1.5 , 2.25])")
    val mm = np.arange(6.0).reshape(2, 3)
    assertEquals((mm @@ mm.T).toList, List(5.0, 14.0, 14.0, 50.0))
    assertEquals(mm.sum(0).toList, List(3.0, 5.0, 7.0))
    assertEquals(mm.mean(), 2.5)
    assertEquals(m(np.array(0, 2), ::).shape, Seq(2, 4))
    val t3 = np.arange(24).reshape(2, 3, 4)
    assertEquals(t3(::, np.array(0, 1), np.array(1, 2)).shape, Seq(2, 2))
    assertEquals(t3(0, ::, np.array(1, 2)).shape, Seq(2, 3))
    assertEquals(t3(---, 1).shape, Seq(2, 3))
    assertEquals(t3(None, ::).shape, Seq(1, 2, 3, 4))
  }

class CreationSmokeSuite extends munit.FunSuite:
  test("array nesting") {
    val a = np.array(Seq(Seq(1.0, 2.0), Seq(3.0, 4.0)))
    assertEquals(a.shape, Seq(2, 2))
    val b = np.array(Seq(1, 2), Seq(3, 4))
    assertEquals(b.shape, Seq(2, 2))
    assertEquals(b.dtype.name, "int32")
    val c = np.array(Seq(1, 2, 3))
    assertEquals(c.shape, Seq(3))
    val s = np.array(5.0)
    assertEquals(s.ndim, 0)
    assertEquals(np.arrayOf(Seq(1, 2), DType.Float64).toList, List(1.0, 2.0))
    assertEquals(np.linspace(0.0, 1.0, 5).toList, List(0.0, 0.25, 0.5, 0.75, 1.0))
    assertEquals(np.eye(2).repr, "array([[1., 0.],\n       [0., 1.]])")
    assertEquals(np.arange(0.0, 1.0, 0.25).size, 4)
    assertEquals(np.indices(2, 3).shape, Seq(2, 2, 3))
    val (x, y) = np.meshgrid(np.arange(3), np.arange(2))
    assertEquals(x.shape, Seq(2, 3))
    assertEquals(np.tril(np.ones(3, 3)).sum(), 6.0)
  }
