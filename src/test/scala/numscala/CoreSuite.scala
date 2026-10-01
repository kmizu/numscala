package numscala

class FormatSuite extends munit.FunSuite:
  test("float arrays") {
    assertEquals(np.array(1.0, 2.0, 3.0).toString, "[1. 2. 3.]")
    assertEquals(np.array(0.1, 0.2).toString, "[0.1 0.2]")
    assertEquals(np.array(1e10, 1.0).toString, "[1.e+10 1.e+00]")
    assertEquals(np.array(Double.NaN, 1.0, Double.NegativeInfinity).toString, "[ nan   1. -inf]")
    assertEquals(np.array(-1.5, 2.0).toString, "[-1.5  2. ]")
    assertEquals(np.array(1.0 / 3).toString, "0.3333333333333333")
    assertEquals(np.array(1.0 / 3, 1.0).toString, "[0.33333333 1.        ]")
    assertEquals(np.array(1.0, 2.0).astype[Float].repr, "array([1., 2.], dtype=float32)")
  }
  test("int, bool, complex, string arrays") {
    assertEquals(np.array(1, 10, 100).toString, "[  1  10 100]")
    assertEquals(np.array(true, false).toString, "[ True False]")
    assertEquals(np.array(Complex(1, 2), Complex(3, -4)).toString, "[1.+2.j 3.-4.j]")
    assertEquals(np.array("a", "bcd").repr, "array(['a', 'bcd'], dtype='<U3')")
    assertEquals(np.array(1L, 2L).repr, "array([1, 2])")
  }
  test("nested, summarised and wrapped") {
    assertEquals(np.array(Seq(Seq(1L, 2L), Seq(3L, 4L))).repr, "array([[1, 2],\n       [3, 4]])")
    assertEquals(np.arange(8).reshape(2, 2, 2).toString, "[[[0 1]\n  [2 3]]\n\n [[4 5]\n  [6 7]]]")
    assertEquals(np.arange(2000).toString, "[   0    1    2 ... 1997 1998 1999]")
    assertEquals(
      np.arange(30).toString,
      "[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23\n 24 25 26 27 28 29]"
    )
  }
  test("empty and 0-d") {
    assertEquals(np.zeros(0).repr, "array([], dtype=float64)")
    assertEquals(np.zeros(0, 3).repr, "array([], shape=(0, 3), dtype=float64)")
    assertEquals(np.zeros(0).toString, "[]")
    assertEquals(np.array(5.0).repr, "array(5.)")
    assertEquals(np.array(5.0).toString, "5.0")
  }
  test("scalar formatting like Python repr") {
    assertEquals(Format.formatFloatShort(1e-5), "1e-05")
    assertEquals(Format.formatFloatShort(1e16), "1e+16")
    assertEquals(Format.formatFloatShort(1.5e16), "1.5e+16")
    assertEquals(Format.formatFloatShort(123.456), "123.456")
    assertEquals(Format.formatFloatShort(0.1 + 0.2), "0.30000000000000004")
    assertEquals(Format.formatFloatShort(1e15), "1000000000000000.0")
    assertEquals(Format.formatFloatShort(0.0001), "0.0001")
    assertEquals(Format.formatFloatShort(-0.0), "-0.0")
    assertEquals(Complex(1, -2).toString, "(1-2j)")
    assertEquals(Complex(0, 2).toString, "2j")
  }

class IndexingCoreSuite extends munit.FunSuite:
  val a = np.arange(24).reshape(2, 3, 4)
  test("basic slicing produces views") {
    val b = np.arange(10.0)
    val v = b("2:5")
    v(0) = 100.0
    assertEquals(b(2), 100.0)
    assert(v.base.isDefined)
    assertEquals(b("::-2").toList, List(9.0, 7.0, 5.0, 3.0, 1.0))
    assertEquals(b("-3:").toList, List(7.0, 8.0, 9.0))
    assertEquals(b(-3 until 0).toList, List(7.0, 8.0, 9.0))
    assertEquals(b(1 until 7 by 2).toList, List(1.0, 3.0, 5.0))
    assertEquals(b("5:2:-1").toList, List(5.0, 4.0, 3.0))
    assertEquals(b("10:20").size, 0)
  }
  test("ellipsis, newaxis, mixed") {
    assertEquals(a(---, 0).shape, Seq(2, 3))
    assertEquals(a(0, ---).shape, Seq(3, 4))
    assertEquals(a(None, 0, None).shape, Seq(1, 1, 3, 4))
    assertEquals(a(1, 2, 3), 23)
    assertEquals(a(-1, -1, -1), 23)
    intercept[IndexOutOfBoundsException](a(2, 0, 0))
    intercept[IndexOutOfBoundsException](a(0, 0))
  }
  test("advanced indexing placement rules") {
    // adjacent advanced indices keep their position
    assertEquals(a(::, np.array(0, 2), np.array(1, 3)).toList, List(1, 11, 13, 23))
    assertEquals(a(::, np.array(0, 2), ::).shape, Seq(2, 2, 4))
    // separated by a slice -> broadcast dims go first
    assertEquals(a(np.array(0, 1), ::, np.array(0, 1)).shape, Seq(2, 3))
    assertEquals(a(0, ::, np.array(0, 1)).shape, Seq(2, 3))
    assertEquals(a(np.array(1), np.array(0, 1, 2)).toList.size, 12)
    val m = a % 5 === 0
    assertEquals(a(m).toList, List(0, 5, 10, 15, 20))
    val row = np.array(true, false)
    assertEquals(a(row).shape, Seq(1, 3, 4))
    assertEquals(a(Seq(1, 0)).shape, Seq(2, 3, 4))
  }
  test("assignment through indices") {
    val b = np.zeros(3, 3)
    b(::, 1) = 1.0
    b(np.array(0, 2), np.array(0, 2)) = 5.0
    b(b > 4.0) = np.array(7.0, 8.0)
    assertEquals(b.toList, List(7.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 8.0))
    b(0, ::) = np.array(1.0, 2.0, 3.0)
    assertEquals(b(0, ::).toList, List(1.0, 2.0, 3.0))
    b(1 until 3, ::) := np.array(9.0, 9.0, 9.0)
    assertEquals(b.sum(), 6.0 + 54.0)
    b.T(0, 2) = -1.0
    assertEquals(b(2, 0), -1.0)
  }

class ViewSuite extends munit.FunSuite:
  test("reshape returns views when possible") {
    val a = np.arange(12.0)
    val r = a.reshape(3, 4)
    r(0, 0) = 42.0
    assertEquals(a(0), 42.0)
    val t = r.T
    assertEquals(t.shape, Seq(4, 3))
    assert(!t.isCContiguous)
    assert(t.isFContiguous)
    val tr = t.reshape(12)
    tr(0) = -1.0
    assertEquals(a(0), 42.0) // copy, not a view
    assertEquals(t.reshape(-1).toList.take(3), List(42.0, 4.0, 8.0))
    val sl = r(::, "1:3")
    assertEquals(sl.reshape(6).toList, List(1.0, 2.0, 5.0, 6.0, 9.0, 10.0))
    val r2 = r.reshape(2, 6)
    r2(1, 0) = 7.0
    assertEquals(a(6), 7.0)
    intercept[IllegalArgumentException](a.reshape(5, -1))
    assertEquals(a.reshape(2, -1).shape, Seq(2, 6))
    assertEquals(r.ravel().base.isDefined, true)
    assertEquals(r.flatten().base, None)
  }
  test("transpose/swapaxes/moveaxis/squeeze/expandDims/broadcastTo") {
    val a = np.zeros(2, 3, 4)
    assertEquals(a.transpose(1, 0, 2).shape, Seq(3, 2, 4))
    assertEquals(a.swapaxes(0, 2).shape, Seq(4, 3, 2))
    assertEquals(a.moveaxis(0, -1).shape, Seq(3, 4, 2))
    assertEquals(np.zeros(1, 3, 1).squeeze().shape, Seq(3))
    assertEquals(np.zeros(1, 3, 1).squeeze(0).shape, Seq(3, 1))
    assertEquals(np.zeros(3).expandDims(0).shape, Seq(1, 3))
    assertEquals(np.zeros(3).expandDims(-1).shape, Seq(3, 1))
    val b = np.arange(3.0).broadcastTo(2, 3)
    assertEquals(b.toList, List(0.0, 1.0, 2.0, 0.0, 1.0, 2.0))
    intercept[IllegalArgumentException](np.arange(3.0).broadcastTo(2, 4))
  }
  test("tolist, item, iterator") {
    val a = np.arange(4).reshape(2, 2)
    assertEquals(a.tolist, List(List(0, 1), List(2, 3)))
    assertEquals(np.array(3.5).item, 3.5)
    assertEquals(a.iterator.map(_.toList).toList, List(List(0, 1), List(2, 3)))
    assertEquals(a.item(3), 3)
  }

class DTypeSuite extends munit.FunSuite:
  test("casting") {
    assertEquals(np.array(1.7, -1.7).astype[Int].toList, List(1, -1))
    assertEquals(np.array(200, 300).astype[Byte].toList, List[Byte](-56, 44))
    assertEquals(np.array(Complex(1.5, 2)).astype[Double].toList, List(1.5))
    assertEquals(np.array("1.5", "2").astype[Double].toList, List(1.5, 2.0))
    assertEquals(np.array(1.5, 0.0).astype[Boolean].toList, List(true, false))
    assertEquals(np.array(1, 2).astype[String].toList, List("1", "2"))
    assertEquals(DType.byName("f8"), DType.Float64)
    assertEquals(DType.byName("int"), DType.Int64)
    assertEquals(DType.promote(DType.Int32, DType.Float32), DType.Float64)
    assertEquals(DType.promote(DType.Int8, DType.Float32), DType.Float32)
    assertEquals(DType.promote(DType.Bool, DType.Int16), DType.Int16)
  }
  test("arithmetic promotion follows NumPy") {
    val i = np.array(1, 2)
    val f = np.array(1.0f, 2.0f)
    val l = np.array(1L, 2L)
    val b = np.array(true, false)
    val s = np.array(1.toShort, 2.toShort)
    val c = np.array(Complex(0, 1), Complex(1, 0))
    val r1: NDArray[Double] = i + f
    val r2: NDArray[Float] = s + f
    val r3: NDArray[Long] = i * l
    val r4: NDArray[Int] = b + i
    val r5: NDArray[Complex] = c * f
    val r6: NDArray[Float] = f / f
    val r7: NDArray[Double] = l / l
    assertEquals(r1.dtype, DType.Float64)
    assertEquals(r2.dtype, DType.Float32)
    assertEquals(r4.toList, List(2, 2))
    assertEquals(r5.toList, List(Complex(0, 1), Complex(2, 0)))
    assertEquals(r6.toList, List(1.0f, 1.0f))
    assertEquals(r7.toList, List(1.0, 1.0))
    assertEquals((i ** i).toList, List(1, 4))
    assertEquals((np.array(7, -7) % 3).toList, List(1, 2))
    assertEquals((np.array(7.0, -7.0) % 3.0).toList, List(1.0, 2.0))
    assertEquals(np.array(7, -7).floorDiv(2).toList, List(3, -4))
    assertEquals((np.array(1, 2) / 2).toList, List(0.5, 1.0))
    assertEquals((1 - np.array(1.0, 2.0)).toList, List(0.0, -1.0))
    assertEquals((1.0 / np.array(2.0, 4.0)).toList, List(0.5, 0.25))
    assertEquals((~np.array(true, false)).toList, List(false, true))
    assertEquals((np.array(6, 3) & np.array(3, 3)).toList, List(2, 3))
    assertEquals((np.array(1, 2) << 2).toList, List(4, 8))
    intercept[ArithmeticException](np.array(2) ** np.array(-1))
  }
  test("comparisons with NaN and broadcasting") {
    val x = np.array(1.0, Double.NaN, 3.0)
    assertEquals((x < 2.0).toList, List(true, false, false))
    assertEquals((x >= np.array(1, 1, 1)).toList, List(true, false, true))
    assertEquals((x === x).toList, List(true, false, true))
    assertEquals((x =!= x).toList, List(false, true, false))
    val col = np.arange(3).reshape(3, 1)
    assertEquals((col < np.arange(3)).sum(), 3L)
  }
  test("in-place operators") {
    val a = np.ones(2, 2)
    a += 1.0
    a *= np.array(1.0, 2.0)
    assertEquals(a.toList, List(2.0, 4.0, 2.0, 4.0))
    a(0, ::) -= 1.0
    assertEquals(a.toList, List(1.0, 3.0, 2.0, 4.0))
    a /= 2.0
    assertEquals(a.toList, List(0.5, 1.5, 1.0, 2.0))
  }

class ReduceCoreSuite extends munit.FunSuite:
  val a = np.arange(12.0).reshape(3, 4)
  test("method reductions") {
    assertEquals(a.sum(), 66.0)
    assertEquals(a.sum(0).toList, List(12.0, 15.0, 18.0, 21.0))
    assertEquals(a.sum(1, keepdims = true).shape, Seq(3, 1))
    assertEquals(a.sum(Seq(0, 1)).item, 66.0)
    assertEquals(a.prod(1).toList, List(0.0, 840.0, 7920.0))
    assertEquals(a.max(), 11.0)
    assertEquals(a.min(0).toList, List(0.0, 1.0, 2.0, 3.0))
    assertEquals(a.argmax(), 11)
    assertEquals(a.argmin(1).toList, List(0, 0, 0))
    assertEquals(a.mean(0).toList, List(4.0, 5.0, 6.0, 7.0))
    assertEqualsDouble(a.std(), 3.452052529534663, 1e-12)
    assertEqualsDouble(a.variance(ddof = 1), 13.0, 1e-12)
    assertEquals(a.cumsum().toList.last, 66.0)
    assertEquals(a.cumsum(0).toList.drop(8), List(12.0, 15.0, 18.0, 21.0))
    assertEquals(a.cumprod(1).toList.take(4), List(0.0, 0.0, 0.0, 0.0))
    assertEquals(np.array(1, 2, 3).sum(), 6L)
    assertEquals(np.array(true, true, false).sum(), 2L)
    assertEquals(np.array(1, 2, 3, 4).mean(), 2.5)
    assert(np.array(1.0, Double.NaN).max().isNaN)
    assertEquals(np.array(1.0, Double.NaN).argmax(), 1)
    assert(np.array(true, false).any())
    assert(!np.array(true, false).all())
    assertEquals(a.ptp(), 11.0)
    assertEquals(np.zeros(0).sum(), 0.0)
    intercept[IllegalArgumentException](np.zeros(0).max())
    assertEquals(np.array(Complex(1, 1), Complex(2, -1)).sum(), Complex(3, 0))
    assertEqualsDouble(np.array(Complex(1, 1), Complex(-1, -1)).variance(), 2.0, 1e-12)
  }
  test("pairwise summation accuracy") {
    val n = 100000
    val x = np.full(Seq(n), 0.1)
    assertEqualsDouble(x.sum(), 10000.0, 1e-9)
  }
  test("matmul and dot") {
    val m = np.arange(6.0).reshape(2, 3)
    assertEquals((m @@ np.array(1.0, 1.0, 1.0)).toList, List(3.0, 12.0))
    assertEquals((np.array(1.0, 1.0) @@ m).toList, List(3.0, 5.0, 7.0))
    val batch = np.arange(12.0).reshape(2, 2, 3)
    assertEquals((batch @@ m.T).shape, Seq(2, 2, 2))
    assertEquals(np.array(1, 2).dot(np.array(3, 4)).item, 11)
    intercept[IllegalArgumentException](m @@ m)
    assertEquals(np.eye[Int](2).dot(np.array(Seq(Seq(1, 2), Seq(3, 4)))).toList, List(1, 2, 3, 4))
  }
  test("sort/argsort/nonzero/diagonal/round") {
    val x = np.array(3.0, Double.NaN, 1.0, 2.0)
    assertEquals(x.sorted.toString, "[ 1.  2.  3. nan]")
    assertEquals(x.argsort().toList, List(2, 3, 0, 1))
    val m = np.array(Seq(Seq(3, 1), Seq(2, 4)))
    m.sort(0)
    assertEquals(m.toList, List(2, 1, 3, 4))
    assertEquals(np.array(0, 1, 0, 2).nonzero.head.toList, List(1, 3))
    assertEquals(np.arange(9).reshape(3, 3).diagonal(1).toList, List(1, 5))
    assertEquals(np.arange(9).reshape(3, 3).trace(), 12L)
    assertEquals(np.array(0.5, 1.5, 2.5, -0.5, 1.2345).round().toList, List(0.0, 2.0, 2.0, -0.0, 1.0))
    assertEquals(np.array(1.2345).round(2).item, 1.23)
    assertEquals(np.array(1234.0).round(-2).item, 1200.0)
  }
