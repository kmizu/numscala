package numscala

class ExtrasSuite extends munit.FunSuite:
  private def ints(a: NDArray[UInt8]): List[Int] = a.astype[Int].toList

  test("packbits / unpackbits") {
    assertEquals(ints(np.packbits(np.array(1, 0, 1, 1, 0, 0, 0, 1, 1))), List(177, 128))
    assertEquals(ints(np.packbits(np.array(true, false, true), bitorder = "little")), List(5))
    val m = np.array(Seq(Seq(1, 0, 1), Seq(0, 1, 1)))
    val p = np.packbits(m, axis = 1)
    assertEquals(p.shape, Seq(2, 1))
    assertEquals(ints(p), List(160, 96))
    val two = np.array(2).astype[UInt8]
    assertEquals(ints(np.unpackbits(two)), List(0, 0, 0, 0, 0, 0, 1, 0))
    assertEquals(ints(np.unpackbits(two, count = 3)), List(0, 0, 0))
    assertEquals(ints(np.unpackbits(two, count = -2)), List(0, 0, 0, 0, 0, 0))
    assertEquals(ints(np.unpackbits(two, bitorder = "little")), List(0, 1, 0, 0, 0, 0, 0, 0))
    val round = np.unpackbits(np.packbits(np.array(1, 1, 0, 1, 0)), count = 5)
    assertEquals(ints(round), List(1, 1, 0, 1, 0))
    intercept[IllegalArgumentException](np.packbits(np.array(1.5)))
  }

  test("matvec / vecmat") {
    val a = np.arange(6.0).reshape(2, 3)
    assertEquals(np.matvec(a, np.array(1.0, 1.0, 1.0)).toList, List(3.0, 12.0))
    assertEquals(np.vecmat(np.array(1.0, 1.0), a).toList, List(3.0, 5.0, 7.0))
    val batch = np.arange(12.0).reshape(2, 2, 3)
    assertEquals(np.matvec(batch, np.ones(3)).shape, Seq(2, 2))
    val c = np.array(Complex(0, 1), Complex(1, 0))
    assertEquals(np.vecmat(c, np.eye[Complex](2)).toList, List(Complex(0, -1), Complex(1, 0)))
  }

  test("r_ / c_ / concat / astype") {
    assertEquals(np.r_(np.array(1, 2), 3, np.array(4)).toList, List(1, 2, 3, 4))
    val cc = np.c_(np.array(1, 2, 3), np.array(4, 5, 6))
    assertEquals(cc.shape, Seq(3, 2))
    assertEquals(cc(0, ::).toList, List(1, 4))
    assertEquals(np.concat(Seq(np.array(Seq(1)), np.array(Seq(2)))).toList, List(1, 2))
    assertEquals(np.astype(np.array(1.5, 2.5), DType.Int32).toList, List(1, 2))
    val x = np.array(1.0, 2.0)
    assert(np.astype(x, DType.Float64, copy = false) eq x)
  }

  test("dtype queries") {
    assertEquals(np.common_type(np.array(1, 2)), DType.Float64)
    assertEquals(np.common_type(np.array(1.0f)), DType.Float32)
    assertEquals(np.common_type(np.array(1.0f), np.array(Complex(1, 0))), DType.Complex128)
    assert(np.isdtype(DType.Int32, "signed integer"))
    assert(np.isdtype(DType.UInt8, "integral"))
    assert(!np.isdtype(DType.Float64, "integral"))
    assert(np.isdtype(DType.Float32, Seq("complex floating", "real floating")))
    assertEquals(np.mintypecode(Seq("d", "f")), "d")
    assertEquals(np.mintypecode(Seq("d", "F")), "D")
    assertEquals(np.mintypecode(Seq("i")), "d")
    assertEquals(np.typename("d"), "double precision")
    assert(np.iterable(Seq(1)))
    assert(!np.iterable(3))
    assert(!np.iterable(np.array(3)))
  }

  test("broadcast / nditer / require / isfortran / chkfinite") {
    val b = np.broadcast(np.zeros(3, 1), np.zeros(4))
    assertEquals(b.shape, Seq(3, 4))
    assertEquals((b.nd, b.size, b.numiter), (2, 12, 2))
    assertEquals(np.broadcast(np.array(1, 2), np.array(10)).iterator.toList, List(Seq(1, 10), Seq(2, 10)))
    val m = np.arange(6).reshape(2, 3)
    assertEquals(np.nditer(m, "F").toList, List(0, 3, 1, 4, 2, 5))
    val f = np.require(m, Seq("F"))
    assert(np.isfortran(f))
    assert(!np.isfortran(m))
    assertEquals(f.toList, m.toList)
    intercept[IllegalArgumentException](np.asarray_chkfinite(Seq(1.0, Double.NaN)))
    assertEquals(np.asarray_chkfinite(Seq(1.0, 2.0)).toList, List(1.0, 2.0))
  }

  test("seterr / errstate / bufsize") {
    val old = np.geterr()
    np.errstate(all = "ignore") {
      assertEquals(np.geterr()("divide"), "ignore")
    }
    assertEquals(np.geterr(), old)
    val prev = np.seterr(divide = "raise")
    assertEquals(np.geterr()("divide"), "raise")
    np.seterr(divide = prev("divide"))
    intercept[IllegalArgumentException](np.seterr(all = "bogus"))
    val bs = np.setbufsize(16384)
    assertEquals(np.getbufsize(), 16384)
    np.setbufsize(bs)
  }

  test("fromregex") {
    val tmp = java.nio.file.Files.createTempFile("rx", ".txt")
    java.nio.file.Files.writeString(tmp, "a=1 b=2\na=3 b=4\n")
    val r = np.fromregex(tmp, "a=(\\d+) b=(\\d+)", DType.Int64)
    assertEquals(r.shape, Seq(2, 2))
    assertEquals(r.toList, List(1L, 2L, 3L, 4L))
    java.nio.file.Files.delete(tmp)
  }
