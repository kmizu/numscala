package com.github.kmizu.numscala

class UnsignedSuite extends munit.FunSuite:
  test("uint8 arithmetic wraps and prints unsigned") {
    val a = np.array(250, 10, 0).astype[UInt8]
    assertEquals(a.dtype, DType.UInt8)
    assertEquals(a.toString, "[250  10   0]")
    assertEquals(a.repr, "array([250,  10,   0], dtype=uint8)")
    val b = a + np.array(10, 10, 10).astype[UInt8]
    assertEquals(b.toString, "[ 4 20 10]")
    assertEquals((a - np.array(1, 1, 1).astype[UInt8]).toString, "[249   9 255]")
    assertEquals(a.max(), UInt8(250))
    assertEquals(a.sorted.toString, "[  0  10 250]")
    assertEquals(a.argsort().toList, List(2, 1, 0))
    val s: UInt64 = a.sum()
    assertEquals(s.unsignedString, "260")
    val m: NDArray[Double] = a.mean(0)
    assertEqualsDouble(m.item, 260.0 / 3, 1e-12)
    assertEquals(a.astype[Double].toList, List(250.0, 10.0, 0.0))
  }
  test("promotion with unsigned types") {
    val u8 = np.array(200).astype[UInt8]
    val i8 = np.array(-1).astype[Byte]
    val r1: NDArray[Short] = u8 + i8
    assertEquals(r1.toList, List[Short](199))
    val u32 = np.array(1).astype[UInt32]
    val r2: NDArray[Long] = u32 + np.array(1)
    val u64 = np.array(1L).astype[UInt64]
    val r3: NDArray[Double] = u64 + np.array(1L)
    val r4: NDArray[Float] = u8 + np.array(1.0f)
    assertEquals(DType.promote(DType.UInt16, DType.Int16), DType.Int32)
    assertEquals(DType.promote(DType.UInt64, DType.Int8), DType.Float64)
    assertEquals(DType.promote(DType.UInt8, DType.UInt32), DType.UInt32)
    assertEquals(DType.byName("u1"), DType.UInt8)
    assertEquals(DType.UInt8.str, "|u1")
    assertEquals(DType.UInt32.str, "<u4")
    assertEquals(r2.dtype, DType.Int64)
    assertEquals(r3.dtype, DType.Float64)
    assertEquals(r4.dtype, DType.Float32)
  }
  test("uint64 large values") {
    val big = np.arrayOf(Seq("18446744073709551615", "1"), DType.UInt64)
    assertEquals(big.toString, "[18446744073709551615                    1]")
    assertEquals(big.sorted.toList.map(_.unsignedString), List("1", "18446744073709551615"))
    assertEqualsDouble(big.astype[Double].item(0), 1.8446744073709552e19, 1e4)
    assertEquals(big.max().unsignedString, "18446744073709551615")
    assertEquals(np.array(-1).astype[UInt32].astype[Long].toList, List(4294967295L))
  }

class UnsignedRandomSuite extends munit.FunSuite:
  test("Generator.integers with unsigned dtypes") {
    val rng = np.random.default_rng(42)
    val u8 = rng.integers(0L, 256L, Seq(2000), DType.UInt8, false)
    assertEquals(u8.dtype.name, "uint8")
    val ints = u8.astype[Int].toArray
    assert(ints.forall(v => v >= 0 && v < 256))
    assert(ints.max > 200 && ints.min < 50)
    val u32 = rng.integers(0L, 4000000000L, Seq(100), DType.UInt32, false)
    assert(u32.astype[Long].toArray.forall(v => v >= 0 && v < 4000000000L))
    intercept[IllegalArgumentException](rng.integers(0L, 257L, Seq(1), DType.UInt8, false))
  }
