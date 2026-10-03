package com.github.kmizu.numscala

/** uint64 values at and above 2^63 (stored as negative JVM Longs) match NumPy 2.4. */
class UnsignedParitySuite extends munit.FunSuite:
  private def u64(xs: Long*): NDArray[UInt64] = NDArray.fromArray(xs.toArray, Array(xs.length)).astype[UInt64]

  test("uint64 floor division, remainder and comparison above 2^63"):
    val u = u64(-1L, Long.MinValue) // 2^64-1, 2^63
    assertEquals(u.`//`(3).toSeq.map(_.toString), Seq("6148914691236517205", "3074457345618258602"))
    assertEquals((u % 7).toSeq.map(_.toString), Seq("1", "1"))
    assertEquals((u > u64(Long.MaxValue)).toSeq, Seq(true, true))
    assertEquals(u.astype[Double].toSeq, Seq(1.8446744073709552e19, 9.223372036854776e18))
    // `toString` on an opaque UInt64 shows the raw Long bits; `unsignedString` is the value
    assertEquals(np.max(u).unsignedString, "18446744073709551615")

  test("uint8 wraps around like NumPy"):
    val a = np.arrayOf(Seq(250), DType.UInt8)
    assertEquals((a + np.arrayOf(Seq(10), DType.UInt8)).toSeq.map(_.toString), Seq("4"))

  test("str of a uint64 array"):
    assertEquals(np.array_str(u64(-1L)), "[18446744073709551615]")
