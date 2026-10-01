package com.github.kmizu.numscala

class MiscSuite extends munit.FunSuite:
  private def close(a: NDArray[Double], expected: Seq[Double], tol: Double = 1e-15): Unit =
    assertEquals(a.size, expected.length)
    a.toList.zip(expected).foreach((x, y) => assertEqualsDouble(x, y, tol * math.max(1.0, math.abs(y))))

  // ---------------------------------------------------------------- constants

  test("constants") {
    assertEquals(np.pi, math.Pi)
    assertEquals(np.e, math.E)
    assert(np.inf.isPosInfinity && np.PINF.isPosInfinity && np.NINF.isNegInfinity)
    assert(np.nan.isNaN && np.NAN.isNaN)
    assertEquals(np.euler_gamma, 0.5772156649015329)
    assertEquals(1.0 / np.NZERO, Double.NegativeInfinity)
    assertEquals(1.0 / np.PZERO, Double.PositiveInfinity)
    assertEquals(np.arange(3)(np.newaxis, ::).shape, Seq(1, 3))
    assertEquals(np.typecodes("AllFloat"), "efdgFDG")
    assertEquals(np.typecodes("All"), "?bhilqnpBHILQNPefdgFDGSUVOMm")
  }

  // ---------------------------------------------------------------- isclose & co

  test("isclose / allclose") {
    val r = np.isclose(Seq(1.0, Double.NaN, Double.PositiveInfinity, 1e10), Seq(1.0 + 1e-9, Double.NaN, Double.PositiveInfinity, 1.00001e10))
    assertEquals(r.toList, List(true, false, true, true))
    assertEquals(np.isclose(Double.NaN, Double.NaN, equal_nan = true).toList, List(true))
    assertEquals(np.isclose(Seq(Double.PositiveInfinity), Seq(Double.NegativeInfinity)).toList, List(false))
    assert(np.allclose(Seq(1, 2), Seq(1.0, 2.00000001)))
    assert(!np.allclose(Seq(1.0, 2.0), Seq(1.0, 2.1)))
    assert(np.allclose(np.array(1.0, 2.0), np.array(1.05, 2.1), rtol = 0.1))
    assert(np.isclose(Complex(1, 1), Complex(1, 1.000001)).item)
    // broadcasting
    val b = np.isclose(np.array(Seq(1.0, 2.0), Seq(3.0, 4.0)), np.array(1.0, 4.0))
    assertEquals(b.shape, Seq(2, 2))
    assertEquals(b.toList, List(true, false, false, true))
    assert(np.isclose(0.0, 1e-9).item)
    assert(!np.isclose(0.0, 1e-9, atol = 0.0).item)
    intercept[IllegalArgumentException](np.isclose(Seq(1.0, 2.0), Seq(1.0, 2.0, 3.0)))
  }

  test("array_equal / array_equiv") {
    assert(np.array_equal(Seq(1, 2), Seq(1, 2)))
    assert(!np.array_equal(Seq(1, 2), Seq(1, 2, 3)))
    assert(!np.array_equal(Seq(1.0, Double.NaN), Seq(1.0, Double.NaN)))
    assert(np.array_equal(Seq(1.0, Double.NaN), Seq(1.0, Double.NaN), equal_nan = true))
    assert(np.array_equal(np.array(1, 2), np.array(1.0, 2.0)))
    assert(np.array_equal(np.array(Long.MaxValue), np.array(Long.MaxValue)))
    assert(!np.array_equal(np.array("1"), np.array(1)))
    assert(np.array_equal(np.array("a", "b"), Seq("a", "b")))
    assert(np.array_equal(Seq(Complex(1, 0)), Seq(1.0)))
    assert(np.array_equiv(Seq(1, 2), Seq(Seq(1, 2), Seq(1, 2))))
    assert(!np.array_equiv(Seq(1, 2), Seq(Seq(1, 2), Seq(1, 3))))
    assert(!np.array_equiv(Seq(1, 2), Seq(1, 2, 3)))
  }

  // ---------------------------------------------------------------- dtype introspection

  test("finfo") {
    val f = np.finfo(DType.Float64)
    assertEquals(f.eps, 2.220446049250313e-16)
    assertEquals(f.max, Double.MaxValue)
    assertEquals(f.min, -Double.MaxValue)
    assertEquals(f.tiny, 2.2250738585072014e-308)
    assertEquals(f.smallest_subnormal, 4.9e-324)
    assertEquals(f.resolution, 1e-15)
    assertEquals((f.precision, f.bits, f.nmant, f.nexp, f.maxexp, f.minexp), (15, 64, 52, 11, 1024, -1022))
    assertEquals(
      f.toString,
      """Machine parameters for float64
        |---------------------------------------------------------------
        |precision = 15   resolution = 1e-15
        |machep = -52   eps =        2.220446049250313e-16
        |negep =  -53   epsneg =     1.1102230246251565e-16
        |minexp = -1022   tiny =       2.2250738585072014e-308
        |maxexp = 1024   max =        1.7976931348623157e+308
        |nexp =   11   min =        -max
        |smallest_normal = 2.2250738585072014e-308   smallest_subnormal = 5e-324
        |---------------------------------------------------------------
        |""".stripMargin
    )
    assertEquals(f.repr, "finfo(resolution=1e-15, min=-1.7976931348623157e+308, max=1.7976931348623157e+308, dtype=float64)")
    val g = np.finfo("float32")
    assertEquals(g.eps, 1.1920928955078125e-07)
    assertEquals(g.repr, "finfo(resolution=1e-06, min=-3.4028235e+38, max=3.4028235e+38, dtype=float32)")
    assert(g.toString.contains("eps =        1.1920929e-07"))
    assert(g.toString.contains("smallest_subnormal = 1e-45"))
    assertEquals(np.finfo(DType.Complex128).dtype, DType.Float64)
    intercept[IllegalArgumentException](np.finfo(DType.Int32))
  }

  test("iinfo") {
    val i = np.iinfo(DType.Int32)
    assertEquals((i.min, i.max, i.bits, i.kind), (BigInt(-2147483648L), BigInt(2147483647L), 32, 'i'))
    assertEquals(
      i.toString,
      "Machine parameters for int32\n---------------------------------------------------------------\nmin = -2147483648\nmax = 2147483647\n---------------------------------------------------------------\n"
    )
    assertEquals(np.iinfo("int8").repr, "iinfo(min=-128, max=127, dtype=int8)")
    assertEquals(np.iinfo(DType.Int64).max, BigInt(Long.MaxValue))
    assertEquals(np.iinfo(DType.UInt64).max.toString, "18446744073709551615")
    assertEquals(np.iinfo("uint8").repr, "iinfo(min=0, max=255, dtype=uint8)")
    assertEquals(np.min_scalar_type(10), DType.UInt8)
    assertEquals(np.min_scalar_type(-1), DType.Int8)
    assert(np.can_cast(DType.UInt8, DType.Int16, "safe"))
    assert(!np.can_cast(DType.UInt8, DType.Int8, "safe"))
    assert(np.issubdtype(DType.UInt32, np.unsignedinteger))
    assertEquals(np.iinfo(DType.Int16).min, BigInt(-32768))
    intercept[IllegalArgumentException](np.iinfo(DType.Float64))
    intercept[IllegalArgumentException](np.iinfo(DType.Bool))
  }

  test("promote_types / result_type") {
    assertEquals(np.promote_types("?", "f4"), DType.Float32)
    assertEquals(np.promote_types(DType.Float32, DType.Int64), DType.Float64)
    assertEquals(np.promote_types("c16", "f8"), DType.Complex128)
    assertEquals(np.promote_types("U", "f8"), DType.Str)
    assertEquals(np.result_type(DType.Int8, 1.0), DType.Float64)
    assertEquals(np.result_type(DType.Int8, 1), DType.Int8)
    assertEquals(np.result_type(DType.Float32, Complex(0, 1)), DType.Complex128)
    assertEquals(np.result_type(1, 2.0), DType.Float64)
    assertEquals(np.result_type(np.array(1.toByte), DType.Int16), DType.Int16)
    assertEquals(np.result_type(true, DType.Int8), DType.Int8)
    assertEquals(np.result_type(3), DType.Int64)
    assertEquals(np.result_type(DType.Float32, 2.0), DType.Float32)
    assertEquals(np.result_type(DType.Bool, 2), DType.Int64)
    assertEquals(np.result_type(true), DType.Bool)
    intercept[IllegalArgumentException](np.result_type())
    intercept[IllegalArgumentException](np.result_type(Seq(1)))
  }

  test("can_cast") {
    val cases = Seq(
      ("i8", "f8", "safe", true), ("f8", "i8", "same_kind", false), ("f8", "f4", "same_kind", true),
      ("f8", "f4", "safe", false), ("i4", "f4", "safe", false), ("i8", "U", "safe", true),
      ("U", "i8", "unsafe", true), ("U", "f8", "same_kind", false), ("?", "i1", "safe", true),
      ("i2", "f4", "safe", true), ("c16", "f8", "same_kind", false), ("f8", "f8", "no", true),
      ("i4", "i8", "equiv", false), ("i8", "U", "same_kind", true), ("?", "U", "safe", true)
    )
    for (f, t, c, exp) <- cases do assertEquals(np.can_cast(f, t, c), exp, s"$f -> $t ($c)")
    assert(np.can_cast(np.array(1, 2), DType.Float64))
    intercept[IllegalArgumentException](np.can_cast("i4", "i8", "bogus"))
  }

  test("min_scalar_type / isscalar / dtype") {
    assertEquals(np.min_scalar_type(10), DType.UInt8)
    assertEquals(np.min_scalar_type(-10), DType.Int8)
    assertEquals(np.min_scalar_type(-200), DType.Int16)
    assertEquals(np.min_scalar_type(100000), DType.UInt32)
    assertEquals(np.min_scalar_type(1L << 40), DType.UInt64)
    assertEquals(np.min_scalar_type(3.1), DType.Float32)
    assertEquals(np.min_scalar_type(1e50), DType.Float64)
    assertEquals(np.min_scalar_type(Complex(0, 1)), DType.Complex128)
    assertEquals(np.min_scalar_type(true), DType.Bool)
    assertEquals(np.min_scalar_type(np.array(1.0)), DType.Float32)
    assertEquals(np.min_scalar_type(np.array(1.0, 2.0)), DType.Float64)
    assert(np.isscalar(3.1) && np.isscalar(1) && np.isscalar("s") && np.isscalar(Complex(1, 1)) && np.isscalar(false))
    assert(!np.isscalar(np.array(3.1)) && !np.isscalar(Seq(1)))
    assertEquals(np.dtype("float64"), DType.Float64)
    assertEquals(np.dtype("<i4"), DType.Int32)
    assertEquals(np.dtype(DType.Bool), DType.Bool)
    intercept[IllegalArgumentException](np.dtype("float128x"))
  }

  test("issubdtype") {
    assert(np.issubdtype(DType.Float64, np.floating))
    assert(np.issubdtype(DType.Float32, np.inexact))
    assert(np.issubdtype(DType.Int8, np.integer))
    assert(np.issubdtype(DType.Int64, np.signedinteger))
    assert(!np.issubdtype(DType.Int64, np.unsignedinteger))
    assert(np.issubdtype(DType.Complex128, np.complexfloating))
    assert(np.issubdtype(DType.Complex128, np.number))
    assert(!np.issubdtype(DType.Bool, np.number))
    assert(np.issubdtype(DType.Bool, np.bool_))
    assert(np.issubdtype(DType.Str, np.character) && np.issubdtype(DType.Str, np.flexible) && np.issubdtype("U", np.str_))
    assert(np.issubdtype(DType.Float64, DType.Float64))
    assert(!np.issubdtype(DType.Float32, DType.Float64))
    assert(np.issubdtype(np.floating, np.inexact))
    assert(!np.issubdtype(np.floating, DType.Float64))
    assert(np.issubdtype("int32", np.generic))
    assertEquals(np.floating.toString, "<class 'numpy.floating'>")
  }

  // ---------------------------------------------------------------- windows

  test("window functions") {
    close(np.bartlett(5), Seq(0.0, 0.5, 1.0, 0.5, 0.0))
    close(np.blackman(5), Seq(-1.3877787807814457e-17, 0.34, 0.9999999999999999, 0.34, -1.3877787807814457e-17))
    close(np.hamming(5), Seq(0.08000000000000002, 0.54, 1.0, 0.54, 0.08000000000000002))
    close(np.hanning(5), Seq(0.0, 0.5, 1.0, 0.5, 0.0))
    close(np.kaiser(5, 14), Seq(7.726866835270368e-06, 0.16493218754795194, 1.0, 0.16493218754795194, 7.726866835270368e-06), 1e-12)
    close(np.kaiser(6, 5.0), Seq(0.036710892271286676, 0.4149036392433668, 0.9138124838692001, 0.9138124838692001, 0.4149036392433668, 0.036710892271286676), 1e-12)
    close(np.kaiser(4, 0), Seq(1.0, 1.0, 1.0, 1.0))
    assertEquals(np.blackman(1).toList, List(1.0))
    assertEquals(np.hanning(0).size, 0)
    assertEquals(np.hamming(-3).size, 0)
    assertEquals(np.kaiser(1, 3).toList, List(1.0))
    assertEquals(np.kaiser(0, 3).size, 0)
  }

  // ---------------------------------------------------------------- memory

  test("may_share_memory / shares_memory") {
    val a = np.arange(10)
    assert(np.may_share_memory(a, a(sliceFrom(2))))
    assert(!np.may_share_memory(a, np.arange(10)))
    val even = a("::2")
    val odd = a("1::2")
    assert(np.may_share_memory(even, odd))
    assert(!np.shares_memory(even, odd))
    assert(np.shares_memory(a("0:5"), a("4:")))
    assert(!np.shares_memory(a("0:5"), a("5:")))
    assert(!np.may_share_memory(a("0:5"), a("5:")))
    val m = np.arange(12.0).reshape(3, 4)
    assert(np.shares_memory(m.T, m(1, ::)))
    assert(!np.may_share_memory(a, a("3:3")))
  }

  // ---------------------------------------------------------------- printing

  test("format_float_positional") {
    assertEquals(np.format_float_positional(0.15, precision = 1), "0.1")
    assertEquals(np.format_float_positional(0.25, precision = 1), "0.2")
    assertEquals(np.format_float_positional(1.0), "1.")
    assertEquals(np.format_float_positional(1.0, trim = "-"), "1")
    assertEquals(np.format_float_positional(1.5, precision = 3, unique = false), "1.500")
    assertEquals(np.format_float_positional(1.5, precision = 3, unique = false, trim = "-"), "1.5")
    assertEquals(np.format_float_positional(1.5, precision = 3, unique = false, trim = "."), "1.5")
    assertEquals(np.format_float_positional(1.0, precision = 3, unique = false, trim = "0"), "1.0")
    assertEquals(np.format_float_positional(0.1f), "0.1")
    assertEquals(np.format_float_positional(0.1), "0.1")
    assertEquals(np.format_float_positional(3.14159, pad_left = 5, pad_right = 6, sign = true), "   +3.14159 ")
    assertEquals(np.format_float_positional(1.0, min_digits = 3), "1.000")
    assertEquals(np.format_float_positional(123.456, precision = 2, fractional = false), "120.")
    assertEquals(np.format_float_positional(123.456, precision = 2, fractional = false, unique = false), "120.")
    assertEquals(np.format_float_positional(Double.PositiveInfinity), "inf")
    assertEquals(np.format_float_positional(Double.NaN), "nan")
    assertEquals(np.format_float_positional(-0.0), "-0.")
    assertEquals(np.format_float_positional(1e20), "100000000000000000000.")
    assertEquals(np.format_float_positional(1e-5), "0.00001")
    intercept[IllegalArgumentException](np.format_float_positional(1.0, unique = false))
    intercept[IllegalArgumentException](np.format_float_positional(1.0, trim = "x"))
  }

  test("format_float_scientific") {
    assertEquals(np.format_float_scientific(1.0), "1.e+00")
    assertEquals(np.format_float_scientific(0.1), "1.e-01")
    assertEquals(np.format_float_scientific(123.456), "1.23456e+02")
    assertEquals(np.format_float_scientific(1.0, precision = 3, unique = false), "1.000e+00")
    assertEquals(np.format_float_scientific(1.0, trim = "-"), "1e+00")
    assertEquals(np.format_float_scientific(1.5, exp_digits = 4), "1.5e+0000")
    assertEquals(np.format_float_scientific(1.5, pad_left = 4, sign = true), "  +1.5e+00")
    assertEquals(np.format_float_scientific(0.0), "0.e+00")
    assertEquals(np.format_float_scientific(123.456, precision = 2), "1.23e+02")
    assertEquals(np.format_float_scientific(1.0, min_digits = 3), "1.000e+00")
    assertEquals(np.format_float_scientific(1e300), "1.e+300")
    assertEquals(np.format_float_scientific(-2.5e-7), "-2.5e-07")
    assertEquals(np.format_float_scientific(Double.NegativeInfinity), "-inf")
  }

  test("format_float with one-digit shortest values and float32") {
    assertEquals(np.format_float_scientific(java.lang.Double.MIN_VALUE), "5.e-324")
    assertEquals(np.format_float_scientific(java.lang.Float.MIN_VALUE), "1.e-45")
    assertEquals(np.format_float_scientific(0.1f, precision = 3, unique = false), "1.000e-01")
    assertEquals(NpMiscFloatFmt.pyRepr(1e16, false), "1e+16")
    assertEquals(NpMiscFloatFmt.pyRepr(-0.0001, false), "-0.0001")
    assertEquals(NpMiscFloatFmt.pyRepr(123.0, false), "123.0")
    assertEquals(NpMiscFloatFmt.pyRepr(Double.NaN, false), "nan")
  }

  test("base_repr / binary_repr") {
    assertEquals(np.base_repr(5, 2, padding = 3), "000101")
    assertEquals(np.base_repr(-5, 2), "-101")
    assertEquals(np.base_repr(-5, 2, padding = 2), "-00101")
    assertEquals(np.base_repr(0, padding = 2), "00")
    assertEquals(np.base_repr(0), "0")
    assertEquals(np.base_repr(255, 16), "FF")
    assertEquals(np.base_repr(35, 36), "Z")
    intercept[IllegalArgumentException](np.base_repr(3, 37))
    intercept[IllegalArgumentException](np.base_repr(3, 1))
    assertEquals(np.binary_repr(3), "11")
    assertEquals(np.binary_repr(-3), "-11")
    assertEquals(np.binary_repr(-3, width = 4), "1101")
    assertEquals(np.binary_repr(-4, width = 3), "100")
    assertEquals(np.binary_repr(0), "0")
    assertEquals(np.binary_repr(0, width = 4), "0000")
    assertEquals(np.binary_repr(5, width = 8), "00000101")
    assertEquals(np.binary_repr(-128, width = 8), "10000000")
    assertEquals(np.binary_repr(Long.MinValue, width = 64), "1" + "0" * 63)
    intercept[IllegalArgumentException](np.binary_repr(8, width = 2))
  }

  test("print options, printoptions context, array2string, array_repr, array_str") {
    val saved = np.get_printoptions()
    try
      assertEquals(saved.precision, 8)
      assertEquals(np.array2string(np.array(1.123456, 2.5), precision = 2, separator = ", "), "[1.12, 2.5 ]")
      assertEquals(np.array2string(np.arange(5), prefix = "x(", suffix = ")"), "[0 1 2 3 4]")
      assertEquals(np.array2string(np.arange(10.0), max_line_width = 20), "[0. 1. 2. 3. 4. 5.\n 6. 7. 8. 9.]")
      assertEquals(np.array2string(np.array(3.0)), "3.")
      assertEquals(np.array2string(np.zeros(0)), "[]")
      assertEquals(np.array_repr(np.array(1.0, 2.0), precision = 3), "array([1., 2.])")
      assertEquals(np.array_repr(np.array(1.toByte, 2.toByte)), "array([1, 2], dtype=int8)")
      assertEquals(np.array_str(np.array(0.000001, 1.0), suppress_small = true), "[0.000001 1.      ]")
      val inside = np.printoptions(precision = 3) {
        assertEquals(np.get_printoptions().precision, 3)
        np.array(math.Pi, 1.0).toString
      }
      assertEquals(inside, "[3.142 1.   ]")
      assertEquals(np.get_printoptions().precision, 8)
      np.set_printoptions(nanstr = "NaN")
      val s = np.array(1.5, Double.NaN).toString
      np.set_printoptions(nanstr = "nan")
      assertEquals(s, "[1.5 NaN]")
      assertEquals(np.get_printoptions(), saved)
      np.set_printoptions(precision = 8, threshold = 1000, edgeitems = 3, linewidth = 75, suppress = false, infstr = "inf")
      assertEquals(np.get_printoptions(), saved)
      assertEquals(np.array2string(np.arange(10), threshold = 5, edgeitems = 1), "[0 ... 9]")
      intercept[IllegalArgumentException](np.set_printoptions(precision = -1))
    finally Format.setPrintOptions(saved)
  }
