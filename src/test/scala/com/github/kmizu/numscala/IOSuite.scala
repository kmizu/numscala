package com.github.kmizu.numscala

import java.nio.file.{Files, Path}
import java.nio.charset.StandardCharsets
import java.util.Base64

/** Fixtures written by NumPy 2.4 (`np.save` / `np.lib.format.write_array` / `np.savez`), base64-encoded. */
object IOSuiteFixtures:
  val npy: Map[String, String] = Map(
    "f8" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDMsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAD4PwAAAAAAAADAAAAAAAAACkA=",
    "i4_2d" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnPGk0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDMpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAAAgAAAAMAAAAEAAAABQAAAAYAAAA=",
    "i8" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAr//////////wAAAAAAAQAA",
    "i2" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnPGkyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAr9/ywB",
    "i1" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnfGkxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAqAfw==",
    "b1" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnfGIxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDMsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAE=",
    "f4" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIArNzMw9AAAgQA==",
    "c16" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnPGMxNicsICdmb3J0cmFuX29yZGVyJzogRmFsc2UsICdzaGFwZSc6ICgyLCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAADwPwAAAAAAAABAAAAAAAAAAIAAAAAAAAAMwA==",
    "U" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnPFUyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDMsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAphAAAAYgAAAGMAAAAAAAAA5WUAACxnAAA=",
    "scalar" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAcQA==",
    "be_f8" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnPmY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAo/+AAAAAAAAEAEAAAAAAAA",
    "be_i4" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnPmk0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAB/////g==",
    "fortran" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBUcnVlLCAnc2hhcGUnOiAoMiwgMyksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAAAAAAAAAhAAAAAAAAA8D8AAAAAAAAQQAAAAAAAAABAAAAAAAAAFEA=",
    "u1" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDMsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAyP8=",
    "u2" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnPHUyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAr//wEA",
    "u4" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnPHU0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAr/////AAAAAA==",
    "f2" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnPGYyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDQsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAPgC0/3sAfA==",
    "c8" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnPGM4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsKSwgfSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAIA/AAAAQA==",
    "v2" -> "k05VTVBZAgB0AAAAeydkZXNjcic6ICc8aTgnLCAnZm9ydHJhbl9vcmRlcic6IEZhbHNlLCAnc2hhcGUnOiAoNCwpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAAAEAAAAAAAAAAgAAAAAAAAADAAAAAAAAAA==",
    "v3" -> "k05VTVBZAwB0AAAAeydkZXNjcic6ICc8ZjgnLCAnZm9ydHJhbl9vcmRlcic6IEZhbHNlLCAnc2hhcGUnOiAoMiwpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAADwPwAAAAAAAABA",
    "empty" -> "k05VTVBZAQB2AHsnZGVzY3InOiAnPGY4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDAsIDMpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAo=",
    "npz" -> "UEsDBC0AAAAAAAAAIQDqxRmd//////////8FABQAdy5ucHkBABAAiAAAAAAAAACIAAAAAAAAAJNOVU1QWQEAdgB7J2Rlc2NyJzogJzxmOCcsICdmb3J0cmFuX29yZGVyJzogRmFsc2UsICdzaGFwZSc6ICgxLCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKAAAAAAAA4D9QSwMELQAAAAAAAAAhAAv5qi3//////////wkAFABhcnJfMC5ucHkBABAAiAAAAAAAAACIAAAAAAAAAJNOVU1QWQEAdgB7J2Rlc2NyJzogJzxpNCcsICdmb3J0cmFuX29yZGVyJzogRmFsc2UsICdzaGFwZSc6ICgyLCksIH0gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKAQAAAAIAAABQSwECLQMtAAAAAAAAACEA6sUZnYgAAACIAAAABQAAAAAAAAAAAAAAgAEAAAAAdy5ucHlQSwECLQMtAAAAAAAAACEAC/mqLYgAAACIAAAACQAAAAAAAAAAAAAAgAG/AAAAYXJyXzAubnB5UEsFBgAAAAACAAIAagAAAIIBAAAAAA==",
    "npzc" -> "UEsDBC0AAAAIAAAAIQAkIeEr//////////8FABQAeC5ucHkBABAAmAAAAAAAAABMAAAAAAAAAJvsF+obEMnIUMZQrZ6SWpxcpG6loG6TZqGuo6Cell9UUpSYF59flJIKEndLzClOBYoXZyQWpAL5GsY6mjoKtQoUAC4GFPDBHspwAABQSwECLQMtAAAACAAAACEAJCHhK0wAAACYAAAABQAAAAAAAAAAAAAAgAEAAAAAeC5ucHlQSwUGAAAAAAEAAQAzAAAAgwAAAAAA"
  )
  def bytes(k: String): Array[Byte] = Base64.getDecoder.decode(npy(k))

class IOSuite extends munit.FunSuite:
  import IOSuiteFixtures.bytes

  private def tmp(): Path = Files.createTempDirectory("numscala-io")
  private def text(p: Path): String = new String(Files.readAllBytes(p), StandardCharsets.UTF_8)
  private def write(p: Path, s: String): Path = Files.write(p, s.getBytes(StandardCharsets.UTF_8))

  // ---------------------------------------------------------------- .npy byte compatibility

  test("np.save writes byte-identical .npy files to NumPy") {
    val d = tmp()
    val cases: Seq[(String, NDArray[?])] = Seq(
      "f8" -> np.array(1.5, -2.0, 3.25),
      "i4_2d" -> np.array(Seq(1, 2, 3), Seq(4, 5, 6)),
      "i8" -> np.array(-1L, 1L << 40),
      "i2" -> np.array(-3.toShort, 300.toShort),
      "i1" -> np.array(-128.toByte, 127.toByte),
      "b1" -> np.array(true, false, true),
      "f4" -> np.array(0.1f, 2.5f),
      "c16" -> np.array(Complex(1, 2), Complex(-0.0, -3.5)),
      "U" -> np.array("ab", "c", "日本"),
      "scalar" -> np.array(7.0),
      "empty" -> np.zeros(0, 3)
    )
    for (k, a) <- cases do
      val p = d.resolve(k + ".npy")
      np.save(p, a)
      assert(Files.readAllBytes(p).sameElements(bytes(k)), s"bytes differ for $k")
  }

  test("np.save appends .npy to string paths and header is 64-byte aligned") {
    val d = tmp()
    np.save(d.resolve("x").toString, np.arange(5))
    assert(Files.exists(d.resolve("x.npy")))
    val b = Files.readAllBytes(d.resolve("x.npy"))
    val hlen = (b(8) & 0xff) | ((b(9) & 0xff) << 8)
    assertEquals((10 + hlen) % 64, 0)
    assertEquals(b(10 + hlen - 1), '\n'.toByte)
    assertEquals(np.load(d.resolve("x.npy")), np.arange(5))
  }

  test("np.load reads NumPy files of every supported dtype") {
    def ld(k: String): NDArray[?] = NpIONpy.fromBytes(bytes(k))
    assertEquals(ld("f8"), np.array(1.5, -2.0, 3.25))
    assertEquals(ld("i4_2d"), np.array(Seq(1, 2, 3), Seq(4, 5, 6)))
    assertEquals(ld("i8"), np.array(-1L, 1L << 40))
    assertEquals(ld("i2").dtype.name, "int16")
    assertEquals(ld("i1").toList: List[Any], List(-128.toByte, 127.toByte))
    assertEquals(ld("b1"), np.array(true, false, true))
    assertEquals(ld("f4"), np.array(0.1f, 2.5f))
    assertEquals(ld("c16"), np.array(Complex(1, 2), Complex(0, -3.5)))
    assertEquals(ld("U").toList: List[Any], List("ab", "c", "日本"))
    val s = ld("scalar")
    assertEquals(s.ndim, 0)
    assertEquals((s.toList: List[Any]), List(7.0))
    assertEquals(ld("empty").shape, Seq(0, 3))
  }

  test("np.load handles big-endian, Fortran order, unsigned, half and complex64") {
    def ld(k: String): NDArray[?] = NpIONpy.fromBytes(bytes(k))
    assertEquals(ld("be_f8"), np.array(1.5, 2.5))
    assertEquals(ld("be_i4"), np.array(1, -2))
    assertEquals(ld("fortran"), np.arange(6.0).reshape(2, 3))
    val u1 = ld("u1")
    assertEquals(u1.dtype.name, "uint8")
    assertEquals(u1.astypeDyn(DType.Int64).toList: List[Any], List(0L, 200L, 255L))
    // round trip reproduces NumPy's bytes exactly
    assertEquals(NpIONpy.toBytes(u1).toSeq, bytes("u1").toSeq)
    assertEquals(ld("u2").dtype.name, "uint16")
    assertEquals(ld("u2").astypeDyn(DType.Int64).toList: List[Any], List(65535L, 1L))
    assertEquals(ld("u4").astypeDyn(DType.Int64).toList: List[Any], List(4294967295L, 0L))
    val f2 = ld("f2")
    assertEquals(f2.dtype.name, "float32")
    assertEquals(f2.toList: List[Any], List(1.5f, -0.25f, 65504f, Float.PositiveInfinity))
    assertEquals(ld("c8"), np.array(Seq(Complex(1, 2))))
  }

  test("np.load reads format versions 2.0 and 3.0") {
    assertEquals(NpIONpy.fromBytes(bytes("v2")), np.arange(4L))
    assertEquals(NpIONpy.fromBytes(bytes("v3")), np.array(1.0, 2.0))
    val b2 = NpIONpy.toBytes(np.array(1, 2), Some((2, 0)))
    assertEquals(b2(6).toInt, 2)
    assertEquals((b2.length - 8) % 64, 0)
    assertEquals(NpIONpy.fromBytes(b2), np.array(1, 2))
    assertEquals(NpIONpy.fromBytes(NpIONpy.toBytes(np.array("x"), Some((3, 0)))).toList: List[Any], List("x"))
    intercept[IllegalArgumentException](NpIONpy.toBytes(np.array(1), Some((4, 0))))
  }

  test("typed load converts with same_kind casting") {
    val d = tmp()
    val p = d.resolve("i.npy")
    np.save(p, np.array(1, 2, 3))
    val a: NDArray[Double] = np.load(p, DType.Float64)
    assertEquals(a, np.array(1.0, 2.0, 3.0))
    val same: NDArray[Int] = np.load(p, DType.Int32)
    assertEquals(same.toList, List(1, 2, 3))
    np.save(p, np.array(1.5))
    intercept[IllegalArgumentException](np.load(p, DType.Int32))
  }

  test("np.load error cases") {
    val d = tmp()
    val p = write(d.resolve("bad.npy"), "not an npy file at all")
    intercept[IllegalArgumentException](np.load(p))
    val z = d.resolve("z.npz")
    np.savez(z, np.arange(2))
    intercept[IllegalArgumentException](np.load(z))
    val trunc = d.resolve("t.npy")
    Files.write(trunc, bytes("f8").take(140))
    intercept[IllegalArgumentException](np.load(trunc))
  }

  // ---------------------------------------------------------------- npz

  test("savez / load_npz round trip with positional and named arrays") {
    val d = tmp()
    np.savez(d.resolve("a").toString, np.arange(3), "w" -> np.array(0.5, 1.5), np.array(true))
    val z = np.load_npz(d.resolve("a.npz"))
    assertEquals(z.files, Seq("w", "arr_0", "arr_1"))
    assertEquals(z("arr_0"), np.arange(3))
    assertEquals(z("w"), np.array(0.5, 1.5))
    assertEquals(z("arr_1").ndim, 0)
    val w: NDArray[Double] = z.get("arr_0", DType.Float64)
    assertEquals(w.toList, List(0.0, 1.0, 2.0))
    intercept[NoSuchElementException](z("nope"))
    np.savez(d.resolve("m.npz"), Map("x" -> np.ones(2)))
    assertEquals(np.load_npz(d.resolve("m.npz"))("x"), np.ones(2))
    intercept[IllegalArgumentException](np.savez(d.resolve("dup.npz"), np.ones(1), "arr_0" -> np.ones(1)))
  }

  test("savez_compressed and NumPy-written archives") {
    val d = tmp()
    np.savez_compressed(d.resolve("c.npz"), "big" -> np.zeros(1000))
    assert(Files.size(d.resolve("c.npz")) < 1000)
    assertEquals(np.load_npz(d.resolve("c.npz"))("big"), np.zeros(1000))
    np.savez_compressed(d.resolve("c2.npz"), Map("q" -> np.arange(4)))
    assertEquals(np.load_npz(d.resolve("c2.npz"))("q"), np.arange(4))
    val p1 = d.resolve("np.npz")
    Files.write(p1, bytes("npz"))
    val z = np.load_npz(p1)
    assertEquals(z.files, Seq("w", "arr_0"))
    assertEquals(z("arr_0"), np.array(1, 2))
    assertEquals(z("w"), np.array(Seq(0.5)))
    val p2 = d.resolve("npc.npz")
    Files.write(p2, bytes("npzc"))
    assertEquals(np.load_npz(p2)("x"), np.arange(3.0))
  }

  // ---------------------------------------------------------------- savetxt

  test("savetxt default format matches NumPy") {
    val d = tmp()
    val p = d.resolve("a.txt")
    np.savetxt(p, np.array(Seq(1.0, 2.5), Seq(0.1, -3.0)))
    assertEquals(
      text(p),
      "1.000000000000000000e+00 2.500000000000000000e+00\n1.000000000000000056e-01 -3.000000000000000000e+00\n"
    )
  }

  test("savetxt header, footer, delimiter, per-column and complex formats") {
    val d = tmp()
    val p = d.resolve("b.txt")
    np.savetxt(p, np.array(1, 2), fmt = "%d", header = "a\nb", footer = "end", delimiter = ",")
    assertEquals(text(p), "# a\n# b\n1\n2\n# end\n")
    np.savetxt(p, np.array(Seq(Seq(Complex(1, 2), Complex(3, -4)))), fmt = "%.2f")
    assertEquals(text(p), " (1.00+2.00j)  (3.00-4.00j)\n")
    np.savetxt(p, np.array(Seq(Seq(1.5, 2.0))), fmt = Seq("%d", "%.3g"), delimiter = ";")
    assertEquals(text(p), "1;2\n")
    np.savetxt(p, np.array(Seq(Seq(Double.NaN, Double.PositiveInfinity, Double.NegativeInfinity))), fmt = "%8.3f")
    assertEquals(text(p), "     nan      inf     -inf\n")
    np.savetxt(p, np.array(Seq(Seq(1, 2), Seq(3, 4))), fmt = "%d|%d", newline = "\r\n", comments = "% ", header = "h")
    assertEquals(text(p), "% h\r\n1|2\r\n3|4\r\n")
    intercept[IllegalArgumentException](np.savetxt(p, np.array(Seq(Seq(1, 2))), fmt = "%d %d %d"))
    intercept[IllegalArgumentException](np.savetxt(p, np.array(Seq(Seq(1, 2))), fmt = Seq("%d")))
    intercept[IllegalArgumentException](np.savetxt(p, np.zeros(2, 2, 2)))
  }

  test("Python %-formatting reproduces CPython") {
    val got = NpIOPyFormat.format(
      "%05f|%-8.2e|%+g|%g|%g|%#g|%x|%#o|%5s|%05d|%.3g",
      Seq(Double.PositiveInfinity, 3.14159, 2.0, 1e-5, 123456789.0, 1.0, 255L, 8L, "ab", -42L, 0.0001234)
    )
    assertEquals(got, "00inf|3.14e+00|+2|1e-05|1.23457e+08|1.00000|ff|0o10|   ab|-0042|0.000123")
    assertEquals(NpIOPyFormat.format("%.18e", Seq(0.1)), "1.000000000000000056e-01")
    assertEquals(NpIOPyFormat.format("%.0f %.0f", Seq(2.5, 3.5)), "2 4")
    assertEquals(NpIOPyFormat.format("%g %g", Seq(100000.0, 1000000.0)), "100000 1e+06")
    assertEquals(NpIOPyFormat.format("%d %s %e", Seq(1.7, true, -0.0)), "1 True -0.000000e+00")
    assertEquals(NpIOPyFormat.format("%E %X %%", Seq(1.5, 255L)), "1.500000E+00 FF %")
    intercept[IllegalArgumentException](NpIOPyFormat.format("%d %d", Seq(1L)))
    intercept[IllegalArgumentException](NpIOPyFormat.format("%d", Seq(Double.NaN)))
  }

  // ---------------------------------------------------------------- loadtxt

  test("loadtxt shapes follow NumPy's squeeze / ndmin rules") {
    assertEquals(np.loadtxt(Seq("")).shape, Seq(0))
    val one = np.loadtxt(Seq("1"))
    assertEquals(one.ndim, 0)
    assertEquals(one.toList, List(1.0))
    assertEquals(np.loadtxt(Seq("1 2 3")), np.array(1.0, 2.0, 3.0))
    assertEquals(np.loadtxt(Seq("1", "2", "3")).shape, Seq(3))
    assertEquals(np.loadtxt(Seq("1", "2"), ndmin = 2).shape, Seq(2, 1))
    assertEquals(np.loadtxt(Seq("1 2"), ndmin = 2).shape, Seq(1, 2))
    assertEquals(np.loadtxt(Seq("1"), ndmin = 1).shape, Seq(1))
  }

  test("loadtxt comments, delimiter, usecols, dtype, unpack, max_rows, skiprows") {
    val d = tmp()
    val p = write(d.resolve("t.csv"), "# c\n1,2 # x\n\n3,4\n")
    assertEquals(np.loadtxt(p, delimiter = ","), np.array(Seq(1.0, 2.0), Seq(3.0, 4.0)))
    assertEquals(np.loadtxt(p.toString, delimiter = ",", skiprows = 2), np.array(3.0, 4.0))
    val ints = np.loadtxt(Seq("a 1", "b 2"), usecols = 1, dtype = DType.Int32)
    assertEquals(ints, np.array(1, 2))
    val i2: NDArray[Long] = np.loadtxt[Long](Seq("1 2", "3 4"), usecols = Seq(-1))
    assertEquals(i2.toList, List(2L, 4L))
    assertEquals(np.loadtxt(Seq("1 2", "3 4", "5 6"), unpack = true, max_rows = 2), np.array(Seq(1.0, 3.0), Seq(2.0, 4.0)))
    assertEquals(np.loadtxt(Seq("1+2j 3"), dtype = DType.Complex128), np.array(Complex(1, 2), Complex(3, 0)))
    assertEquals(np.loadtxt(Seq("nan inf -inf")).toList.map(_.toString), List("NaN", "Infinity", "-Infinity"))
    assertEquals(np.loadtxt(Seq("1;2 // c"), delimiter = ";", comments = Seq("//", "#")), np.array(1.0, 2.0))
    assertEquals(np.loadtxt(Seq("True False"), dtype = DType.Bool), np.array(true, false))
    val crlf = write(d.resolve("w.txt"), "1 2\r\n3 4\r\n")
    assertEquals(np.loadtxt(crlf).shape, Seq(2, 2))
  }

  test("loadtxt quotechar and str dtype") {
    val a = np.loadtxt(Seq("\"1,5\",2", "\"3\",4"), delimiter = ",", quotechar = '"', dtype = DType.Str)
    assertEquals(a.toList, List("1,5", "2", "3", "4"))
    val b = np.loadtxt(Seq("\"a \"\"q\"\" # no\" 1 # yes"), quotechar = '"', dtype = DType.Str)
    assertEquals(b.toList, List("a \"q\" # no", "1"))
  }

  test("loadtxt errors use NumPy's messages") {
    val e1 = intercept[IllegalArgumentException](np.loadtxt(Seq("1 2", "3")))
    assertEquals(
      e1.getMessage,
      "the number of columns changed from 2 to 1 at row 2; use `usecols` to select a subset and avoid this error"
    )
    val e2 = intercept[IllegalArgumentException](np.loadtxt(Seq("1 x")))
    assertEquals(e2.getMessage, "could not convert string 'x' to float64 at row 0, column 2.")
    intercept[IllegalArgumentException](np.loadtxt(Seq("1 2"), usecols = 5))
    intercept[IllegalArgumentException](np.loadtxt(Seq("1.5"), dtype = DType.Int32))
    intercept[IllegalArgumentException](np.loadtxt(Seq("1"), ndmin = 3))
  }

  // ---------------------------------------------------------------- genfromtxt

  test("genfromtxt fills missing and invalid values") {
    val a = np.genfromtxt(Seq("1,,3", "4,abc,6"), delimiter = ",")
    assertEquals(a, np.array(Seq(1.0, Double.NaN, 3.0), Seq(4.0, Double.NaN, 6.0)))
    val b = np.genfromtxt(Seq("1,,3", "4,5,6"), delimiter = ",", dtype = DType.Int32)
    assertEquals(b, np.array(Seq(1, -1, 3), Seq(4, 5, 6)))
    val c = np.genfromtxt(Seq("1,N/A,3", "4,5,6"), delimiter = ",", missing_values = "N/A", filling_values = 0)
    assertEquals(c, np.array(Seq(1.0, 0.0, 3.0), Seq(4.0, 5.0, 6.0)))
    val per = np.genfromtxt(Seq("1,,", "4,5,6"), delimiter = ",", filling_values = Seq(0, 10, 20))
    assertEquals(per, np.array(Seq(1.0, 10.0, 20.0), Seq(4.0, 5.0, 6.0)))
  }

  test("genfromtxt skip_header, skip_footer, shapes, fixed widths, invalid rows") {
    val a = np.genfromtxt(Seq("h", "1 2", "3 4", "5 6", "foot"), skip_header = 1, skip_footer = 1)
    assertEquals(a, np.array(Seq(1.0, 2.0), Seq(3.0, 4.0), Seq(5.0, 6.0)))
    assertEquals(np.genfromtxt(Seq("1 2")), np.array(1.0, 2.0))
    assertEquals(np.genfromtxt(Seq("1", "2")), np.array(1.0, 2.0))
    assertEquals(np.genfromtxt(Seq("12345"), delimiter = Seq(2, 3)), np.array(12.0, 345.0))
    assertEquals(np.genfromtxt(Seq("1122"), delimiter = 2), np.array(11.0, 22.0))
    val e = intercept[IllegalArgumentException](np.genfromtxt(Seq("1 2", "3", "4 5")))
    assert(e.getMessage.contains("Line #2 (got 1 columns instead of 2)"), e.getMessage)
    assertEquals(np.genfromtxt(Seq("1 2", "3", "4 5"), invalid_raise = false), np.array(Seq(1.0, 2.0), Seq(4.0, 5.0)))
    assertEquals(np.genfromtxt(Seq("1 2 3", "4 5 6"), usecols = Seq(0, 2), unpack = true), np.array(Seq(1.0, 4.0), Seq(3.0, 6.0)))
    assertEquals(np.genfromtxt(Seq("1 2", "# c", "3 4", "5 6"), max_rows = 2).shape, Seq(2, 2))
  }

  test("gzip text input, str savetxt, NpzFile map operations") {
    val d = tmp()
    val gz = d.resolve("a.txt.gz")
    val out = new java.util.zip.GZIPOutputStream(Files.newOutputStream(gz))
    out.write("1 2\n3 4\n".getBytes(StandardCharsets.UTF_8))
    out.close()
    assertEquals(np.loadtxt(gz), np.array(Seq(1.0, 2.0), Seq(3.0, 4.0)))
    val p = d.resolve("s.txt")
    np.savetxt(p, np.array("a", "bc"), fmt = "%s")
    assertEquals(text(p), "a\nbc\n")
    np.savetxt(p, np.array(true, false), fmt = "%d")
    assertEquals(text(p), "1\n0\n")
    np.savez(d.resolve("z.npz"), "a" -> np.ones(1))
    val z = np.load_npz(d.resolve("z.npz"))
    assertEquals(z.get("a.npy").map(_.size), Some(1))
    assertEquals((z - "a").size, 0)
    assertEquals(z.updated("b", np.zeros(1)).keySet, Set("a", "b"))
    val g = np.genfromtxt(Seq("1,,", "2,,"), delimiter = ",", filling_values = Map(1 -> 7, 2 -> 8))
    assertEquals(g, np.array(Seq(1.0, 7.0, 8.0), Seq(2.0, 7.0, 8.0)))
  }

  // ---------------------------------------------------------------- raw binary

  test("tofile / fromfile binary and text round trips") {
    val d = tmp()
    val p = d.resolve("raw.bin")
    val a = np.array(1.5, -2.0, 3.0)
    a.tofile(p)
    assertEquals(Files.size(p), 24L)
    assertEquals(np.fromfile(p), a)
    assertEquals(np.fromfile(p, count = 2), np.array(1.5, -2.0))
    assertEquals(np.fromfile(p, offset = 8), np.array(-2.0, 3.0))
    np.tofile(np.array(1, 2, 3), p)
    assertEquals(np.fromfile(p, dtype = DType.Int32), np.array(1, 2, 3))
    val t = d.resolve("t.txt")
    np.tofile(np.array(1.0, 2.5), t, sep = ", ")
    assertEquals(text(t), "1.0, 2.5")
    assertEquals(np.fromfile(t, sep = ","), np.array(1.0, 2.5))
    np.array(1, 2, 3).tofile(t, sep = " ", format = "%03d")
    assertEquals(text(t), "001 002 003")
    assertEquals(np.fromfile(t, dtype = DType.Int64, sep = " ", count = 2), np.array(1L, 2L))
    intercept[IllegalArgumentException](np.fromfile(t, sep = " ", offset = 1))
    intercept[IllegalArgumentException](np.tofile(np.array("a"), p))
  }

  test("frombuffer decodes little-endian bytes") {
    val bytes = Array[Byte](1, 0, 0, 0, 2, 0, 0, 0, -1, -1, -1, -1)
    assertEquals(np.frombuffer(bytes, dtype = DType.Int32), np.array(1, 2, -1))
    assertEquals(np.frombuffer(bytes, dtype = DType.Int32, count = 1, offset = 4).toList, List(2))
    assertEquals(np.frombuffer(bytes, dtype = DType.Int16).toList, List[Short](1, 0, 2, 0, -1, -1))
    assertEquals(np.frombuffer(bytes, dtype = DType.Bool).toList.take(2), List(true, false))
    val d = np.array(0.25, 4.0)
    assertEquals(np.frombuffer(d.tobytes()), d)
    val c = np.array(Seq(Complex(1, -1)))
    assertEquals(np.frombuffer(c.tobytes(), dtype = DType.Complex128), c)
    assertEquals(np.array(1.0f).tobytes().length, 4)
    intercept[IllegalArgumentException](np.frombuffer(Array[Byte](1, 2, 3)))
    intercept[IllegalArgumentException](np.frombuffer(bytes, dtype = DType.Int64, count = 2))
    intercept[IllegalArgumentException](np.frombuffer(bytes, dtype = DType.Str))
  }
