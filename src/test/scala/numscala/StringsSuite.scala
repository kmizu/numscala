package numscala

class StringsSuite extends munit.FunSuite:
  private val s = np.strings
  private def arr(xs: String*): NDArray[String] = s.array(xs)

  private val sample =
    arr("hello world", "  Abc Def ", "ß straße", "x\ty\tz", "-42", "+7", "", "ǆemal")

  test("np.char is np.strings; array/asarray helpers") {
    assert(np.char eq np.strings)
    val a = s.array(Seq("abc", "de"))
    assertEquals(a.dtype.name, "str")
    assertEquals(a.shape, Seq(2))
    assertEquals(s.array(Seq("abcdef", "xy"), 3).toList, List("abc", "xy"))
    val c = s.array(a)
    c.flatSet(0, "zz")
    assertEquals(a.toList, List("abc", "de"))
    assert(s.asarray(a) eq a)
    assertEquals(s.asarray(Seq("q")).toList, List("q"))
    intercept[IllegalArgumentException](s.upper(np.array(1, 2).asInstanceOf[NDArray[String]]))
  }

  test("case conversion") {
    assertEquals(
      s.capitalize(sample).toList,
      List("Hello world", "  abc def ", "Ss straße", "X\ty\tz", "-42", "+7", "", "ǅemal")
    )
    assertEquals(
      s.title(sample).toList,
      List("Hello World", "  Abc Def ", "Ss Straße", "X\tY\tZ", "-42", "+7", "", "ǅemal")
    )
    assertEquals(
      s.swapcase(sample).toList,
      List("HELLO WORLD", "  aBC dEF ", "SS STRASSE", "X\tY\tZ", "-42", "+7", "", "ǄEMAL")
    )
    assertEquals(
      s.upper(sample).toList,
      List("HELLO WORLD", "  ABC DEF ", "SS STRASSE", "X\tY\tZ", "-42", "+7", "", "ǄEMAL")
    )
    assertEquals(s.lower(arr("ABC", "ǄEMAL")).toList, List("abc", "ǆemal"))
    assertEquals(s.title(arr("they're bill's friends")).toList, List("They'Re Bill'S Friends"))
    assertEquals(s.upper("abc").shape, Seq.empty[Int])
    assertEquals(s.upper("abc").item, "ABC")
  }

  test("predicates") {
    assertEquals(s.str_len(sample).toList, List(11, 10, 8, 5, 3, 2, 0, 5))
    assertEquals(s.isalpha(sample).toList, List(false, false, false, false, false, false, false, true))
    assertEquals(s.isupper(sample).toList, List(false, false, false, false, false, false, false, false))
    assertEquals(s.islower(sample).toList, List(true, false, true, true, false, false, false, true))
    assertEquals(s.istitle(sample).toList, List(false, true, false, false, false, false, false, false))
    assertEquals(s.isupper(arr("ABC1", "AbC", "1")).toList, List(true, false, false))
    val t = arr("123", "½", "²", "①", "Ⅻ", "一二", "٣", "12a", "a1", "  ", " ", " \u001c", "")
    assertEquals(
      s.isdecimal(t).toList,
      List(true, false, false, false, false, false, true, false, false, false, false, false, false)
    )
    assertEquals(
      s.isdigit(t).toList,
      List(true, false, true, true, false, false, true, false, false, false, false, false, false)
    )
    assertEquals(
      s.isnumeric(t).toList,
      List(true, true, true, true, true, true, true, false, false, false, false, false, false)
    )
    assertEquals(
      s.isalnum(t).toList,
      List(true, true, true, true, true, true, true, true, true, false, false, false, false)
    )
    assertEquals(
      s.isspace(t).toList,
      List(false, false, false, false, false, false, false, false, false, true, true, true, false)
    )
  }

  test("padding: center/ljust/rjust/zfill/expandtabs") {
    assertEquals(
      s.center(sample, 15, "*").toList,
      List("**hello world**", "***  Abc Def **", "****ß straße***", "*****x\ty\tz*****", "******-42******",
        "*******+7******", "***************", "*****ǆemal*****")
    )
    assertEquals(s.center(arr("a", "ab"), 4).toList, List(" a  ", " ab "))
    assertEquals(s.center(arr("abc"), np.array(1, 5, 6)).toList, List("abc", " abc ", " abc  "))
    assertEquals(s.ljust(arr("ab"), 4, "-").toList, List("ab--"))
    assertEquals(s.rjust(arr("ab", "abcde"), 4).toList, List("  ab", "abcde"))
    intercept[IllegalArgumentException](s.center(arr("a"), 5, "ab"))
    assertEquals(
      s.zfill(sample, 6).toList,
      List("hello world", "  Abc Def ", "ß straße", "0x\ty\tz", "-00042", "+00007", "000000", "0ǆemal")
    )
    assertEquals(s.expandtabs(arr("x\ty\tz"), 4).toList, List("x   y   z"))
    assertEquals(s.expandtabs("a\tbc\td\n\tx", 3).item, "a  bc d\n   x")
    assertEquals(s.expandtabs(arr("a\tb")).toList, List("a       b"))
    assertEquals(s.expandtabs(arr("a\tb"), 0).toList, List("ab"))
    // non-BMP characters count as one
    assertEquals(s.center(arr("😀"), 3, "-").toList, List("-😀-"))
    assertEquals(s.str_len(arr("😀a")).toList, List(2))
  }

  test("strip family") {
    assertEquals(
      s.strip(sample).toList,
      List("hello world", "Abc Def", "ß straße", "x\ty\tz", "-42", "+7", "", "ǆemal")
    )
    assertEquals(
      s.strip(sample, "hd -").toList,
      List("ello worl", "Abc Def", "ß straße", "x\ty\tz", "42", "+7", "", "ǆemal")
    )
    assertEquals(s.lstrip(arr("  a  ")).toList, List("a  "))
    assertEquals(s.rstrip(arr("  a  ", "abee "), "e ").toList, List("  a", "ab"))
    assertEquals(s.strip(arr("xxaxx", "yay"), arr("x", "y")).toList, List("a", "a"))
    assertEquals(s.strip(arr(" 　a ")).toList, List("a"))
  }

  test("find / rfind / count / index / startswith / endswith") {
    assertEquals(s.find(sample, "l", 2).toList, List(2, -1, -1, -1, -1, -1, -1, 4))
    assertEquals(s.rfind(sample, "l").toList, List(9, -1, -1, -1, -1, -1, -1, 4))
    assertEquals(s.count(sample, "l", 0, 10).toList, List(3, 0, 0, 0, 0, 0, 0, 1))
    assertEquals(s.find("abc", "", 3).item, 3)
    assertEquals(s.find("abc", "", 4).item, -1)
    assertEquals(s.count("abc", "").item, 4)
    assertEquals(s.count("abc", "", 2, 1).item, 0)
    assertEquals(s.count(arr("aaaa"), "aa").toList, List(2))
    assertEquals(s.startswith("abc", "", 3).item, true)
    assertEquals(s.startswith("abc", "", 4).item, false)
    assertEquals(s.endswith("abcd", "bc", -4, -1).item, true)
    assertEquals(s.startswith(arr("abc", "xbc"), arr("a", "x")).toList, List(true, true))
    assertEquals(s.endswith(arr("abc", "xbd"), "bc").toList, List(true, false))
    assertEquals(s.find(arr("abcabc"), "c", np.array(0, 3, 6)).toList, List(2, 5, -1))
    assertEquals(s.find(arr("abcabc"), "bc", 0, -3).toList, List(1))
    assertEquals(s.rfind(arr("abcabc"), "bc", 0, -1).toList, List(1))
    assertEquals(s.index(arr("abc"), "c").toList, List(2))
    assertEquals(s.rindex(arr("abcc"), "c").toList, List(3))
    val e = intercept[IllegalArgumentException](s.index(arr("abc", "x"), "c"))
    assertEquals(e.getMessage, "substring not found")
    intercept[IllegalArgumentException](s.rindex(arr("abc"), "z"))
    // code point indices
    assertEquals(s.find(arr("😀ab"), "b").toList, List(2))
    assertEquals(s.rfind(arr("😀b😀b"), "b", 0, 3).toList, List(1))
  }

  test("replace / join / multiply / add") {
    assertEquals(s.replace(arr("aaaa", "abab"), "a", "X", np.array(2, -1)).toList, List("XXaa", "XbXb"))
    assertEquals(s.replace("abc", "", "-").item, "-a-b-c-")
    assertEquals(s.replace("abc", "", "-", 2).item, "-a-bc")
    assertEquals(s.replace(arr("abc"), "b", "", 0).toList, List("abc"))
    assertEquals(s.join("-", arr("abc", "x")).toList, List("a-b-c", "x"))
    assertEquals(s.join(arr("-", "::"), "ab").toList, List("a-b", "a::b"))
    val m = s.multiply(arr("ab", "c"), np.array(3, 0, -1).reshape(3, 1))
    assertEquals(m.shape, Seq(3, 2))
    assertEquals(m.toList, List("ababab", "ccc", "", "", "", ""))
    assertEquals(s.multiply("ab", 2L).item, "abab")
    val ad = s.add(arr("a", "b"), arr("x", "y", "z").reshape(3, 1))
    assertEquals(ad.shape, Seq(3, 2))
    assertEquals(ad.toList, List("ax", "bx", "ay", "by", "az", "bz"))
    assertEquals(s.add("a", arr("1")).toList, List("a1"))
  }

  test("translate / maketrans") {
    val tbl = s.maketrans("ab", "xy", "c")
    assertEquals(s.translate(arr("abcabd"), tbl).toList, List("xyxyd"))
    assertEquals(s.translate(arr("hello"), Map('l' -> "LL", 'o' -> '0')).toList, List("heLLLL0"))
    intercept[IllegalArgumentException](s.maketrans("ab", "x", ""))
  }

  test("partition / rpartition") {
    val (a, b, c) = s.partition(arr("a-b-c", "abc"), "-")
    assertEquals((a.toList, b.toList, c.toList), (List("a", "abc"), List("-", ""), List("b-c", "")))
    val (d, e, f) = s.rpartition(arr("a-b-c", "abc"), "-")
    assertEquals((d.toList, e.toList, f.toList), (List("a-b", ""), List("-", ""), List("c", "abc")))
    intercept[IllegalArgumentException](s.partition(arr("a"), ""))
  }

  test("split / rsplit / splitlines") {
    val sp = s.split(arr("a b  c ", " x "))
    assertEquals(sp.shape, Seq(2))
    assertEquals(sp(0), List("a", "b", "c"))
    assertEquals(sp(1), List("x"))
    assertEquals(s.split(arr("a b  c "), null, 1)(0), List("a", "b  c "))
    assertEquals(s.rsplit(arr(" a b  c "), null, 1)(0), List(" a b", "c"))
    assertEquals(s.split(arr("a,b,,c"), ",", 2)(0), List("a", "b", ",c"))
    assertEquals(s.rsplit(arr("a,b,,c"), ",", 2)(0), List("a,b", "", "c"))
    assertEquals(s.split(arr("a,b,,c"), ",")(0), List("a", "b", "", "c"))
    assertEquals(s.rsplit(arr("aaa"), "aa")(0), List("a", ""))
    assertEquals(s.split(arr("   "))(0), Nil)
    assertEquals(s.split(arr(""), ",")(0), List(""))
    intercept[IllegalArgumentException](s.split(arr("a"), ""))
    assertEquals(
      s.splitlines(arr("l1\nl2\r\nl3\rl4\u000bl5\n"))(0),
      List("l1", "l2", "l3", "l4", "l5")
    )
    assertEquals(s.splitlines(arr("l1\r\nl2\n"), true)(0), List("l1\r\n", "l2\n"))
    val two = s.split(arr("a b", "c", "d e f", "g").reshape(2, 2))
    assertEquals(two.shape, Seq(2, 2))
    assertEquals(two(1, 0), List("d", "e", "f"))
    assertEquals(two(-1, -1), List("g"))
    assertEquals(two.tolist, List(List(List("a", "b"), List("c")), List(List("d", "e", "f"), List("g"))))
    assertEquals(two.toString, "ObjectArray([[['a', 'b'], ['c']], [['d', 'e', 'f'], ['g']]])")
    assertEquals(two.map(_.length).toList, List(2, 1, 3, 1))
    assertEquals(two, s.split(arr("a b", "c", "d e f", "g").reshape(2, 2)))
    intercept[IndexOutOfBoundsException](two(2, 0))
  }

  test("slice") {
    assertEquals(s.slice(arr("abcdef", "xyz"), 1, None, 2).toList, List("bdf", "y"))
    assertEquals(s.slice(arr("abcdef"), 2).toList, List("ab"))
    assertEquals(s.slice(arr("abcdef"), None, None, -1).toList, List("fedcba"))
    assertEquals(s.slice(arr("abcdef"), -2, 0, -1).toList, List("edcb"))
    assertEquals(s.slice(arr("abcdef"), np.array(0, 1, 2), None).toList, List("abcdef", "bcdef", "cdef"))
    assertEquals(s.slice(arr("a😀b"), 1, 2).toList, List("😀"))
    intercept[IllegalArgumentException](s.slice(arr("a"), 0, 1, 0))
  }

  test("comparisons") {
    assertEquals(s.less(arr("a", "b", "abc"), "ab").toList, List(true, false, false))
    assertEquals(s.less_equal(arr("a", "ab"), "ab").toList, List(true, true))
    assertEquals(s.greater(arr("b", "ab"), "ab").toList, List(true, false))
    assertEquals(s.greater_equal(arr("b", "aa"), "ab").toList, List(true, false))
    assertEquals(s.equal(arr("a", "a "), "a").toList, List(true, false))
    assertEquals(s.not_equal(arr("a", "a "), "a").toList, List(false, true))
    // code-point order: U+1F600 > U+FFFF (UTF-16 order would say otherwise)
    assertEquals(s.greater(arr("😀"), "￿").toList, List(true))
    assertEquals(s.compare_chararrays(arr("a ", "b"), arr("a", "a"), "==", true).toList, List(true, false))
    assertEquals(s.compare_chararrays(arr("a ", "b"), arr("a", "a"), ">", false).toList, List(true, true))
    assertEquals(s.compare_chararrays(arr("a", "b"), arr("a", "c"), "<=", false).toList, List(true, true))
    assertEquals(s.compare_chararrays(arr("a", "b"), arr("a", "c"), "!=", false).toList, List(false, true))
    assertEquals(s.compare_chararrays(arr("a", "b"), arr("a", "c"), ">=", false).toList, List(true, false))
    assertEquals(s.compare_chararrays(arr("a", "b"), arr("a", "c"), "<", false).toList, List(false, true))
    intercept[IllegalArgumentException](s.compare_chararrays(arr("a"), arr("b"), "<>", false))
    intercept[IllegalArgumentException](s.compare_chararrays(arr("a"), arr("b", "c"), "==", false))
  }

  test("encode / decode") {
    val e = s.encode(arr("héllo", "abc"))
    assertEquals(e(0).toList, List[Byte](104, -61, -87, 108, 108, 111))
    assertEquals(s.decode(e).toList, List("héllo", "abc"))
    val l1 = s.encode(arr("é"), "latin-1")
    assertEquals(l1(0).toList, List[Byte](-23))
    assertEquals(s.decode(l1, "latin-1").toList, List("é"))
    intercept[IllegalArgumentException](s.encode(arr("é"), "ascii"))
    assertEquals(s.encode(arr("aéb"), "ascii", "ignore")(0).toList, List[Byte](97, 98))
    assertEquals(s.encode(arr("aéb"), "ascii", "replace")(0).toList, List[Byte](97, 63, 98))
    assertEquals(s.decode(Seq(Array[Byte](97, -1)), "utf-8", "replace").toList, List("a�"))
    intercept[IllegalArgumentException](s.decode(Seq(Array[Byte](97, -1))))
    intercept[IllegalArgumentException](s.encode(arr("a"), "no-such-codec"))
    intercept[IllegalArgumentException](s.encode(arr("a"), "utf-8", "bogus"))
    assertEquals(s.encode(arr("ab")).toString, "ObjectArray([b'ab'])")
    assertEquals(s.encode(arr("ab")), s.encode(arr("ab")))
  }

  private def fmt(f: String, v: NDArray[?]): String = s.mod(f, v).item
  private def d(x: Double) = NDArray.scalar(x)
  private def l(x: Long) = NDArray.scalar(x)
  private def st(x: String) = NDArray.scalar(x)

  test("mod: integers") {
    assertEquals(fmt("%d", l(42)), "42")
    assertEquals(fmt("%5d", l(-42)), "  -42")
    assertEquals(fmt("%-5d|", l(42)), "42   |")
    assertEquals(fmt("%05d", l(-42)), "-0042")
    assertEquals(fmt("%+d", l(5)), "+5")
    assertEquals(fmt("% d", l(5)), " 5")
    assertEquals(fmt("%.3d", l(5)), "005")
    assertEquals(fmt("%x", l(255)), "ff")
    assertEquals(fmt("%#X", l(255)), "0XFF")
    assertEquals(fmt("%#o", l(8)), "0o10")
    assertEquals(fmt("%#08x", l(255)), "0x0000ff")
    assertEquals(fmt("%d", d(2.9)), "2")
    assertEquals(fmt("%d", d(-2.9)), "-2")
    assertEquals(fmt("%d", d(1e30)), "1000000000000000019884624838656")
    assertEquals(fmt("%i", l(7)), "7")
    assertEquals(fmt("%d%%", l(5)), "5%")
    assertEquals(fmt("%d", NDArray.scalar(true)), "1")
    assertEquals(fmt("%c", l(65)), "A")
    assertEquals(fmt("%c", st("z")), "z")
    assertEquals(fmt("abc", l(1)), "abc")
  }

  test("mod: floats") {
    assertEquals(fmt("%f", d(3.14159)), "3.141590")
    assertEquals(fmt("%.2f", d(2.675)), "2.67")
    assertEquals(fmt("%.2f", d(0.125)), "0.12")
    assertEquals(fmt("%10.3f", d(-1.5)), "    -1.500")
    assertEquals(fmt("%-10.1f|", d(2.25)), "2.2       |")
    assertEquals(fmt("%+.1f", d(0.0)), "+0.0")
    assertEquals(fmt("%f", d(-0.0)), "-0.000000")
    assertEquals(fmt("%e", d(12345.678)), "1.234568e+04")
    assertEquals(fmt("%.0e", d(5e-310)), "5e-310")
    assertEquals(fmt("%#.0f", d(3.0)), "3.")
    assertEquals(fmt("%E", d(1e100)), "1.000000E+100")
    assertEquals(fmt("%g", d(0.0001)), "0.0001")
    assertEquals(fmt("%g", d(0.00001)), "1e-05")
    assertEquals(fmt("%g", d(123456789.0)), "1.23457e+08")
    assertEquals(fmt("%.3g", d(1234.5)), "1.23e+03")
    assertEquals(fmt("%#g", d(1.0)), "1.00000")
    assertEquals(fmt("%G", d(1e-10)), "1E-10")
    assertEquals(fmt("%g", d(100000.0)), "100000")
    assertEquals(fmt("%g", d(1e6)), "1e+06")
    assertEquals(fmt("%.0g", d(0.5)), "0.5")
    assertEquals(fmt("%g", d(0.0)), "0")
    assertEquals(fmt("%f", d(Double.PositiveInfinity)), "inf")
    assertEquals(fmt("%05f", d(Double.PositiveInfinity)), "00inf")
    assertEquals(fmt("%F", d(Double.NaN)), "NAN")
    assertEquals(fmt("%+e", d(Double.NegativeInfinity)), "-inf")
    assertEquals(fmt("%.1f", d(1e22)), "10000000000000000000000.0")
    assertEquals(fmt("%g", d(1e22)), "1e+22")
    assertEquals(fmt("%e", l(3)), "3.000000e+00")
    assertEquals(fmt("%f", NDArray.scalar(Complex(1, 0))), "1.000000")
  }

  test("mod: strings, repr and errors") {
    assertEquals(fmt("%s", d(1.5)), "1.5")
    assertEquals(fmt("%r", d(1.5)), "np.float64(1.5)")
    assertEquals(fmt("%s", st("abc")), "abc")
    assertEquals(fmt("%r", st("abc")), "np.str_('abc')")
    assertEquals(fmt("%a", st("é")), "np.str_('\\xe9')")
    assertEquals(fmt("%r", NDArray.scalar(true)), "np.True_")
    assertEquals(fmt("%r", NDArray.scalar(3)), "np.int32(3)")
    assertEquals(fmt("%r", NDArray.scalar(Complex(1, 2))), "np.complex128(1+2j)")
    assertEquals(fmt("%s", NDArray.scalar(Complex(1, -2))), "(1-2j)")
    assertEquals(fmt("%.2s", st("abcdef")), "ab")
    assertEquals(fmt("%5s|", st("ab")), "   ab|")
    assertEquals(fmt("%-5s|", st("ab")), "ab   |")
    assertEquals(fmt("%s", NDArray.scalar(true)), "True")
    intercept[IllegalArgumentException](fmt("%x", d(1.5)))
    intercept[IllegalArgumentException](fmt("%x", NDArray.scalar(true)))
    intercept[IllegalArgumentException](fmt("%d", st("a")))
    intercept[IllegalArgumentException](fmt("%f", st("a")))
    intercept[IllegalArgumentException](fmt("%c", st("ab")))
    intercept[IllegalArgumentException](fmt("%d %d", l(1)))
    intercept[IllegalArgumentException](fmt("%d", d(Double.NaN)))
    intercept[IllegalArgumentException](fmt("%y", l(1)))
    intercept[IllegalArgumentException](fmt("%(a)s", l(1)))
    intercept[IllegalArgumentException](fmt("%", l(1)))
  }

  test("mod: arrays and tuples") {
    assertEquals(s.mod(arr("%d", "%x"), np.array(1, 2)).toList, List("1", "2"))
    assertEquals(s.mod("%.2f", np.array(1.0, 2.5)).toList, List("1.00", "2.50"))
    val t = s.mod(arr("%s-%05.1f", "%s|%.1e"), Seq(arr("x", "y"), np.array(2.5, 1000.0)))
    assertEquals(t.toList, List("x-002.5", "y|1.0e+03"))
    assertEquals(s.mod("%*d|", Seq(np.array(5), np.array(3))).toList, List("    3|"))
    assertEquals(s.mod("%-*d|", Seq(np.array(5), np.array(3))).toList, List("3    |"))
    assertEquals(s.mod("%*d|", Seq(np.array(-5), np.array(3))).toList, List("3    |"))
    assertEquals(s.mod("%.*f", Seq(np.array(2), np.array(3.14159))).toList, List("3.14"))
    intercept[IllegalArgumentException](s.mod("%d", Seq(np.array(1), np.array(2))))
    intercept[IllegalArgumentException](s.mod("%*d", Seq(np.array(1.5), np.array(2))))
  }

  test("integer-array arguments are validated") {
    intercept[IllegalArgumentException](s.zfill(arr("1"), np.array(1.5)))
    assertEquals(s.zfill(arr("1"), np.array(3L)).toList, List("001"))
  }
