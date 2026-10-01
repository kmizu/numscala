package numscala

import scala.compiletime.{erasedValue, summonFrom, summonInline}

/** Minimal JSON reader for the differential-test data (numbers are kept as strings). */
private[numscala] object DiffJson:
  final case class Num(text: String)

  def parse(s: String): Any =
    val p = new Parser(s)
    val v = p.value()
    p.ws()
    if p.i != s.length then throw new IllegalArgumentException(s"trailing JSON at ${p.i}")
    v

  private final class Parser(s: String):
    var i = 0
    def ws(): Unit = while i < s.length && s(i).isWhitespace do i += 1
    def value(): Any =
      ws()
      s(i) match
        case '{' =>
          i += 1
          val m = scala.collection.mutable.LinkedHashMap.empty[String, Any]
          ws()
          if s(i) == '}' then i += 1
          else
            var more = true
            while more do
              ws()
              val k = str()
              ws(); expect(':')
              m(k) = value()
              ws()
              if s(i) == ',' then i += 1 else { expect('}'); more = false }
          m.toMap
        case '[' =>
          i += 1
          val b = Vector.newBuilder[Any]
          ws()
          if s(i) == ']' then i += 1
          else
            var more = true
            while more do
              b += value()
              ws()
              if s(i) == ',' then i += 1 else { expect(']'); more = false }
          b.result()
        case '"' => str()
        case 't' => i += 4; true
        case 'f' => i += 5; false
        case 'n' => i += 4; null
        case _ =>
          val st = i
          while i < s.length && "+-0123456789.eE".indexOf(s(i)) >= 0 do i += 1
          Num(s.substring(st, i))
    def expect(c: Char): Unit =
      if s(i) != c then throw new IllegalArgumentException(s"expected '$c' at $i")
      i += 1
    def str(): String =
      expect('"')
      val sb = new StringBuilder
      while s(i) != '"' do
        if s(i) == '\\' then
          i += 1
          s(i) match
            case 'n' => sb += '\n'
            case 't' => sb += '\t'
            case 'u' => sb += Integer.parseInt(s.substring(i + 1, i + 5), 16).toChar; i += 4
            case c => sb += c
        else sb += s(i)
        i += 1
      i += 1
      sb.toString

/** Runtime lookup of the statically resolved promotion typeclasses, keyed by dtype name. */
private[numscala] object DiffTypes:
  type Ts = (Boolean, Byte, Short, Int, Long, UInt8, UInt32, Float, Double, Complex)

  private inline def promoteRow[A, R <: Tuple](na: String): List[((String, String), Promote[?, ?])] =
    inline erasedValue[R] match
      case _: EmptyTuple => Nil
      case _: (b *: rest) =>
        ((na, summonInline[DType[b]].name), summonInline[Promote[A, b]]) :: promoteRow[A, rest](na)
  private inline def numRow[A, R <: Tuple](na: String): List[((String, String), NumPromote[?, ?])] =
    inline erasedValue[R] match
      case _: EmptyTuple => Nil
      case _: (b *: rest) =>
        summonFrom {
          case p: NumPromote[A, `b`] => ((na, summonInline[DType[b]].name), p) :: numRow[A, rest](na)
          case _ => numRow[A, rest](na)
        }
  private inline def divRow[A, R <: Tuple](na: String): List[((String, String), DivPromote[?, ?])] =
    inline erasedValue[R] match
      case _: EmptyTuple => Nil
      case _: (b *: rest) =>
        summonFrom {
          case p: DivPromote[A, `b`] => ((na, summonInline[DType[b]].name), p) :: divRow[A, rest](na)
          case _ => divRow[A, rest](na)
        }

  final case class Unary(sum: SumOf[?], inexact: ToInexact[?], real: RealOf[?])

  private inline def all[R <: Tuple]: List[(String, Unary, List[((String, String), Promote[?, ?])],
      List[((String, String), NumPromote[?, ?])], List[((String, String), DivPromote[?, ?])])] =
    inline erasedValue[R] match
      case _: EmptyTuple => Nil
      case _: (a *: rest) =>
        val n = summonInline[DType[a]].name
        (n, Unary(summonInline[SumOf[a]], summonInline[ToInexact[a]], summonInline[RealOf[a]]),
          promoteRow[a, Ts](n), numRow[a, Ts](n), divRow[a, Ts](n)) :: all[rest]

  private val table = all[Ts]
  val dtypes: Map[String, DType[?]] =
    Map("bool" -> DType.Bool, "int8" -> DType.Int8, "int16" -> DType.Int16, "int32" -> DType.Int32,
      "int64" -> DType.Int64, "uint8" -> DType.UInt8, "uint32" -> DType.UInt32, "uint16" -> DType.UInt16, "uint64" -> DType.UInt64, "float32" -> DType.Float32,
      "float64" -> DType.Float64, "complex128" -> DType.Complex128)
  val unary: Map[String, Unary] = table.map(t => t._1 -> t._2).toMap
  val promote: Map[(String, String), Promote[?, ?]] = table.flatMap(_._3).toMap
  val numPromote: Map[(String, String), NumPromote[?, ?]] = table.flatMap(_._4).toMap
  val divPromote: Map[(String, String), DivPromote[?, ?]] = table.flatMap(_._5).toMap

/** Decoding of arrays / indices and comparison against NumPy's recorded results. */
private[numscala] object DiffData:
  type J = Map[String, Any]
  type A = NDArray[Any]

  def int(x: Any): Int = x.asInstanceOf[DiffJson.Num].text.toInt
  def ints(x: Any): Seq[Int] = x.asInstanceOf[Vector[Any]].map(int)
  def optInt(x: Any): Option[Int] = Option(x).map(int)

  def dbl(s: String): Double = s match
    case "nan" => Double.NaN
    case "inf" => Double.PositiveInfinity
    case "-inf" => Double.NegativeInfinity
    case t => t.toDouble

  private def elem(d: DType[Any], x: Any): Any = x match
    case b: Boolean => d.fromBoolean(b)
    case n: DiffJson.Num => d.fromLong(BigInt(n.text).toLong)
    case s: String =>
      if d eq DType.Float32 then d.fromDouble(if s.endsWith("inf") || s == "nan" then dbl(s) else s.toFloat.toDouble)
      else d.fromDouble(dbl(s))
    case v: Vector[?] => d.fromComplex(Complex(dbl(v(0).asInstanceOf[String]), dbl(v(1).asInstanceOf[String])))
    case other => throw new IllegalArgumentException(s"bad value $other")

  /** Contiguous array from the `d`/`s`/`v` fields (ignores view steps). */
  def plain(j: Any): A =
    val m = j.asInstanceOf[J]
    val d = DiffTypes.dtypes(m("d").asInstanceOf[String])
    build(d, m("v").asInstanceOf[Vector[Any]], ints(m("s")).toArray).asInstanceOf[A]

  // generic in T so that the primitive buffer is never cast to Array[Object]
  private def build[T](d: DType[T], vs: Vector[Any], shape: Array[Int]): NDArray[T] =
    val data = d.newArray(vs.length)
    var k = 0
    while k < vs.length do
      data(k) = elem(d.asInstanceOf[DType[Any]], vs(k)).asInstanceOf[T]
      k += 1
    NDArray.fromArray(data, shape)(using d)

  /** Array with its view steps applied. */
  def array(j: Any): A = applySteps(plain(j), j.asInstanceOf[J].getOrElse("w", Vector.empty))

  def applySteps(a0: A, steps: Any): A =
    steps.asInstanceOf[Vector[Any]].foldLeft(a0) { (a, st) =>
      val m = st.asInstanceOf[J]
      if m.contains("T") then a.transpose(ints(m("T"))*)
      else a.index(index(m("ix")))
    }

  def index(j: Any): Seq[Index] =
    j.asInstanceOf[Vector[Any]].map { it =>
      val m = it.asInstanceOf[J]
      if m.contains("i") then Index.At(int(m("i")))
      else if m.contains("sl") then
        val v = m("sl").asInstanceOf[Vector[Any]]
        Index.Slice(optInt(v(0)), optInt(v(1)), optInt(v(2)).getOrElse(1))
      else if m.contains("na") then Index.NewAxis
      else if m.contains("el") then Index.Ellipsis
      else Index.from(array(m("a")))
    }

  /** Element comparison options: `rtol` (0 = exact) and an absolute tolerance. */
  final case class Tol(rtol: Double, atol: Double = 0.0, idx: Boolean = false)
  val Exact: Tol = Tol(0.0)

  private def close(x: Double, y: Double, t: Tol): Boolean =
    if x.isNaN || y.isNaN then x.isNaN && y.isNaN
    else if x.isInfinite || y.isInfinite then x == y
    else if t.rtol == 0.0 then x == y
    else math.abs(x - y) <= t.atol + t.rtol * math.max(math.abs(x), math.abs(y))

  /** Returns `None` when `actual` matches the recorded NumPy array `exp`, else a description. */
  def check(actual: NDArray[?], exp: Any, tol: Tol): Option[String] =
    val e = plain(exp)
    val en = e.dtype.name
    val an = actual.dtype.name
    val dtOk = en == an || (tol.idx && en == "int64" && an == "int32")
    if !dtOk then return Some(s"dtype $an != numpy $en")
    if actual.shape != e.shape then return Some(s"shape ${actual.shape.mkString("(", ",", ")")} != numpy ${e.shape.mkString("(", ",", ")")}")
    val ad = actual.dtype.asInstanceOf[DType[Any]]
    val aa = actual.asInstanceOf[A]
    val ed = e.dtype.asInstanceOf[DType[Any]]
    // tolerance for float32 results is relative to float32 precision
    val t = if an == "float32" && tol.rtol > 0 then tol.copy(rtol = math.max(tol.rtol, 2e-6)) else tol
    var k = 0
    while k < e.size do
      val (xa, xe) = (aa.flatGet(k), e.flatGet(k))
      val ok = ad.kind match
        case 'c' =>
          val (x, y) = (ad.toComplex(xa), ed.toComplex(xe))
          close(x.re, y.re, t) && close(x.im, y.im, t)
        case 'f' => close(ad.toDouble(xa), ed.toDouble(xe), t)
        case 'b' => ad.toBoolean(xa) == ed.toBoolean(xe)
        case _ => ad.toLong(xa) == ed.toLong(xe)
      if !ok then
        return Some(s"element $k: ${ad.format(xa)} != numpy ${ed.format(xe)}\n  got:   ${actual.asInstanceOf[A].repr}\n  numpy: ${e.repr}")
      k += 1
    None
