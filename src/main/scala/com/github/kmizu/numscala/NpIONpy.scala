package com.github.kmizu.numscala

import java.io.{ByteArrayOutputStream, InputStream}
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets

/** Reader/writer for NumPy's `.npy` binary format (versions 1.0, 2.0 and 3.0). */
private[numscala] object NpIONpy:
  val Magic: Array[Byte] = Array(0x93.toByte, 'N', 'U', 'M', 'P', 'Y').map(_.toByte)
  private val ArrayAlign = 64
  private val GrowthAxisMaxDigits = 21

  // ------------------------------------------------------------------ descr

  /** Number of code points of a string (the `n` of `<U{n}`). */
  private def codePoints(s: String): Int = s.codePointCount(0, s.length)

  /** The `descr` of an array as NumPy writes it, plus the element size in bytes. */
  def descrOf(a: NDArray[?]): (String, Int) =
    val d = a.dtype
    d.kind match
      case 'b' => ("|b1", 1)
      case 'U' =>
        val n = math.max(1, a.toSeq.map(x => codePoints(x.asInstanceOf[String])).maxOption.getOrElse(1))
        (s"<U$n", 4 * n)
      case k if d.itemSize == 1 => (s"|${k}1", 1)
      case k => (s"<$k${d.itemSize}", d.itemSize)

  /** Parsed `descr`: dtype in memory, byte order, on-disk kind and on-disk item size. */
  final case class Descr(dtype: DType[?], order: ByteOrder, kind: Char, size: Int)

  def parseDescr(descr: String): Descr =
    val bad = new IllegalArgumentException(s"data type '$descr' not understood")
    if descr.isEmpty then throw bad
    val (orderCh, body) = descr.head match
      case c @ ('<' | '>' | '|' | '=') => (c, descr.tail)
      case _ => ('=', descr)
    val order = if orderCh == '>' then ByteOrder.BIG_ENDIAN else ByteOrder.LITTLE_ENDIAN
    if body.isEmpty then throw bad
    val kind = body.head
    val size =
      try body.tail.toInt
      catch case _: NumberFormatException => throw bad
    val dtype: DType[?] = (kind, size) match
      case ('b', 1) => DType.Bool
      case ('i', 1) => DType.Int8
      case ('i', 2) => DType.Int16
      case ('i', 4) => DType.Int32
      case ('i', 8) => DType.Int64
      case ('u', 1) => DType.UInt8
      case ('u', 2) => DType.UInt16
      case ('u', 4) => DType.UInt32
      case ('u', 8) => DType.UInt64
      case ('f', 2) => DType.Float32
      case ('f', 4) => DType.Float32
      case ('f', 8) => DType.Float64
      case ('c', 8) => DType.Complex128
      case ('c', 16) => DType.Complex128
      case ('U', n) if n >= 0 => DType.Str
      case _ => throw new IllegalArgumentException(s"unsupported dtype '$descr' in .npy file")
    Descr(dtype, order, kind, if kind == 'U' then 4 * size else size)

  // ------------------------------------------------------------------ header

  private def shapeRepr(shape: Array[Int]): String =
    if shape.isEmpty then "()"
    else if shape.length == 1 then s"(${shape(0)},)"
    else shape.mkString("(", ", ", ")")

  /** Builds the full header (magic, version, length, dict, padding) exactly like NumPy. */
  def header(descr: String, fortranOrder: Boolean, shape: Array[Int], version: Option[(Int, Int)] = None): Array[Byte] =
    val fo = if fortranOrder then "True" else "False"
    var dict = s"{'descr': '$descr', 'fortran_order': $fo, 'shape': ${shapeRepr(shape)}, }"
    if shape.nonEmpty then
      val growth = shape(if fortranOrder then shape.length - 1 else 0).toString.length
      dict += " " * (GrowthAxisMaxDigits - growth)
    def wrap(major: Int, minor: Int): Option[Array[Byte]] =
      val utf8 = major >= 3
      val bytes =
        if utf8 then dict.getBytes(StandardCharsets.UTF_8)
        else dict.getBytes(StandardCharsets.ISO_8859_1)
      val lenBytes = if major == 1 then 2 else 4
      val hlen = bytes.length + 1
      val padlen = ArrayAlign - ((Magic.length + 2 + lenBytes + hlen) % ArrayAlign)
      val total = hlen + padlen
      if major == 1 && total > 65535 then None
      else
        val out = new ByteArrayOutputStream()
        out.write(Magic)
        out.write(major)
        out.write(minor)
        val bb = ByteBuffer.allocate(lenBytes).order(ByteOrder.LITTLE_ENDIAN)
        if lenBytes == 2 then bb.putShort(total.toShort) else bb.putInt(total)
        out.write(bb.array())
        out.write(bytes)
        out.write(Array.fill[Byte](padlen)(' '.toByte))
        out.write('\n')
        Some(out.toByteArray)
    version match
      case Some((ma, mi)) =>
        if !Seq((1, 0), (2, 0), (3, 0)).contains((ma, mi)) then
          throw new IllegalArgumentException(s"we only support format version (1,0), (2,0), and (3,0), not ($ma, $mi)")
        wrap(ma, mi).getOrElse(throw new IllegalArgumentException(s"header does not fit inside ${65535} bytes required by the 1.0 format"))
      case None =>
        val latin1 = StandardCharsets.ISO_8859_1.newEncoder().canEncode(dict)
        if !latin1 then wrap(3, 0).get
        else wrap(1, 0).orElse(wrap(2, 0)).get

  /** Minimal parser for the Python dict literal of a `.npy` header. */
  private final class DictParser(s: String):
    private var i = 0
    private def ws(): Unit = while i < s.length && s.charAt(i).isWhitespace do i += 1
    private def fail(msg: String) = new IllegalArgumentException(s"Cannot parse header: $msg: ${s.trim}")
    private def expect(c: Char): Unit =
      ws()
      if i >= s.length || s.charAt(i) != c then throw fail(s"expected '$c'")
      i += 1
    private def peek: Char = { ws(); if i < s.length then s.charAt(i) else '\u0000' }

    def value(): Any =
      peek match
        case '\'' | '"' =>
          val q = s.charAt(i); i += 1
          val sb = new StringBuilder
          while i < s.length && s.charAt(i) != q do
            if s.charAt(i) == '\\' && i + 1 < s.length then i += 1
            sb.append(s.charAt(i)); i += 1
          if i >= s.length then throw fail("unterminated string")
          i += 1
          sb.toString
        case '(' | '[' =>
          val close = if s.charAt(i) == '(' then ')' else ']'
          i += 1
          val items = scala.collection.mutable.ArrayBuffer.empty[Any]
          while peek != close do
            items += value()
            if peek == ',' then i += 1
            else if peek != close then throw fail("expected ',' in tuple")
          i += 1
          items.toList
        case '{' =>
          i += 1
          val m = scala.collection.mutable.LinkedHashMap.empty[String, Any]
          while peek != '}' do
            val k = value() match
              case ks: String => ks
              case other => throw fail(s"non-string key $other")
            expect(':')
            m(k) = value()
            if peek == ',' then i += 1
            else if peek != '}' then throw fail("expected ',' in dict")
          i += 1
          m.toMap
        case _ =>
          val st = i
          while i < s.length && !",)]}: \t\n".contains(s.charAt(i)) do i += 1
          s.substring(st, i) match
            case "True" => true
            case "False" => false
            case tok =>
              val t = tok.stripSuffix("L").stripSuffix("l")
              try t.toLong
              catch case _: NumberFormatException => throw fail(s"unexpected token '$tok'")

  final case class Header(descr: Descr, fortranOrder: Boolean, shape: Array[Int])

  private def readFully(in: InputStream, n: Int): Array[Byte] =
    val b = in.readNBytes(n)
    if b.length != n then throw new IllegalArgumentException("EOF: reading array header/data, file is truncated")
    b

  def readHeader(in: InputStream): Header =
    val magic = readFully(in, 8)
    if !magic.take(6).sameElements(Magic) then
      throw new IllegalArgumentException("the magic string is not correct; expected b'\\x93NUMPY'")
    val major = magic(6) & 0xff
    val minor = magic(7) & 0xff
    val lenBytes = (major, minor) match
      case (1, 0) => 2
      case (2, 0) | (3, 0) => 4
      case _ => throw new IllegalArgumentException(s"we only support format version (1,0), (2,0), and (3,0), not ($major, $minor)")
    val lb = ByteBuffer.wrap(readFully(in, lenBytes)).order(ByteOrder.LITTLE_ENDIAN)
    val hlen = if lenBytes == 2 then lb.getShort() & 0xffff else lb.getInt()
    val hb = readFully(in, hlen)
    val text = new String(hb, if major >= 3 then StandardCharsets.UTF_8 else StandardCharsets.ISO_8859_1)
    val d = new DictParser(text).value() match
      case m: Map[?, ?] => m.asInstanceOf[Map[String, Any]]
      case _ => throw new IllegalArgumentException(s"Header is not a dictionary: ${text.trim}")
    if d.keySet != Set("descr", "fortran_order", "shape") then
      throw new IllegalArgumentException(s"Header does not contain the correct keys: ${d.keys.toList.sorted}")
    val descr = d("descr") match
      case s: String => parseDescr(s)
      case _ => throw new IllegalArgumentException("unsupported (structured) descr in .npy header")
    val fo = d("fortran_order") match
      case b: Boolean => b
      case _ => throw new IllegalArgumentException("fortran_order is not a valid bool")
    val shape = d("shape") match
      case l: List[?] => l.map {
          case v: Long if v >= 0 && v <= Int.MaxValue => v.toInt
          case v => throw new IllegalArgumentException(s"shape is not valid: $v")
        }.toArray
      case _ => throw new IllegalArgumentException("shape is not valid")
    Header(descr, fo, shape)

  // ------------------------------------------------------------------ data

  /** Encodes the elements of `a` in C order (little-endian). */
  def encodeData(a: NDArray[?], itemBytes: Int): Array[Byte] =
    val n = a.size
    val bb = ByteBuffer.allocate(n * itemBytes).order(ByteOrder.LITTLE_ENDIAN)
    writeElems(a, bb, itemBytes)
    bb.array()

  def writeElems(a: NDArray[?], bb: ByteBuffer, itemBytes: Int): Unit =
    a.toArray.asInstanceOf[Any] match
      case xs: Array[Boolean] => xs.foreach(b => bb.put(if b then 1.toByte else 0.toByte))
      case xs: Array[Byte] => bb.put(xs)
      case xs: Array[Short] => xs.foreach(bb.putShort)
      case xs: Array[Int] => xs.foreach(bb.putInt)
      case xs: Array[Long] => xs.foreach(bb.putLong)
      case xs: Array[Float] => xs.foreach(bb.putFloat)
      case xs: Array[Double] => xs.foreach(bb.putDouble)
      case xs: Array[Complex] => xs.foreach { c => bb.putDouble(c.re); bb.putDouble(c.im) }
      case xs: Array[String] =>
        val n = itemBytes / 4
        xs.foreach { s =>
          var k = 0
          var i = 0
          while i < s.length do
            val cp = s.codePointAt(i)
            bb.putInt(cp)
            i += Character.charCount(cp)
            k += 1
          while k < n do
            bb.putInt(0)
            k += 1
        }
      case other => throw new IllegalArgumentException(s"cannot serialize $other")

  /** IEEE 754 half precision to float. */
  def halfToFloat(h: Int): Float =
    val sign = (h >>> 15) & 1
    val exp = (h >>> 10) & 0x1f
    val mant = h & 0x3ff
    val v: Float =
      if exp == 0 then (mant.toFloat * math.pow(2, -24).toFloat)
      else if exp == 31 then (if mant == 0 then Float.PositiveInfinity else Float.NaN)
      else java.lang.Float.intBitsToFloat(((exp - 15 + 127) << 23) | (mant << 13))
    if sign == 1 then -v else v

  /** Decodes `count` elements described by `descr` from `bb` (positioned at the data). */
  def decodeData(bb: ByteBuffer, descr: Descr, count: Int): Array[?] =
    bb.order(descr.order)
    if bb.remaining() < count.toLong * descr.size then
      throw new IllegalArgumentException(
        s"EOF: reading array data, expected ${count.toLong * descr.size} bytes got ${bb.remaining()}"
      )
    (descr.kind, descr.size) match
      case ('b', _) => Array.fill(count)(bb.get() != 0)
      case ('i', 1) => Array.fill(count)(bb.get())
      case ('i', 2) => Array.fill(count)(bb.getShort())
      case ('i', 4) => Array.fill(count)(bb.getInt())
      case ('i', 8) => Array.fill(count)(bb.getLong())
      // unsigned types share the bit patterns of the signed buffers (opaque types)
      case ('u', 1) => Array.fill(count)(bb.get())
      case ('u', 2) => Array.fill(count)(bb.getShort())
      case ('u', 4) => Array.fill(count)(bb.getInt())
      case ('u', 8) => Array.fill(count)(bb.getLong())
      case ('f', 2) => Array.fill(count)(halfToFloat(bb.getShort() & 0xffff))
      case ('f', 4) => Array.fill(count)(bb.getFloat())
      case ('f', 8) => Array.fill(count)(bb.getDouble())
      case ('c', 8) => Array.fill(count) { val r = bb.getFloat(); Complex(r.toDouble, bb.getFloat().toDouble) }
      case ('c', 16) => Array.fill(count) { val r = bb.getDouble(); Complex(r, bb.getDouble()) }
      case ('U', sz) =>
        val n = sz / 4
        Array.fill(count) {
          val sb = new java.lang.StringBuilder
          var k = 0
          while k < n do
            val cp = bb.getInt()
            if cp != 0 then sb.appendCodePoint(cp)
            k += 1
          // NumPy strips trailing NULs only; interior NULs are rare and kept out here too
          sb.toString
        }
      case _ => throw new IllegalArgumentException(s"unsupported dtype in .npy data")

  /** Wraps a decoded buffer into an array of the header's shape (handling Fortran order). */
  def toNDArray(data: Array[?], h: Header): NDArray[?] =
    wrap(data, h.descr.dtype, h)

  private def wrap[T](data: Array[?], d: DType[T], h: Header): NDArray[T] =
    val buf = data.asInstanceOf[Array[T]]
    if h.fortranOrder && h.shape.length > 1 then
      NDArray.fromArray(buf, h.shape.reverse)(using d).transpose().copy()
    else NDArray.fromArray(buf, h.shape.clone())(using d)

  /** Serialises an array to `.npy` bytes. */
  def toBytes(a: NDArray[?], version: Option[(Int, Int)] = None): Array[Byte] =
    val (descr, item) = descrOf(a)
    val h = header(descr, false, a.shapeArr, version)
    val out = new ByteArrayOutputStream(h.length + a.size * item)
    out.write(h)
    out.write(encodeData(a, item))
    out.toByteArray

  /** Parses `.npy` bytes. */
  def fromBytes(bytes: Array[Byte]): NDArray[?] = read(new java.io.ByteArrayInputStream(bytes))

  def read(in: InputStream): NDArray[?] =
    val h = readHeader(in)
    val count = Shape.size(h.shape)
    val need = count.toLong * h.descr.size
    if need > Int.MaxValue then throw new IllegalArgumentException("array too large to load")
    val bytes = in.readNBytes(need.toInt)
    toNDArray(decodeData(ByteBuffer.wrap(bytes), h.descr, count), h)
