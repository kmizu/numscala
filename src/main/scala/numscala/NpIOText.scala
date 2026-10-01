package numscala

import java.math.{BigDecimal as JBigDecimal, BigInteger, MathContext, RoundingMode}

/** Python `%`-style formatting (`'%.18e' % x`) with CPython's exact, correctly rounded output. */
private[numscala] object NpIOPyFormat:
  private val Spec = "%([-+ #0]*)(\\d+)?(?:\\.(\\d+))?([diouxXeEfFgGsrc%])".r

  /** Number of conversion specifiers (excluding `%%`) in `fmt`. */
  def countSpecs(fmt: String): Int = Spec.findAllMatchIn(fmt).count(_.group(4) != "%")

  /** Formats `args` with the Python format string `fmt`. */
  def format(fmt: String, args: Seq[Any]): String =
    val sb = new StringBuilder
    var last = 0
    var k = 0
    for m <- Spec.findAllMatchIn(fmt) do
      sb.append(fmt.substring(last, m.start))
      last = m.end
      val conv = m.group(4).head
      if conv == '%' then sb.append('%')
      else
        if k >= args.length then throw new IllegalArgumentException("not enough arguments for format string")
        val flags = m.group(1)
        val width = Option(m.group(2)).map(_.toInt).getOrElse(0)
        val prec = Option(m.group(3)).map(_.toInt)
        sb.append(one(args(k), flags, width, prec, conv))
        k += 1
    sb.append(fmt.substring(last))
    if k < args.length then throw new IllegalArgumentException("not all arguments converted during string formatting")
    sb.toString

  /** Python `str()` of a scalar value. */
  def pyStr(v: Any): String = v match
    case b: Boolean => if b then "True" else "False"
    case d: Double => Format.formatFloatShort(d)
    case f: Float => Format.formatFloat32Short(f)
    case c: Complex => Complex.format(c)
    case other => other.toString

  private def toDouble(v: Any): Double = v match
    case b: Boolean => if b then 1.0 else 0.0
    case b: Byte => b.toDouble
    case s: Short => s.toDouble
    case i: Int => i.toDouble
    case l: Long => l.toDouble
    case f: Float => f.toDouble
    case d: Double => d
    case s: String => throw new IllegalArgumentException(s"must be real number, not str ('$s')")
    case c: Complex => throw new IllegalArgumentException(s"must be real number, not complex ($c)")
    case other => throw new IllegalArgumentException(s"must be real number, not $other")

  private def toBigInt(v: Any): BigInteger = v match
    case b: Boolean => if b then BigInteger.ONE else BigInteger.ZERO
    case b: Byte => BigInteger.valueOf(b.toLong)
    case s: Short => BigInteger.valueOf(s.toLong)
    case i: Int => BigInteger.valueOf(i.toLong)
    case l: Long => BigInteger.valueOf(l)
    case other =>
      val d = toDouble(other)
      if d.isNaN then throw new IllegalArgumentException("cannot convert float NaN to integer")
      if d.isInfinite then throw new IllegalArgumentException("cannot convert float infinity to integer")
      new JBigDecimal(d).toBigInteger

  private def pad(sign: String, body: String, flags: String, width: Int, zeroOk: Boolean): String =
    val len = sign.length + body.length
    if len >= width then sign + body
    else if flags.contains('-') then sign + body + " " * (width - len)
    else if flags.contains('0') && zeroOk then sign + "0" * (width - len) + body
    else " " * (width - len) + sign + body

  private def signOf(neg: Boolean, flags: String): String =
    if neg then "-" else if flags.contains('+') then "+" else if flags.contains(' ') then " " else ""

  private def one(v: Any, flags: String, width: Int, prec: Option[Int], conv: Char): String = conv match
    case 's' | 'r' =>
      val s0 = pyStr(v)
      val s = prec.fold(s0)(p => s0.take(p))
      pad("", s, flags.filter(_ == '-'), width, zeroOk = false)
    case 'c' =>
      val s = v match
        case s: String => s
        case other => new String(Character.toChars(toBigInt(other).intValue))
      pad("", s, flags.filter(_ == '-'), width, zeroOk = false)
    case 'd' | 'i' | 'u' | 'o' | 'x' | 'X' =>
      val n = toBigInt(v)
      val a = n.abs
      var body = conv match
        case 'o' => (if flags.contains('#') then "0o" else "") + a.toString(8)
        case 'x' => (if flags.contains('#') then "0x" else "") + a.toString(16)
        case 'X' => (if flags.contains('#') then "0X" else "") + a.toString(16).toUpperCase
        case _ => a.toString
      prec.foreach(p => if body.length < p then body = "0" * (p - body.length) + body)
      pad(signOf(n.signum < 0, flags), body, flags, width, zeroOk = true)
    case _ =>
      val d = toDouble(v)
      val neg = d < 0 || (d == 0.0 && 1.0 / d < 0)
      val upper = conv.isUpper
      val body =
        if d.isNaN then "nan"
        else if d.isInfinite then "inf"
        else
          val p = prec.getOrElse(6)
          conv.toLower match
            case 'e' => fmtE(math.abs(d), p, flags.contains('#'))
            case 'f' => fmtF(math.abs(d), p, flags.contains('#'))
            case _ => fmtG(math.abs(d), p, flags.contains('#'))
      pad(signOf(neg, flags), if upper then body.toUpperCase else body, flags, width, zeroOk = true)

  /** Exact `%.{p}e` of a non-negative finite double. */
  def fmtE(d: Double, p: Int, alt: Boolean): String =
    val (digits, exp) =
      if d == 0.0 then ("0" * (p + 1), 0)
      else
        val bd = new JBigDecimal(d).round(new MathContext(p + 1, RoundingMode.HALF_EVEN))
        val u = bd.unscaledValue().toString
        val e = u.length - 1 - bd.scale()
        (if u.length < p + 1 then u + "0" * (p + 1 - u.length) else u.substring(0, p + 1), e)
    val mant = if p == 0 then digits.take(1) + (if alt then "." else "") else digits.head.toString + "." + digits.tail
    val ae = math.abs(exp).toString
    mant + "e" + (if exp < 0 then "-" else "+") + (if ae.length < 2 then "0" + ae else ae)

  /** Exact `%.{p}f` of a non-negative finite double. */
  def fmtF(d: Double, p: Int, alt: Boolean): String =
    val s = new JBigDecimal(d).setScale(p, RoundingMode.HALF_EVEN).toPlainString
    if p == 0 && alt then s + "." else s

  /** `%.{p}g` of a non-negative finite double. */
  def fmtG(d: Double, p0: Int, alt: Boolean): String =
    val p = if p0 == 0 then 1 else p0
    val e = fmtE(d, p - 1, false)
    val x = e.substring(e.indexOf('e') + 1).toInt
    val s =
      if x >= -4 && x < p then fmtF(d, p - 1 - x, alt)
      else fmtE(d, p - 1, alt)
    if alt then s
    else
      val k = s.indexOf('e')
      val (m, ex) = if k >= 0 then (s.substring(0, k), s.substring(k)) else (s, "")
      val mt = if m.contains('.') then m.reverse.dropWhile(_ == '0').reverse.stripSuffix(".") else m
      mt + ex

/** Line splitting and value parsing shared by `loadtxt` / `genfromtxt` / `fromfile`. */
private[numscala] object NpIOText:
  /** Removes a trailing comment (earliest occurrence of any comment marker outside quotes). */
  def stripComment(line: String, comments: Seq[String], quote: Option[Char]): String =
    if comments.isEmpty then line
    else
      var inQ = false
      var i = 0
      var cut = -1
      while cut < 0 && i < line.length do
        val c = line.charAt(i)
        if quote.contains(c) then inQ = !inQ
        else if !inQ && comments.exists(m => m.nonEmpty && line.startsWith(m, i)) then cut = i
        i += 1
      if cut < 0 then line else line.substring(0, cut)

  /** Splits a line into fields; `delimiter == null` means runs of whitespace. */
  def split(line: String, delimiter: String | Null, quote: Option[Char]): Seq[String] =
    val delim = delimiter
    if delim == null then
      if quote.isEmpty then line.trim.split("\\s+").toSeq.filter(_.nonEmpty)
      else splitQuoted(line.trim, None, quote.get).filter(_.nonEmpty)
    else if quote.isEmpty then
      line.split(java.util.regex.Pattern.quote(delim), -1).toSeq
    else splitQuoted(line, Some(delim), quote.get)

  private def splitQuoted(line: String, delim: Option[String], q: Char): Seq[String] =
    val out = scala.collection.mutable.ArrayBuffer.empty[String]
    val cur = new StringBuilder
    var i = 0
    var inQ = false
    while i < line.length do
      val c = line.charAt(i)
      if inQ then
        if c == q then
          if i + 1 < line.length && line.charAt(i + 1) == q then
            cur.append(q); i += 1
          else inQ = false
        else cur.append(c)
        i += 1
      else if c == q then
        inQ = true
        i += 1
      else
        delim match
          case Some(d) if line.startsWith(d, i) =>
            out += cur.toString; cur.clear(); i += d.length
          case None if c.isWhitespace =>
            out += cur.toString; cur.clear(); i += 1
          case _ =>
            cur.append(c); i += 1
    out += cur.toString
    out.toSeq

  /** Splits text into lines (universal newlines); a final empty line after the last newline is dropped. */
  def lines(text: String): Seq[String] =
    val ls = text.split("\r\n|\r|\n", -1).toSeq
    if ls.nonEmpty && ls.last.isEmpty then ls.init else ls

  /** Parses one field for `dtype`, throwing NumberFormatException-like errors on failure. */
  def parse[T](d: DType[T], s: String): T =
    if d.isString then d.fromString(s)
    else
      val t = s.trim
      if t.isEmpty then throw new IllegalArgumentException("empty field")
      d.kind match
        case 'i' =>
          try d.fromLong(t.stripPrefix("+").toLong)
          catch
            case _: NumberFormatException =>
              throw new IllegalArgumentException(s"invalid int '$t'")
        case 'f' =>
          if !t.matches("(?i)[+-]?(nan|inf|infinity|(\\d+\\.?\\d*|\\.\\d+)(e[+-]?\\d+)?)") then
            throw new IllegalArgumentException(s"invalid float '$t'")
          d.fromDouble(Format.parseDouble(t))
        case 'c' =>
          if !t.matches("(?i)\\(?[0-9eE.+\\-naifjty]+\\)?") then throw new IllegalArgumentException(s"invalid complex '$t'")
          d.fromComplex(Format.parseComplex(t))
        case _ => d.fromString(t)
