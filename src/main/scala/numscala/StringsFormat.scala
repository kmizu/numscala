package numscala

import java.math.{BigDecimal as JBigDecimal, MathContext, RoundingMode}

/** Python printf-style (`%`) formatting used by [[Strings.mod]].
  *
  * Values are NumPy scalars taken from arrays, so `%s` prints them like `str()` and
  * `%r` like NumPy 2's `repr()` (`np.float64(1.5)`, `np.str_('a')`, `np.True_`).
  */
private[numscala] object StringsFormat:

  /** One formatting argument: an element together with its dtype. */
  final case class Arg(dtype: DType[Any], value: Any):
    def typeName: String = if dtype.isBool then "numpy.bool" else if dtype.isString then "numpy.str_" else s"numpy.${dtype.name}"
    def str: String = dtype.format(value)
    def repr: String =
      if dtype.isBool then (if value.asInstanceOf[Boolean] then "np.True_" else "np.False_")
      else if dtype.isString then s"np.str_(${StringsPy.repr(value.asInstanceOf[String])})"
      else if dtype.isComplex then
        val s = str
        s"np.${dtype.name}(${if s.startsWith("(") && s.endsWith(")") then s.substring(1, s.length - 1) else s})"
      else s"np.${dtype.name}($str)"

  private def typeError(msg: String) = new IllegalArgumentException(msg)

  /** Formats `fmt % args` (`isTuple` = the arguments came as a Python tuple). */
  def format(fmt: String, args: IndexedSeq[Arg], isTuple: Boolean): String =
    val sb = new java.lang.StringBuilder
    var argi = 0
    def next(): Arg =
      if argi >= args.length then throw typeError("not enough arguments for format string")
      argi += 1
      args(argi - 1)
    val n = fmt.length
    var i = 0
    while i < n do
      val c = fmt.charAt(i)
      if c != '%' then
        sb.append(c)
        i += 1
      else
        i += 1
        if i >= n then throw new IllegalArgumentException("incomplete format")
        if fmt.charAt(i) == '(' then
          throw new IllegalArgumentException("format requires a mapping (mapping keys are not supported)")
        var left = false
        var plus = false
        var space = false
        var alt = false
        var zero = false
        var flagsDone = false
        while !flagsDone && i < n do
          fmt.charAt(i) match
            case '-' => left = true; i += 1
            case '+' => plus = true; i += 1
            case ' ' => space = true; i += 1
            case '#' => alt = true; i += 1
            case '0' => zero = true; i += 1
            case _ => flagsDone = true
        var width = -1
        if i < n && fmt.charAt(i) == '*' then
          val w = next()
          if !w.dtype.isInteger then throw typeError("* wants int")
          val wv = w.dtype.toLong(w.value).toInt
          if wv < 0 then
            left = true
            width = -wv
          else width = wv
          i += 1
        else
          val b = i
          while i < n && fmt.charAt(i).isDigit do i += 1
          if i > b then width = fmt.substring(b, i).toInt
        var prec = -1
        if i < n && fmt.charAt(i) == '.' then
          i += 1
          if i < n && fmt.charAt(i) == '*' then
            val p = next()
            if !p.dtype.isInteger then throw typeError("* wants int")
            prec = math.max(0, p.dtype.toLong(p.value).toInt)
            i += 1
          else
            val b = i
            while i < n && fmt.charAt(i).isDigit do i += 1
            prec = if i > b then fmt.substring(b, i).toInt else 0
        while i < n && (fmt.charAt(i) == 'h' || fmt.charAt(i) == 'l' || fmt.charAt(i) == 'L') do i += 1
        if i >= n then throw new IllegalArgumentException("incomplete format")
        val conv = fmt.charAt(i)
        i += 1
        val spec = Spec(left, plus, space, alt, zero && !left, width, prec, conv)
        conv match
          case '%' => sb.append('%')
          case 's' => sb.append(padText(truncate(next().str, prec), spec))
          case 'r' => sb.append(padText(truncate(next().repr, prec), spec))
          case 'a' => sb.append(padText(truncate(ascii(next().repr), prec), spec))
          case 'c' => sb.append(padText(charOf(next()), spec))
          case 'd' | 'i' | 'u' => sb.append(formatInt(intOf(next(), conv, allowFloat = true), spec))
          case 'o' | 'x' | 'X' => sb.append(formatInt(intOf(next(), conv, allowFloat = false), spec))
          case 'e' | 'E' | 'f' | 'F' | 'g' | 'G' => sb.append(formatFloat(floatOf(next()), spec))
          case other =>
            throw new IllegalArgumentException(
              f"unsupported format character '$other' (0x${other.toInt}%x) at index ${i - 1}"
            )
    if isTuple && argi < args.length then throw typeError("not all arguments converted during string formatting")
    sb.toString

  private final case class Spec(
      left: Boolean,
      plus: Boolean,
      space: Boolean,
      alt: Boolean,
      zero: Boolean,
      width: Int,
      prec: Int,
      conv: Char
  )

  private def truncate(s: String, prec: Int): String =
    if prec < 0 || StringsPy.len(s) <= prec then s else s.substring(0, StringsPy.cpToIdx(s, prec))

  private def padText(s: String, sp: Spec): String =
    val l = StringsPy.len(s)
    if sp.width <= l then s
    else if sp.left then s + " " * (sp.width - l)
    else " " * (sp.width - l) + s

  private def ascii(s: String): String =
    val sb = new java.lang.StringBuilder
    s.codePoints().forEach { c =>
      if c < 0x80 then sb.appendCodePoint(c)
      else if c <= 0xff then sb.append(f"\\x$c%02x")
      else if c <= 0xffff then sb.append(f"\\u$c%04x")
      else sb.append(f"\\U$c%08x")
      ()
    }
    sb.toString

  private def charOf(a: Arg): String =
    if a.dtype.isString then
      val s = a.value.asInstanceOf[String]
      if StringsPy.len(s) != 1 then
        throw typeError(s"%c requires an int or a unicode character, not a string of length ${StringsPy.len(s)}")
      s
    else if a.dtype.isInteger || a.dtype.isBool then
      val v = a.dtype.toLong(a.value)
      if v < 0 || v > 0x10ffff then throw new IllegalArgumentException("%c arg not in range(0x110000)")
      new String(Character.toChars(v.toInt))
    else throw typeError(s"%c requires an int or a unicode character, not ${a.typeName}")

  private def intOf(a: Arg, conv: Char, allowFloat: Boolean): BigInt =
    val d = a.dtype
    if d.isInteger then BigInt(d.toLong(a.value))
    else if d.isBool && allowFloat then BigInt(d.toLong(a.value))
    else if (d.isFloating || d.isComplex) && allowFloat then
      val x = d.toDouble(a.value)
      if x.isNaN then throw new IllegalArgumentException("cannot convert float NaN to integer")
      if x.isInfinite then throw new ArithmeticException("cannot convert float infinity to integer")
      BigInt(new JBigDecimal(x).toBigInteger)
    else if allowFloat then throw typeError(s"%$conv format: a real number is required, not ${a.typeName}")
    else throw typeError(s"%$conv format: an integer is required, not ${a.typeName}")

  private def floatOf(a: Arg): Double =
    if a.dtype.isNumeric || a.dtype.isBool then a.dtype.toDouble(a.value)
    else throw typeError(s"must be real number, not ${a.typeName}")

  private def signOf(neg: Boolean, sp: Spec): String =
    if neg then "-" else if sp.plus then "+" else if sp.space then " " else ""

  /** Assembles sign, prefix and body with width/zero padding. */
  private def assemble(sign: String, prefix: String, body: String, sp: Spec): String =
    val l = sign.length + prefix.length + body.length
    if sp.width <= l then sign + prefix + body
    else if sp.left then sign + prefix + body + " " * (sp.width - l)
    else if sp.zero then sign + prefix + "0" * (sp.width - l) + body
    else " " * (sp.width - l) + sign + prefix + body

  private def formatInt(v: BigInt, sp: Spec): String =
    val radix = sp.conv match
      case 'o' => 8
      case 'x' | 'X' => 16
      case _ => 10
    var digits = v.abs.toString(radix)
    if sp.conv == 'X' then digits = digits.toUpperCase
    if sp.prec > digits.length then digits = "0" * (sp.prec - digits.length) + digits
    val prefix =
      if !sp.alt then ""
      else
        sp.conv match
          case 'o' => "0o"
          case 'x' => "0x"
          case 'X' => "0X"
          case _ => ""
    assemble(signOf(v < 0, sp), prefix, digits, sp)

  /** Digits and decimal exponent of `|x|` rounded to `sig` significant digits (round-half-even). */
  private def sigDigits(x: Double, sig: Int): (String, Int) =
    if x == 0.0 then ("0" * sig, 0)
    else
      val bd = new JBigDecimal(math.abs(x)).round(new MathContext(sig, RoundingMode.HALF_EVEN))
      val u = bd.unscaledValue().toString
      val exp = u.length - 1 - bd.scale()
      val stripped = if u.length > sig then u.substring(0, sig) else u
      (stripped + "0" * (sig - stripped.length), exp)

  private def fixed(x: Double, prec: Int, alt: Boolean): String =
    val s = new JBigDecimal(math.abs(x)).setScale(prec, RoundingMode.HALF_EVEN).toPlainString
    if prec == 0 && alt then s + "." else s

  private def expo(x: Double, prec: Int, alt: Boolean): String =
    val (d, e) = sigDigits(x, prec + 1)
    val mant = d.substring(0, 1) + (if prec > 0 || alt then "." + d.substring(1) else "")
    val ae = math.abs(e)
    mant + "e" + (if e < 0 then "-" else "+") + (if ae < 10 then "0" + ae else ae.toString)

  private def stripZeros(s: String): String =
    if !s.contains('.') then s
    else
      var t = s
      while t.endsWith("0") do t = t.substring(0, t.length - 1)
      if t.endsWith(".") then t.substring(0, t.length - 1) else t

  private def formatFloat(x: Double, sp: Spec): String =
    val neg = x < 0 || (x == 0.0 && 1.0 / x < 0)
    val lower = sp.conv.toLower
    val body =
      if x.isNaN then "nan"
      else if x.isInfinite then "inf"
      else
        val p = if sp.prec < 0 then 6 else sp.prec
        lower match
          case 'f' => fixed(x, p, sp.alt)
          case 'e' => expo(x, p, sp.alt)
          case _ =>
            val pg = if p == 0 then 1 else p
            val (_, e) = sigDigits(x, pg)
            if e >= -4 && e < pg then
              val s = fixed(x, pg - 1 - e, sp.alt)
              if sp.alt then s else stripZeros(s)
            else
              val s = expo(x, pg - 1, sp.alt)
              if sp.alt then s
              else
                val k = s.indexOf('e')
                stripZeros(s.substring(0, k)) + s.substring(k)
    val out = if sp.conv.isUpper then body.toUpperCase else body
    assemble(signOf(neg && !x.isNaN, sp), "", out, sp)
