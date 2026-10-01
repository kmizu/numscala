package numscala

import java.math.{BigDecimal as JBigDecimal, MathContext, RoundingMode}

/** Dragon4-style scalar float formatting (`np.format_float_positional` / `_scientific`). */
private[numscala] object NpMiscFloatFmt:
  private def exactOf(x: Double): JBigDecimal = new JBigDecimal(math.abs(x))
  /** Shortest round-tripping decimal of |x|. Java's `toString` never emits a single digit
    * (`4.9E-324` instead of `5E-324`), so a one-digit candidate is tried as well.
    */
  def shortestOf(x: Double, single: Boolean): JBigDecimal =
    val ax = math.abs(x)
    val s = new JBigDecimal(Format.shortestDecimal(if single then ax.toFloat.toDouble else ax, single))
      .stripTrailingZeros()
    if s.precision() == 2 then
      val one = s.round(new MathContext(1, RoundingMode.HALF_EVEN))
      val back = if single then one.floatValue() == ax.toFloat else one.doubleValue() == ax
      if back then one.stripTrailingZeros() else s
    else s

  /** Python/NumPy scalar `repr` of a float (float32 digits when `single`). */
  def pyRepr(x: Double, single: Boolean): String =
    if x.isNaN then "nan"
    else if x.isInfinite then (if x > 0 then "inf" else "-inf")
    else if x == 0.0 then (if 1.0 / x < 0 then "-0.0" else "0.0")
    else
      val v = shortestOf(x, single)
      val digits = v.unscaledValue().toString
      val exp = digits.length - 1 - v.scale()
      val sign = if x < 0 then "-" else ""
      if exp < -4 || exp >= 16 then
        val mant = if digits.length == 1 then digits else digits.head.toString + "." + digits.tail
        val ae = math.abs(exp).toString
        sign + mant + "e" + (if exp < 0 then "-" else "+") + (if ae.length < 2 then "0" + ae else ae)
      else if exp < 0 then sign + "0." + "0" * (-exp - 1) + digits
      else if digits.length <= exp + 1 then sign + digits + "0" * (exp + 1 - digits.length) + ".0"
      else sign + digits.substring(0, exp + 1) + "." + digits.substring(exp + 1)

  private def nonFinite(x: Double, sign: Boolean): Option[String] =
    if x.isNaN then Some(if sign then "+nan" else "nan")
    else if x.isInfinite then Some(if x < 0 then "-inf" else if sign then "+inf" else "inf")
    else None

  private def checkTrim(trim: String): Unit =
    if !Set("k", ".", "0", "-").contains(trim) then
      throw new IllegalArgumentException("trim must be one of 'k', '.', '0' or '-'")

  /** Applies NumPy's trim mode to a fraction-digit string; returns (fraction, keepDot). */
  private def trimFrac(frac: String, trim: String, trailingZerosAllowed: Boolean): (String, Boolean) =
    trim match
      case "k" => (frac, true)
      case "." => (if trailingZerosAllowed then frac.reverse.dropWhile(_ == '0').reverse else frac, true)
      case "0" =>
        val f = frac.reverse.dropWhile(_ == '0').reverse
        (if f.isEmpty then "0" else f, true)
      case _ =>
        val f = frac.reverse.dropWhile(_ == '0').reverse
        (f, f.nonEmpty)

  private def padL(s: String, w: Int): String = if w > s.length then " " * (w - s.length) + s else s
  private def padR(s: String, w: Int): String = if w > s.length then s + " " * (w - s.length) else s

  def positional(
      x: Double,
      single: Boolean,
      precision: Int,
      unique: Boolean,
      fractional: Boolean,
      trim: String,
      sign: Boolean,
      padLeft: Int,
      padRight: Int,
      minDigits: Int
  ): String =
    checkTrim(trim)
    if !unique && precision < 0 then
      throw new IllegalArgumentException("precision must be specified when unique is False")
    nonFinite(x, sign) match
      case Some(s) => s
      case None =>
        val neg = x < 0 || (x == 0.0 && 1.0 / x < 0)
        val exact = exactOf(x)
        var v =
          if unique then
            val sh = shortestOf(x, single).stripTrailingZeros()
            if precision < 0 then sh
            else if fractional then (if sh.scale() > precision then exact.setScale(precision, RoundingMode.HALF_EVEN).stripTrailingZeros() else sh)
            else if sh.precision() > precision && precision > 0 then
              exact.round(new MathContext(precision, RoundingMode.HALF_EVEN)).stripTrailingZeros()
            else sh
          else if fractional then exact.setScale(precision, RoundingMode.HALF_EVEN)
          else
            val r = exact.round(new MathContext(math.max(precision, 1), RoundingMode.HALF_EVEN))
            if r.precision() < precision then r.setScale(r.scale() + precision - r.precision()) else r
        if v.scale() < 0 then v = v.setScale(0)
        val plain = v.toPlainString
        val dot = plain.indexOf('.')
        val (ip, fp0) = if dot < 0 then (plain, "") else (plain.substring(0, dot), plain.substring(dot + 1))
        val fp1 = if minDigits > fp0.length then fp0 + "0" * (minDigits - fp0.length) else fp0
        val (fp, keepDot) = trimFrac(fp1, trim, !unique)
        val intPart = (if neg then "-" else if sign then "+" else "") + ip
        val fracPart = if keepDot then "." + fp else ""
        val left = padL(intPart, padLeft)
        if padRight > 0 then left + (if keepDot then "." + padR(fp, padRight) else padR("", padRight + 1))
        else left + fracPart

  def scientific(
      x: Double,
      single: Boolean,
      precision: Int,
      unique: Boolean,
      trim: String,
      sign: Boolean,
      padLeft: Int,
      expDigits: Int,
      minDigits: Int
  ): String =
    checkTrim(trim)
    if !unique && precision < 0 then
      throw new IllegalArgumentException("precision must be specified when unique is False")
    nonFinite(x, sign) match
      case Some(s) => s
      case None =>
        val neg = x < 0 || (x == 0.0 && 1.0 / x < 0)
        val (digits, exp) =
          if x == 0.0 then ((if unique then "0" else "0" * (precision + 1)), 0)
          else
            val v =
              if unique then
                val sh = shortestOf(x, single).stripTrailingZeros()
                if precision >= 0 && sh.precision() > precision + 1 then
                  exactOf(x).round(new MathContext(precision + 1, RoundingMode.HALF_EVEN)).stripTrailingZeros()
                else sh
              else exactOf(x).round(new MathContext(precision + 1, RoundingMode.HALF_EVEN))
            val u0 = v.unscaledValue().toString
            val e = u0.length - 1 - v.scale()
            val u = if !unique && u0.length < precision + 1 then u0 + "0" * (precision + 1 - u0.length) else u0
            (u, e)
        val frac0 = digits.tail
        val frac1 = if minDigits > frac0.length then frac0 + "0" * (minDigits - frac0.length) else frac0
        val (frac, keepDot) = trimFrac(frac1, trim, !unique)
        val ae = math.abs(exp).toString
        val ed = math.max(expDigits, 2)
        val expStr = "e" + (if exp < 0 then "-" else "+") + ("0" * math.max(0, ed - ae.length)) + ae
        val intPart = (if neg then "-" else if sign then "+" else "") + digits.head
        padL(intPart, padLeft) + (if keepDot then "." + frac else "") + expStr

/** Printing: print options, `array2string`, `array_repr`, scalar float formatting, `base_repr`, `binary_repr`. */
trait NpMiscPrint:
  private def merged(
      precision: Int | Null,
      threshold: Int | Null,
      edgeitems: Int | Null,
      linewidth: Int | Null,
      suppress: Boolean | Null,
      nanstr: String | Null,
      infstr: String | Null
  ): PrintOptions =
    var o = Format.printOptions
    precision match
      case p: Int =>
        if p < 0 then throw new IllegalArgumentException("precision must be >= 0")
        o = o.copy(precision = p)
      case _ =>
    threshold match
      case t: Int => o = o.copy(threshold = t)
      case _ =>
    edgeitems match
      case t: Int => o = o.copy(edgeitems = t)
      case _ =>
    linewidth match
      case t: Int => o = o.copy(linewidth = t)
      case _ =>
    suppress match
      case b: Boolean => o = o.copy(suppress = b)
      case _ =>
    if nanstr != null then o = o.copy(nanstr = nanstr.asInstanceOf[String])
    if infstr != null then o = o.copy(infstr = infstr.asInstanceOf[String])
    o

  /** `np.set_printoptions`: changes only the options that are given. */
  def set_printoptions(
      precision: Int | Null = null,
      threshold: Int | Null = null,
      edgeitems: Int | Null = null,
      linewidth: Int | Null = null,
      suppress: Boolean | Null = null,
      nanstr: String | Null = null,
      infstr: String | Null = null
  ): Unit = Format.setPrintOptions(merged(precision, threshold, edgeitems, linewidth, suppress, nanstr, infstr))

  /** `np.get_printoptions`: the current print options. */
  def get_printoptions(): PrintOptions = Format.printOptions

  /** `with np.printoptions(...)`: runs `body` with temporarily changed print options. */
  def printoptions[A](
      precision: Int | Null = null,
      threshold: Int | Null = null,
      edgeitems: Int | Null = null,
      linewidth: Int | Null = null,
      suppress: Boolean | Null = null,
      nanstr: String | Null = null,
      infstr: String | Null = null
  )(body: => A): A =
    Format.withPrintOptions(merged(precision, threshold, edgeitems, linewidth, suppress, nanstr, infstr))(body)

  /** `np.array2string`: the array body as `str()` prints it, with optional overrides. */
  def array2string[T](
      a: NDArray[T],
      max_line_width: Int | Null = null,
      precision: Int | Null = null,
      suppress_small: Boolean | Null = null,
      separator: String = " ",
      prefix: String = "",
      suffix: String = "",
      threshold: Int | Null = null,
      edgeitems: Int | Null = null
  ): String =
    Format.withPrintOptions(merged(precision, threshold, edgeitems, max_line_width, suppress_small, null, null)) {
      if a.size == 0 then "[]" else Format.formatArray(a, separator, prefix, suffix.length)
    }

  /** `np.array_repr`: like `repr(a)`, with optional overrides. */
  def array_repr[T](
      arr: NDArray[T],
      max_line_width: Int | Null = null,
      precision: Int | Null = null,
      suppress_small: Boolean | Null = null
  ): String =
    Format.withPrintOptions(merged(precision, null, null, max_line_width, suppress_small, null, null))(Format.repr(arr))

  /** `np.array_str`: like `str(a)`, with optional overrides. */
  def array_str[T](
      a: NDArray[T],
      max_line_width: Int | Null = null,
      precision: Int | Null = null,
      suppress_small: Boolean | Null = null
  ): String =
    Format.withPrintOptions(merged(precision, null, null, max_line_width, suppress_small, null, null))(Format.str(a))

  private def floatArg(x: Double | Float): (Double, Boolean) = x match
    case f: Float => (f.toDouble, true)
    case d: Double => (d, false)

  /** `np.format_float_positional`: positional notation (Dragon4). `precision = -1` means unlimited. */
  def format_float_positional(
      x: Double | Float,
      precision: Int = -1,
      unique: Boolean = true,
      fractional: Boolean = true,
      trim: String = "k",
      sign: Boolean = false,
      pad_left: Int = -1,
      pad_right: Int = -1,
      min_digits: Int = -1
  ): String =
    val (d, single) = floatArg(x)
    NpMiscFloatFmt.positional(d, single, precision, unique, fractional, trim, sign, pad_left, pad_right, min_digits)

  /** `np.format_float_scientific`: scientific notation (Dragon4). `precision = -1` means unlimited. */
  def format_float_scientific(
      x: Double | Float,
      precision: Int = -1,
      unique: Boolean = true,
      trim: String = "k",
      sign: Boolean = false,
      pad_left: Int = -1,
      exp_digits: Int = -1,
      min_digits: Int = -1
  ): String =
    val (d, single) = floatArg(x)
    NpMiscFloatFmt.scientific(d, single, precision, unique, trim, sign, pad_left, exp_digits, min_digits)

  /** `np.base_repr`: string representation of an integer in `base` (2..36). */
  def base_repr(number: Long, base: Int = 2, padding: Int = 0): String =
    if base > 36 then throw new IllegalArgumentException("Bases greater than 36 not handled in base_repr.")
    if base < 2 then throw new IllegalArgumentException("Bases less than 2 not handled in base_repr.")
    val digits = BigInt(number).abs.toString(base).toUpperCase
    val body = if number == 0 then "" else digits
    val padded = "0" * math.max(padding, 0) + body
    val res = if padded.isEmpty then "0" else padded
    if number < 0 then "-" + res else res

  /** `np.binary_repr`: binary string; with `width`, negative numbers use two's complement. */
  def binary_repr(num: Long, width: Int = -1): String =
    def check(binwidth: Int): Unit =
      if width >= 0 && width < binwidth then
        throw new IllegalArgumentException(s"Insufficient bit width=$width provided for binwidth=$binwidth")
    if num == 0 then "0" * (if width > 0 then width else 1)
    else if num > 0 then
      val b = java.lang.Long.toBinaryString(num)
      check(b.length)
      "0" * math.max(0, width - b.length) + b
    else if width < 0 then "-" + BigInt(num).abs.toString(2)
    else
      val mag = BigInt(num).abs
      var poswidth = mag.toString(2).length
      if BigInt(2).pow(poswidth - 1) == mag then poswidth -= 1
      val b = (BigInt(2).pow(poswidth + 1) + BigInt(num)).toString(2)
      check(b.length)
      "1" * math.max(0, width - b.length) + b
