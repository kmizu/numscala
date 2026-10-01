package numscala

import java.math.{BigDecimal as JBigDecimal, RoundingMode}

/** Global print options, mirroring `numpy.set_printoptions`. */
final case class PrintOptions(
    precision: Int = 8,
    threshold: Int = 1000,
    edgeitems: Int = 3,
    linewidth: Int = 75,
    suppress: Boolean = false,
    nanstr: String = "nan",
    infstr: String = "inf"
)

/** Number and array formatting that reproduces NumPy's `str`/`repr` output. */
object Format:
  @volatile private var globalOpts: PrintOptions = PrintOptions()
  private val localOpts = new scala.util.DynamicVariable[Option[PrintOptions]](None)
  private def opts: PrintOptions = localOpts.value.getOrElse(globalOpts)

  def printOptions: PrintOptions = opts
  /** Sets the process-wide print options (`np.set_printoptions`). */
  def setPrintOptions(o: PrintOptions): Unit = globalOpts = o

  /** Runs `body` with temporary print options (`np.printoptions` context manager).
    * The override is thread-local (inherited by threads started inside `body`).
    */
  def withPrintOptions[A](o: PrintOptions)(body: => A): A =
    localOpts.withValue(Some(o))(body)

  /** Total order on doubles with NaN sorted last (NumPy's sort order). */
  def compareDouble(a: Double, b: Double): Int =
    if a < b then -1
    else if a > b then 1
    else if a == b then 0
    else if a.isNaN then (if b.isNaN then 0 else 1)
    else -1

  def parseDouble(s: String): Double = s.trim.toLowerCase match
    case "nan" | "+nan" | "-nan" => Double.NaN
    case "inf" | "+inf" | "infinity" | "+infinity" => Double.PositiveInfinity
    case "-inf" | "-infinity" => Double.NegativeInfinity
    case t => t.toDouble

  def parseComplex(s: String): Complex =
    val t = s.trim.stripPrefix("(").stripSuffix(")").replace(" ", "")
    if !t.endsWith("j") && !t.endsWith("J") then Complex(parseDouble(t), 0.0)
    else
      val body = t.dropRight(1)
      // find the split between real and imaginary parts: last +/- not following 'e'
      var i = body.length - 1
      var split = -1
      while i > 0 && split < 0 do
        val c = body.charAt(i)
        if (c == '+' || c == '-') && body.charAt(i - 1) != 'e' && body.charAt(i - 1) != 'E' then split = i
        i -= 1
      def im(x: String): Double = if x == "" || x == "+" then 1.0 else if x == "-" then -1.0 else parseDouble(x)
      if split < 0 then Complex(0.0, im(body))
      else Complex(parseDouble(body.substring(0, split)), im(body.substring(split)))

  // ---------------------------------------------------------------- scalars

  /** Shortest decimal digits of |d|: (digits, exponent) with value = d1.d2d3... x 10^exp. */
  private def shortestDigits(s: String): (String, Int) =
    val bd = new JBigDecimal(s).stripTrailingZeros()
    val u = bd.unscaledValue().abs().toString
    (u, u.length - 1 - bd.scale())

  /** The shortest decimal string that rounds back to `x` (as a double, or as a float when
    * `single`), choosing the closest such string — Python's `repr` / NumPy's Dragon4 "unique"
    * mode. Implemented here because `Double.toString` is only guaranteed shortest from JDK 19.
    */
  private[numscala] def shortestDecimal(x: Double, single: Boolean): String =
    if x == 0.0 || x.isNaN || x.isInfinite then java.lang.Double.toString(x)
    else
      val exact = new JBigDecimal(x)
      val maxP = if single then 9 else 17
      var p = 1
      var result: JBigDecimal = null
      while result == null && p <= maxP do
        val r = exact.round(new java.math.MathContext(p, RoundingMode.HALF_EVEN))
        val back = if single then r.floatValue().toDouble else r.doubleValue()
        if back == x then result = r
        p += 1
      (if result == null then exact else result).stripTrailingZeros().toString

  private def pyRepr(neg: Boolean, digits: String, exp: Int, sci: Boolean): String =
    val sign = if neg then "-" else ""
    if sci then
      val mant = if digits.length == 1 then digits else s"${digits.head}.${digits.tail}"
      val es = if exp < 0 then f"-${-exp}%02d" else f"+$exp%02d"
      s"$sign${mant}e$es"
    else if exp < 0 then s"${sign}0.${"0" * (-exp - 1)}$digits"
    else if digits.length <= exp + 1 then s"$sign$digits${"0" * (exp + 1 - digits.length)}.0"
    else s"$sign${digits.substring(0, exp + 1)}.${digits.substring(exp + 1)}"

  /** Formats a double like Python's `repr(float)` (and NumPy's float64 scalar `str`). */
  def formatFloatShort(d: Double): String =
    if d.isNaN then "nan"
    else if d.isInfinite then (if d > 0 then "inf" else "-inf")
    else if d == 0.0 then (if 1.0 / d < 0 then "-0.0" else "0.0")
    else
      val (dig, e) = shortestDigits(shortestDecimal(math.abs(d), single = false))
      pyRepr(d < 0, dig, e, e < -4 || e >= 16)

  /** Formats a float32 scalar with its own shortest representation; like NumPy, float32
    * scalars switch to scientific notation from 1e6 on (and below 1e-4).
    */
  def formatFloat32Short(f: Float): String =
    if f.isNaN then "nan"
    else if f.isInfinite then (if f > 0 then "inf" else "-inf")
    else if f == 0.0f then (if 1.0f / f < 0 then "-0.0" else "0.0")
    else
      val (dig, e) = shortestDigits(shortestDecimal(math.abs(f).toDouble, single = true))
      val a = math.abs(f.toDouble)
      pyRepr(f < 0, dig, e, a >= 1e6 || a < 1e-4)

  private def shortestOf(d: Double, single: Boolean): String =
    shortestDecimal(math.abs(if single then d.toFloat.toDouble else d), single)

  /** The exact decimal value of |d| (of the float32 value when `single`). */
  private def exactOf(d: Double, single: Boolean): JBigDecimal =
    new JBigDecimal(math.abs(if single then d.toFloat.toDouble else d))

  /** Dragon4 positional formatting (unique, at most `precision` fraction digits, trim '.').
    * Like Dragon4, digits beyond the shortest representation (cut off at `precision`, or
    * required by `minDigits`) are those of the exact binary value, correctly rounded.
    */
  private[numscala] def positional(d: Double, precision: Int, single: Boolean, minDigits: Int = 0): String =
    val neg = d < 0 || (d == 0.0 && 1.0 / d < 0)
    var bd = new JBigDecimal(shortestOf(d, single))
    if bd.scale() > precision then bd = exactOf(d, single).setScale(precision, RoundingMode.HALF_EVEN)
    bd = bd.stripTrailingZeros()
    if bd.scale() < 0 then bd = bd.setScale(0)
    if bd.scale() < minDigits then
      bd = exactOf(d, single).setScale(math.min(minDigits, precision), RoundingMode.HALF_EVEN)
    var s = bd.toPlainString
    if !s.contains('.') then s = s + "."
    val frac = s.length - s.indexOf('.') - 1
    if frac < minDigits then s = s + "0" * (minDigits - frac)
    (if neg then "-" else "") + s

  /** Dragon4 scientific formatting. */
  private[numscala] def scientific(d: Double, precision: Int, single: Boolean, minDigits: Int = 0, expDigits: Int = 2): String =
    val neg = d < 0 || (d == 0.0 && 1.0 / d < 0)
    var (dig, e) =
      if d == 0.0 then ("0", 0)
      else shortestDigits(shortestOf(d, single))
    // digits beyond the shortest representation are those of the exact value (Dragon4)
    val ndig =
      if dig.length > precision + 1 then precision + 1
      else if dig.length < minDigits + 1 then minDigits + 1
      else 0
    if ndig > 0 && d != 0.0 then
      val bd = exactOf(d, single).round(new java.math.MathContext(ndig, RoundingMode.HALF_EVEN))
      val nd = bd.unscaledValue().toString
      e = nd.length - 1 - bd.scale()
      val stripped = nd.reverse.dropWhile(_ == '0').reverse
      dig = if stripped.isEmpty then "0" else stripped
    var frac = dig.tail
    if frac.length < minDigits then frac = frac + "0" * (minDigits - frac.length)
    val ae = math.abs(e).toString
    val es = (if e < 0 then "-" else "+") + ("0" * math.max(0, expDigits - ae.length)) + ae
    (if neg then "-" else "") + dig.head + "." + frac + "e" + es

  // ---------------------------------------------------------------- arrays

  /** A function that formats elements with uniform width for one array. */
  private[numscala] def elementFormatter[T](dtype: DType[T], values: Seq[T], repr: Boolean): T => String =
    dtype.kind match
      case 'b' =>
        v => padLeft(dtype.format(v), 5)
      case 'i' | 'u' =>
        val w = if values.isEmpty then 0 else values.map(v => dtype.format(v).length).max
        v => padLeft(dtype.format(v), w)
      case 'f' =>
        val ff = new FloatFormatter(values.map(dtype.toDouble), dtype eq DType.Float32, sign = '-')
        v => ff(dtype.toDouble(v))
      case 'c' =>
        val cs = values.map(dtype.toComplex)
        val rf = new FloatFormatter(cs.map(_.re), false, sign = '-')
        val imf = new FloatFormatter(cs.map(_.im), false, sign = '+')
        v =>
          val c = dtype.toComplex(v)
          val r = rf(c.re)
          val i = imf(c.im)
          val sp = i.replaceAll("\\s+$", "").length
          r + i.substring(0, sp) + "j" + " " * (i.length - sp)
      case _ =>
        v => "'" + dtype.format(v).replace("\\", "\\\\").replace("'", "\\'") + "'"

  private def padLeft(s: String, w: Int): String = if s.length >= w then s else " " * (w - s.length) + s
  private def padRight(s: String, w: Int): String = if s.length >= w then s else s + " " * (w - s.length)

  /** NumPy's FloatingFormat (floatmode='maxprec'). */
  private final class FloatFormatter(data: Seq[Double], single: Boolean, sign: Char) extends (Double => String):
    private val o = opts
    private val finite = data.filter(d => !d.isNaN && !d.isInfinite)
    private val absNonZero = finite.map(math.abs).filter(_ != 0.0)
    private val expFormat: Boolean =
      if absNonZero.isEmpty then false
      else
        val mx = absNonZero.max
        val mn = absNonZero.min
        // NumPy >= 2.3: the cutoff is 10**min(8, finfo(dtype).precision), i.e. 1e6 for float32
        mx >= (if single then 1e6 else 1e8) || (!o.suppress && (mn < 0.0001 || mx / mn > 1000.0))
    private var padL = 0
    private var padR = 0
    private var precision = o.precision
    private var minDigits = 0
    private var expSize = -1
    private def withSign(s: String): String = if sign == '+' && !s.startsWith("-") then "+" + s else s

    if finite.nonEmpty then
      if expFormat then
        val strs = finite.map(d => withSign(scientific(d, o.precision, single)))
        val parts = strs.map { s => val k = s.indexOf('e'); (s.substring(0, k), s.substring(k + 1)) }
        expSize = parts.map(_._2.length).max - 1
        val ip = parts.map(p => p._1.substring(0, p._1.indexOf('.')))
        val fp = parts.map(p => p._1.substring(p._1.indexOf('.') + 1))
        precision = fp.map(_.length).max
        minDigits = precision
        padL = ip.map(_.length).max
        padR = expSize + 2 + precision
      else
        val strs = finite.map(d => withSign(positional(d, o.precision, single)))
        padL = strs.map(s => s.indexOf('.')).max
        padR = strs.map(s => s.length - s.indexOf('.') - 1).max
    if finite.length != data.length then
      val neginf = sign != '-' || data.exists(d => d.isInfinite && d < 0)
      val offset = padR + 1
      padL = math.max(padL, math.max(o.nanstr.length - offset, o.infstr.length + (if neginf then 1 else 0) - offset))

    def apply(d: Double): String =
      if d.isNaN then
        val s = if sign == '+' then "+" + o.nanstr else o.nanstr
        padLeft(s, padL + padR + 1)
      else if d.isInfinite then
        val s = (if d > 0 then (if sign == '+' then "+" else "") else "-") + o.infstr
        padLeft(s, padL + padR + 1)
      else if expFormat then
        val s = withSign(scientific(d, precision, single, minDigits, math.max(expSize, 2)))
        val k = s.indexOf('.')
        padLeft(s.substring(0, k), padL) + s.substring(k)
      else
        val s = withSign(positional(d, precision, single, minDigits))
        val k = s.indexOf('.')
        padLeft(s.substring(0, k), padL) + padRight(s.substring(k), padR + 1)

  /** Formats an array as NumPy's `str()` (sep " ") or the body of `repr()` (sep ", "). */
  private[numscala] def formatArray[T](a: NDArray[T], separator: String, prefix: String, suffixLen: Int): String =
    val o = opts
    val nd = a.ndim
    val summarize = a.size > o.threshold
    val sample: Seq[T] =
      if !summarize then a.toSeq
      else
        // only elements that will be printed participate in width computation
        val buf = scala.collection.mutable.ArrayBuffer.empty[T]
        def collect(idx: Vector[Int]): Unit =
          if idx.length == nd then buf += a.getAt(idx.toArray)
          else
            val n = a.shapeArr(idx.length)
            val ks =
              if 2 * o.edgeitems < n then (0 until o.edgeitems) ++ (n - o.edgeitems until n)
              else 0 until n
            ks.foreach(k => collect(idx :+ k))
        collect(Vector.empty)
        buf.toSeq
    val fmt = elementFormatter(a.dtype, sample, separator != " ")
    if nd == 0 then
      val x = a.getAt(Array.emptyIntArray)
      return if a.dtype.isBool then a.dtype.format(x) else fmt(x)
    val lineWidth = o.linewidth - suffixLen
    val nextLinePrefix = " " + " " * prefix.length
    val summaryInsert = if summarize then "..." else ""
    val idx = new Array[Int](nd)

    def extendLine(s: StringBuilder, line: String, word: String, width: Int, hanging: String): String =
      var needsWrap = line.length + word.length > width
      if line.length <= hanging.length then needsWrap = false
      if needsWrap then
        s.append(line.replaceAll("\\s+$", "")).append('\n')
        hanging + word
      else line + word

    def recurse(axis: Int, hanging: String, currWidth: Int): String =
      val axesLeft = nd - axis
      val nextHanging = hanging + " "
      val nextWidth = currWidth - 1
      val n = a.shapeArr(axis)
      val showSummary = summaryInsert.nonEmpty && 2 * o.edgeitems < n
      val leading = if showSummary then o.edgeitems else 0
      val trailing = if showSummary then o.edgeitems else n
      val s = new StringBuilder
      def elem(i: Int): String =
        idx(axis) = if i < 0 then n + i else i
        if axis + 1 == nd then fmt(a.getAt(idx)) else recurse(axis + 1, nextHanging, nextWidth)
      if n == 0 then return "[]"
      if axesLeft == 1 then
        val elemWidth = currWidth - math.max(separator.replaceAll("\\s+$", "").length, 1)
        var line = hanging
        for i <- 0 until leading do
          line = extendLine(s, line, elem(i), elemWidth, hanging) + separator
        if showSummary then line = extendLine(s, line, summaryInsert, elemWidth, hanging) + separator
        for i <- trailing until 1 by -1 do
          line = extendLine(s, line, elem(-i), elemWidth, hanging) + separator
        line = extendLine(s, line, elem(-1), elemWidth, hanging)
        s.append(line)
      else
        val lineSep = separator.replaceAll("\\s+$", "") + "\n" * (axesLeft - 1)
        for i <- 0 until leading do s.append(hanging).append(elem(i)).append(lineSep)
        if showSummary then s.append(hanging).append(summaryInsert).append(lineSep)
        for i <- trailing until 1 by -1 do s.append(hanging).append(elem(-i)).append(lineSep)
        s.append(hanging).append(elem(-1))
      "[" + s.toString.substring(hanging.length) + "]"

    recurse(0, nextLinePrefix, lineWidth)

  private def impliedDType(d: DType[?]): Boolean =
    (d eq DType.Float64) || (d eq DType.Int64) || (d eq DType.Bool) || (d eq DType.Complex128)

  private[numscala] def str[T](a: NDArray[T]): String =
    if a.ndim == 0 then a.dtype.format(a.item)
    else if a.size == 0 then "[]"
    else formatArray(a, " ", "", 0)

  /** `repr(a)`, like NumPy's `_array_repr_implementation` (`shape=`/`dtype=` extras). */
  private[numscala] def repr[T](a: NDArray[T]): String =
    val dtypeStr = if a.dtype.isString then "<U" + math.max(1, a.toSeq.map(x => a.dtype.format(x).length).maxOption.getOrElse(1)) else a.dtype.name
    val prefix = "array("
    val lst = if a.size == 0 && a.ndim > 0 then "[]" else formatArray(a, ", ", prefix, 1)
    val extras = Seq(
      Option.when((a.size == 0 && a.ndim != 1) || a.size > opts.threshold)(s"shape=${Shape.str(a.shapeArr)}"),
      Option.when(!impliedDType(a.dtype) || a.size == 0)(
        s"dtype=${if a.dtype.isString then "'" + dtypeStr + "'" else dtypeStr}")
    ).flatten
    if extras.isEmpty then prefix + lst + ")"
    else
      val arrStr = prefix + lst + ","
      val extraStr = extras.mkString(", ") + ")"
      val lastLineLen = arrStr.length - (arrStr.lastIndexOf('\n') + 1)
      val spacer = if lastLineLen + extraStr.length + 1 > opts.linewidth then "\n" + " " * prefix.length else " "
      arrStr + spacer + extraStr
