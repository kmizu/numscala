package numscala

/** NumPy-exact `str`/`repr` of masked arrays. Like NumPy, an array with a mask is printed
  * as an object array: each element as its Python `repr`, masked ones as `--`, no padding.
  */
private[numscala] object MAFormat:

  /** Python `repr` of an element converted to a Python object (`astype(object)`). */
  def pyRepr[T](d: DType[T], x: T): String = d.kind match
    case 'f' => Format.formatFloatShort(d.toDouble(x))
    case 'U' =>
      val s = d.format(x)
      if s.contains('\'') && !s.contains('"') then "\"" + s.replace("\\", "\\\\") + "\""
      else "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"
    case _ => d.format(x)

  /** Element strings of `a` (masked as `--`) in C order. */
  private def elementStrings[T](a: MaskedArray[T]): NDArray[String] =
    val d = a.dtype
    NDArray.zipMap(a._data, a.maskArray)((x, m) => if m then MA.masked_print_option.display() else pyRepr(d, x))(using DType.Str)

  /** NumPy's `_formatArray` for pre-formatted elements (object dtype: no padding). */
  def formatStrings(a: NDArray[String], separator: String, prefix: String, suffixLen: Int): String =
    val o = Format.printOptions
    val nd = a.ndim
    if nd == 0 then return a.item
    val summarize = a.size > o.threshold
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
        if axis + 1 == nd then a.getAt(idx) else recurse(axis + 1, nextHanging, nextWidth)
      if n == 0 then return "[]"
      if axesLeft == 1 then
        val elemWidth = currWidth - math.max(separator.replaceAll("\\s+$", "").length, 1)
        var line = hanging
        for i <- 0 until leading do line = extendLine(s, line, elem(i), elemWidth, hanging) + separator
        if showSummary then line = extendLine(s, line, summaryInsert, elemWidth, hanging) + separator
        for i <- trailing until 1 by -1 do line = extendLine(s, line, elem(-i), elemWidth, hanging) + separator
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

  /** NumPy's `BoolFormat`: `' True'` / `'False'` (unpadded only for 0-d arrays). */
  private def boolStrings(b: NDArray[Boolean]): NDArray[String] =
    if b.ndim == 0 then b.map(x => if x then "True" else "False")(using DType.Str)
    else b.map(x => if x then " True" else "False")(using DType.Str)

  /** Plain (unmasked) array formatting, with NumPy's padded booleans. */
  private def plain[T](a: NDArray[T], separator: String, prefix: String, suffixLen: Int): String =
    if a.dtype eq DType.Bool then formatStrings(boolStrings(a.asInstanceOf[NDArray[Boolean]]), separator, prefix, suffixLen)
    else Format.formatArray(a, separator, prefix, suffixLen)

  /** `str(masked_array)`. */
  def str[T](a: MaskedArray[T]): String =
    if a._mask == null then
      if a.ndim == 0 || a.size == 0 || !(a.dtype eq DType.Bool) then Format.str(a._data) else plain(a._data, " ", "", 0)
    else if a.size == 0 && a.ndim > 0 then "[]"
    else formatStrings(elementStrings(a), " ", "", 0)

  private def impliedDType(d: DType[?]): Boolean =
    (d eq DType.Float64) || (d eq DType.Int64) || (d eq DType.Bool) || (d eq DType.Complex128)

  private def dtypeRepr[T](a: MaskedArray[T]): String =
    if a.dtype.isString then
      "'<U" + math.max(1, a._data.toSeq.map(x => a.dtype.format(x).length).maxOption.getOrElse(1)) + "'"
    else a.dtype.name

  /** The `fill_value=` field as NumPy prints it. */
  def fillRepr[T](a: MaskedArray[T]): String =
    val d = a.dtype
    a._fill match
      case Some(v) => if d.isString then pyRepr(d, v) else d.format(v)
      case None =>
        val v = a.fill_value
        d match
          case DType.Int8 | DType.Int16 | DType.Int32 => "np.int64(999999)"
          case DType.Float32 => "np.float64(1e+20)"
          case _ => if d.isString then pyRepr(d, v) else d.format(v)

  /** `repr(masked_array)`. */
  def repr[T](a: MaskedArray[T]): String =
    val prefix0 = "masked_array("
    val allMasked = a._mask != null && a._mask.all()
    val dtypeNeeded = !impliedDType(a.dtype) || allMasked || a.size == 0
    val keys = Seq("data", "mask", "fill_value") ++ (if dtypeNeeded then Seq("dtype") else Nil)
    val isOneRow = a.shape.dropRight(1).forall(_ == 1)
    val (indents, prefix) =
      if isOneRow then
        val m = keys.tail.map(k => k -> " " * math.max(2, (prefix0 + keys.head).length - k.length)).toMap
        (m + (keys.head -> prefix0), "")
      else (keys.map(_ -> "  ").toMap, prefix0 + "\n")
    val dataStr =
      val pre = indents("data") + "data="
      if a._mask == null then
        if a.size == 0 && a.ndim > 0 then "[]" else plain(a._data, ", ", pre, 1)
      else if a.size == 0 && a.ndim > 0 then "[]"
      else formatStrings(elementStrings(a), ", ", pre, 1)
    val maskStr =
      if a._mask == null then "False"
      else if a.size == 0 && a.ndim > 0 then "[]"
      else plain(a._mask, ", ", indents("mask") + "mask=", 1)
    val reprs = Map("data" -> dataStr, "mask" -> maskStr, "fill_value" -> fillRepr(a), "dtype" -> dtypeRepr(a))
    prefix + keys.map(k => s"${indents(k)}$k=${reprs(k)}").mkString(",\n") + ")"
