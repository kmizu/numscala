package numscala

import java.util.Locale

/** Scalar implementations of Python `str` methods used by [[Strings]].
  *
  * Python strings are sequences of Unicode code points while JVM strings are UTF-16;
  * every length, index and width below is measured in code points so that results
  * match CPython for text outside the Basic Multilingual Plane as well.
  */
private[numscala] object StringsPy:

  // ------------------------------------------------------------------ code point helpers

  private def hasSurrogates(s: String): Boolean =
    var i = 0
    var found = false
    while !found && i < s.length do
      if Character.isSurrogate(s.charAt(i)) then found = true
      i += 1
    found

  /** Length in code points (`len(s)`). */
  def len(s: String): Int = if hasSurrogates(s) then s.codePointCount(0, s.length) else s.length

  /** UTF-16 index of code point number `cp` (0 <= cp <= len). */
  def cpToIdx(s: String, cp: Int): Int = if cp == 0 then 0 else s.offsetByCodePoints(0, cp)

  /** Code point number of UTF-16 index `i`. */
  def idxToCp(s: String, i: Int): Int = s.codePointCount(0, i)

  def codePoints(s: String): Array[Int] = s.codePoints().toArray

  private def cpStr(cp: Int): String = new String(Character.toChars(cp))

  private def fromCodePoints(cps: Array[Int], from: Int, until: Int): String =
    if from >= until then "" else new String(cps, from, until - from)

  /** Python `slice.indices`-style normalisation of `start`/`end` for search methods. */
  private def adjust(start: Long, end: Long, n: Int): (Long, Long) =
    var s = start
    var e = end
    if e > n then e = n
    else if e < 0 then
      e += n
      if e < 0 then e = 0
    if s < 0 then
      s += n
      if s < 0 then s = 0
    (s, e)

  // ------------------------------------------------------------------ character classes

  /** Python `str.isspace` for a single code point. */
  def isSpaceCp(c: Int): Boolean = c match
    case 0x09 | 0x0a | 0x0b | 0x0c | 0x0d | 0x1c | 0x1d | 0x1e | 0x1f | 0x20 | 0x85 | 0xa0 | 0x1680 |
        0x2028 | 0x2029 | 0x202f | 0x205f | 0x3000 =>
      true
    case _ => c >= 0x2000 && c <= 0x200a

  private def isLineBreak(c: Int): Boolean = c match
    case 0x0a | 0x0b | 0x0c | 0x0d | 0x1c | 0x1d | 0x1e | 0x85 | 0x2028 | 0x2029 => true
    case _ => false

  private def isCased(c: Int): Boolean =
    Character.isUpperCase(c) || Character.isLowerCase(c) || Character.isTitleCase(c)

  private def isDecimalCp(c: Int): Boolean = Character.getType(c) == Character.DECIMAL_DIGIT_NUMBER

  private def isDigitCp(c: Int): Boolean =
    isDecimalCp(c) || (Character.getType(c) == Character.OTHER_NUMBER && {
      val v = Character.getNumericValue(c)
      v >= 0 && v <= 9
    })

  private val cjkNumerals: Set[Int] =
    "〇一二三四五六七八九十百千万萬億兆零壱壹弐貳贰参參叁肆伍陸陆柒捌玖拾佰仟廿卅".codePoints().toArray.toSet

  private def isNumericCp(c: Int): Boolean =
    val t = Character.getType(c)
    t == Character.DECIMAL_DIGIT_NUMBER || t == Character.LETTER_NUMBER || t == Character.OTHER_NUMBER ||
    cjkNumerals.contains(c)

  private def isAlphaCp(c: Int): Boolean = Character.isLetter(c)

  private def allCp(s: String)(p: Int => Boolean): Boolean =
    s.nonEmpty && s.codePoints().allMatch(c => p(c))

  def isalpha(s: String): Boolean = allCp(s)(isAlphaCp)
  def isdecimal(s: String): Boolean = allCp(s)(isDecimalCp)
  def isdigit(s: String): Boolean = allCp(s)(isDigitCp)
  def isnumeric(s: String): Boolean = allCp(s)(isNumericCp)
  def isalnum(s: String): Boolean = allCp(s)(c => isAlphaCp(c) || isNumericCp(c))
  def isspace(s: String): Boolean = allCp(s)(isSpaceCp)

  def isupper(s: String): Boolean =
    var cased = false
    var ok = true
    val it = s.codePoints().iterator()
    while ok && it.hasNext do
      val c = it.nextInt()
      if Character.isLowerCase(c) || Character.isTitleCase(c) then ok = false
      else if Character.isUpperCase(c) then cased = true
    ok && cased

  def islower(s: String): Boolean =
    var cased = false
    var ok = true
    val it = s.codePoints().iterator()
    while ok && it.hasNext do
      val c = it.nextInt()
      if Character.isUpperCase(c) || Character.isTitleCase(c) then ok = false
      else if Character.isLowerCase(c) then cased = true
    ok && cased

  def istitle(s: String): Boolean =
    var cased = false
    var prevCased = false
    var ok = true
    val it = s.codePoints().iterator()
    while ok && it.hasNext do
      val c = it.nextInt()
      if Character.isUpperCase(c) || Character.isTitleCase(c) then
        if prevCased then ok = false
        prevCased = true
        cased = true
      else if Character.isLowerCase(c) then
        if !prevCased then ok = false
        prevCased = true
        cased = true
      else prevCased = false
    ok && cased

  // ------------------------------------------------------------------ case conversion

  def lower(s: String): String = s.toLowerCase(Locale.ROOT)
  def upper(s: String): String = s.toUpperCase(Locale.ROOT)

  private def lowerCp(c: Int): String = cpStr(c).toLowerCase(Locale.ROOT)
  private def upperCp(c: Int): String = cpStr(c).toUpperCase(Locale.ROOT)

  /** Full title-case mapping of one code point (`'ß'` -> `"Ss"`). */
  private def titleCp(c: Int): String =
    val t = Character.toTitleCase(c)
    if t != c then cpStr(t)
    else
      val up = upperCp(c)
      if up.codePointCount(0, up.length) <= 1 then cpStr(t)
      else
        val first = up.codePointAt(0)
        cpStr(first) + up.substring(Character.charCount(first)).toLowerCase(Locale.ROOT)

  def capitalize(s: String): String =
    if s.isEmpty then s
    else
      val first = s.codePointAt(0)
      titleCp(first) + lower(s.substring(Character.charCount(first)))

  def swapcase(s: String): String =
    val sb = new java.lang.StringBuilder(s.length)
    s.codePoints().forEach { c =>
      if Character.isUpperCase(c) then sb.append(lowerCp(c))
      else if Character.isLowerCase(c) then sb.append(upperCp(c))
      else sb.appendCodePoint(c)
      ()
    }
    sb.toString

  def title(s: String): String =
    val sb = new java.lang.StringBuilder(s.length)
    var prevCased = false
    s.codePoints().forEach { c =>
      if prevCased then sb.append(lowerCp(c)) else sb.append(titleCp(c))
      prevCased = isCased(c)
    }
    sb.toString

  // ------------------------------------------------------------------ padding

  private def checkFill(fill: String): Int =
    if len(fill) != 1 then
      throw new IllegalArgumentException("The fill character must be exactly one character long")
    fill.codePointAt(0)

  private def rep(fill: String, n: Int): String = if n <= 0 then "" else fill * n

  def center(s: String, width: Long, fill: String): String =
    checkFill(fill)
    val n = len(s)
    if width <= n then s
    else
      val marg = (width - n).toInt
      val left = marg / 2 + (marg & width.toInt & 1)
      rep(fill, left) + s + rep(fill, marg - left)

  def ljust(s: String, width: Long, fill: String): String =
    checkFill(fill)
    val n = len(s)
    if width <= n then s else s + rep(fill, (width - n).toInt)

  def rjust(s: String, width: Long, fill: String): String =
    checkFill(fill)
    val n = len(s)
    if width <= n then s else rep(fill, (width - n).toInt) + s

  def zfill(s: String, width: Long): String =
    val n = len(s)
    if width <= n then s
    else
      val pad = "0" * (width - n).toInt
      if s.nonEmpty && (s.charAt(0) == '+' || s.charAt(0) == '-') then s.substring(0, 1) + pad + s.substring(1)
      else pad + s

  def expandtabs(s: String, tabsize: Long): String =
    val sb = new java.lang.StringBuilder(s.length)
    var col = 0L
    s.codePoints().forEach { c =>
      if c == '\t' then
        if tabsize > 0 then
          val k = tabsize - (col % tabsize)
          var i = 0L
          while i < k do
            sb.append(' ')
            i += 1
          col += k
      else
        sb.appendCodePoint(c)
        if c == '\n' || c == '\r' then col = 0 else col += 1
    }
    sb.toString

  def repeat(s: String, n: Long): String =
    if n <= 0 || s.isEmpty then ""
    else
      if s.length.toLong * n > Int.MaxValue then throw new OutOfMemoryError("repeated string is too long")
      s * n.toInt

  // ------------------------------------------------------------------ stripping

  private def stripPred(chars: String | Null): Int => Boolean =
    if chars == null then isSpaceCp
    else
      val set = chars.codePoints().toArray.toSet
      c => set.contains(c)

  def strip(s: String, chars: String | Null, left: Boolean, right: Boolean): String =
    val p = stripPred(chars)
    var i = 0
    var j = s.length
    if left then
      while i < j && p(s.codePointAt(i)) do i += Character.charCount(s.codePointAt(i))
    if right then
      while j > i && p(s.codePointBefore(j)) do j -= Character.charCount(s.codePointBefore(j))
    s.substring(i, j)

  // ------------------------------------------------------------------ searching

  /** Python `str.find`; returns a code-point index or -1. */
  def find(s: String, sub: String, start: Long, end: Long): Int =
    val n = len(s)
    val (st, en) = adjust(start, end, n)
    val m = len(sub)
    if st > n || en - st < m then -1
    else
      val si = cpToIdx(s, st.toInt)
      val ei = cpToIdx(s, en.toInt)
      val r = s.indexOf(sub, si)
      if r < 0 || r + sub.length > ei then -1 else idxToCp(s, r)

  /** Python `str.rfind`; returns a code-point index or -1. */
  def rfind(s: String, sub: String, start: Long, end: Long): Int =
    val n = len(s)
    val (st, en) = adjust(start, end, n)
    val m = len(sub)
    if st > n || en - st < m then -1
    else
      val si = cpToIdx(s, st.toInt)
      val ei = cpToIdx(s, en.toInt)
      val r = s.lastIndexOf(sub, ei - sub.length)
      if r < si then -1 else idxToCp(s, r)

  /** Python `str.count` (non-overlapping occurrences). */
  def count(s: String, sub: String, start: Long, end: Long): Int =
    val n = len(s)
    val (st, en) = adjust(start, end, n)
    val m = len(sub)
    if st > n || en - st < m then 0
    else if m == 0 then (en - st + 1).toInt
    else
      val si = cpToIdx(s, st.toInt)
      val ei = cpToIdx(s, en.toInt)
      var c = 0
      var i = s.indexOf(sub, si)
      while i >= 0 && i + sub.length <= ei do
        c += 1
        i = s.indexOf(sub, i + sub.length)
      c

  def startswith(s: String, prefix: String, start: Long, end: Long): Boolean =
    val n = len(s)
    val (st, en) = adjust(start, end, n)
    val m = len(prefix)
    if st > n || en - st < m then false
    else s.startsWith(prefix, cpToIdx(s, st.toInt))

  def endswith(s: String, suffix: String, start: Long, end: Long): Boolean =
    val n = len(s)
    val (st, en) = adjust(start, end, n)
    val m = len(suffix)
    if st > n || en - st < m then false
    else
      val ei = cpToIdx(s, en.toInt)
      s.startsWith(suffix, ei - suffix.length)

  // ------------------------------------------------------------------ replace / join / slice

  def replace(s: String, old: String, nw: String, count: Long): String =
    val limit = if count < 0 then Long.MaxValue else count
    if limit == 0 then s
    else if old.isEmpty then
      val cps = codePoints(s)
      val sb = new java.lang.StringBuilder
      var done = 0L
      var i = 0
      while i <= cps.length do
        if done < limit then
          sb.append(nw)
          done += 1
        if i < cps.length then sb.appendCodePoint(cps(i))
        i += 1
      sb.toString
    else
      val sb = new java.lang.StringBuilder
      var done = 0L
      var from = 0
      var i = s.indexOf(old)
      while i >= 0 && done < limit do
        sb.append(s, from, i).append(nw)
        from = i + old.length
        done += 1
        i = s.indexOf(old, from)
      sb.append(s, from, s.length)
      sb.toString

  /** `sep.join(s)`: `sep` inserted between the characters of `s`. */
  def join(sep: String, s: String): String =
    if sep.isEmpty then s
    else
      val sb = new java.lang.StringBuilder
      var first = true
      s.codePoints().forEach { c =>
        if !first then sb.append(sep)
        sb.appendCodePoint(c)
        first = false
      }
      sb.toString

  /** Python `s[start:stop:step]` over code points (`None` = absent bound). */
  def slice(s: String, start: Option[Long], stop: Option[Long], step: Option[Long]): String =
    val st = step.getOrElse(1L)
    if st == 0 then throw new IllegalArgumentException("slice step cannot be zero")
    val cps = codePoints(s)
    val n = cps.length.toLong
    def norm(v: Option[Long], dflt: Long, lo: Long, hi: Long): Long = v match
      case None => dflt
      case Some(x0) =>
        val x = if x0 < 0 then x0 + n else x0
        if x < lo then lo else if x > hi then hi else x
    if st > 0 then
      val b = norm(start, 0, 0, n)
      val e = norm(stop, n, 0, n)
      if st == 1 then fromCodePoints(cps, b.toInt, e.toInt)
      else
        val sb = new java.lang.StringBuilder
        var i = b
        while i < e do
          sb.appendCodePoint(cps(i.toInt))
          i += st
        sb.toString
    else
      val b = norm(start, n - 1, -1, n - 1)
      val e = norm(stop, -1, -1, n - 1)
      val sb = new java.lang.StringBuilder
      var i = b
      while i > e do
        sb.appendCodePoint(cps(i.toInt))
        i += st
      sb.toString

  // ------------------------------------------------------------------ partition / split

  def partition(s: String, sep: String): (String, String, String) =
    if sep.isEmpty then throw new IllegalArgumentException("empty separator")
    val i = s.indexOf(sep)
    if i < 0 then (s, "", "") else (s.substring(0, i), sep, s.substring(i + sep.length))

  def rpartition(s: String, sep: String): (String, String, String) =
    if sep.isEmpty then throw new IllegalArgumentException("empty separator")
    val i = s.lastIndexOf(sep)
    if i < 0 then ("", "", s) else (s.substring(0, i), sep, s.substring(i + sep.length))

  /** Python `str.split(sep, maxsplit)`; `sep == null` splits on whitespace runs. */
  def split(s: String, sep: String | Null, maxsplit: Long): List[String] =
    val limit = if maxsplit < 0 then Long.MaxValue else maxsplit
    val out = List.newBuilder[String]
    if sep == null then
      var i = 0
      val n = s.length
      var done = 0L
      var finished = false
      while !finished do
        while i < n && isSpaceCp(s.codePointAt(i)) do i += Character.charCount(s.codePointAt(i))
        if i >= n then finished = true
        else if done >= limit then
          out += s.substring(i)
          finished = true
        else
          val b = i
          while i < n && !isSpaceCp(s.codePointAt(i)) do i += Character.charCount(s.codePointAt(i))
          out += s.substring(b, i)
          done += 1
    else
      val sp: String = sep
      if sp.isEmpty then throw new IllegalArgumentException("empty separator")
      var from = 0
      var done = 0L
      var i = s.indexOf(sp)
      while i >= 0 && done < limit do
        out += s.substring(from, i)
        from = i + sp.length
        done += 1
        i = s.indexOf(sp, from)
      out += s.substring(from)
    out.result()

  /** Python `str.rsplit(sep, maxsplit)`. */
  def rsplit(s: String, sep: String | Null, maxsplit: Long): List[String] =
    val limit = if maxsplit < 0 then Long.MaxValue else maxsplit
    var out = List.empty[String]
    if sep == null then
      var j = s.length
      var done = 0L
      var finished = false
      while !finished do
        while j > 0 && isSpaceCp(s.codePointBefore(j)) do j -= Character.charCount(s.codePointBefore(j))
        if j <= 0 then finished = true
        else if done >= limit then
          out = s.substring(0, j) :: out
          finished = true
        else
          val e = j
          while j > 0 && !isSpaceCp(s.codePointBefore(j)) do j -= Character.charCount(s.codePointBefore(j))
          out = s.substring(j, e) :: out
          done += 1
    else
      val sp: String = sep
      if sp.isEmpty then throw new IllegalArgumentException("empty separator")
      var end = s.length
      var done = 0L
      var i = if end - sp.length >= 0 then s.lastIndexOf(sp, end - sp.length) else -1
      while i >= 0 && done < limit do
        out = s.substring(i + sp.length, end) :: out
        end = i
        done += 1
        i = if end - sp.length >= 0 then s.lastIndexOf(sp, end - sp.length) else -1
      out = s.substring(0, end) :: out
    out

  /** Python `str.splitlines(keepends)`. */
  def splitlines(s: String, keepends: Boolean): List[String] =
    val out = List.newBuilder[String]
    val n = s.length
    var i = 0
    var b = 0
    while i < n do
      val c = s.charAt(i).toInt
      if isLineBreak(c) then
        var e = i + 1
        if c == '\r' && e < n && s.charAt(e) == '\n' then e += 1
        out += (if keepends then s.substring(b, e) else s.substring(b, i))
        b = e
        i = e
      else i += 1
    if b < n then out += s.substring(b)
    out.result()

  // ------------------------------------------------------------------ comparison / misc

  /** Code-point lexicographic comparison (NumPy compares UCS4 code units). */
  def compare(a: String, b: String): Int =
    var i = 0
    var r = 0
    while r == 0 && i < a.length && i < b.length do
      val ca = a.codePointAt(i)
      val cb = b.codePointAt(i)
      if ca != cb then r = Integer.compare(ca, cb)
      else i += Character.charCount(ca)
    if r != 0 then r else Integer.compare(a.length - i, b.length - i)

  def rstripWs(s: String): String = strip(s, null, left = false, right = true)

  def translate(s: String, table: Map[Char, String | Char]): String =
    val sb = new java.lang.StringBuilder(s.length)
    var i = 0
    while i < s.length do
      val c = s.charAt(i)
      table.get(c) match
        case Some(r: String) => sb.append(r)
        case Some(r: Char) => sb.append(r)
        case None => sb.append(c)
      i += 1
    sb.toString

  /** Python-style `repr` of a string (single quotes unless the text contains only `'`). */
  def repr(s: String): String =
    val q = if s.contains('\'') && !s.contains('"') then '"' else '\''
    val sb = new java.lang.StringBuilder
    sb.append(q)
    s.codePoints().forEach { c =>
      c match
        case '\\' => sb.append("\\\\")
        case '\n' => sb.append("\\n")
        case '\r' => sb.append("\\r")
        case '\t' => sb.append("\\t")
        case x if x == q => sb.append('\\').append(q)
        case x if x < 0x20 || x == 0x7f => sb.append(f"\\x$x%02x")
        case x => sb.appendCodePoint(x)
      ()
    }
    sb.append(q)
    sb.toString
