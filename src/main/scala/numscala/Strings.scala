package numscala

import java.nio.{ByteBuffer, CharBuffer}
import java.nio.charset.{CharacterCodingException, Charset, CodingErrorAction}

/** `numpy.strings` / `numpy.char` (vectorized string operations).
  *
  * Every function applies the corresponding Python `str` method to each element of a
  * `NDArray[String]`, broadcasting array-valued arguments against each other.  Arguments
  * typed [[Strings.StrLike]] accept either a string array or a plain `String`; those typed
  * [[Strings.IntLike]] accept an `Int`, a `Long` or an integer array.  Lengths, widths and
  * indices are counted in Unicode code points, like Python.
  *
  * Design choices (where NumPy returns object or bytes arrays, which have no dtype here):
  *  - `split`, `rsplit`, `splitlines` and `encode` return an [[Strings.ObjectArray]]
  *    (an immutable n-d container of arbitrary Scala values in C order).
  *  - `partition` / `rpartition` follow `numpy.strings` (NumPy >= 2): a tuple of three
  *    arrays, not the legacy `numpy.char` array with a trailing axis of length 3.
  *  - `equal` ... `greater_equal` follow `numpy.strings` (no trailing-whitespace stripping);
  *    use `compare_chararrays(..., rstrip = true)` for the legacy `numpy.char` behaviour.
  *  - Index-like results (`find`, `count`, `str_len`, ...) are `NDArray[Int]`.
  */
object Strings:

  /** A string array or a single string (broadcast as a 0-d array). */
  type StrLike = NDArray[String] | String
  /** An integer scalar or an integer array. */
  type IntLike = Int | Long | NDArray[?]
  /** A slice bound: an integer, an integer array or `None`. */
  type SliceArg = Int | Long | NDArray[?] | None.type

  // ------------------------------------------------------------------ object arrays

  /** An immutable n-dimensional container of arbitrary values (NumPy `dtype=object`),
    * returned by functions whose per-element result is not a string or number.
    */
  final class ObjectArray[A] private[numscala] (private val elems: IndexedSeq[A], shapeArr: Array[Int]):
    /** The array dimensions. */
    val shape: Seq[Int] = shapeArr.toSeq
    def ndim: Int = shapeArr.length
    def size: Int = elems.length
    /** Element at a full multi-index (negative indices count from the end). */
    def apply(idx: Int*): A =
      if idx.length != ndim then
        throw new IndexOutOfBoundsException(s"expected $ndim indices but got ${idx.length}")
      var flat = 0
      var k = 0
      while k < ndim do
        val n = shapeArr(k)
        val i = if idx(k) < 0 then idx(k) + n else idx(k)
        if i < 0 || i >= n then
          throw new IndexOutOfBoundsException(s"index ${idx(k)} is out of bounds for axis $k with size $n")
        flat = flat * n + i
        k += 1
      elems(flat)
    /** The elements in C order. */
    def toSeq: IndexedSeq[A] = elems
    def toList: List[A] = elems.toList
    /** The element of a size-1 array. */
    def item: A =
      if size != 1 then throw new IllegalArgumentException("can only convert an array of size 1 to a Scala scalar")
      elems(0)
    def map[B](f: A => B): ObjectArray[B] = new ObjectArray(elems.map(f), shapeArr)
    /** Nested Scala lists (`a.tolist()`); a 0-d array yields its element. */
    def tolist: Any =
      def go(axis: Int, base: Int, stride: Int): Any =
        if axis == ndim then elems(base)
        else
          val inner = stride / math.max(shapeArr(axis), 1)
          List.tabulate(shapeArr(axis))(i => go(axis + 1, base + i * inner, inner))
      go(0, 0, elems.length)
    private def elemEq(x: Any, y: Any): Boolean = java.util.Objects.deepEquals(x, y)
    override def equals(o: Any): Boolean = o match
      case that: ObjectArray[?] =>
        shape == that.shape && elems.length == that.elems.length &&
        elems.indices.forall(i => elemEq(elems(i), that.elems(i)))
      case _ => false
    override def hashCode: Int = shape.hashCode
    override def toString: String =
      def show(x: Any): String = x match
        case b: Array[Byte] => "b'" + b.map(v => if v >= 32 && v < 127 then v.toChar.toString else f"\\x${v & 0xff}%02x").mkString + "'"
        case l: List[?] => l.map(show).mkString("[", ", ", "]")
        case s: String => StringsPy.repr(s)
        case other => String.valueOf(other)
      def nest(axis: Int, base: Int, stride: Int): String =
        if axis == ndim then show(elems(base))
        else
          val inner = stride / math.max(shapeArr(axis), 1)
          (0 until shapeArr(axis)).map(i => nest(axis + 1, base + i * inner, inner)).mkString("[", ", ", "]")
      s"ObjectArray(${nest(0, 0, elems.length)})"

  private def objOf[A](a: NDArray[?], f: Int => A): ObjectArray[A] =
    new ObjectArray(IndexedSeq.tabulate(a.size)(f), a.shapeArr.clone())

  // ------------------------------------------------------------------ argument helpers

  private def sa(x: StrLike): NDArray[String] = x match
    case s: String => NDArray.scalar(s)
    case a: NDArray[?] =>
      if !a.dtype.isString then
        throw new IllegalArgumentException(s"string operation not supported for dtype ${a.dtype.name}")
      a.asInstanceOf[NDArray[String]]

  private def ia(x: IntLike): NDArray[Long] = x match
    case i: Int => NDArray.scalar(i.toLong)
    case l: Long => NDArray.scalar(l)
    case a: NDArray[?] =>
      val d = a.dtype.asInstanceOf[DType[Any]]
      if !(d.isInteger || d.isBool) then
        throw new IllegalArgumentException(s"expected an integer array, got dtype ${d.name}")
      a.asInstanceOf[NDArray[Any]].map(v => d.toLong(v))

  private val NoneVal: Long = Long.MinValue

  private def sl(x: SliceArg): NDArray[Long] = x match
    case None => NDArray.scalar(NoneVal)
    case other => ia(other.asInstanceOf[IntLike])

  private def opt(v: Long): Option[Long] = if v == NoneVal then None else Some(v)

  private def map1[U](a: StrLike)(f: String => U)(using ev: DType[U]): NDArray[U] = sa(a).map(f)

  /** Elementwise `f` over four broadcast arrays. */
  private def map4[A, B, C, D, U](a: NDArray[A], b: NDArray[B], c: NDArray[C], d: NDArray[D])(
      f: (A, B, C, D) => U
  )(using u: DType[U]): NDArray[U] =
    val sh = Shape.broadcast(a.shapeArr, b.shapeArr, c.shapeArr, d.shapeArr)
    val xa = a.broadcastTo(sh.toSeq*).toArray
    val xb = b.broadcastTo(sh.toSeq*).toArray
    val xc = c.broadcastTo(sh.toSeq*).toArray
    val xd = d.broadcastTo(sh.toSeq*).toArray
    val out = u.newArray(xa.length)
    var i = 0
    while i < out.length do
      out(i) = f(xa(i), xb(i), xc(i), xd(i))
      i += 1
    NDArray.fromArray(out, sh)

  // ------------------------------------------------------------------ construction

  /** `np.char.array(list_of_str)`: a new 1-D string array. */
  def array(obj: Seq[String]): NDArray[String] = NDArray.fromArray(obj.toArray)

  /** `np.char.array(list_of_str, itemsize=n)`: strings truncated to `itemsize` characters. */
  def array(obj: Seq[String], itemsize: Int): NDArray[String] =
    if itemsize < 0 then throw new IllegalArgumentException("itemsize must be non-negative")
    NDArray.fromArray(obj.map(s => StringsPy.slice(s, None, Some(itemsize.toLong), None)).toArray)

  /** `np.char.array(arr)`: a copy of a string array. */
  def array(obj: NDArray[String]): NDArray[String] = sa(obj).copy()

  /** `np.char.asarray(list_of_str)`. */
  def asarray(obj: Seq[String]): NDArray[String] = array(obj)

  /** `np.char.asarray(arr)`: the array itself (no copy). */
  def asarray(obj: NDArray[String]): NDArray[String] = sa(obj)

  // ------------------------------------------------------------------ concatenation / formatting

  /** `np.strings.add`: elementwise concatenation. */
  def add(x1: StrLike, x2: StrLike): NDArray[String] = NDArray.zipMap(sa(x1), sa(x2))(_ + _)

  /** `np.strings.multiply`: each string repeated `i` times (`""` for `i <= 0`). */
  def multiply(a: StrLike, i: IntLike): NDArray[String] =
    NDArray.zipMap(sa(a), ia(i))((s, n) => StringsPy.repeat(s, n))

  private def argOf[T](a: NDArray[T]): Int => StringsFormat.Arg =
    val d = a.dtype.asInstanceOf[DType[Any]]
    val arr: Array[T] = a.toArray
    i => StringsFormat.Arg(d, arr(i))

  /** `np.strings.mod`: Python `%` formatting, `a[i] % values[i]` (one value per element). */
  def mod(a: StrLike, values: NDArray[?]): NDArray[String] =
    val s = sa(a)
    val sh = Shape.broadcast(s.shapeArr, values.shapeArr)
    val fmts = s.broadcastTo(sh.toSeq*).toArray
    val vals = argOf(values.broadcastTo(sh.toSeq*))
    NDArray.fromArray(Array.tabulate(fmts.length)(i => StringsFormat.format(fmts(i), IndexedSeq(vals(i)), false)), sh)

  /** `np.strings.mod` with a tuple of values per element: `a[i] % (v0[i], v1[i], ...)`. */
  def mod(a: StrLike, values: Seq[NDArray[?]]): NDArray[String] =
    val s = sa(a)
    val sh = Shape.broadcast((s.shapeArr +: values.map(_.shapeArr))*)
    val fmts = s.broadcastTo(sh.toSeq*).toArray
    val vals = values.map(v => argOf(v.broadcastTo(sh.toSeq*))).toIndexedSeq
    NDArray.fromArray(
      Array.tabulate(fmts.length)(i => StringsFormat.format(fmts(i), vals.map(_(i)), true)),
      sh
    )

  /** `np.strings.join`: `sep` inserted between the characters of each string. */
  def join(sep: StrLike, seq: StrLike): NDArray[String] = NDArray.zipMap(sa(sep), sa(seq))(StringsPy.join)

  // ------------------------------------------------------------------ case conversion

  /** `np.strings.capitalize`: first character title-cased, the rest lower-cased. */
  def capitalize(a: StrLike): NDArray[String] = map1(a)(StringsPy.capitalize)
  /** `np.strings.lower`. */
  def lower(a: StrLike): NDArray[String] = map1(a)(StringsPy.lower)
  /** `np.strings.upper`. */
  def upper(a: StrLike): NDArray[String] = map1(a)(StringsPy.upper)
  /** `np.strings.swapcase`. */
  def swapcase(a: StrLike): NDArray[String] = map1(a)(StringsPy.swapcase)
  /** `np.strings.title`: words start upper-case, remaining cased characters lower-case. */
  def title(a: StrLike): NDArray[String] = map1(a)(StringsPy.title)

  // ------------------------------------------------------------------ padding

  /** `np.strings.center(a, width, fillchar=' ')`. */
  def center(a: StrLike, width: IntLike, fillchar: StrLike = " "): NDArray[String] =
    NDArray.zipMap3(sa(a), ia(width), sa(fillchar))(StringsPy.center)

  /** `np.strings.ljust(a, width, fillchar=' ')`. */
  def ljust(a: StrLike, width: IntLike, fillchar: StrLike = " "): NDArray[String] =
    NDArray.zipMap3(sa(a), ia(width), sa(fillchar))(StringsPy.ljust)

  /** `np.strings.rjust(a, width, fillchar=' ')`. */
  def rjust(a: StrLike, width: IntLike, fillchar: StrLike = " "): NDArray[String] =
    NDArray.zipMap3(sa(a), ia(width), sa(fillchar))(StringsPy.rjust)

  /** `np.strings.zfill`: left-pads with zeros, after a leading sign. */
  def zfill(a: StrLike, width: IntLike): NDArray[String] = NDArray.zipMap(sa(a), ia(width))(StringsPy.zfill)

  /** `np.strings.expandtabs(a, tabsize=8)`. */
  def expandtabs(a: StrLike, tabsize: IntLike = 8): NDArray[String] =
    NDArray.zipMap(sa(a), ia(tabsize))(StringsPy.expandtabs)

  // ------------------------------------------------------------------ stripping

  private def stripWith(a: StrLike, chars: StrLike | Null, l: Boolean, r: Boolean): NDArray[String] =
    if chars == null then map1(a)(s => StringsPy.strip(s, null, l, r))
    else NDArray.zipMap(sa(a), sa(chars.asInstanceOf[StrLike]))((s, c) => StringsPy.strip(s, c, l, r))

  /** `np.strings.strip(a, chars=None)`: removes leading and trailing whitespace (or `chars`). */
  def strip(a: StrLike, chars: StrLike | Null = null): NDArray[String] = stripWith(a, chars, true, true)
  /** `np.strings.lstrip(a, chars=None)`. */
  def lstrip(a: StrLike, chars: StrLike | Null = null): NDArray[String] = stripWith(a, chars, true, false)
  /** `np.strings.rstrip(a, chars=None)`. */
  def rstrip(a: StrLike, chars: StrLike | Null = null): NDArray[String] = stripWith(a, chars, false, true)

  // ------------------------------------------------------------------ partition / split

  /** `np.strings.partition`: `(before, sep, after)` around the first `sep`, as three arrays. */
  def partition(a: StrLike, sep: StrLike): (NDArray[String], NDArray[String], NDArray[String]) =
    partition3(a, sep, StringsPy.partition)

  /** `np.strings.rpartition`: `(before, sep, after)` around the last `sep`, as three arrays. */
  def rpartition(a: StrLike, sep: StrLike): (NDArray[String], NDArray[String], NDArray[String]) =
    partition3(a, sep, StringsPy.rpartition)

  private def partition3(a: StrLike, sep: StrLike, f: (String, String) => (String, String, String)) =
    val x = sa(a)
    val y = sa(sep)
    val sh = Shape.broadcast(x.shapeArr, y.shapeArr)
    val xs = x.broadcastTo(sh.toSeq*).toArray
    val ys = y.broadcastTo(sh.toSeq*).toArray
    val res = Array.tabulate(xs.length)(i => f(xs(i), ys(i)))
    (
      NDArray.fromArray(res.map(_._1), sh.clone()),
      NDArray.fromArray(res.map(_._2), sh.clone()),
      NDArray.fromArray(res.map(_._3), sh.clone())
    )

  /** `np.strings.split(a, sep=None, maxsplit=-1)`: a list of words per element. */
  def split(a: StrLike, sep: String | Null = null, maxsplit: Int = -1): ObjectArray[List[String]] =
    val x = sa(a)
    val xs = x.toArray
    objOf(x, i => StringsPy.split(xs(i), sep, maxsplit.toLong))

  /** `np.strings.rsplit(a, sep=None, maxsplit=-1)`: splits from the right. */
  def rsplit(a: StrLike, sep: String | Null = null, maxsplit: Int = -1): ObjectArray[List[String]] =
    val x = sa(a)
    val xs = x.toArray
    objOf(x, i => StringsPy.rsplit(xs(i), sep, maxsplit.toLong))

  /** `np.strings.splitlines(a, keepends=False)`. */
  def splitlines(a: StrLike, keepends: Boolean = false): ObjectArray[List[String]] =
    val x = sa(a)
    val xs = x.toArray
    objOf(x, i => StringsPy.splitlines(xs(i), keepends))

  // ------------------------------------------------------------------ replace / translate / slice

  /** `np.strings.replace(a, old, new, count=-1)`: replaces (at most `count`) occurrences. */
  def replace(a: StrLike, old: StrLike, `new`: StrLike, count: IntLike = -1): NDArray[String] =
    map4(sa(a), sa(old), sa(`new`), ia(count))(StringsPy.replace)

  /** `np.strings.translate(a, table)`: maps characters through `table` (`""` deletes). */
  def translate(a: StrLike, table: Map[Char, String | Char]): NDArray[String] =
    map1(a)(s => StringsPy.translate(s, table))

  /** Python `str.maketrans(x, y, z)`: `x(i) -> y(i)`, characters of `z` deleted. */
  def maketrans(x: String, y: String, z: String): Map[Char, String | Char] =
    if x.length != y.length then
      throw new IllegalArgumentException("the first two maketrans arguments must have equal length")
    x.zip(y).toMap[Char, String | Char] ++ z.map(c => c -> ("": String | Char))

  /** `np.strings.slice(a, stop)`: `a[i][:stop]`. */
  def slice(a: StrLike, stop: SliceArg): NDArray[String] = slice(a, None, stop, None)

  /** `np.strings.slice(a, start, stop, step=None)`: Python slicing of every element. */
  def slice(a: StrLike, start: SliceArg, stop: SliceArg, step: SliceArg = None): NDArray[String] =
    map4(sa(a), sl(start), sl(stop), sl(step))((s, b, e, st) => StringsPy.slice(s, opt(b), opt(e), opt(st)))

  // ------------------------------------------------------------------ searching

  /** `np.strings.str_len`: length in characters (code points). */
  def str_len(a: StrLike): NDArray[Int] = map1(a)(StringsPy.len)

  /** `np.strings.count(a, sub, start=0, end=None)`: non-overlapping occurrences. */
  def count(a: StrLike, sub: StrLike, start: IntLike = 0, end: IntLike = Long.MaxValue): NDArray[Int] =
    map4(sa(a), sa(sub), ia(start), ia(end))(StringsPy.count)

  /** `np.strings.find(a, sub, start=0, end=None)`: lowest index of `sub`, or -1. */
  def find(a: StrLike, sub: StrLike, start: IntLike = 0, end: IntLike = Long.MaxValue): NDArray[Int] =
    map4(sa(a), sa(sub), ia(start), ia(end))(StringsPy.find)

  /** `np.strings.rfind(a, sub, start=0, end=None)`: highest index of `sub`, or -1. */
  def rfind(a: StrLike, sub: StrLike, start: IntLike = 0, end: IntLike = Long.MaxValue): NDArray[Int] =
    map4(sa(a), sa(sub), ia(start), ia(end))(StringsPy.rfind)

  private def notFound(i: Int): Int =
    if i < 0 then throw new IllegalArgumentException("substring not found") else i

  /** `np.strings.index`: like `find` but throws when `sub` is not found. */
  def index(a: StrLike, sub: StrLike, start: IntLike = 0, end: IntLike = Long.MaxValue): NDArray[Int] =
    map4(sa(a), sa(sub), ia(start), ia(end))((s, u, b, e) => notFound(StringsPy.find(s, u, b, e)))

  /** `np.strings.rindex`: like `rfind` but throws when `sub` is not found. */
  def rindex(a: StrLike, sub: StrLike, start: IntLike = 0, end: IntLike = Long.MaxValue): NDArray[Int] =
    map4(sa(a), sa(sub), ia(start), ia(end))((s, u, b, e) => notFound(StringsPy.rfind(s, u, b, e)))

  /** `np.strings.startswith(a, prefix, start=0, end=None)`. */
  def startswith(a: StrLike, prefix: StrLike, start: IntLike = 0, end: IntLike = Long.MaxValue): NDArray[Boolean] =
    map4(sa(a), sa(prefix), ia(start), ia(end))(StringsPy.startswith)

  /** `np.strings.endswith(a, suffix, start=0, end=None)`. */
  def endswith(a: StrLike, suffix: StrLike, start: IntLike = 0, end: IntLike = Long.MaxValue): NDArray[Boolean] =
    map4(sa(a), sa(suffix), ia(start), ia(end))(StringsPy.endswith)

  // ------------------------------------------------------------------ predicates

  /** `np.strings.isalnum`. */
  def isalnum(a: StrLike): NDArray[Boolean] = map1(a)(StringsPy.isalnum)
  /** `np.strings.isalpha`. */
  def isalpha(a: StrLike): NDArray[Boolean] = map1(a)(StringsPy.isalpha)
  /** `np.strings.isdecimal`: Unicode category Nd only. */
  def isdecimal(a: StrLike): NDArray[Boolean] = map1(a)(StringsPy.isdecimal)
  /** `np.strings.isdigit`: decimals plus digit-valued characters such as superscripts. */
  def isdigit(a: StrLike): NDArray[Boolean] = map1(a)(StringsPy.isdigit)
  /** `np.strings.islower`. */
  def islower(a: StrLike): NDArray[Boolean] = map1(a)(StringsPy.islower)
  /** `np.strings.isnumeric`: any numeric character (fractions, roman numerals, CJK numerals). */
  def isnumeric(a: StrLike): NDArray[Boolean] = map1(a)(StringsPy.isnumeric)
  /** `np.strings.isspace`. */
  def isspace(a: StrLike): NDArray[Boolean] = map1(a)(StringsPy.isspace)
  /** `np.strings.istitle`. */
  def istitle(a: StrLike): NDArray[Boolean] = map1(a)(StringsPy.istitle)
  /** `np.strings.isupper`. */
  def isupper(a: StrLike): NDArray[Boolean] = map1(a)(StringsPy.isupper)

  // ------------------------------------------------------------------ comparison

  private def cmpWith(x1: StrLike, x2: StrLike)(p: Int => Boolean): NDArray[Boolean] =
    NDArray.zipMap(sa(x1), sa(x2))((a, b) => p(StringsPy.compare(a, b)))

  /** `np.strings.equal`. */
  def equal(x1: StrLike, x2: StrLike): NDArray[Boolean] = NDArray.zipMap(sa(x1), sa(x2))(_ == _)
  /** `np.strings.not_equal`. */
  def not_equal(x1: StrLike, x2: StrLike): NDArray[Boolean] = NDArray.zipMap(sa(x1), sa(x2))(_ != _)
  /** `np.strings.less` (code-point order). */
  def less(x1: StrLike, x2: StrLike): NDArray[Boolean] = cmpWith(x1, x2)(_ < 0)
  /** `np.strings.less_equal`. */
  def less_equal(x1: StrLike, x2: StrLike): NDArray[Boolean] = cmpWith(x1, x2)(_ <= 0)
  /** `np.strings.greater`. */
  def greater(x1: StrLike, x2: StrLike): NDArray[Boolean] = cmpWith(x1, x2)(_ > 0)
  /** `np.strings.greater_equal`. */
  def greater_equal(x1: StrLike, x2: StrLike): NDArray[Boolean] = cmpWith(x1, x2)(_ >= 0)

  /** `np.char.compare_chararrays(a1, a2, cmp, rstrip)`; `cmp` is one of `== != < <= > >=`. */
  def compare_chararrays(a1: StrLike, a2: StrLike, cmp: String, rstrip: Boolean): NDArray[Boolean] =
    val p: Int => Boolean = cmp match
      case "==" => _ == 0
      case "!=" => _ != 0
      case "<" => _ < 0
      case "<=" => _ <= 0
      case ">" => _ > 0
      case ">=" => _ >= 0
      case _ => throw new IllegalArgumentException("comparison must be '==', '!=', '<', '>', '<=', '>='")
    val x = sa(a1)
    val y = sa(a2)
    if x.shapeArr.toSeq != y.shapeArr.toSeq then throw new IllegalArgumentException("Incompatible shapes")
    val f: String => String = if rstrip then StringsPy.rstripWs else identity
    NDArray.zipMap(x, y)((a, b) => p(StringsPy.compare(f(a), f(b))))

  // ------------------------------------------------------------------ encode / decode

  private def charset(encoding: String): Charset =
    encoding.trim.toLowerCase.replace('_', '-') match
      case "utf-8" | "utf8" | "u8" => java.nio.charset.StandardCharsets.UTF_8
      case "ascii" | "us-ascii" | "646" => java.nio.charset.StandardCharsets.US_ASCII
      case "latin-1" | "latin1" | "iso-8859-1" | "iso8859-1" | "l1" => java.nio.charset.StandardCharsets.ISO_8859_1
      case "utf-16-le" | "utf-16le" => java.nio.charset.StandardCharsets.UTF_16LE
      case "utf-16-be" | "utf-16be" => java.nio.charset.StandardCharsets.UTF_16BE
      case other =>
        try Charset.forName(other)
        catch case _: Exception => throw new IllegalArgumentException(s"unknown encoding: $encoding")

  private def action(errors: String): CodingErrorAction = errors match
    case "strict" => CodingErrorAction.REPORT
    case "ignore" => CodingErrorAction.IGNORE
    case "replace" => CodingErrorAction.REPLACE
    case other => throw new IllegalArgumentException(s"unknown error handler name '$other'")

  /** `np.strings.encode(a, encoding='utf-8', errors='strict')`: the bytes of every element. */
  def encode(a: StrLike, encoding: String = "utf-8", errors: String = "strict"): ObjectArray[Array[Byte]] =
    val cs = charset(encoding)
    val act = action(errors)
    val x = sa(a)
    val xs = x.toArray
    objOf(
      x,
      i =>
        val enc = cs.newEncoder().onMalformedInput(act).onUnmappableCharacter(act)
        try
          val bb = enc.encode(CharBuffer.wrap(xs(i)))
          val out = new Array[Byte](bb.remaining())
          bb.get(out)
          out
        catch
          case e: CharacterCodingException =>
            throw new IllegalArgumentException(s"'$encoding' codec can't encode ${StringsPy.repr(xs(i))}: $e")
    )

  /** `np.strings.decode(a, encoding='utf-8', errors='strict')`: bytes back to strings. */
  def decode(a: ObjectArray[Array[Byte]], encoding: String = "utf-8", errors: String = "strict"): NDArray[String] =
    val cs = charset(encoding)
    val act = action(errors)
    val out = a.toSeq.map { b =>
      val dec = cs.newDecoder().onMalformedInput(act).onUnmappableCharacter(act)
      try dec.decode(ByteBuffer.wrap(b)).toString
      catch
        case e: CharacterCodingException =>
          throw new IllegalArgumentException(s"'$encoding' codec can't decode bytes: $e")
    }
    NDArray.fromArray(out.toArray, a.shape.toArray)

  /** `np.strings.decode` for a plain sequence of byte strings (1-D result). */
  def decode(a: Seq[Array[Byte]]): NDArray[String] = decode(a, "utf-8", "strict")

  /** `np.strings.decode` for a plain sequence of byte strings with an explicit codec. */
  def decode(a: Seq[Array[Byte]], encoding: String, errors: String): NDArray[String] =
    decode(new ObjectArray(a.toIndexedSeq, Array(a.length)), encoding, errors)
