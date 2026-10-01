package com.github.kmizu.numscala

import NpShapeOps.*

/** `np.pad`. */
trait NpShapePad:

  /** Pads an array (`np.pad`).
    *
    * `pad_width` is an `Int`, a `(before, after)` pair, a `Seq` of per-axis pairs (or one pair
    * broadcast to every axis), or a `Seq` of one or two ints.  `mode` is one of `"constant"`,
    * `"edge"`, `"linear_ramp"`, `"maximum"`, `"mean"`, `"median"`, `"minimum"`, `"reflect"`,
    * `"symmetric"`, `"wrap"`, `"empty"`.  `constant_values` / `end_values` accept the same forms
    * as `pad_width` (with values of any numeric type); `stat_length` likewise, or `None` for the
    * whole axis.  `reflect_type` is `"even"` or `"odd"`.
    */
  def pad[T](
      array: NDArray[T],
      pad_width: Int | (Int, Int) | Seq[?],
      mode: String = "constant",
      constant_values: Any = 0,
      end_values: Any = 0,
      stat_length: Int | (Int, Int) | Seq[?] | None.type = None,
      reflect_type: String = "even"
  ): NDArray[T] =
    val modes = Set("constant", "edge", "linear_ramp", "maximum", "mean", "median", "minimum", "reflect",
      "symmetric", "wrap", "empty")
    if !modes(mode) then throw new IllegalArgumentException(s"mode '$mode' is not supported")
    if reflect_type != "even" && reflect_type != "odd" then
      throw new IllegalArgumentException(s"unsupported reflect_type '$reflect_type'")
    val nd = array.ndim
    val d = array.dtype
    val pw = PadPairs(pad_width, nd).map((b, a) => (toNonNegInt(b), toNonNegInt(a)))
    val inShape = array.shapeArr
    val outShape = Array.tabulate(nd)(i => inShape(i) + pw(i)._1 + pw(i)._2)
    val padded = NDArray.zerosOf(d, outShape)
    // the original area
    var center = padded
    var i = 0
    while i < nd do
      center = sliceAxis(center, i, pw(i)._1, inShape(i))
      i += 1
    center := array
    if mode == "empty" then return padded
    if mode != "constant" then
      i = 0
      while i < nd do
        if inShape(i) == 0 && (pw(i)._1 > 0 || pw(i)._2 > 0) then
          throw new IllegalArgumentException(
            s"can't extend empty axis $i using modes other than 'constant' or 'empty'"
          )
        i += 1
      if array.size == 0 then return padded
    val cvals = if mode == "constant" then PadPairs(constant_values, nd).map((b, a) => (d.coerce(b), d.coerce(a))) else null
    val evals = if mode == "linear_ramp" then PadPairs(end_values, nd).map((b, a) => (toDouble(b), toDouble(a))) else null
    val slens: Array[(Int, Int)] = stat_length match
      case None => Array.tabulate(nd)(k => (inShape(k), inShape(k)))
      case other => PadPairs(other, nd).map((b, a) => (toNonNegInt(b), toNonNegInt(a)))
    var ax = 0
    while ax < nd do
      val (left, right) = pw(ax)
      if left > 0 || right > 0 then
        // region of interest: full extent on axes <= ax, original area on later axes
        var roi = padded
        var j = ax + 1
        while j < nd do
          roi = sliceAxis(roi, j, pw(j)._1, inShape(j))
          j += 1
        val n = inShape(ax)
        val fill: Array[T] => Unit = mode match
          case "constant" =>
            val (cb, ca) = cvals(ax)
            buf => PadLane.constant(buf, left, n, right, cb, ca)
          case "edge" => buf => PadLane.edge(buf, left, n, right)
          case "wrap" => buf => PadLane.wrap(buf, left, n, right)
          case "reflect" | "symmetric" =>
            if n == 1 then (buf => PadLane.edge(buf, left, n, right))
            else
              val num = if reflect_type == "odd" then numDType(d, "reflect_type='odd'") else null
              buf => PadLane.reflect(buf, left, n, right, mode == "symmetric", num)
          case "linear_ramp" =>
            val (eb, ea) = evals(ax)
            buf => PadLane.linearRamp(buf, left, n, right, eb, ea, d)
          case stat =>
            val (lb, la) = slens(ax)
            buf => PadLane.stat(buf, left, n, right, math.min(lb, n), math.min(la, n), stat, d)
        updateLanes(roi, ax)(fill)
      ax += 1
    padded

  private def toNonNegInt(v: Any): Int =
    val i = v match
      case i: Int => i
      case l: Long => l.toInt
      case s: Short => s.toInt
      case b: Byte => b.toInt
      case other => throw new IllegalArgumentException(s"`pad_width` must be of integral type (got $other)")
    if i < 0 then throw new IllegalArgumentException("index can't contain negative values")
    i

  private def toDouble(v: Any): Double = v match
    case i: Int => i.toDouble
    case l: Long => l.toDouble
    case d: Double => d
    case f: Float => f.toDouble
    case s: Short => s.toDouble
    case b: Byte => b.toDouble
    case other => throw new IllegalArgumentException(s"invalid end value $other")

  private def numDType[T](d: DType[T], what: String): NumDType[T] = d match
    case n: NumDType[?] => n.asInstanceOf[NumDType[T]]
    case _ => throw new IllegalArgumentException(s"$what requires a numeric array (got ${d.name})")

/** NumPy's `_as_pairs`: broadcasts a scalar / pair / per-axis sequence of pairs to `nd` pairs. */
private[numscala] object PadPairs:
  def apply(v: Any, nd: Int): Array[(Any, Any)] =
    def pairOf(x: Any): (Any, Any) = x match
      case (b, a) => (b, a)
      case s: Seq[?] if s.length == 2 && !s.head.isInstanceOf[Seq[?]] && !s.head.isInstanceOf[Tuple2[?, ?]] => (s(0), s(1))
      case s: Seq[?] if s.length == 1 && !s.head.isInstanceOf[Seq[?]] && !s.head.isInstanceOf[Tuple2[?, ?]] => (s(0), s(0))
      case s: Seq[?] => throw mismatch(s)
      case x => (x, x)
    def mismatch(s: Any) = new IllegalArgumentException(
      s"operands could not be broadcast together with remapped shapes [original->remapped]: $s and requested shape ($nd, 2)"
    )
    v match
      case s: Seq[?] if s.nonEmpty && s.forall(e => e.isInstanceOf[Seq[?]] || e.isInstanceOf[Tuple2[?, ?]]) =>
        if s.length == 1 then Array.fill(nd)(pairOf(s.head))
        else if s.length == nd then s.map(pairOf).toArray
        else throw mismatch(s)
      case s: Seq[?] =>
        if s.isEmpty then throw mismatch(s)
        Array.fill(nd)(pairOf(s))
      case x => Array.fill(nd)(pairOf(x))

/** One-dimensional padding kernels; `buf` holds `left` pad slots, `n` original values, `right` pad slots. */
private[numscala] object PadLane:
  def constant[T](buf: Array[T], left: Int, n: Int, right: Int, cb: T, ca: T): Unit =
    var i = 0
    while i < left do
      buf(i) = cb
      i += 1
    i = left + n
    while i < buf.length do
      buf(i) = ca
      i += 1

  def edge[T](buf: Array[T], left: Int, n: Int, right: Int): Unit =
    constant(buf, left, n, right, buf(left), buf(left + n - 1))

  def wrap[T](buf: Array[T], left: Int, n: Int, right: Int): Unit =
    var i = 0
    while i < left do
      buf(i) = buf(left + Math.floorMod(i - left, n))
      i += 1
    var j = 0
    while j < right do
      buf(left + n + j) = buf(left + j % n)
      j += 1

  /** Port of NumPy's iterative `_set_reflect_both` (`num` non-null selects `reflect_type='odd'`). */
  def reflect[T](buf: Array[T], left: Int, n: Int, right: Int, includeEdge: Boolean, num: NumDType[T] | Null): Unit =
    val len = buf.length
    var lp = left
    var rp = right
    val tmp = buf.clone()
    def odd(edgeV: T, v: T): T =
      val d = num.nn
      d.minus(d.plus(edgeV, edgeV), v)
    while lp > 0 || rp > 0 do
      val cur = len - rp - lp
      val (oldLen, eo) =
        if includeEdge then (cur / n * n, 1)
        else ((cur - 1) / (n - 1) * (n - 1), 0)
      if lp > 0 then
        val chunk = math.min(oldLen, lp)
        val start = lp - eo + chunk
        val edgeV = buf(lp)
        var j = 0
        while j < chunk do
          val v = buf(start - j)
          tmp(j) = if num == null then v else odd(edgeV, v)
          j += 1
        j = 0
        while j < chunk do
          buf(lp - chunk + j) = tmp(j)
          j += 1
        lp -= chunk
      if rp > 0 then
        val chunk = math.min(oldLen, rp)
        val start = len - rp + eo - 2
        val edgeV = buf(len - rp - 1)
        var j = 0
        while j < chunk do
          val v = buf(start - j)
          tmp(j) = if num == null then v else odd(edgeV, v)
          j += 1
        j = 0
        while j < chunk do
          buf(len - rp + j) = tmp(j)
          j += 1
        rp -= chunk

  def linearRamp[T](buf: Array[T], left: Int, n: Int, right: Int, eb: Double, ea: Double, d: DType[T]): Unit =
    val isInt = d.isInteger || d.isBool
    def conv(y: Double): T = if isInt then d.fromDouble(math.floor(y)) else d.fromDouble(y)
    // linspace(start=end, stop=edge, num=width, endpoint=False)
    def ramp(start: Double, stop: Double, num: Int)(put: (Int, T) => Unit): Unit =
      if num > 0 then
        val step = (stop - start) / num
        var i = 0
        while i < num do
          val y = if step == 0.0 then i.toDouble / num * (stop - start) + start else i * step + start
          put(i, conv(y))
          i += 1
    ramp(eb, d.toDouble(buf(left)), left)((i, v) => buf(i) = v)
    ramp(ea, d.toDouble(buf(left + n - 1)), right)((i, v) => buf(buf.length - 1 - i) = v)

  def stat[T](buf: Array[T], left: Int, n: Int, right: Int, lenB: Int, lenA: Int, mode: String, d: DType[T]): Unit =
    def compute(from: Int, len: Int): T =
      mode match
        case "maximum" | "minimum" =>
          if len == 0 then throw new IllegalArgumentException("stat_length of 0 yields no value for padding")
          val chunk = d.newArray(len)
          System.arraycopy(buf, from, chunk, 0, len)
          Reduce.extremeBuf(d, chunk, len, mode == "maximum")
        case "mean" => fromStat(d, meanOf(buf, from, len, d))
        case _ => fromStat(d, medianOf(buf, from, len, d))
    val vb = if left > 0 then compute(left, lenB) else buf(left)
    val va = if right > 0 then compute(left + n - lenA, lenA) else buf(left)
    constant(buf, left, n, right, vb, va)

  /** Casts a floating statistic back to `d`, rounding half-to-even for integer dtypes like NumPy. */
  private def fromStat[T](d: DType[T], v: Any): T = v match
    case c: Complex => d.fromComplex(c)
    case x: Double => if d.isInteger || d.isBool then d.fromDouble(math.rint(x)) else d.fromDouble(x)
    case other => d.coerce(other)

  private def meanOf[T](buf: Array[T], from: Int, len: Int, d: DType[T]): Any =
    if d.isComplex then
      var re = 0.0
      var im = 0.0
      var i = 0
      while i < len do
        val c = d.toComplex(buf(from + i))
        re += c.re
        im += c.im
        i += 1
      Complex(re / len, im / len)
    else
      val xs = Array.tabulate(len)(i => d.toDouble(buf(from + i)))
      Reduce.pairwiseSum(xs, 0, len) / len

  private def medianOf[T](buf: Array[T], from: Int, len: Int, d: DType[T]): Any =
    if len == 0 then Double.NaN
    else if d.isComplex then
      val xs = Array.tabulate(len)(i => d.toComplex(buf(from + i))).sorted(using Complex.ordering)
      if len % 2 == 1 then xs(len / 2) else (xs(len / 2 - 1) + xs(len / 2)) / Complex(2.0, 0.0)
    else
      val xs = Array.tabulate(len)(i => d.toDouble(buf(from + i)))
      if xs.exists(_.isNaN) then Double.NaN
      else
        java.util.Arrays.sort(xs)
        if len % 2 == 1 then xs(len / 2) else (xs(len / 2 - 1) + xs(len / 2)) / 2.0
