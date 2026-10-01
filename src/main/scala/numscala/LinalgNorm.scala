package numscala

/** Implementation of `np.linalg.norm`, `vector_norm` and `matrix_norm`. */
private[numscala] object LinalgNorm:
  private val F64 = DType.Float64
  type Ord = Option[Either[String, Double]]

  def ordOf(o: Linalg.NormOrd): Ord = o match
    case None => None
    case s: String => Some(Left(s))
    case i: Int => Some(Right(i.toDouble))
    case d: Double => Some(Right(d))

  private def ordStr(o: Ord): String = o match
    case None => "None"
    case Some(Left(s)) => s
    case Some(Right(d)) => if d == math.rint(d) && !d.isInfinite then d.toLong.toString else d.toString

  /** |x| as float64. */
  def absD(x: NDArray[?]): NDArray[Double] =
    val d = x.dtype.asInstanceOf[DType[Any]]
    val xa = x.asInstanceOf[NDArray[Any]]
    if d.isComplex then xa.map(v => d.toComplex(v).abs)(using F64)
    else xa.map(v => math.abs(d.toDouble(v)))(using F64)

  /** |x|^2 as float64 (re^2 + im^2 for complex). */
  def sqAbsD(x: NDArray[?]): NDArray[Double] =
    val d = x.dtype.asInstanceOf[DType[Any]]
    val xa = x.asInstanceOf[NDArray[Any]]
    if d.isComplex then xa.map(v => { val c = d.toComplex(v); c.re * c.re + c.im * c.im })(using F64)
    else xa.map(v => { val r = d.toDouble(v); r * r })(using F64)

  private def maxNaN(b: Array[Double], n: Int, op: String): Double =
    if n == 0 then throw new IllegalArgumentException(s"zero-size array to reduction operation $op which has no identity")
    var r = b(0)
    var i = 1
    while i < n do
      val v = b(i)
      if v.isNaN || r.isNaN then r = Double.NaN
      else if op == "maximum" && v > r then r = v
      else if op == "minimum" && v < r then r = v
      i += 1
    r

  /** Vector norm of order `ord` of a lane of absolute values. */
  def vecLane(ord: Double)(b: Array[Double], n: Int): Double =
    if ord == Double.PositiveInfinity then maxNaN(b, n, "maximum")
    else if ord == Double.NegativeInfinity then maxNaN(b, n, "minimum")
    else if ord == 0.0 then
      var c = 0
      var i = 0
      while i < n do
        if b(i) != 0.0 then c += 1
        i += 1
      c.toDouble
    else if ord == 1.0 then Reduce.pairwiseSum(b, 0, n)
    else if ord == 2.0 then
      var s = 0.0
      var i = 0
      while i < n do
        s += b(i) * b(i)
        i += 1
      math.sqrt(s)
    else
      var s = 0.0
      var i = 0
      while i < n do
        s += math.pow(b(i), ord)
        i += 1
      math.pow(s, 1.0 / ord)

  private def fullTwoNorm(x: NDArray[?], keepdims: Boolean): NDArray[Double] =
    val sq = sqAbsD(x).toArray
    var s = 0.0
    var i = 0
    while i < sq.length do
      s += sq(i)
      i += 1
    val shape = if keepdims then Array.fill(x.ndim)(1) else Array.emptyIntArray
    NDArray.fromArray(Array(math.sqrt(s)), shape)

  private def axesOf(axis: Axis | None.type, ndim: Int): Seq[Int] = axis match
    case None => 0 until ndim
    case i: Int => Seq(i)
    case s: Seq[?] => s.asInstanceOf[Seq[Int]]

  def norm(x: NDArray[?], ord0: Linalg.NormOrd, axis: Axis | None.type, keepdims: Boolean): NDArray[Double] =
    val ord = ordOf(ord0)
    val nd = x.ndim
    if axis == None then
      val plain = ord match
        case None => true
        case Some(Left("fro" | "f")) => nd == 2
        case Some(Right(2.0)) => nd == 1
        case _ => false
      if plain then return fullTwoNorm(x, keepdims)
    val axes = axesOf(axis, nd)
    axes.length match
      case 1 => vectorAlong(x, axes, keepdims, ord)
      case 2 => matrixNorm(x, axes(0), axes(1), ord, keepdims)
      case _ => throw new IllegalArgumentException("Improper number of dimensions to norm.")

  private def vectorAlong(x: NDArray[?], axes: Seq[Int], keepdims: Boolean, ord: Ord): NDArray[Double] =
    val p = ord match
      case None => 2.0
      case Some(Right(v)) => v
      case Some(Left(s)) => throw new IllegalArgumentException(s"Invalid norm order '$s' for vectors")
    val ax: Axis = if axes.length == 1 then axes.head else axes
    if x.ndim == 0 then
      throw new IllegalArgumentException(s"axis ${axes.head} is out of bounds for array of dimension 0")
    Lanes.reduce(absD(x), ax, keepdims)((b, n) => vecLane(p)(b, n))(using F64)

  def vectorNorm(x: NDArray[?], axis: Axis | None.type, keepdims: Boolean, ord0: Linalg.NormOrd): NDArray[Double] =
    val ord = ordOf(ord0) match
      case None => Some(Right(2.0))
      case o => o
    axis match
      case None =>
        val p = ord match
          case Some(Right(v)) => v
          case _ => throw new IllegalArgumentException(s"Invalid norm order '${ordStr(ord)}' for vectors")
        val a = absD(x).toArray
        val shape = if keepdims then Array.fill(x.ndim)(1) else Array.emptyIntArray
        NDArray.fromArray(Array(vecLane(p)(a, a.length)), shape)
      case _ => vectorAlong(x, axesOf(axis, x.ndim), keepdims, ord)

  /** Singular values of each matrix of a C-contiguous (batch, r, c) stack. */
  private def stackSvals(m: NDArray[?], nb: Int, r: Int, c: Int): Array[Array[Double]] =
    if m.dtype.isComplex then
      val (re, im) = LinalgSupport.complexData(m)
      Array.tabulate(nb) { b =>
        val z = ZMat(r, c, java.util.Arrays.copyOfRange(re, b * r * c, (b + 1) * r * c), java.util.Arrays.copyOfRange(im, b * r * c, (b + 1) * r * c))
        LinalgCplx.svd(z, false, false).s
      }
    else
      val d = LinalgSupport.realData(m)
      Array.tabulate(nb)(b => LinalgReal.svd(java.util.Arrays.copyOfRange(d, b * r * c, (b + 1) * r * c), r, c, false, false).s)

  def matrixNorm(x: NDArray[?], r0: Int, c0: Int, ord: Ord, keepdims: Boolean): NDArray[Double] =
    val nd = x.ndim
    val ra = Shape.normAxis(r0, nd)
    val ca = Shape.normAxis(c0, nd)
    if ra == ca then throw new IllegalArgumentException("Duplicate axes given.")
    val others = (0 until nd).filter(i => i != ra && i != ca)
    val moved = x.asInstanceOf[NDArray[Any]].transpose((others :+ ra :+ ca)*)
    val r = x.shapeArr(ra)
    val c = x.shapeArr(ca)
    val bs = others.map(x.shapeArr(_)).toArray
    val nb = Shape.size(bs)
    val out = new Array[Double](nb)
    def withAbs(f: (Array[Double], Int) => Double): Unit =
      val a = absD(moved).toArray
      var b = 0
      while b < nb do
        out(b) = f(a, b * r * c)
        b += 1
    // sums over rows (per column) or over columns (per row), then max/min
    def lineSums(a: Array[Double], off: Int, perColumn: Boolean): Array[Double] =
      if perColumn then
        Array.tabulate(c) { j =>
          var s = 0.0
          var i = 0
          while i < r do
            s += a(off + i * c + j)
            i += 1
          s
        }
      else
        Array.tabulate(r) { i =>
          var s = 0.0
          var j = 0
          while j < c do
            s += a(off + i * c + j)
            j += 1
          s
        }
    ord match
      case None | Some(Left("fro" | "f")) =>
        val a = sqAbsD(moved).toArray
        var b = 0
        while b < nb do
          var s = 0.0
          var i = 0
          while i < r * c do
            s += a(b * r * c + i)
            i += 1
          out(b) = math.sqrt(s)
          b += 1
      case Some(Right(1.0)) => withAbs((a, off) => { val s = lineSums(a, off, true); maxNaN(s, s.length, "maximum") })
      case Some(Right(-1.0)) => withAbs((a, off) => { val s = lineSums(a, off, true); maxNaN(s, s.length, "minimum") })
      case Some(Right(Double.PositiveInfinity)) =>
        withAbs((a, off) => { val s = lineSums(a, off, false); maxNaN(s, s.length, "maximum") })
      case Some(Right(Double.NegativeInfinity)) =>
        withAbs((a, off) => { val s = lineSums(a, off, false); maxNaN(s, s.length, "minimum") })
      case Some(Right(2.0)) | Some(Right(-2.0)) | Some(Left("nuc")) =>
        val sv = stackSvals(moved.contiguous, nb, r, c)
        var b = 0
        while b < nb do
          val s = sv(b)
          out(b) = ord match
            case Some(Right(2.0)) => maxNaN(s, s.length, "maximum")
            case Some(Right(-2.0)) => maxNaN(s, s.length, "minimum")
            case _ => s.sum
          b += 1
      case _ => throw new IllegalArgumentException("Invalid norm order for matrices.")
    val shape =
      if keepdims then (0 until nd).map(i => if i == ra || i == ca then 1 else x.shapeArr(i)).toArray
      else bs
    NDArray.fromArray(out, shape)

  /** Whether each matrix (last two axes) of `x` contains a NaN. */
  def matrixHasNaN(x: NDArray[?]): Array[Boolean] =
    val (re, im) =
      if x.dtype.isComplex then LinalgSupport.complexData(x)
      else
        val r = LinalgSupport.realData(x)
        (r, r)
    val per = x.shapeArr(x.ndim - 1) * x.shapeArr(x.ndim - 2)
    val nb = Shape.size(x.shapeArr.dropRight(2))
    Array.tabulate(nb) { b =>
      var found = false
      var i = 0
      while !found && i < per do
        if re(b * per + i).isNaN || im(b * per + i).isNaN then found = true
        i += 1
      found
    }
