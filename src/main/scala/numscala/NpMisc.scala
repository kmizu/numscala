package numscala

/** Miscellaneous NumPy routines: constants, closeness/equality tests, dtype introspection
  * (`finfo`, `iinfo`, `result_type`, `can_cast`, `issubdtype`, ...), window functions and
  * memory-overlap queries. Printing helpers come from [[NpMiscPrint]].
  */
trait NpMisc extends NpMiscPrint:
  // ------------------------------------------------------------------ constants

  /** `np.pi`. */
  val pi: Double = math.Pi
  /** `np.e`. */
  val e: Double = math.E
  /** `np.inf`. */
  val inf: Double = Double.PositiveInfinity
  /** `np.nan`. */
  val nan: Double = Double.NaN
  /** `np.euler_gamma` (Euler–Mascheroni constant). */
  val euler_gamma: Double = 0.5772156649015329
  /** `np.newaxis`: inserts a length-one axis when indexing. */
  val newaxis: Index = Index.NewAxis
  /** `np.PINF` (positive infinity). */
  val PINF: Double = Double.PositiveInfinity
  /** `np.NINF` (negative infinity). */
  val NINF: Double = Double.NegativeInfinity
  /** `np.PZERO` (`0.0`). */
  val PZERO: Double = 0.0
  /** `np.NZERO` (`-0.0`). */
  val NZERO: Double = -0.0
  /** `np.NAN`. */
  val NAN: Double = Double.NaN

  // ------------------------------------------------------------------ abstract scalar types

  /** `np.generic`. */
  val generic: DTypeCategory = DTypeCategory.generic
  /** `np.number`. */
  val number: DTypeCategory = DTypeCategory.number
  /** `np.integer`. */
  val integer: DTypeCategory = DTypeCategory.integer
  /** `np.signedinteger`. */
  val signedinteger: DTypeCategory = DTypeCategory.signedinteger
  /** `np.unsignedinteger` (no num-scala dtype belongs to it). */
  val unsignedinteger: DTypeCategory = DTypeCategory.unsignedinteger
  /** `np.inexact`. */
  val inexact: DTypeCategory = DTypeCategory.inexact
  /** `np.floating`. */
  val floating: DTypeCategory = DTypeCategory.floating
  /** `np.complexfloating`. */
  val complexfloating: DTypeCategory = DTypeCategory.complexfloating
  /** `np.bool_`. */
  val bool_ : DTypeCategory = DTypeCategory.bool_
  /** `np.flexible`. */
  val flexible: DTypeCategory = DTypeCategory.flexible
  /** `np.character`. */
  val character: DTypeCategory = DTypeCategory.character
  /** `np.str_`. */
  val str_ : DTypeCategory = DTypeCategory.str_

  /** `np.typecodes`. */
  val typecodes: Map[String, String] = Map(
    "Character" -> "c",
    "Integer" -> "bhilqnp",
    "UnsignedInteger" -> "BHILQNP",
    "Float" -> "efdg",
    "Complex" -> "FDG",
    "AllInteger" -> "bBhHiIlLqQnNpP",
    "AllFloat" -> "efdgFDG",
    "Datetime" -> "Mm",
    "All" -> "?bhilqnpBHILQNPefdgFDGSUVOMm"
  )

  // ------------------------------------------------------------------ closeness / equality

  private def arr[A, E](a: A)(using n: Nested[A, E]): NDArray[E] = a match
    case x: NDArray[?] => x.asInstanceOf[NDArray[E]]
    case _ =>
      val shape = n.shapeOf(a).toArray
      val out = n.dtype.newArray(Shape.size(shape))
      n.write(a, out, 0)
      NDArray.fromArray(out, shape)(using n.dtype)

  /** `np.isclose`: elementwise `|a - b| <= atol + rtol * |b|` with broadcasting; infinities
    * are close only to themselves, NaNs only when `equal_nan`.
    */
  def isclose[A, EA, B, EB](a: A, b: B, rtol: Double = 1e-5, atol: Double = 1e-8, equal_nan: Boolean = false)(using
      na: Nested[A, EA],
      nb: Nested[B, EB]
  ): NDArray[Boolean] =
    val x = arr(a)
    val y = arr(b)
    val dx = x.dtype
    val dy = y.dtype
    if dx.isString || dy.isString then throw new IllegalArgumentException("isclose is not defined for str arrays")
    if dx.isComplex || dy.isComplex then
      NDArray.zipMap(x, y) { (p, q) =>
        val u = dx.toComplex(p)
        val v = dy.toComplex(q)
        if u.isNaN || v.isNaN then equal_nan && u.isNaN && v.isNaN
        else if u.isInfinite || v.isInfinite then u == v
        else (u - v).abs <= atol + rtol * v.abs
      }
    else
      NDArray.zipMap(x, y) { (p, q) =>
        val u = dx.toDouble(p)
        val v = dy.toDouble(q)
        if u.isNaN || v.isNaN then equal_nan && u.isNaN && v.isNaN
        else if u.isInfinite || v.isInfinite then u == v
        else math.abs(u - v) <= atol + rtol * math.abs(v)
      }

  /** `np.allclose`: true if every element pair is close (see [[isclose]]). */
  def allclose[A, EA, B, EB](a: A, b: B, rtol: Double = 1e-5, atol: Double = 1e-8, equal_nan: Boolean = false)(using
      na: Nested[A, EA],
      nb: Nested[B, EB]
  ): Boolean = isclose(a, b, rtol, atol, equal_nan).toArray.forall(identity)

  private def elemEq[A, B](da: DType[A], x: A, db: DType[B], y: B, equalNan: Boolean): Boolean =
    if da.isString || db.isString then
      da.isString && db.isString && x == y
    else if da.isComplex || db.isComplex then
      val u = da.toComplex(x)
      val v = db.toComplex(y)
      if u.isNaN || v.isNaN then equalNan && u.isNaN && v.isNaN else u == v
    else if da.isFloating || db.isFloating then
      val u = da.toDouble(x)
      val v = db.toDouble(y)
      if u.isNaN || v.isNaN then equalNan && u.isNaN && v.isNaN else u == v
    else da.toLong(x) == db.toLong(y)

  /** `np.array_equal`: same shape and equal elements (NaNs equal only with `equal_nan`). */
  def array_equal[A, EA, B, EB](a1: A, a2: B, equal_nan: Boolean = false)(using
      na: Nested[A, EA],
      nb: Nested[B, EB]
  ): Boolean =
    val x = arr(a1)
    val y = arr(a2)
    java.util.Arrays.equals(x.shapeArr, y.shapeArr) && {
      val xs = x.toArray
      val ys = y.toArray
      xs.indices.forall(i => elemEq(x.dtype, xs(i), y.dtype, ys(i), equal_nan))
    }

  /** `np.array_equiv`: shapes broadcast together and all elements are equal. */
  def array_equiv[A, EA, B, EB](a1: A, a2: B)(using na: Nested[A, EA], nb: Nested[B, EB]): Boolean =
    val x = arr(a1)
    val y = arr(a2)
    val ok =
      try
        Shape.broadcast(x.shapeArr, y.shapeArr)
        true
      catch case _: IllegalArgumentException => false
    ok && NDArray.zipMap(x, y)((p, q) => elemEq(x.dtype, p, y.dtype, q, false))(using DType.Bool).toArray.forall(identity)

  // ------------------------------------------------------------------ dtype introspection

  /** `np.dtype("float64")`: looks a dtype up by name or type code. */
  def dtype(spec: String | DType[?]): DType[?] = NpMiscTypes.resolve(spec)

  /** `np.finfo`: machine limits of a float or complex dtype (complex reports its real part type). */
  def finfo(dtype: DType[?] | String): FInfo =
    val d = NpMiscTypes.resolve(dtype)
    if (d eq DType.Float64) || (d eq DType.Complex128) then NpMiscTypes.F64
    else if d eq DType.Float32 then NpMiscTypes.F32
    else throw new IllegalArgumentException(s"data type $d not inexact")

  /** `np.iinfo`: machine limits of an integer dtype. */
  def iinfo(dtype: DType[?] | String): IInfo =
    NpMiscTypes.resolve(dtype) match
      case d: IntDType[?] => IInfo(d, d.bits, d.toLong(d.minValue), d.toLong(d.maxValue))
      case d => throw new IllegalArgumentException(s"Invalid integer data type '${d.kind}'.")

  /** `np.promote_types`: the smallest dtype to which both can be safely cast. */
  def promote_types(type1: DType[?] | String, type2: DType[?] | String): DType[?] =
    NpMiscTypes.promote(NpMiscTypes.resolve(type1), NpMiscTypes.resolve(type2))

  /** `np.result_type`: promotion of arrays, dtypes (strong) and Scala scalars (weak, like Python scalars). */
  def result_type(arrays_and_dtypes: Any*): DType[?] = NpMiscTypes.resultType(arrays_and_dtypes)

  /** `np.can_cast(from, to, casting)` with `casting` one of `no`, `equiv`, `safe`, `same_kind`, `unsafe`. */
  def can_cast(from_ : DType[?] | String | NDArray[?], to: DType[?] | String, casting: String = "safe"): Boolean =
    val f = from_ match
      case a: NDArray[?] => a.dtype
      case other => NpMiscTypes.resolve(other.asInstanceOf[DType[?] | String])
    NpMiscTypes.canCast(f, NpMiscTypes.resolve(to), casting)

  /** `np.min_scalar_type`: the smallest dtype that can hold a scalar's value (arrays: their dtype).
    * num-scala has no unsigned or 16-bit float types, so the closest signed/float32 type is returned.
    */
  def min_scalar_type(a: Any): DType[?] = NpMiscTypes.minScalarType(a)

  /** `np.isscalar`: true for numbers, booleans, complex values and strings (not arrays). */
  def isscalar(element: Any): Boolean = element match
    case _: Boolean | _: Byte | _: Short | _: Int | _: Long | _: Float | _: Double | _: Complex | _: String | _: Char =>
      true
    case _ => false

  /** `np.issubdtype`: whether `arg1` is lower in the type hierarchy than `arg2`. */
  def issubdtype(arg1: DType[?] | DTypeCategory | String, arg2: DType[?] | DTypeCategory | String): Boolean =
    (NpMiscTypes.category(arg1), NpMiscTypes.category(arg2)) match
      case (Left(d1), Left(d2)) => d1 eq d2
      case (Left(d1), Right(c2)) => DTypeCategory.of(d1) <:< c2
      case (Right(c1), Right(c2)) => c1 <:< c2
      case (Right(_), Left(_)) => false

  // ------------------------------------------------------------------ windows

  private def symWindow(M: Int)(f: (Double, Double) => Double): NDArray[Double] =
    if M < 1 then NDArray.fromArray(Array.emptyDoubleArray)
    else if M == 1 then NDArray.fromArray(Array(1.0))
    else
      val m1 = (M - 1).toDouble
      NDArray.fromArray(Array.tabulate(M)(i => f((1 - M + 2 * i).toDouble, m1)))

  /** `np.bartlett`: triangular window of `M` points. */
  def bartlett(M: Int): NDArray[Double] =
    symWindow(M)((n, m1) => if n <= 0 then 1 + n / m1 else 1 - n / m1)

  /** `np.blackman`: Blackman window of `M` points. */
  def blackman(M: Int): NDArray[Double] =
    symWindow(M)((n, m1) => 0.42 + 0.5 * math.cos(math.Pi * n / m1) + 0.08 * math.cos(2.0 * math.Pi * n / m1))

  /** `np.hamming`: Hamming window of `M` points. */
  def hamming(M: Int): NDArray[Double] =
    symWindow(M)((n, m1) => 0.54 + 0.46 * math.cos(math.Pi * n / m1))

  /** `np.hanning`: Hann window of `M` points. */
  def hanning(M: Int): NDArray[Double] =
    symWindow(M)((n, m1) => 0.5 + 0.5 * math.cos(math.Pi * n / m1))

  /** Modified Bessel function of the first kind, order 0 (power series; terms are all positive). */
  private def besselI0(x: Double): Double =
    val y = x * x / 4.0
    var term = 1.0
    var sum = 1.0
    var k = 1
    while term > sum * 1e-17 && k < 2000 && !sum.isInfinite do
      term *= y / (k.toDouble * k)
      sum += term
      k += 1
    sum

  /** `np.kaiser`: Kaiser window of `M` points with shape parameter `beta`. */
  def kaiser(M: Int, beta: Double): NDArray[Double] =
    if M < 1 then NDArray.fromArray(Array.emptyDoubleArray)
    else if M == 1 then NDArray.fromArray(Array(1.0))
    else
      val alpha = (M - 1) / 2.0
      val denom = besselI0(beta)
      NDArray.fromArray(Array.tabulate(M) { n =>
        val r = (n - alpha) / alpha
        besselI0(beta * math.sqrt(1 - r * r)) / denom
      })

  // ------------------------------------------------------------------ memory

  private def extent(a: NDArray[?]): (Int, Int) =
    var lo = a.offset
    var hi = a.offset
    var k = 0
    while k < a.ndim do
      val span = (a.shapeArr(k) - 1) * a.stridesArr(k)
      if span < 0 then lo += span else hi += span
      k += 1
    (lo, hi)

  /** `np.may_share_memory`: true if the arrays use the same buffer and their memory bounds overlap. */
  def may_share_memory(a: NDArray[?], b: NDArray[?]): Boolean =
    (a.data.asInstanceOf[AnyRef] eq b.data.asInstanceOf[AnyRef]) && a.size > 0 && b.size > 0 && {
      val (l1, h1) = extent(a)
      val (l2, h2) = extent(b)
      l1 <= h2 && l2 <= h1
    }

  /** `np.shares_memory`: exact test whether some element is shared by both arrays. */
  def shares_memory(a: NDArray[?], b: NDArray[?]): Boolean =
    may_share_memory(a, b) && {
      val (small, big) = if a.size <= b.size then (a, b) else (b, a)
      val set = new java.util.HashSet[Integer]()
      small.foreachOffset(o => set.add(o))
      var found = false
      big.foreachOffset(o => if !found && set.contains(o) then found = true)
      found
    }
