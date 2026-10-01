package numscala

/** `numpy.testing`: assertion helpers for array-valued tests. Failures throw `AssertionError`
  * with NumPy-style messages (mismatch counts, max absolute/relative difference).
  */
object Testing:

  private def fail(header: String, errMsg: String, lines: Seq[String]): Nothing =
    val msg = (Seq("", header) ++ (if errMsg.nonEmpty then Seq(errMsg) else Nil) ++ lines).mkString("\n")
    throw new AssertionError(msg)

  private def toD[T](a: NDArray[T]): NDArray[Double] = a.asType[Double]

  private def shapesMatch(a: NDArray[?], b: NDArray[?], header: String, errMsg: String): (NDArray[?], NDArray[?]) =
    if a.shape == b.shape then (a, b)
    else if a.ndim == 0 then (a.broadcastTo(b.shape*), b)
    else if b.ndim == 0 then (a, b.broadcastTo(a.shape*))
    else fail(header, errMsg, Seq(s"(shapes ${Shape.str(a.shape)}, ${Shape.str(b.shape)} mismatch)"))

  /** `np.testing.assert_allclose`: `|actual - desired| <= atol + rtol * |desired|` elementwise. */
  def assert_allclose[A, B](
      actual: NDArray[A],
      desired: NDArray[B],
      rtol: Double = 1e-7,
      atol: Double = 0.0,
      equal_nan: Boolean = true,
      err_msg: String = ""
  ): Unit =
    val header = s"Not equal to tolerance rtol=${Format.formatFloatShort(rtol)}, atol=${Format.formatFloatShort(atol)}"
    val (a0, b0) = shapesMatch(actual, desired, header, err_msg)
    if a0.dtype.isComplex || b0.dtype.isComplex then
      val a = a0.asInstanceOf[NDArray[Any]].astypeDyn(DType.Complex128).asInstanceOf[NDArray[Complex]].toArray
      val b = b0.asInstanceOf[NDArray[Any]].astypeDyn(DType.Complex128).asInstanceOf[NDArray[Complex]].toArray
      val bad = a.indices.filter { i =>
        val bothNan = a(i).isNaN && b(i).isNaN
        !(bothNan && equal_nan) && !((a(i) - b(i)).abs <= atol + rtol * b(i).abs)
      }
      if bad.nonEmpty then
        fail(header, err_msg, Seq(s"Mismatched elements: ${bad.length} / ${a.length}", s"ACTUAL: ${a0.repr}", s"DESIRED: ${b0.repr}"))
    else
      val a = toD(a0.asInstanceOf[NDArray[Any]]).toArray
      val b = toD(b0.asInstanceOf[NDArray[Any]]).toArray
      checkClose(a, b, rtol, atol, equal_nan, header, err_msg, a0, b0)

  def assert_allclose(actual: Double, desired: Double): Unit =
    assert_allclose(NDArray.scalar(actual), NDArray.scalar(desired))

  private def checkClose(
      a: Array[Double],
      b: Array[Double],
      rtol: Double,
      atol: Double,
      equalNan: Boolean,
      header: String,
      errMsg: String,
      a0: NDArray[?],
      b0: NDArray[?]
  ): Unit =
    var bad = 0
    var maxAbs = 0.0
    var maxRel = 0.0
    var i = 0
    while i < a.length do
      val x = a(i)
      val y = b(i)
      val ok =
        if x.isNaN || y.isNaN then equalNan && x.isNaN && y.isNaN
        else if x.isInfinite || y.isInfinite then x == y
        else math.abs(x - y) <= atol + rtol * math.abs(y)
      if !ok then
        bad += 1
        val d = math.abs(x - y)
        if !d.isNaN then
          maxAbs = math.max(maxAbs, d)
          if y != 0.0 then maxRel = math.max(maxRel, d / math.abs(y))
      i += 1
    if bad > 0 then
      val pct = g3(100.0 * bad / a.length)
      fail(
        header,
        errMsg,
        Seq(
          s"Mismatched elements: $bad / ${a.length} ($pct%)",
          s"Max absolute difference among violations: ${Format.formatFloatShort(maxAbs)}",
          s"Max relative difference among violations: ${Format.formatFloatShort(maxRel)}",
          s" ACTUAL: ${a0.repr}",
          s" DESIRED: ${b0.repr}"
        )
      )

  /** `np.testing.assert_array_equal`: same shape (after scalar broadcasting) and equal elements; NaNs compare equal. */
  def assert_array_equal[A, B](actual: NDArray[A], desired: NDArray[B], err_msg: String = "", strict: Boolean = false): Unit =
    val header = "Arrays are not equal"
    if strict && (actual.shape != desired.shape || (actual.dtype ne desired.dtype)) then
      fail(header, err_msg, Seq(s"(shapes ${Shape.str(actual.shape)}, ${Shape.str(desired.shape)} or dtypes ${actual.dtype}, ${desired.dtype} mismatch)"))
    val (a0, b0) = shapesMatch(actual, desired, header, err_msg)
    val a = a0.toArray
    val b = b0.toArray
    var bad = 0
    var i = 0
    val da = a0.dtype.asInstanceOf[DType[Any]]
    val db = b0.dtype.asInstanceOf[DType[Any]]
    while i < a.length do
      val x: Any = a(i)
      val y: Any = b(i)
      val eq =
        if da.isString || db.isString then x == y
        else if da.isComplex || db.isComplex then
          val cx = da.toComplex(x)
          val cy = db.toComplex(y)
          cx == cy || (cx.isNaN && cy.isNaN)
        else
          val dx = da.toDouble(x)
          val dy = db.toDouble(y)
          dx == dy || (dx.isNaN && dy.isNaN)
      if !eq then bad += 1
      i += 1
    if bad > 0 then
      fail(header, err_msg, Seq(s"Mismatched elements: $bad / ${a.length}", s" ACTUAL: ${a0.repr}", s" DESIRED: ${b0.repr}"))

  /** `np.testing.assert_equal` for arrays (delegates to `assert_array_equal`) and plain values. */
  def assert_equal(actual: Any, desired: Any, err_msg: String = ""): Unit =
    (actual, desired) match
      case (a: NDArray[?], b: NDArray[?]) => assert_array_equal(a, b, err_msg)
      case (a: Double, b: Double) if a.isNaN && b.isNaN => ()
      case (a, b) =>
        if a != b then fail("Items are not equal:", err_msg, Seq(s" ACTUAL: $a", s" DESIRED: $b"))

  /** `np.testing.assert_array_almost_equal`: `|actual - desired| < 1.5 * 10**(-decimal)`. */
  def assert_array_almost_equal[A, B](actual: NDArray[A], desired: NDArray[B], decimal: Int = 6, err_msg: String = ""): Unit =
    val header = s"Arrays are not almost equal to $decimal decimals"
    val (a0, b0) = shapesMatch(actual, desired, header, err_msg)
    val a = toD(a0.asInstanceOf[NDArray[Any]]).toArray
    val b = toD(b0.asInstanceOf[NDArray[Any]]).toArray
    val tol = 1.5 * math.pow(10.0, -decimal)
    val bad = a.indices.count { i =>
      val x = a(i)
      val y = b(i)
      if x.isNaN || y.isNaN then !(x.isNaN && y.isNaN)
      else if x.isInfinite || y.isInfinite then x != y
      else !(math.abs(x - y) < tol)
    }
    if bad > 0 then fail(header, err_msg, Seq(s"Mismatched elements: $bad / ${a.length}", s" ACTUAL: ${a0.repr}", s" DESIRED: ${b0.repr}"))

  /** `np.testing.assert_almost_equal` for scalars. */
  def assert_almost_equal(actual: Double, desired: Double, decimal: Int = 7, err_msg: String = ""): Unit =
    val ok =
      if actual.isNaN || desired.isNaN then actual.isNaN && desired.isNaN
      else if actual.isInfinite || desired.isInfinite then actual == desired
      else math.abs(desired - actual) < 1.5 * math.pow(10.0, -decimal)
    if !ok then fail(s"Arrays are not almost equal to $decimal decimals", err_msg, Seq(s" ACTUAL: ${Format.formatFloatShort(actual)}", s" DESIRED: ${Format.formatFloatShort(desired)}"))

  /** `np.testing.assert_approx_equal`: equal to `significant` significant digits. */
  def assert_approx_equal(actual: Double, desired: Double, significant: Int = 7, err_msg: String = ""): Unit =
    if actual == desired then return
    val scale = 0.5 * (math.abs(desired) + math.abs(actual))
    val sc = if scale == 0.0 then 1.0 else math.pow(10.0, math.floor(math.log10(scale)))
    val ok = math.abs(desired / sc - actual / sc) < math.pow(10.0, -(significant - 1))
    if !ok then fail(s"Items are not equal to $significant significant digits:", err_msg, Seq(s" ACTUAL: $actual", s" DESIRED: $desired"))

  /** `np.testing.assert_array_less`: strictly `x < y` elementwise. */
  def assert_array_less[A, B](x: NDArray[A], y: NDArray[B], err_msg: String = ""): Unit =
    val header = "Arrays are not less-ordered"
    val (a0, b0) = shapesMatch(x, y, header, err_msg)
    val a = toD(a0.asInstanceOf[NDArray[Any]]).toArray
    val b = toD(b0.asInstanceOf[NDArray[Any]]).toArray
    val bad = a.indices.count(i => !(a(i) < b(i)) && !(a(i).isNaN && b(i).isNaN))
    if bad > 0 then fail(header, err_msg, Seq(s"Mismatched elements: $bad / ${a.length}", s" x: ${a0.repr}", s" y: ${b0.repr}"))

  /** `np.testing.assert_array_max_ulp`: returns the max ULP distance, failing above `maxulp`. */
  def assert_array_max_ulp(a: NDArray[Double], b: NDArray[Double], maxulp: Int = 1): NDArray[Double] =
    val ulps = NDArray.zipMap(a, b)((x, y) => ulpDistance(x, y).toDouble)
    val worst = if ulps.size == 0 then 0.0 else ulps.max()
    if worst > maxulp then
      fail(s"Arrays are not almost equal up to $maxulp ULP (max difference is ${worst.toLong} ULP)", "", Nil)
    ulps

  /** `np.testing.assert_array_almost_equal_nulp`. */
  def assert_array_almost_equal_nulp(x: NDArray[Double], y: NDArray[Double], nulp: Int = 1): Unit =
    val ok = NDArray.zipMap(x, y) { (a, b) =>
      val ref = nulp * math.max(math.ulp(a), math.ulp(b))
      math.abs(a - b) <= ref
    }
    if !ok.all() then fail(s"Arrays are not equal to $nulp ULP", "", Nil)

  /** `np.testing.assert_string_equal`. */
  def assert_string_equal(actual: String, desired: String): Unit =
    if actual != desired then fail("Differences in strings:", "", Seq(s"- $desired", s"+ $actual"))

  /** `np.testing.assert_raises`: runs `body` expecting an exception of type `E`. */
  def assert_raises[E <: Throwable](body: => Any)(using ct: scala.reflect.ClassTag[E]): E =
    try
      body
      throw new AssertionError(s"${ct.runtimeClass.getSimpleName} not raised")
    catch
      case e: AssertionError if !ct.runtimeClass.isInstance(e) => throw e
      case e: Throwable if ct.runtimeClass.isInstance(e) => e.asInstanceOf[E]

  /** Python's `'{:.3g}'` for non-negative values below 1e3. */
  private def g3(x: Double): String =
    if x == 0.0 then "0"
    else
      val bd = new java.math.BigDecimal(x).round(new java.math.MathContext(3)).stripTrailingZeros()
      bd.toPlainString

  private def ulpDistance(a: Double, b: Double): Long =
    if a.isNaN || b.isNaN then Long.MaxValue
    else
      def ordered(d: Double): Long =
        val bits = java.lang.Double.doubleToLongBits(d)
        if bits < 0 then Long.MinValue - bits else bits
      math.abs(ordered(a) - ordered(b))
