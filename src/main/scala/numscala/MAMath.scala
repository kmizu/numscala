package numscala

/** `np.ma` elementwise functions. Like NumPy's domained ufuncs, invalid inputs (e.g. the log
  * of a non-positive number, division by zero) and non-finite results are masked instead of
  * producing NaN/inf; masked entries keep their original data.
  */
trait MAMath:
  import MA.toMA

  // ------------------------------------------------------------------ domains

  /** Lexicographic `<` on complex (and real) values, like NumPy's `less` on complex. */
  private def ltC(c: Complex, v: Double): Boolean = c.re < v || (c.re == v && c.im < 0)
  private def gtC(c: Complex, v: Double): Boolean = c.re > v || (c.re == v && c.im > 0)
  private def leC(c: Complex, v: Double): Boolean = c.re < v || (c.re == v && c.im <= 0)

  private def uf[T](x: MaskedArrayLike[T], t: ToInexact[T])(f: Double => Double, fc: Complex => Complex,
      domain: Option[Complex => Boolean]): MaskedArray[t.Out] =
    val m = toMA(x)
    val od = t.dtype
    val src = m.dtype
    val g = od.lift(f, fc)
    MaskedArray.unary(m, od, (v: T) => g(od.castFrom(src, v)), domain.map(p => (v: T) => p(src.toComplex(v))))

  private val ln2 = math.log(2.0)
  private val ln10 = math.log(10.0)

  // ------------------------------------------------------------------ unary, domained

  /** `np.ma.sqrt`: masks negative inputs. */
  def sqrt[T](x: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] =
    uf(x, t)(math.sqrt, _.sqrt, Some(ltC(_, 0.0)))
  /** `np.ma.log`: masks non-positive inputs. */
  def log[T](x: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] =
    uf(x, t)(math.log, _.log, Some(leC(_, 0.0)))
  /** `np.ma.log2`. */
  def log2[T](x: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] =
    uf(x, t)(v => math.log(v) / ln2, _.log / ln2, Some(leC(_, 0.0)))
  /** `np.ma.log10`. */
  def log10[T](x: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] =
    uf(x, t)(math.log10, _.log / ln10, Some(leC(_, 0.0)))
  /** `np.ma.arcsin`: masks inputs outside [-1, 1]. */
  def arcsin[T](x: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] =
    uf(x, t)(math.asin, _.asin, Some(c => ltC(c, -1.0) || gtC(c, 1.0)))
  /** `np.ma.arccos`: masks inputs outside [-1, 1]. */
  def arccos[T](x: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] =
    uf(x, t)(math.acos, _.acos, Some(c => ltC(c, -1.0) || gtC(c, 1.0)))
  /** `np.ma.arccosh`: masks inputs below 1. */
  def arccosh[T](x: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] =
    uf(x, t)(v => math.log(v + math.sqrt(v * v - 1.0)), _.acosh, Some(ltC(_, 1.0)))
  /** `np.ma.arctanh`: masks inputs outside (-1, 1). */
  def arctanh[T](x: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] =
    uf(x, t)(v => 0.5 * math.log((1.0 + v) / (1.0 - v)), _.atanh, Some(c => ltC(c, -1.0 + 1e-15) || gtC(c, 1.0 - 1e-15)))
  /** `np.ma.tan`: masks inputs where `|cos(x)| < 1e-35`. */
  def tan[T](x: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] =
    uf(x, t)(math.tan, _.tan, Some(c => c.cos.abs < 1e-35))

  // ------------------------------------------------------------------ unary, no domain

  /** `np.ma.exp`. */
  def exp[T](x: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] = uf(x, t)(math.exp, _.exp, None)
  /** `np.ma.sin`. */
  def sin[T](x: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] = uf(x, t)(math.sin, _.sin, None)
  /** `np.ma.cos`. */
  def cos[T](x: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] = uf(x, t)(math.cos, _.cos, None)
  /** `np.ma.arctan`. */
  def arctan[T](x: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] = uf(x, t)(math.atan, _.atan, None)
  /** `np.ma.sinh`. */
  def sinh[T](x: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] = uf(x, t)(math.sinh, _.sinh, None)
  /** `np.ma.cosh`. */
  def cosh[T](x: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] = uf(x, t)(math.cosh, _.cosh, None)
  /** `np.ma.tanh`. */
  def tanh[T](x: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] = uf(x, t)(math.tanh, _.tanh, None)
  /** `np.ma.arcsinh`. */
  def arcsinh[T](x: MaskedArrayLike[T])(using t: ToInexact[T]): MaskedArray[t.Out] =
    uf(x, t)(v => if v.isInfinite then v else math.log(v + math.sqrt(v * v + 1.0)), _.asinh, None)

  /** `np.ma.absolute` (complex -> float64). */
  def absolute[T](x: MaskedArrayLike[T])(using a: AbsOf[T]): MaskedArray[a.Out] =
    val m = toMA(x)
    val src = m.dtype
    val out = a.dtype
    val f: T => a.Out = src match
      case _: ComplexDType => v => out.fromDouble(src.toComplex(v).abs)
      case r: RealDType[T] => v => out.castFrom(src, r.abs(v))
      case _ => v => out.castFrom(src, v)
    MaskedArray.unary(m, out, f, None)
  /** `np.ma.abs` (alias of [[absolute]]). */
  def abs[T](x: MaskedArrayLike[T])(using a: AbsOf[T]): MaskedArray[a.Out] = absolute(x)
  /** `np.ma.fabs`: absolute value as float. */
  def fabs[T](x: MaskedArrayLike[T])(using t: ToFloat[T]): MaskedArray[t.Out] = floatUnary(x, t)(math.abs)
  /** `np.ma.floor`. */
  def floor[T](x: MaskedArrayLike[T])(using t: ToFloat[T]): MaskedArray[t.Out] = floatUnary(x, t)(math.floor)
  /** `np.ma.ceil`. */
  def ceil[T](x: MaskedArrayLike[T])(using t: ToFloat[T]): MaskedArray[t.Out] = floatUnary(x, t)(math.ceil)
  /** `np.ma.negative`. */
  def negative[T](x: MaskedArrayLike[T])(using d: NumDType[T]): MaskedArray[T] = MaskedArray.unary(toMA(x), d, d.negate, None)
  /** `np.ma.conjugate`. */
  def conjugate[T](x: MaskedArrayLike[T]): MaskedArray[T] =
    val m = toMA(x)
    MaskedArray.unary(m, m.dtype, (v: T) => if m.dtype.isComplex then v.asInstanceOf[Complex].conj.asInstanceOf[T] else v, None)
  /** `np.ma.around(a, decimals)`: rounds the data, keeping the mask. */
  def around[T](a: MaskedArrayLike[T], decimals: Int = 0)(using d: NumDType[T]): MaskedArray[T] = toMA(a).round(decimals)
  /** `np.ma.round` (alias of [[around]]). */
  def round[T](a: MaskedArrayLike[T], decimals: Int = 0)(using d: NumDType[T]): MaskedArray[T] = toMA(a).round(decimals)
  /** `np.ma.logical_not`. */
  def logical_not[T](x: MaskedArrayLike[T]): MaskedArray[Boolean] =
    val m = toMA(x)
    MaskedArray.unary(m, DType.Bool, (v: T) => !m.dtype.toBoolean(v), None)

  private def floatUnary[T](x: MaskedArrayLike[T], t: ToFloat[T])(f: Double => Double): MaskedArray[t.Out] =
    val m = toMA(x)
    val od = t.dtype
    val src = m.dtype
    MaskedArray.unary(m, od, (v: T) => od.fromDouble(f(src.toDouble(v))), None)

  // ------------------------------------------------------------------ binary

  /** `np.ma.add`. */
  def add[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using p: NumPromote[A, B]): MaskedArray[p.Out] =
    MaskedArray.arith(toMA(a), toMA(b), p.dtype, Arith.Add)
  /** `np.ma.subtract`. */
  def subtract[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using p: NumPromote[A, B]): MaskedArray[p.Out] =
    MaskedArray.arith(toMA(a), toMA(b), p.dtype, Arith.Sub)
  /** `np.ma.multiply`. */
  def multiply[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using p: NumPromote[A, B]): MaskedArray[p.Out] =
    MaskedArray.arith(toMA(a), toMA(b), p.dtype, Arith.Mul)
  /** `np.ma.divide`: masks division by zero and non-finite results. */
  def divide[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using p: DivPromote[A, B]): MaskedArray[p.Out] =
    MaskedArray.domained(toMA(a), toMA(b), p.dtype, Arith.Div)
  /** `np.ma.true_divide` (alias of [[divide]]). */
  def true_divide[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using p: DivPromote[A, B]): MaskedArray[p.Out] =
    divide(a, b)
  /** `np.ma.floor_divide`: masks division by zero. */
  def floor_divide[T](a: MaskedArrayLike[T], b: MaskedArrayLike[T])(using d: RealDType[T]): MaskedArray[T] =
    MaskedArray.domainedF(toMA(a), toMA(b), d)(d.floorDiv)
  /** `np.ma.remainder`: masks division by zero. */
  def remainder[T](a: MaskedArrayLike[T], b: MaskedArrayLike[T])(using d: RealDType[T]): MaskedArray[T] =
    MaskedArray.domainedF(toMA(a), toMA(b), d)(d.mod)
  /** `np.ma.mod` (alias of [[remainder]]). */
  def mod[T](a: MaskedArrayLike[T], b: MaskedArrayLike[T])(using d: RealDType[T]): MaskedArray[T] = remainder(a, b)
  /** `np.ma.fmod`: C-style remainder, masking division by zero. */
  def fmod[T](a: MaskedArrayLike[T], b: MaskedArrayLike[T])(using d: RealDType[T]): MaskedArray[T] =
    MaskedArray.domainedF(toMA(a), toMA(b), d)(d.fmod)
  /** `np.ma.power`: masks non-finite results (their data is set to the fill value). */
  def power[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using p: NumPromote[A, B]): MaskedArray[p.Out] =
    MaskedArray.power(toMA(a), toMA(b), p.dtype)
  /** `np.ma.maximum` (elementwise). */
  def maximum[T](a: MaskedArrayLike[T], b: MaskedArrayLike[T])(using d: RealDType[T]): MaskedArray[T] =
    MaskedArray.binaryWith(toMA(a), toMA(b), d)((x, y) => NDArray.zipMap(x, y)(d.max))
  /** `np.ma.minimum` (elementwise). */
  def minimum[T](a: MaskedArrayLike[T], b: MaskedArrayLike[T])(using d: RealDType[T]): MaskedArray[T] =
    MaskedArray.binaryWith(toMA(a), toMA(b), d)((x, y) => NDArray.zipMap(x, y)(d.min))
  /** `np.ma.hypot`. */
  def hypot[T](a: MaskedArrayLike[T], b: MaskedArrayLike[T])(using t: ToFloat[T]): MaskedArray[t.Out] = floatBinary(a, b, t)(math.hypot)
  /** `np.ma.arctan2`. */
  def arctan2[T](a: MaskedArrayLike[T], b: MaskedArrayLike[T])(using t: ToFloat[T]): MaskedArray[t.Out] = floatBinary(a, b, t)(math.atan2)

  private def floatBinary[T](a: MaskedArrayLike[T], b: MaskedArrayLike[T], t: ToFloat[T])(f: (Double, Double) => Double): MaskedArray[t.Out] =
    val od = t.dtype
    val (ma, mb) = (toMA(a), toMA(b))
    val src = ma.dtype
    MaskedArray.binaryWith(ma, mb, od)((x, y) => NDArray.zipMap(x, y)((u, v) => od.fromDouble(f(src.toDouble(u), src.toDouble(v))))(using od))

  private def cmp[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B], d: DType[?], op: CmpOp): MaskedArray[Boolean] =
    MaskedArray.binaryWith(toMA(a), toMA(b), DType.Bool)((x, y) => Ops.compare(x, y, d, op))
  private def eqOp[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B], d: DType[?], eq: Boolean): MaskedArray[Boolean] =
    MaskedArray.binaryWith(toMA(a), toMA(b), DType.Bool)((x, y) => Ops.equal(x, y, d, eq))

  /** `np.ma.equal` (a masked binary ufunc: masked where either input is masked). */
  def equal[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using p: Promote[A, B]): MaskedArray[Boolean] = eqOp(a, b, p.dtype, true)
  /** `np.ma.not_equal`. */
  def not_equal[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using p: Promote[A, B]): MaskedArray[Boolean] = eqOp(a, b, p.dtype, false)
  /** `np.ma.less`. */
  def less[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using p: Promote[A, B]): MaskedArray[Boolean] = cmp(a, b, p.dtype, CmpOp.Lt)
  /** `np.ma.less_equal`. */
  def less_equal[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using p: Promote[A, B]): MaskedArray[Boolean] = cmp(a, b, p.dtype, CmpOp.Le)
  /** `np.ma.greater`. */
  def greater[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using p: Promote[A, B]): MaskedArray[Boolean] = cmp(a, b, p.dtype, CmpOp.Gt)
  /** `np.ma.greater_equal`. */
  def greater_equal[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using p: Promote[A, B]): MaskedArray[Boolean] = cmp(a, b, p.dtype, CmpOp.Ge)

  private def logical[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(f: (Boolean, Boolean) => Boolean): MaskedArray[Boolean] =
    val (ma, mb) = (toMA(a), toMA(b))
    val (da, db) = (ma.dtype, mb.dtype)
    MaskedArray.binaryWith(ma, mb, DType.Bool)((x, y) => NDArray.zipMap(x, y)((u, v) => f(da.toBoolean(u), db.toBoolean(v)))(using DType.Bool))

  /** `np.ma.logical_and`. */
  def logical_and[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B]): MaskedArray[Boolean] = logical(a, b)(_ && _)
  /** `np.ma.logical_or`. */
  def logical_or[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B]): MaskedArray[Boolean] = logical(a, b)(_ || _)
  /** `np.ma.logical_xor`. */
  def logical_xor[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B]): MaskedArray[Boolean] = logical(a, b)(_ ^ _)

  // ------------------------------------------------------------------ mask hardness

  /** `np.ma.harden_mask(a)`. */
  def harden_mask[T](a: MaskedArray[T]): MaskedArray[T] = a.harden_mask()
  /** `np.ma.soften_mask(a)`. */
  def soften_mask[T](a: MaskedArray[T]): MaskedArray[T] = a.soften_mask()
