package numscala

/** An abstract scalar type of NumPy's type hierarchy (`np.number`, `np.integer`, `np.floating`, ...),
  * usable with [[NpMisc.issubdtype]].
  */
final class DTypeCategory private[numscala] (val name: String, val parent: Option[DTypeCategory]):
  /** True if this category is `other` or one of its descendants. */
  def <:<(other: DTypeCategory): Boolean = (this eq other) || parent.exists(_ <:< other)
  override def toString: String = s"<class 'numpy.$name'>"

object DTypeCategory:
  val generic: DTypeCategory = DTypeCategory("generic", None)
  val number: DTypeCategory = DTypeCategory("number", Some(generic))
  val integer: DTypeCategory = DTypeCategory("integer", Some(number))
  val signedinteger: DTypeCategory = DTypeCategory("signedinteger", Some(integer))
  val unsignedinteger: DTypeCategory = DTypeCategory("unsignedinteger", Some(integer))
  val inexact: DTypeCategory = DTypeCategory("inexact", Some(number))
  val floating: DTypeCategory = DTypeCategory("floating", Some(inexact))
  val complexfloating: DTypeCategory = DTypeCategory("complexfloating", Some(inexact))
  val bool_ : DTypeCategory = DTypeCategory("bool", Some(generic))
  val flexible: DTypeCategory = DTypeCategory("flexible", Some(generic))
  val character: DTypeCategory = DTypeCategory("character", Some(flexible))
  val str_ : DTypeCategory = DTypeCategory("str_", Some(character))

  /** The most specific category of a concrete dtype. */
  def of(d: DType[?]): DTypeCategory = d.kind match
    case 'b' => bool_
    case 'i' => signedinteger
    case 'u' => unsignedinteger
    case 'f' => floating
    case 'c' => complexfloating
    case _ => str_

/** Machine limits of a floating point type (`np.finfo`). Values are given as `Double`s
  * (float32 limits are exact float32 values widened to double).
  */
final case class FInfo(
    dtype: DType[?],
    bits: Int,
    eps: Double,
    epsneg: Double,
    max: Double,
    min: Double,
    tiny: Double,
    smallest_normal: Double,
    smallest_subnormal: Double,
    resolution: Double,
    precision: Int,
    machep: Int,
    negep: Int,
    maxexp: Int,
    minexp: Int,
    nexp: Int,
    nmant: Int
):
  /** `iexp` (same as `nexp`). */
  def iexp: Int = nexp
  private def s(v: Double): String = NpMiscFloatFmt.pyRepr(v, dtype eq DType.Float32)
  /** NumPy's `repr(np.finfo(...))`. */
  def repr: String = s"finfo(resolution=${s(resolution)}, min=${s(min)}, max=${s(max)}, dtype=$dtype)"
  /** NumPy's `str(np.finfo(...))`. */
  override def toString: String =
    s"""Machine parameters for $dtype
       |---------------------------------------------------------------
       |precision = $precision   resolution = ${s(resolution)}
       |machep = $machep   eps =        ${s(eps)}
       |negep =  $negep   epsneg =     ${s(epsneg)}
       |minexp = $minexp   tiny =       ${s(tiny)}
       |maxexp = $maxexp   max =        ${s(max)}
       |nexp =   $nexp   min =        -max
       |smallest_normal = ${s(smallest_normal)}   smallest_subnormal = ${s(smallest_subnormal)}
       |---------------------------------------------------------------
       |""".stripMargin

/** Machine limits of an integer type (`np.iinfo`). */
final case class IInfo(dtype: DType[?], bits: Int, min: BigInt, max: BigInt):
  /** The kind character, `'i'` or `'u'`. */
  def kind: Char = dtype.kind
  /** NumPy's `repr(np.iinfo(...))`. */
  def repr: String = s"iinfo(min=$min, max=$max, dtype=$dtype)"
  override def toString: String =
    s"""Machine parameters for $dtype
       |---------------------------------------------------------------
       |min = $min
       |max = $max
       |---------------------------------------------------------------
       |""".stripMargin

private[numscala] object NpMiscTypes:
  val F64: FInfo = FInfo(
    DType.Float64, 64, math.ulp(1.0), math.pow(2, -53), Double.MaxValue, -Double.MaxValue,
    java.lang.Double.MIN_NORMAL, java.lang.Double.MIN_NORMAL, java.lang.Double.MIN_VALUE, 1e-15,
    15, -52, -53, 1024, -1022, 11, 52
  )
  val F32: FInfo = FInfo(
    DType.Float32, 32, math.ulp(1.0f).toDouble, math.pow(2, -24), Float.MaxValue.toDouble, -Float.MaxValue.toDouble,
    java.lang.Float.MIN_NORMAL.toDouble, java.lang.Float.MIN_NORMAL.toDouble, java.lang.Float.MIN_VALUE.toDouble,
    1e-6f.toDouble, 6, -23, -24, 128, -126, 8, 23
  )

  def resolve(d: DType[?] | String): DType[?] = d match
    case s: String => DType.byName(s)
    case t: DType[?] => t

  private val rank: Seq[DType[?]] =
    Seq(DType.Bool, DType.Int8, DType.Int16, DType.Int32, DType.Int64, DType.Float32, DType.Float64, DType.Complex128)
  private def kindOrder(d: DType[?]): Int = d.kind match
    case 'b' => 0
    case 'u' | 'i' => 1
    case 'f' => 2
    case 'c' => 3
    case _ => 4

  /** Safe casts between numeric dtypes (NumPy's `can_cast(..., 'safe')`). */
  private def safe(f: DType[?], t: DType[?]): Boolean =
    if f eq t then true
    else if t.isString then true
    else if f.isString then false
    else if f.isBool then true
    else if t.isBool then false
    else
      (f.kind, t.kind) match
        case ('i', 'i') | ('u', 'u') => f.itemSize <= t.itemSize
        case ('u', 'i') => f.itemSize < t.itemSize
        case ('i' | 'u', 'f') => (t eq DType.Float64) || f.itemSize <= 2
        case ('i' | 'u', 'c') => true
        case ('f', 'f') => f.itemSize <= t.itemSize
        case ('f', 'c') => true
        case ('c', 'c') => true
        case _ => false

  def canCast(f: DType[?], t: DType[?], casting: String): Boolean = casting match
    case "no" | "equiv" => f eq t
    case "safe" => safe(f, t)
    case "same_kind" =>
      safe(f, t) || (!f.isString && !t.isString && kindOrder(f) <= kindOrder(t))
    case "unsafe" => true
    case other =>
      throw new IllegalArgumentException(
        s"casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe' (got '$other')"
      )

  def promote(a: DType[?], b: DType[?]): DType[?] =
    if a.isString || b.isString then DType.Str else DType.promote(a, b)

  /** NEP 50 `result_type`: arrays and dtypes are strong, Scala scalars are weak (Python scalars). */
  def resultType(args: Seq[Any]): DType[?] =
    if args.isEmpty then throw new IllegalArgumentException("at least one array or dtype is required")
    val strong = args.collect {
      case a: NDArray[?] => a.dtype
      case d: DType[?] => d
      case s: String => DType.byName(s)
    }
    def weakKind(v: Any): Int = v match
      case _: Boolean => 0
      case _: Byte | _: Short | _: Int | _: Long => 1
      case _: Float | _: Double => 2
      case _: Complex => 3
      case _: NDArray[?] | _: DType[?] | _: String => -1
      case other => throw new IllegalArgumentException(s"invalid argument to result_type: $other")
    val weak = args.map(weakKind).filter(_ >= 0)
    val base: DType[?] =
      if strong.nonEmpty then strong.reduce(promote)
      else
        weak.max match
          case 0 => DType.Bool
          case 1 => DType.Int64
          case 2 => DType.Float64
          case _ => DType.Complex128
    weak.foldLeft(base) { (d, w) =>
      if d.isString then d
      else
        val k = kindOrder(d)
        if w <= k then d
        else
          w match
            case 1 => DType.Int64
            case 2 => DType.Float64
            case _ => DType.Complex128
    }

  def minScalarType(v: Any): DType[?] = v match
    case a: NDArray[?] if a.ndim == 0 => minScalarType(a.getAt(Array.emptyIntArray))
    case a: NDArray[?] => a.dtype
    case _: Boolean => DType.Bool
    case x: (Byte | Short | Int | Long) =>
      val l = x match
        case b: Byte => b.toLong
        case s: Short => s.toLong
        case i: Int => i.toLong
        case l: Long => l
      if l >= 0 then
        if l <= 0xffL then DType.UInt8
        else if l <= 0xffffL then DType.UInt16
        else if l <= 0xffffffffL then DType.UInt32
        else DType.UInt64
      else if l >= Byte.MinValue then DType.Int8
      else if l >= Short.MinValue then DType.Int16
      else if l >= Int.MinValue then DType.Int32
      else DType.Int64
    case f: Float => DType.Float32
    case d: Double =>
      if d.isNaN || d.isInfinite || math.abs(d) <= Float.MaxValue then DType.Float32 else DType.Float64
    case _: Complex => DType.Complex128
    case _: String => DType.Str
    case other => throw new IllegalArgumentException(s"invalid argument to min_scalar_type: $other")

  def category(x: DType[?] | DTypeCategory | String): Either[DType[?], DTypeCategory] = x match
    case c: DTypeCategory => Right(c)
    case d: DType[?] => Left(d)
    case s: String => Left(DType.byName(s))
