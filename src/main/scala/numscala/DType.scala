package numscala

import scala.reflect.ClassTag

/** Runtime description of an element type, analogous to `numpy.dtype`.
  *
  * Every [[NDArray]] carries a `DType[T]`.  The hierarchy mirrors NumPy's
  * abstract scalar types:
  *
  *  - [[DType]]          : any element type (`bool`, `str`, numbers)
  *  - [[NumDType]]       : numbers (`number`): integers, floats, complex
  *  - [[RealDType]]      : ordered numbers: integers and floats
  *  - [[IntDType]]       : integers (`integer`)
  *  - [[InexactDType]]   : floats and complex (`inexact`)
  *  - [[FloatDType]]     : real floats (`floating`)
  *  - [[ComplexDType]]   : complex (`complexfloating`)
  */
sealed abstract class DType[T](val name: String, val kind: Char, val itemSize: Int)(using
    val classTag: ClassTag[T]
):
  def zero: T
  def one: T
  def fromBoolean(b: Boolean): T
  def fromLong(v: Long): T
  def fromDouble(v: Double): T
  def fromComplex(c: Complex): T
  def fromString(s: String): T
  def toBoolean(x: T): Boolean
  def toLong(x: T): Long
  def toDouble(x: T): Double
  def toComplex(x: T): Complex
  def ordering: Ordering[T]
  /** Formats a single element the way NumPy's `str()` prints a scalar. */
  def format(x: T): String

  def newArray(n: Int): Array[T] = classTag.newArray(n)
  def isNaN(x: T): Boolean = false
  def equiv(x: T, y: T): Boolean = x == y
  def compare(x: T, y: T): Int = ordering.compare(x, y)

  /** Converts an element of another dtype to this dtype (`astype` semantics, `casting='unsafe'`). */
  def castFrom[S](src: DType[S], x: S): T

  /** Converts an arbitrary Scala value to this dtype. */
  def coerce(v: Any): T = v match
    case b: Boolean => fromBoolean(b)
    case b: Byte => fromLong(b.toLong)
    case s: Short => fromLong(s.toLong)
    case i: Int => fromLong(i.toLong)
    case l: Long => fromLong(l)
    case f: Float => fromDouble(f.toDouble)
    case d: Double => fromDouble(d)
    case c: Complex => fromComplex(c)
    case s: String => fromString(s)
    case c: Char => fromString(c.toString)
    case other => throw new IllegalArgumentException(s"cannot convert $other to $name")

  def isBool: Boolean = kind == 'b'
  def isInteger: Boolean = kind == 'i'
  def isFloating: Boolean = kind == 'f'
  def isComplex: Boolean = kind == 'c'
  def isString: Boolean = kind == 'U'
  def isNumeric: Boolean = kind == 'i' || kind == 'f' || kind == 'c'
  def isInexact: Boolean = kind == 'f' || kind == 'c'

  /** NumPy-style type character string, e.g. `<f8`. */
  def str: String = kind match
    case 'b' => "|b1"
    case 'U' => "<U"
    case k => s"<$k$itemSize"

  override def toString: String = name

/** Numbers: integers, floats and complex. */
sealed abstract class NumDType[T](name: String, kind: Char, itemSize: Int)(using ClassTag[T])
    extends DType[T](name, kind, itemSize):
  def plus(x: T, y: T): T
  def minus(x: T, y: T): T
  def times(x: T, y: T): T
  def negate(x: T): T
  /** `x ** y` within this dtype. */
  def power(x: T, y: T): T
  /** NumPy `sign`: -1, 0, +1 (complex: `x/|x|` per NumPy >= 2). */
  def sign(x: T): T
  def square(x: T): T = times(x, x)
  def fromInt(i: Int): T = fromLong(i.toLong)

/** Ordered numbers (integers and floats). */
sealed abstract class RealDType[T](name: String, kind: Char, itemSize: Int)(using ClassTag[T])
    extends NumDType[T](name, kind, itemSize):
  def abs(x: T): T
  def lt(x: T, y: T): Boolean = compare(x, y) < 0
  def max(x: T, y: T): T
  def min(x: T, y: T): T
  def floorDiv(x: T, y: T): T
  /** Python-style modulo (result has the sign of the divisor). */
  def mod(x: T, y: T): T
  /** C-style remainder (result has the sign of the dividend). */
  def fmod(x: T, y: T): T
  def minValue: T
  def maxValue: T

sealed abstract class IntDType[T](name: String, itemSize: Int)(using ClassTag[T])
    extends RealDType[T](name, 'i', itemSize):
  def bits: Int = itemSize * 8
  def and(x: T, y: T): T = fromLong(toLong(x) & toLong(y))
  def or(x: T, y: T): T = fromLong(toLong(x) | toLong(y))
  def xor(x: T, y: T): T = fromLong(toLong(x) ^ toLong(y))
  def invert(x: T): T = fromLong(~toLong(x))
  def shiftLeft(x: T, y: T): T =
    val s = toLong(y); if s >= bits then zero else fromLong(toLong(x) << s)
  def shiftRight(x: T, y: T): T =
    val s = toLong(y); val v = toLong(x)
    if s >= bits then (if v < 0 then fromLong(-1L) else zero) else fromLong(v >> s)
  def plus(x: T, y: T): T = fromLong(toLong(x) + toLong(y))
  def minus(x: T, y: T): T = fromLong(toLong(x) - toLong(y))
  def times(x: T, y: T): T = fromLong(toLong(x) * toLong(y))
  def negate(x: T): T = fromLong(-toLong(x))
  def abs(x: T): T = fromLong(math.abs(toLong(x)))
  def sign(x: T): T = fromLong(java.lang.Long.signum(toLong(x)).toLong)
  def max(x: T, y: T): T = if toLong(x) >= toLong(y) then x else y
  def min(x: T, y: T): T = if toLong(x) <= toLong(y) then x else y
  def floorDiv(x: T, y: T): T =
    val b = toLong(y)
    if b == 0 then zero else fromLong(Math.floorDiv(toLong(x), b))
  def mod(x: T, y: T): T =
    val b = toLong(y)
    if b == 0 then zero else fromLong(Math.floorMod(toLong(x), b))
  def fmod(x: T, y: T): T =
    val b = toLong(y)
    if b == 0 then zero else fromLong(toLong(x) % b)
  def power(x: T, y: T): T =
    var e = toLong(y)
    if e < 0 then throw new ArithmeticException("Integers to negative integer powers are not allowed.")
    var base = toLong(x)
    var acc = 1L
    while e > 0 do
      if (e & 1L) == 1L then acc *= base
      base *= base
      e >>= 1
    fromLong(acc)
  def fromBoolean(b: Boolean): T = if b then one else zero
  def fromDouble(v: Double): T = fromLong(v.toLong)
  def fromComplex(c: Complex): T = fromLong(c.re.toLong)
  def fromString(s: String): T = fromLong(s.trim.toLong)
  def toBoolean(x: T): Boolean = toLong(x) != 0L
  def toDouble(x: T): Double = toLong(x).toDouble
  def toComplex(x: T): Complex = Complex(toLong(x).toDouble, 0.0)
  def format(x: T): String = toLong(x).toString
  def castFrom[S](src: DType[S], x: S): T = src.kind match
    case 'f' => fromLong(src.toDouble(x).toLong)
    case 'c' => fromLong(src.toComplex(x).re.toLong)
    case 'U' => fromString(x.asInstanceOf[String])
    case _ => fromLong(src.toLong(x))

/** Floats and complex numbers. */
sealed abstract class InexactDType[T](name: String, kind: Char, itemSize: Int)(using ClassTag[T])
    extends NumDType[T](name, kind, itemSize):
  def div(x: T, y: T): T
  def reciprocal(x: T): T = div(one, x)
  /** Applies a mathematical function: `f` for real dtypes, `fc` for complex. */
  def lift(f: Double => Double, fc: Complex => Complex): T => T
  def nan: T
  def isInf(x: T): Boolean
  def isFinite(x: T): Boolean
  def sqrt(x: T): T = lift(math.sqrt, _.sqrt)(x)
  def exp(x: T): T = lift(math.exp, _.exp)(x)
  def log(x: T): T = lift(math.log, _.log)(x)
  /** The real dtype of the same precision (`float64` for `complex128`). */
  def realDType: FloatDType[?]

sealed abstract class FloatDType[T](name: String, itemSize: Int)(using ClassTag[T])
    extends InexactDType[T](name, 'f', itemSize):
  def plus(x: T, y: T): T = fromDouble(toDouble(x) + toDouble(y))
  def minus(x: T, y: T): T = fromDouble(toDouble(x) - toDouble(y))
  def times(x: T, y: T): T = fromDouble(toDouble(x) * toDouble(y))
  def div(x: T, y: T): T = fromDouble(toDouble(x) / toDouble(y))
  def negate(x: T): T = fromDouble(-toDouble(x))
  def power(x: T, y: T): T = fromDouble(math.pow(toDouble(x), toDouble(y)))
  def sign(x: T): T =
    val d = toDouble(x)
    if d.isNaN then x else fromDouble(math.signum(d))
  def lift(f: Double => Double, fc: Complex => Complex): T => T = x => fromDouble(f(toDouble(x)))
  def liftD(f: Double => Double): T => T = x => fromDouble(f(toDouble(x)))
  def liftD2(f: (Double, Double) => Double): (T, T) => T = (x, y) => fromDouble(f(toDouble(x), toDouble(y)))
  def abs(x: T): T = fromDouble(math.abs(toDouble(x)))
  def max(x: T, y: T): T =
    val a = toDouble(x); val b = toDouble(y)
    if a.isNaN then x else if b.isNaN then y else if a >= b then x else y
  def min(x: T, y: T): T =
    val a = toDouble(x); val b = toDouble(y)
    if a.isNaN then x else if b.isNaN then y else if a <= b then x else y
  def floorDiv(x: T, y: T): T =
    val a = toDouble(x); val b = toDouble(y)
    if b == 0.0 then fromDouble(if a == 0.0 || a.isNaN then Double.NaN else a / b)
    else fromDouble(math.floor(a / b))
  def mod(x: T, y: T): T =
    val a = toDouble(x); val b = toDouble(y)
    if b == 0.0 then nan
    else
      val r = a % b
      fromDouble(if r != 0.0 && ((r < 0) != (b < 0)) then r + b else if r == 0.0 then math.copySign(0.0, b) else r)
  def fmod(x: T, y: T): T = fromDouble(toDouble(x) % toDouble(y))
  override def isNaN(x: T): Boolean = toDouble(x).isNaN
  def isInf(x: T): Boolean = toDouble(x).isInfinite
  def isFinite(x: T): Boolean = { val d = toDouble(x); !d.isNaN && !d.isInfinite }
  override def equiv(x: T, y: T): Boolean = toDouble(x) == toDouble(y)
  def fromBoolean(b: Boolean): T = if b then one else zero
  def fromLong(v: Long): T = fromDouble(v.toDouble)
  def fromComplex(c: Complex): T = fromDouble(c.re)
  def fromString(s: String): T = fromDouble(Format.parseDouble(s))
  def toBoolean(x: T): Boolean = toDouble(x) != 0.0
  def toLong(x: T): Long = toDouble(x).toLong
  def toComplex(x: T): Complex = Complex(toDouble(x), 0.0)
  def format(x: T): String = Format.formatFloatShort(toDouble(x))
  def castFrom[S](src: DType[S], x: S): T = src.kind match
    case 'c' => fromDouble(src.toComplex(x).re)
    case 'U' => fromString(x.asInstanceOf[String])
    case _ => fromDouble(src.toDouble(x))
  def realDType: FloatDType[?] = this
  /** Machine epsilon. */
  def eps: Double
  def tiny: Double

final class ComplexDType private[numscala] () extends InexactDType[Complex]("complex128", 'c', 16):
  def zero: Complex = Complex.Zero
  def one: Complex = Complex.One
  def plus(x: Complex, y: Complex): Complex = x + y
  def minus(x: Complex, y: Complex): Complex = x - y
  def times(x: Complex, y: Complex): Complex = x * y
  def div(x: Complex, y: Complex): Complex = x / y
  def negate(x: Complex): Complex = -x
  def power(x: Complex, y: Complex): Complex = x.pow(y)
  def sign(x: Complex): Complex = if x.re == 0.0 && x.im == 0.0 then Complex.Zero else x / x.abs
  def lift(f: Double => Double, fc: Complex => Complex): Complex => Complex = fc
  def nan: Complex = Complex(Double.NaN, Double.NaN)
  override def isNaN(x: Complex): Boolean = x.isNaN
  def isInf(x: Complex): Boolean = x.isInfinite
  def isFinite(x: Complex): Boolean = x.isFinite
  def fromBoolean(b: Boolean): Complex = if b then Complex.One else Complex.Zero
  def fromLong(v: Long): Complex = Complex(v.toDouble, 0.0)
  def fromDouble(v: Double): Complex = Complex(v, 0.0)
  def fromComplex(c: Complex): Complex = c
  def fromString(s: String): Complex = Format.parseComplex(s)
  def toBoolean(x: Complex): Boolean = x.re != 0.0 || x.im != 0.0
  def toLong(x: Complex): Long = x.re.toLong
  def toDouble(x: Complex): Double = x.re
  def toComplex(x: Complex): Complex = x
  def ordering: Ordering[Complex] = Complex.ordering
  def format(x: Complex): String = Complex.format(x)
  def castFrom[S](src: DType[S], x: S): Complex = src.kind match
    case 'U' => fromString(x.asInstanceOf[String])
    case _ => src.toComplex(x)
  def realDType: FloatDType[?] = DType.Float64

/** Lexicographic-with-NaN-last ordering for floating point values (NumPy sort order). */
private[numscala] final class NaNLastOrdering[T](toD: T => Double) extends Ordering[T]:
  def compare(a: T, b: T): Int = Format.compareDouble(toD(a), toD(b))

object DType:
  object Bool extends DType[Boolean]("bool", 'b', 1):
    def zero = false
    def one = true
    def fromBoolean(b: Boolean): Boolean = b
    def fromLong(v: Long): Boolean = v != 0L
    def fromDouble(v: Double): Boolean = v != 0.0
    def fromComplex(c: Complex): Boolean = c.re != 0.0 || c.im != 0.0
    def fromString(s: String): Boolean = s.trim match
      case "True" | "true" | "1" => true
      case "False" | "false" | "0" | "" => false
      case other => throw new IllegalArgumentException(s"invalid literal for bool: '$other'")
    def toBoolean(x: Boolean): Boolean = x
    def toLong(x: Boolean): Long = if x then 1L else 0L
    def toDouble(x: Boolean): Double = if x then 1.0 else 0.0
    def toComplex(x: Boolean): Complex = if x then Complex.One else Complex.Zero
    val ordering: Ordering[Boolean] = Ordering.Boolean
    def format(x: Boolean): String = if x then "True" else "False"
    def castFrom[S](src: DType[S], x: S): Boolean = src.kind match
      case 'U' => x.asInstanceOf[String].nonEmpty
      case _ => src.toBoolean(x)

  object Int8 extends IntDType[Byte]("int8", 1):
    def zero: Byte = 0
    def one: Byte = 1
    def fromLong(v: Long): Byte = v.toByte
    def toLong(x: Byte): Long = x.toLong
    val ordering: Ordering[Byte] = Ordering.Byte
    def minValue: Byte = Byte.MinValue
    def maxValue: Byte = Byte.MaxValue

  object Int16 extends IntDType[Short]("int16", 2):
    def zero: Short = 0
    def one: Short = 1
    def fromLong(v: Long): Short = v.toShort
    def toLong(x: Short): Long = x.toLong
    val ordering: Ordering[Short] = Ordering.Short
    def minValue: Short = Short.MinValue
    def maxValue: Short = Short.MaxValue

  object Int32 extends IntDType[Int]("int32", 4):
    def zero: Int = 0
    def one: Int = 1
    def fromLong(v: Long): Int = v.toInt
    def toLong(x: Int): Long = x.toLong
    val ordering: Ordering[Int] = Ordering.Int
    def minValue: Int = Int.MinValue
    def maxValue: Int = Int.MaxValue
    override def plus(x: Int, y: Int): Int = x + y
    override def minus(x: Int, y: Int): Int = x - y
    override def times(x: Int, y: Int): Int = x * y

  object Int64 extends IntDType[Long]("int64", 8):
    def zero: Long = 0L
    def one: Long = 1L
    def fromLong(v: Long): Long = v
    def toLong(x: Long): Long = x
    val ordering: Ordering[Long] = Ordering.Long
    def minValue: Long = Long.MinValue
    def maxValue: Long = Long.MaxValue
    override def plus(x: Long, y: Long): Long = x + y
    override def minus(x: Long, y: Long): Long = x - y
    override def times(x: Long, y: Long): Long = x * y

  object Float32 extends FloatDType[Float]("float32", 4):
    def zero: Float = 0f
    def one: Float = 1f
    def nan: Float = Float.NaN
    def fromDouble(v: Double): Float = v.toFloat
    def toDouble(x: Float): Double = x.toDouble
    val ordering: Ordering[Float] = NaNLastOrdering[Float](_.toDouble)
    def minValue: Float = Float.MinValue
    def maxValue: Float = Float.MaxValue
    def eps: Double = java.lang.Math.ulp(1.0f).toDouble
    def tiny: Double = java.lang.Float.MIN_NORMAL.toDouble
    override def format(x: Float): String = Format.formatFloat32Short(x)

  object Float64 extends FloatDType[Double]("float64", 8):
    def zero: Double = 0.0
    def one: Double = 1.0
    def nan: Double = Double.NaN
    def fromDouble(v: Double): Double = v
    def toDouble(x: Double): Double = x
    val ordering: Ordering[Double] = NaNLastOrdering[Double](identity)
    def minValue: Double = Double.MinValue
    def maxValue: Double = Double.MaxValue
    def eps: Double = java.lang.Math.ulp(1.0)
    def tiny: Double = java.lang.Double.MIN_NORMAL
    override def plus(x: Double, y: Double): Double = x + y
    override def minus(x: Double, y: Double): Double = x - y
    override def times(x: Double, y: Double): Double = x * y
    override def div(x: Double, y: Double): Double = x / y
    override def negate(x: Double): Double = -x

  val Complex128: ComplexDType = new ComplexDType()

  object Str extends DType[String]("str", 'U', 0):
    def zero = ""
    def one = "1"
    def fromBoolean(b: Boolean): String = if b then "True" else "False"
    def fromLong(v: Long): String = v.toString
    def fromDouble(v: Double): String = Format.formatFloatShort(v)
    def fromComplex(c: Complex): String = Complex.format(c)
    def fromString(s: String): String = s
    def toBoolean(x: String): Boolean = x.nonEmpty
    def toLong(x: String): Long = x.trim.toLong
    def toDouble(x: String): Double = Format.parseDouble(x)
    def toComplex(x: String): Complex = Format.parseComplex(x)
    val ordering: Ordering[String] = Ordering.String
    def format(x: String): String = x
    def castFrom[S](src: DType[S], x: S): String = src.format(x)

  given bool: Bool.type = Bool
  given int8: Int8.type = Int8
  given int16: Int16.type = Int16
  given int32: Int32.type = Int32
  given int64: Int64.type = Int64
  given float32: Float32.type = Float32
  given float64: Float64.type = Float64
  given complex128: ComplexDType = Complex128
  given str: Str.type = Str

  def of[T](using d: DType[T]): DType[T] = d

  /** All built-in dtypes. */
  val all: Seq[DType[?]] = Seq(Bool, Int8, Int16, Int32, Int64, Float32, Float64, Complex128, Str)

  /** Looks a dtype up by NumPy name or type code (`"float64"`, `"f8"`, `"int"`, `"complex"`, ...). */
  def byName(n: String): DType[?] = n.trim.stripPrefix("<").stripPrefix("|").stripPrefix("=") match
    case "bool" | "b1" | "?" | "bool_" => Bool
    case "int8" | "i1" | "b" | "byte" => Int8
    case "int16" | "i2" | "h" | "short" => Int16
    case "int32" | "i4" | "i" | "intc" => Int32
    case "int64" | "i8" | "l" | "q" | "int" | "intp" | "long" | "longlong" => Int64
    case "float32" | "f4" | "f" | "single" => Float32
    case "float64" | "f8" | "d" | "float" | "double" => Float64
    case "complex128" | "c16" | "D" | "complex" | "cdouble" => Complex128
    case s if s == "str" || s == "U" || s.startsWith("U") || s == "str_" => Str
    case other => throw new IllegalArgumentException(s"data type '$other' not understood")

  /** NumPy-style type promotion of two dtypes (`np.result_type`). */
  def promote(a: DType[?], b: DType[?]): DType[?] =
    if a eq b then a
    else if a.isString || b.isString then
      throw new IllegalArgumentException(s"cannot promote $a and $b")
    else
      val rank = Seq[DType[?]](Bool, Int8, Int16, Int32, Int64, Float32, Float64, Complex128)
      val (lo, hi) = if rank.indexOf(a) <= rank.indexOf(b) then (a, b) else (b, a)
      if hi eq Complex128 then Complex128
      else if (hi eq Float32) && ((lo eq Int32) || (lo eq Int64)) then Float64
      else hi
