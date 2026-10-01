package numscala

/** Floating-point error handling modes (`np.seterr`). num-scala follows IEEE arithmetic
  * silently (NumPy's `'ignore'`); with `'raise'`, division by zero / overflow / invalid
  * results of `np.divide`-like float operations are not trapped — the settings are kept for
  * API compatibility and returned by `np.geterr`.
  */
final case class ErrState(divide: String = "warn", over: String = "warn", under: String = "ignore", invalid: String = "warn"):
  def toMap: Map[String, String] = Map("divide" -> divide, "over" -> over, "under" -> under, "invalid" -> invalid)

/** `np.broadcast`: the result of broadcasting several arrays against each other. */
final class Broadcast private[numscala] (val arrays: Seq[NDArray[?]]):
  val shape: Seq[Int] = Shape.broadcast(arrays.map(_.shapeArr)*).toSeq
  def nd: Int = shape.length
  def ndim: Int = shape.length
  def size: Int = Shape.size(shape.toArray)
  def numiter: Int = arrays.length
  /** The broadcast operands as views of the common shape (`b.iters` as arrays). */
  def views: Seq[NDArray[?]] = arrays.map(a => a.asInstanceOf[NDArray[Any]].broadcastTo(shape*))
  /** Iterates element tuples in C order (`for (x, y) in np.broadcast(a, b)`). */
  def iterator: Iterator[Seq[Any]] =
    val vs = views.map(_.asInstanceOf[NDArray[Any]].toSeq)
    Iterator.range(0, size).map(i => vs.map(_(i)))

/** Additional top-level NumPy routines (type queries, bit packing, index tricks, ...). */
trait NpExtras:
  /** `np.ndarray` as a type: `np.ndarray[Double]`. */
  type ndarray[T] = NDArray[T]
  /** `np.ufunc` as a type. */
  type ufunc = Ufunc[?]

  /** The JVM is little-endian on every supported platform for array data written by num-scala. */
  val little_endian: Boolean = java.nio.ByteOrder.nativeOrder() == java.nio.ByteOrder.LITTLE_ENDIAN

  val True_ : Boolean = true
  val False_ : Boolean = false

  // ------------------------------------------------------------------ conversions

  /** `np.astype(x, dtype)` (NumPy 2 function form of `x.astype`). */
  def astype[T, U](x: NDArray[T], dtype: DType[U], copy: Boolean = true): NDArray[U] =
    if !copy then x.asType(using dtype) else x.astype(using dtype)

  /** `np.concat`, the array-API alias of `np.concatenate`. */
  def concat[T](arrays: Seq[NDArray[T]], axis: Int | None.type = 0): NDArray[T] =
    np.concatenate(arrays, axis)

  /** `np.asanyarray` (no subclasses on the JVM: same as `np.asarray`). */
  def asanyarray[A, E](a: A)(using n: Nested[A, E]): NDArray[E] = np.asarray(a)

  /** `np.asarray_chkfinite`: like `asarray` but rejects NaN and infinities. */
  def asarray_chkfinite[A, E](a: A)(using n: Nested[A, E]): NDArray[E] =
    val arr = np.asarray(a)
    val d = arr.dtype
    if d.isInexact then
      val bad = arr.toSeq.exists { x =>
        d match
          case i: InexactDType[E] @unchecked => !i.isFinite(x)
          case _ => false
      }
      if bad then throw new IllegalArgumentException("array must not contain infs or NaNs")
    arr

  /** `np.require`: returns `a` (or a copy) satisfying requirements `"C"`, `"F"`, `"O"`/`"OWNDATA"`, `"W"`, `"A"`. */
  def require[T](a: NDArray[T], requirements: Seq[String] = Nil): NDArray[T] =
    val req = requirements.map(_.toUpperCase)
    var r = a
    if req.exists(Set("C", "C_CONTIGUOUS", "CONTIGUOUS")) && !r.isCContiguous then r = r.copy()
    if req.exists(Set("F", "F_CONTIGUOUS", "FORTRAN")) && !r.isFContiguous then r = r.T.copy().T
    if req.exists(Set("O", "OWNDATA")) && !r.owndata then r = r.copy()
    r

  /** `np.isfortran`: Fortran-contiguous and not C-contiguous. */
  def isfortran(a: NDArray[?]): Boolean = a.isFContiguous && !a.isCContiguous

  /** `np.iterable`. */
  def iterable(x: Any): Boolean = x match
    case _: IterableOnce[?] | _: Array[?] => true
    case a: NDArray[?] => a.ndim > 0
    case _: String => true
    case _ => false

  // ------------------------------------------------------------------ dtype queries

  /** `np.common_type`: the inexact dtype all inputs can be cast to (integers -> float64). */
  def common_type(arrays: NDArray[?]*): DType[?] =
    // NumPy returns float16 for no arguments; num-scala has no float16, so float32 is the floor
    arrays.map(_.dtype).foldLeft[DType[?]](DType.Float32) { (acc, d) =>
      val t: DType[?] = if d.isInexact then d else DType.Float64
      if (acc eq DType.Complex128) || (t eq DType.Complex128) then DType.Complex128
      else if (acc eq DType.Float64) || (t eq DType.Float64) then DType.Float64
      else DType.Float32
    }

  /** `np.mintypecode`: the character of the smallest type in `typeset` that all `typechars` can be cast to. */
  def mintypecode(typechars: Seq[String], typeset: String = "GDFgdf", default: String = "d"): String =
    val chars = typechars.flatMap(s => s.map(_.toString)).filter(c => typeset.contains(c))
    if chars.isEmpty then default
    else
      val order = "GDFgdfQqLlIiHhBb?" // NumPy's _typecodes_by_elsize (largest first)
      if chars.contains("F") && chars.contains("d") then "D"
      else chars.minBy(c => order.indexOf(c))

  private val typenames = Map(
    "S1" -> "character", "?" -> "bool", "B" -> "unsigned char", "D" -> "complex double precision",
    "G" -> "complex long double precision", "F" -> "complex single precision", "I" -> "unsigned integer",
    "H" -> "unsigned short", "L" -> "unsigned long integer", "O" -> "object", "Q" -> "unsigned long long integer",
    "S" -> "string", "U" -> "unicode", "V" -> "void", "b" -> "signed char", "d" -> "double precision",
    "g" -> "long precision", "f" -> "single precision", "i" -> "integer", "h" -> "short",
    "l" -> "long integer", "q" -> "long long integer"
  )

  /** `np.typename`: description of a type character. */
  def typename(char: String): String =
    typenames.getOrElse(char, throw new NoSuchElementException(char))

  /** `np.isdtype(dtype, kind)` (array API): `kind` is a dtype or one of `"bool"`, `"signed integer"`,
    * `"unsigned integer"`, `"integral"`, `"real floating"`, `"complex floating"`, `"numeric"`.
    */
  def isdtype(dtype: DType[?], kind: DType[?] | String | Seq[DType[?] | String]): Boolean = kind match
    case d: DType[?] => d eq dtype
    case s: String =>
      s match
        case "bool" => dtype.isBool
        case "signed integer" => dtype.kind == 'i'
        case "unsigned integer" => dtype.kind == 'u'
        case "integral" => dtype.isInteger
        case "real floating" => dtype.isFloating
        case "complex floating" => dtype.isComplex
        case "numeric" => dtype.isNumeric
        case other => throw new IllegalArgumentException(s"kind argument must be comprised of NumPy dtypes or strings only, but is a '$other'")
    case seq: Seq[?] => seq.exists(k => isdtype(dtype, k.asInstanceOf[DType[?] | String]))

  // ------------------------------------------------------------------ bit packing

  /** `np.packbits`: packs a boolean/integer array into `uint8` bits (flattened, or along `axis`). */
  def packbits[T](a: NDArray[T], axis: Int | None.type = None, bitorder: String = "big"): NDArray[UInt8] =
    if bitorder != "big" && bitorder != "little" then throw new IllegalArgumentException("'order' must be either 'little' or 'big'")
    if !(a.dtype.isBool || a.dtype.isInteger) then
      throw new IllegalArgumentException("Expected an input array of integer or boolean data type")
    val d = a.dtype
    val src = axis match
      case _: None.type => a.ravel()
      case ax: Int => a.moveaxis(ax, -1)
    val bits = src.map(x => d.toBoolean(x))(using DType.Bool)
    val n = bits.shapeArr.last
    val m = (n + 7) / 8
    val lanes = Lanes.transform(bits, -1, m) { (in: Array[Boolean], len: Int, out: Array[UInt8]) =>
      var k = 0
      while k < m do
        var byte = 0
        var b = 0
        while b < 8 do
          val i = k * 8 + b
          if i < len && in(i) then byte |= (if bitorder == "big" then 0x80 >> b else 1 << b)
          b += 1
        out(k) = UInt8(byte.toLong)
        k += 1
    }(using DType.UInt8)
    axis match
      case _: None.type => lanes
      case ax: Int => lanes.moveaxis(-1, ax)

  /** `np.unpackbits`: unpacks `uint8` elements into bits (flattened, or along `axis`); `count` limits/pads the bits. */
  def unpackbits(a: NDArray[UInt8], axis: Int | None.type = None, count: Int = Int.MinValue, bitorder: String = "big"): NDArray[UInt8] =
    if bitorder != "big" && bitorder != "little" then throw new IllegalArgumentException("'order' must be either 'little' or 'big'")
    val src = axis match
      case _: None.type => a.ravel()
      case ax: Int => a.moveaxis(ax, -1)
    val n = src.shapeArr.last
    // a negative count removes that many trailing bits (NumPy semantics: count=-k keeps 8n-k)
    val outLen = if count == Int.MinValue then n * 8 else if count < 0 then math.max(0, n * 8 + count) else count
    val lanes = Lanes.transform(src, -1, outLen) { (in: Array[UInt8], len: Int, out: Array[UInt8]) =>
      var i = 0
      while i < outLen do
        val k = i / 8
        val b = i % 8
        val bit =
          if k < len then
            val byte = in(k).toInt
            if bitorder == "big" then (byte >> (7 - b)) & 1 else (byte >> b) & 1
          else 0
        out(i) = UInt8(bit.toLong)
        i += 1
    }(using DType.UInt8)
    axis match
      case _: None.type => lanes
      case ax: Int => lanes.moveaxis(-1, ax)

  // ------------------------------------------------------------------ products

  /** `np.matvec`: matrix-vector product over the last two/one axes with broadcasting. */
  def matvec[A, B](x1: NDArray[A], x2: NDArray[B])(using p: NumPromote[A, B]): NDArray[p.Out] =
    if x1.ndim < 2 || x2.ndim < 1 then throw new IllegalArgumentException("matvec: operands have too few dimensions")
    LinAlgCore.matmulD(x1, x2.expandDims(-1), p.dtype).squeeze(-1)

  /** `np.vecmat`: vector-matrix product `conj(x1) @ x2` over the last axes with broadcasting. */
  def vecmat[A, B](x1: NDArray[A], x2: NDArray[B])(using p: NumPromote[A, B]): NDArray[p.Out] =
    if x1.ndim < 1 || x2.ndim < 2 then throw new IllegalArgumentException("vecmat: operands have too few dimensions")
    val c = if x1.dtype.isComplex then x1.conj else x1
    LinAlgCore.matmulD(c.expandDims(-2), x2, p.dtype).squeeze(-2)

  // ------------------------------------------------------------------ index tricks

  /** `np.r_[a, b, ...]`: concatenates 1-D pieces (scalars are allowed) along the first axis. */
  def r_[T](parts: (NDArray[T] | T)*)(using d: DType[T]): NDArray[T] =
    val arrays = parts.map {
      case a: NDArray[?] => a.asInstanceOf[NDArray[T]].reshape(-1)
      case v => NDArray.scalar(v.asInstanceOf[T]).reshape(1)
    }
    if arrays.isEmpty then NDArray.zerosOf(d, Array(0)) else np.concatenate(arrays, 0)

  /** `np.c_[a, b, ...]`: stacks 1-D arrays as columns / concatenates 2-D arrays along the last axis. */
  def c_[T](parts: NDArray[T]*): NDArray[T] =
    val arrays = parts.map(a => if a.ndim == 1 then a.reshape(-1, 1) else a)
    np.concatenate(arrays, -1)

  // ------------------------------------------------------------------ broadcasting / iteration

  /** `np.broadcast(a, b, ...)`. */
  def broadcast(arrays: NDArray[?]*): Broadcast = new Broadcast(arrays)

  /** `np.nditer(a)`: iterates over the elements in memory-friendly (`"K"`), C or Fortran order. */
  def nditer[T](a: NDArray[T], order: String = "K"): Iterator[T] = order match
    case "C" | "K" | "A" => a.flat
    case "F" => a.T.flat
    case other => throw new IllegalArgumentException(s"order must be one of 'C', 'F', 'A', or 'K' (got '$other')")

  /** `np.frompyfunc` for unary functions (use `np.vectorize` for typed results). */
  def frompyfunc[A, U](f: A => U)(using u: DType[U]): NDArray[A] => NDArray[U] = a => a.map(f)

  // ------------------------------------------------------------------ floating point error handling

  @volatile private var errState: ErrState = ErrState()
  @volatile private var errCall: Option[(String, Int) => Unit] = None
  @volatile private var bufSize: Int = 8192

  /** `np.geterr`. */
  def geterr(): Map[String, String] = errState.toMap

  /** `np.seterr`: records the error-handling modes and returns the previous ones (see [[ErrState]]). */
  def seterr(all: String = null, divide: String = null, over: String = null, under: String = null, invalid: String = null): Map[String, String] =
    val old = errState
    def pick(v: String, cur: String): String =
      val x = if v != null then v else if all != null then all else cur
      if !Set("ignore", "warn", "raise", "call", "print", "log")(x) then throw new IllegalArgumentException(s"invalid error mode '$x'")
      x
    errState = ErrState(pick(divide, old.divide), pick(over, old.over), pick(under, old.under), pick(invalid, old.invalid))
    old.toMap

  /** `np.errstate(...) { body }`: temporarily changes the error-handling modes. */
  def errstate[A](all: String = null, divide: String = null, over: String = null, under: String = null, invalid: String = null)(body: => A): A =
    val saved = errState
    seterr(all, divide, over, under, invalid)
    try body
    finally errState = saved

  def geterrcall(): Option[(String, Int) => Unit] = errCall
  def seterrcall(f: Option[(String, Int) => Unit]): Option[(String, Int) => Unit] =
    val old = errCall
    errCall = f
    old

  /** `np.getbufsize` / `np.setbufsize` (kept for compatibility; no effect on computations). */
  def getbufsize(): Int = bufSize
  def setbufsize(size: Int): Int =
    if size <= 0 then throw new IllegalArgumentException("buffer size must be positive")
    val old = bufSize
    bufSize = size
    old

  // ------------------------------------------------------------------ text parsing

  /** `np.fromregex`: one column per regex group; a single group gives a 1-D array, several a 2-D array. */
  def fromregex[T](file: String | java.nio.file.Path, regexp: String, dtype: DType[T]): NDArray[T] =
    val path = file match
      case s: String => java.nio.file.Paths.get(s)
      case p: java.nio.file.Path => p
    val text = new String(java.nio.file.Files.readAllBytes(path), java.nio.charset.StandardCharsets.UTF_8)
    val m = java.util.regex.Pattern.compile(regexp).matcher(text)
    val rows = scala.collection.mutable.ArrayBuffer.empty[Seq[T]]
    while m.find() do
      val g = m.groupCount()
      rows += (if g == 0 then Seq(dtype.fromString(m.group())) else (1 to g).map(i => dtype.fromString(m.group(i))))
    val cols = if rows.isEmpty then 1 else rows.head.length
    val flat = rows.flatten.toArray(using dtype.classTag)
    if cols == 1 then NDArray.fromArray(flat)(using dtype) else NDArray.fromArray(flat, Array(rows.length, cols))(using dtype)

  /** `np.show_config` / `np.show_runtime`: prints build/runtime information. */
  def show_config(): Unit =
    println(s"num-scala on Scala ${scala.util.Properties.versionNumberString}, Java ${System.getProperty("java.version")}")
  def show_runtime(): Unit =
    println(s"Java ${System.getProperty("java.version")} (${System.getProperty("java.vm.name")}), " +
      s"${Runtime.getRuntime.availableProcessors()} processors")
