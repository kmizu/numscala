package numscala

import UfuncKind.*

/** The binary ufunc instances (shared singletons). */
private[numscala] object Ufuncs:
  private def k(k2: K2): DType[?] => K2 = _ => k2
  private def bits(d: DType[?]): Int = d match
    case i: IntDType[?] => i.bits
    case _ => 0
  private val NoIdentity: Any = null

  // ------------------------------------------------------------------ arithmetic

  val add: Ufunc[Sum] = SameUfunc[Sum](
    "add", 0L, true,
    k(K2(b = _ || _, i = _ + _, f = _ + _, c = _ + _, s = _ + _)),
    Arith.Add, "sum"
  )
  val multiply: Ufunc[Sum] = SameUfunc[Sum](
    "multiply", 1L, true,
    k(K2(b = _ && _, i = _ * _, f = _ * _, c = _ * _)),
    Arith.Mul, "prod"
  )
  val subtract: Ufunc[Arith] = SameUfunc[Arith](
    "subtract", NoIdentity, false, k(K2(i = _ - _, f = _ - _, c = _ - _)), Arith.Sub
  )
  val power: Ufunc[Arith] = SameUfunc[Arith](
    "power", NoIdentity, false, k(K2(i = MathK.ipow, f = MathK.cpow, c = CMath.pow))
  )
  val floor_divide: Ufunc[Arith] = SameUfunc[Arith](
    "floor_divide", NoIdentity, false, k(K2(i = MathK.floorDivL, f = MathK.floorDivD))
  )
  val remainder: Ufunc[Arith] = SameUfunc[Arith](
    "remainder", NoIdentity, false, k(K2(i = MathK.modL, f = MathK.modD))
  )
  val fmod: Ufunc[Arith] = SameUfunc[Arith](
    "fmod", NoIdentity, false, k(K2(i = MathK.fmodL, f = (a, b) => a % b))
  )
  val divide: Ufunc[Div] = SameUfunc[Div](
    "divide", NoIdentity, false, k(K2(f = _ / _, c = _ / _))
  )
  val float_power: Ufunc[FPow] = SameUfunc[FPow](
    "float_power", NoIdentity, false, k(K2(f = MathK.cpow, c = CMath.pow))
  )

  // ------------------------------------------------------------------ float functions

  val arctan2: Ufunc[Div] = SameUfunc[Div]("arctan2", NoIdentity, false, k(K2(f = math.atan2)))
  val hypot: Ufunc[Div] = SameUfunc[Div]("hypot", 0L, true, k(K2(f = math.hypot)))
  val logaddexp: Ufunc[Div] =
    SameUfunc[Div]("logaddexp", Double.NegativeInfinity, true, k(K2(f = MathK.logaddexp)))
  val logaddexp2: Ufunc[Div] =
    SameUfunc[Div]("logaddexp2", Double.NegativeInfinity, true, k(K2(f = MathK.logaddexp2)))
  val copysign: Ufunc[Div] = SameUfunc[Div](
    "copysign", NoIdentity, false,
    k(K2(f = (a, b) => java.lang.Math.copySign(a, b), f32 = (a, b) => java.lang.Math.copySign(a, b)))
  )
  val nextafter: Ufunc[Div] = SameUfunc[Div](
    "nextafter", NoIdentity, false,
    k(K2(f = (a, b) => math.nextAfter(a, b), f32 = MathK.nextafterF))
  )
  val heaviside: Ufunc[Div] = SameUfunc[Div]("heaviside", NoIdentity, false, k(K2(f = MathK.heaviside)))

  // ------------------------------------------------------------------ extrema

  val maximum: Ufunc[Same] = SameUfunc[Same](
    "maximum", NoIdentity, true,
    k(K2(b = _ || _, i = math.max, f = MathK.maximumD, c = MathK.maximumC, s = (x, y) => if x >= y then x else y))
  )
  val minimum: Ufunc[Same] = SameUfunc[Same](
    "minimum", NoIdentity, true,
    k(K2(b = _ && _, i = math.min, f = MathK.minimumD, c = MathK.minimumC, s = (x, y) => if x <= y then x else y))
  )
  val fmax: Ufunc[Same] = SameUfunc[Same](
    "fmax", NoIdentity, true, k(K2(b = _ || _, i = math.max, f = MathK.fmaxD, c = MathK.fmaxC))
  )
  val fmin: Ufunc[Same] = SameUfunc[Same](
    "fmin", NoIdentity, true, k(K2(b = _ && _, i = math.min, f = MathK.fminD, c = MathK.fminC))
  )

  // ------------------------------------------------------------------ integer / bitwise

  val bitwise_and: Ufunc[Same] = SameUfunc[Same]("bitwise_and", -1L, true, k(K2(b = _ && _, i = _ & _)))
  val bitwise_or: Ufunc[Same] = SameUfunc[Same]("bitwise_or", 0L, true, k(K2(b = _ || _, i = _ | _)))
  val bitwise_xor: Ufunc[Same] = SameUfunc[Same]("bitwise_xor", 0L, true, k(K2(b = _ ^ _, i = _ ^ _)))
  val left_shift: Ufunc[Same] = SameUfunc[Same](
    "left_shift", NoIdentity, false, d => K2(i = MathK.shiftLeft(bits(d)))
  )
  val right_shift: Ufunc[Same] = SameUfunc[Same](
    "right_shift", NoIdentity, false, d => K2(i = MathK.shiftRight(bits(d)))
  )
  val gcd: Ufunc[Same] = SameUfunc[Same]("gcd", 0L, true, k(K2(i = MathK.gcdL)))
  val lcm: Ufunc[Same] = SameUfunc[Same]("lcm", NoIdentity, true, k(K2(i = MathK.lcmL)))

  // ------------------------------------------------------------------ comparisons and logic

  private def cmp(name: String, op: CmpOp): BoolUfunc =
    BoolUfunc(name, NoIdentity, false, d => (x, y) => op.test(d, x, y))

  val equal: Ufunc[Cmp] = BoolUfunc("equal", NoIdentity, false, d => (x, y) => d.equiv(x, y))
  val not_equal: Ufunc[Cmp] = BoolUfunc("not_equal", NoIdentity, false, d => (x, y) => !d.equiv(x, y))
  val less: Ufunc[Cmp] = cmp("less", CmpOp.Lt)
  val less_equal: Ufunc[Cmp] = cmp("less_equal", CmpOp.Le)
  val greater: Ufunc[Cmp] = cmp("greater", CmpOp.Gt)
  val greater_equal: Ufunc[Cmp] = cmp("greater_equal", CmpOp.Ge)
  val logical_and: Ufunc[Cmp] =
    BoolUfunc("logical_and", true, true, d => (x, y) => d.toBoolean(x) && d.toBoolean(y))
  val logical_or: Ufunc[Cmp] =
    BoolUfunc("logical_or", false, true, d => (x, y) => d.toBoolean(x) || d.toBoolean(y))
  val logical_xor: Ufunc[Cmp] =
    BoolUfunc("logical_xor", false, true, d => (x, y) => d.toBoolean(x) != d.toBoolean(y))

/** Binary ufuncs of the `np` namespace. */
trait NpMathUfuncs:
  /** Elementwise addition (`np.add`); bool inputs give logical or, strings concatenate. */
  val add: Ufunc[Sum] = Ufuncs.add
  /** Elementwise subtraction (`np.subtract`). */
  val subtract: Ufunc[Arith] = Ufuncs.subtract
  /** Elementwise product (`np.multiply`). */
  val multiply: Ufunc[Sum] = Ufuncs.multiply
  /** True division, always inexact (`np.divide`). */
  val divide: Ufunc[Div] = Ufuncs.divide
  /** Alias of [[divide]] (`np.true_divide`). */
  val true_divide: Ufunc[Div] = Ufuncs.divide
  /** Floor division; integer division by zero gives 0 (`np.floor_divide`). */
  val floor_divide: Ufunc[Arith] = Ufuncs.floor_divide
  /** Python-style modulo, sign of the divisor (`np.remainder`). */
  val remainder: Ufunc[Arith] = Ufuncs.remainder
  /** Alias of [[remainder]] (`np.mod`). */
  val mod: Ufunc[Arith] = Ufuncs.remainder
  /** C-style remainder, sign of the dividend (`np.fmod`). */
  val fmod: Ufunc[Arith] = Ufuncs.fmod
  /** Elementwise power (`np.power`); integers to negative powers are an error. */
  val power: Ufunc[Arith] = Ufuncs.power
  /** Alias of [[power]] (`np.pow`). */
  val pow: Ufunc[Arith] = Ufuncs.power
  /** Power computed in float64 / complex128 (`np.float_power`). */
  val float_power: Ufunc[FPow] = Ufuncs.float_power
  /** Quadrant-aware arctangent of `x1/x2` (`np.arctan2`). */
  val arctan2: Ufunc[Div] = Ufuncs.arctan2
  /** Alias of [[arctan2]] (`np.atan2`). */
  val atan2: Ufunc[Div] = Ufuncs.arctan2
  /** `sqrt(x1**2 + x2**2)` without overflow (`np.hypot`). */
  val hypot: Ufunc[Div] = Ufuncs.hypot
  /** `log(exp(x1) + exp(x2))` (`np.logaddexp`). */
  val logaddexp: Ufunc[Div] = Ufuncs.logaddexp
  /** `log2(2**x1 + 2**x2)` (`np.logaddexp2`). */
  val logaddexp2: Ufunc[Div] = Ufuncs.logaddexp2
  /** Magnitude of `x1` with the sign of `x2` (`np.copysign`). */
  val copysign: Ufunc[Div] = Ufuncs.copysign
  /** Next representable float after `x1` towards `x2` (`np.nextafter`). */
  val nextafter: Ufunc[Div] = Ufuncs.nextafter
  /** Heaviside step function with value `x2` at zero (`np.heaviside`). */
  val heaviside: Ufunc[Div] = Ufuncs.heaviside
  /** Elementwise maximum, propagating NaN (`np.maximum`). */
  val maximum: Ufunc[Same] = Ufuncs.maximum
  /** Elementwise minimum, propagating NaN (`np.minimum`). */
  val minimum: Ufunc[Same] = Ufuncs.minimum
  /** Elementwise maximum, ignoring NaN (`np.fmax`). */
  val fmax: Ufunc[Same] = Ufuncs.fmax
  /** Elementwise minimum, ignoring NaN (`np.fmin`). */
  val fmin: Ufunc[Same] = Ufuncs.fmin
  /** Bitwise / logical AND of integers or bools (`np.bitwise_and`). */
  val bitwise_and: Ufunc[Same] = Ufuncs.bitwise_and
  /** Bitwise / logical OR (`np.bitwise_or`). */
  val bitwise_or: Ufunc[Same] = Ufuncs.bitwise_or
  /** Bitwise / logical XOR (`np.bitwise_xor`). */
  val bitwise_xor: Ufunc[Same] = Ufuncs.bitwise_xor
  /** Shift bits left (`np.left_shift`). */
  val left_shift: Ufunc[Same] = Ufuncs.left_shift
  /** Alias of [[left_shift]] (`np.bitwise_left_shift`). */
  val bitwise_left_shift: Ufunc[Same] = Ufuncs.left_shift
  /** Arithmetic shift right (`np.right_shift`). */
  val right_shift: Ufunc[Same] = Ufuncs.right_shift
  /** Alias of [[right_shift]] (`np.bitwise_right_shift`). */
  val bitwise_right_shift: Ufunc[Same] = Ufuncs.right_shift
  /** Greatest common divisor of integers (`np.gcd`). */
  val gcd: Ufunc[Same] = Ufuncs.gcd
  /** Least common multiple of integers (`np.lcm`). */
  val lcm: Ufunc[Same] = Ufuncs.lcm
  /** Elementwise `x1 == x2` (`np.equal`). */
  val equal: Ufunc[Cmp] = Ufuncs.equal
  /** Elementwise `x1 != x2` (`np.not_equal`). */
  val not_equal: Ufunc[Cmp] = Ufuncs.not_equal
  /** Elementwise `x1 < x2` (`np.less`). */
  val less: Ufunc[Cmp] = Ufuncs.less
  /** Elementwise `x1 <= x2` (`np.less_equal`). */
  val less_equal: Ufunc[Cmp] = Ufuncs.less_equal
  /** Elementwise `x1 > x2` (`np.greater`). */
  val greater: Ufunc[Cmp] = Ufuncs.greater
  /** Elementwise `x1 >= x2` (`np.greater_equal`). */
  val greater_equal: Ufunc[Cmp] = Ufuncs.greater_equal
  /** Elementwise truth-value AND (`np.logical_and`). */
  val logical_and: Ufunc[Cmp] = Ufuncs.logical_and
  /** Elementwise truth-value OR (`np.logical_or`). */
  val logical_or: Ufunc[Cmp] = Ufuncs.logical_or
  /** Elementwise truth-value XOR (`np.logical_xor`). */
  val logical_xor: Ufunc[Cmp] = Ufuncs.logical_xor
