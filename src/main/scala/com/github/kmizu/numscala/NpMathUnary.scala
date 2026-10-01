package com.github.kmizu.numscala

import MathK.{mapInexact, mapFloat, mapSame, mapPred}

/** Unary elementwise math functions of the `np` namespace (NumPy's unary ufuncs). */
trait NpMathUnary:

  // ------------------------------------------------------------------ arithmetic

  /** Numerical negative (`np.negative`). */
  def negative[T](x: NDArray[T])(using ev: NumDType[T]): NDArray[T] =
    mapSame(x, "negative")(i = v => -v, f = v => -v, c = z => -z)
  def negative(x: Double): Double = -x

  /** Numerical positive, a copy (`np.positive`). */
  def positive[T](x: NDArray[T])(using ev: NumDType[T]): NDArray[T] = x.copy()
  def positive(x: Double): Double = x

  /** Absolute value; complex gives the modulus as float64 (`np.absolute`). */
  def absolute[T](x: NDArray[T])(using r: AbsOf[T]): NDArray[r.Out] =
    if x.dtype.isComplex then
      x.asInstanceOf[NDArray[Complex]].map(z => math.hypot(z.re, z.im)).asInstanceOf[NDArray[r.Out]]
    else
      mapSame(x, "absolute")(b = v => v, i = v => math.abs(v), f = v => math.abs(v)).asInstanceOf[NDArray[r.Out]]
  def absolute(x: Double): Double = math.abs(x)
  def absolute(x: Complex): Double = math.hypot(x.re, x.im)

  /** Alias of [[absolute]] (`np.abs`). */
  def abs[T](x: NDArray[T])(using r: AbsOf[T]): NDArray[r.Out] = absolute(x)
  def abs(x: Double): Double = math.abs(x)
  def abs(x: Complex): Double = math.hypot(x.re, x.im)

  /** Absolute value of reals as floats (`np.fabs`). */
  def fabs[T](x: NDArray[T])(using t: ToFloat[T]): NDArray[t.Out] = mapFloat(x, t.dtype, "fabs")(v => math.abs(v))
  def fabs(x: Double): Double = math.abs(x)

  /** Sign: -1, 0, 1 (NaN for NaN); `z/|z|` for complex (`np.sign`). */
  def sign[T](x: NDArray[T])(using ev: NumDType[T]): NDArray[T] =
    mapSame(x, "sign")(
      i = v => java.lang.Long.signum(v).toLong,
      f = MathK.sign,
      c = MathK.signC
    )
  def sign(x: Double): Double = MathK.sign(x)

  /** Reciprocal `1/x`; for integers `1/x` with integer division, and `1/0` gives the
    * dtype's minimum for int32/int64 like NumPy (`np.reciprocal`).
    */
  def reciprocal[T](x: NDArray[T])(using d: NumDType[T]): NDArray[T] =
    val zeroRes: Long = (d: DType[?]) match
      case id: IntDType[?] if id.bits >= 32 => if id.bits == 32 then Int.MinValue.toLong else Long.MinValue
      case _ => 0L
    mapSame(x, "reciprocal")(
      i = v => if v == 1L then 1L else if v == -1L then -1L else if v == 0L then zeroRes else 0L,
      f = v => 1.0 / v,
      c = z => Complex.One / z
    )
  def reciprocal(x: Double): Double = 1.0 / x

  /** Elementwise square (`np.square`). */
  def square[T](x: NDArray[T])(using ev: NumDType[T]): NDArray[T] =
    mapSame(x, "square")(i = v => v * v, f = v => v * v, c = z => z * z)
  def square(x: Double): Double = x * x

  // ------------------------------------------------------------------ powers, exponentials, logarithms

  /** Square root; integers give float64, negative reals give NaN (`np.sqrt`). */
  def sqrt[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "sqrt")(math.sqrt, CMath.sqrt)
  def sqrt(x: Double): Double = math.sqrt(x)
  def sqrt(x: Complex): Complex = CMath.sqrt(x)

  /** Cube root (`np.cbrt`). */
  def cbrt[T](x: NDArray[T])(using t: ToFloat[T]): NDArray[t.Out] = mapFloat(x, t.dtype, "cbrt")(math.cbrt)
  def cbrt(x: Double): Double = math.cbrt(x)

  /** Exponential (`np.exp`). */
  def exp[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "exp")(math.exp, CMath.exp)
  def exp(x: Double): Double = math.exp(x)
  def exp(x: Complex): Complex = CMath.exp(x)

  /** `2**x` (`np.exp2`). */
  def exp2[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "exp2")(MathK.exp2, CMath.exp2)
  def exp2(x: Double): Double = MathK.exp2(x)
  def exp2(x: Complex): Complex = CMath.exp2(x)

  /** `exp(x) - 1`, accurate for small `x` (`np.expm1`). */
  def expm1[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "expm1")(math.expm1, CMath.expm1)
  def expm1(x: Double): Double = math.expm1(x)
  def expm1(x: Complex): Complex = CMath.expm1(x)

  /** Natural logarithm (`np.log`). */
  def log[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "log")(math.log, CMath.log)
  def log(x: Double): Double = math.log(x)
  def log(x: Complex): Complex = CMath.log(x)

  /** Base-2 logarithm (`np.log2`). */
  def log2[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "log2")(MathK.log2, CMath.log2)
  def log2(x: Double): Double = MathK.log2(x)
  def log2(x: Complex): Complex = CMath.log2(x)

  /** Base-10 logarithm (`np.log10`). */
  def log10[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "log10")(math.log10, CMath.log10)
  def log10(x: Double): Double = math.log10(x)
  def log10(x: Complex): Complex = CMath.log10(x)

  /** `log(1 + x)`, accurate for small `x` (`np.log1p`). */
  def log1p[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "log1p")(math.log1p, CMath.log1p)
  def log1p(x: Double): Double = math.log1p(x)
  def log1p(x: Complex): Complex = CMath.log1p(x)

  // ------------------------------------------------------------------ trigonometric

  /** Sine (`np.sin`). */
  def sin[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "sin")(math.sin, CMath.sin)
  def sin(x: Double): Double = math.sin(x)
  def sin(x: Complex): Complex = CMath.sin(x)

  /** Cosine (`np.cos`). */
  def cos[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "cos")(math.cos, CMath.cos)
  def cos(x: Double): Double = math.cos(x)
  def cos(x: Complex): Complex = CMath.cos(x)

  /** Tangent (`np.tan`). */
  def tan[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "tan")(math.tan, CMath.tan)
  def tan(x: Double): Double = math.tan(x)
  def tan(x: Complex): Complex = CMath.tan(x)

  /** Inverse sine (`np.arcsin`). */
  def arcsin[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "arcsin")(math.asin, CMath.asin)
  def arcsin(x: Double): Double = math.asin(x)
  def arcsin(x: Complex): Complex = CMath.asin(x)

  /** Inverse cosine (`np.arccos`). */
  def arccos[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "arccos")(math.acos, CMath.acos)
  def arccos(x: Double): Double = math.acos(x)
  def arccos(x: Complex): Complex = CMath.acos(x)

  /** Inverse tangent (`np.arctan`). */
  def arctan[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "arctan")(math.atan, CMath.atan)
  def arctan(x: Double): Double = math.atan(x)
  def arctan(x: Complex): Complex = CMath.atan(x)

  /** Hyperbolic sine (`np.sinh`). */
  def sinh[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "sinh")(math.sinh, CMath.sinh)
  def sinh(x: Double): Double = math.sinh(x)
  def sinh(x: Complex): Complex = CMath.sinh(x)

  /** Hyperbolic cosine (`np.cosh`). */
  def cosh[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "cosh")(math.cosh, CMath.cosh)
  def cosh(x: Double): Double = math.cosh(x)
  def cosh(x: Complex): Complex = CMath.cosh(x)

  /** Hyperbolic tangent (`np.tanh`). */
  def tanh[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "tanh")(math.tanh, CMath.tanh)
  def tanh(x: Double): Double = math.tanh(x)
  def tanh(x: Complex): Complex = CMath.tanh(x)

  /** Inverse hyperbolic sine (`np.arcsinh`). */
  def arcsinh[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "arcsinh")(MathK.asinh, CMath.asinh)
  def arcsinh(x: Double): Double = MathK.asinh(x)
  def arcsinh(x: Complex): Complex = CMath.asinh(x)

  /** Inverse hyperbolic cosine (`np.arccosh`). */
  def arccosh[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "arccosh")(MathK.acosh, CMath.acosh)
  def arccosh(x: Double): Double = MathK.acosh(x)
  def arccosh(x: Complex): Complex = CMath.acosh(x)

  /** Inverse hyperbolic tangent (`np.arctanh`). */
  def arctanh[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "arctanh")(MathK.atanh, CMath.atanh)
  def arctanh(x: Double): Double = MathK.atanh(x)
  def arctanh(x: Complex): Complex = CMath.atanh(x)

  /** Alias of [[arcsin]] (`np.asin`). */
  def asin[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = arcsin(x)
  def asin(x: Double): Double = math.asin(x)
  /** Alias of [[arccos]] (`np.acos`). */
  def acos[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = arccos(x)
  def acos(x: Double): Double = math.acos(x)
  /** Alias of [[arctan]] (`np.atan`). */
  def atan[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = arctan(x)
  def atan(x: Double): Double = math.atan(x)
  /** Alias of [[arcsinh]] (`np.asinh`). */
  def asinh[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = arcsinh(x)
  def asinh(x: Double): Double = MathK.asinh(x)
  /** Alias of [[arccosh]] (`np.acosh`). */
  def acosh[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = arccosh(x)
  def acosh(x: Double): Double = MathK.acosh(x)
  /** Alias of [[arctanh]] (`np.atanh`). */
  def atanh[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = arctanh(x)
  def atanh(x: Double): Double = MathK.atanh(x)

  /** Radians to degrees (`np.degrees`). */
  def degrees[T](x: NDArray[T])(using t: ToFloat[T]): NDArray[t.Out] = mapFloat(x, t.dtype, "degrees")(MathK.degrees)
  def degrees(x: Double): Double = MathK.degrees(x)
  /** Degrees to radians (`np.radians`). */
  def radians[T](x: NDArray[T])(using t: ToFloat[T]): NDArray[t.Out] = mapFloat(x, t.dtype, "radians")(MathK.radians)
  def radians(x: Double): Double = MathK.radians(x)
  /** Alias of [[radians]] (`np.deg2rad`). */
  def deg2rad[T](x: NDArray[T])(using t: ToFloat[T]): NDArray[t.Out] = mapFloat(x, t.dtype, "deg2rad")(MathK.radians)
  def deg2rad(x: Double): Double = MathK.radians(x)
  /** Alias of [[degrees]] (`np.rad2deg`). */
  def rad2deg[T](x: NDArray[T])(using t: ToFloat[T]): NDArray[t.Out] = mapFloat(x, t.dtype, "rad2deg")(MathK.degrees)
  def rad2deg(x: Double): Double = MathK.degrees(x)

  /** Normalized sinc `sin(pi x)/(pi x)` (`np.sinc`). */
  def sinc[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "sinc")(MathK.sinc, MathK.sincC)
  def sinc(x: Double): Double = MathK.sinc(x)

  /** Modified Bessel function of the first kind, order 0 (`np.i0`). */
  def i0[T](x: NDArray[T])(using t: ToFloat[T]): NDArray[t.Out] = mapFloat(x, t.dtype, "i0")(MathK.i0)
  def i0(x: Double): Double = MathK.i0(x)

  // ------------------------------------------------------------------ rounding

  /** Floor; integer and bool inputs keep their dtype (`np.floor`). */
  def floor[T](x: NDArray[T])(using ev: RealDType[T]): NDArray[T] = mapSame(x, "floor")(i = v => v, f = math.floor)
  def floor(x: Double): Double = math.floor(x)
  /** Ceiling (`np.ceil`). */
  def ceil[T](x: NDArray[T])(using ev: RealDType[T]): NDArray[T] = mapSame(x, "ceil")(i = v => v, f = math.ceil)
  def ceil(x: Double): Double = math.ceil(x)
  /** Truncation towards zero (`np.trunc`). */
  def trunc[T](x: NDArray[T])(using ev: RealDType[T]): NDArray[T] = mapSame(x, "trunc")(i = v => v, f = MathK.trunc)
  def trunc(x: Double): Double = MathK.trunc(x)
  /** Round towards zero (`np.fix`). */
  def fix[T](x: NDArray[T])(using ev: RealDType[T]): NDArray[T] = mapSame(x, "fix")(i = v => v, f = MathK.fix)
  def fix(x: Double): Double = MathK.fix(x)
  /** Round to nearest integer, ties to even; ints give float64 (`np.rint`). */
  def rint[T](x: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] = mapInexact(x, t.dtype, "rint")(math.rint, CMath.rint)
  def rint(x: Double): Double = math.rint(x)

  /** Round to `decimals` places, ties to even (`np.round`). */
  def round[T](a: NDArray[T], decimals: Int = 0)(using ev: NumDType[T]): NDArray[T] = Ops.round(a, decimals)
  def round(x: Double): Double = math.rint(x)
  def round(x: Double, decimals: Int): Double = Ops.round(NDArray.scalar(x), decimals).item
  /** Alias of [[round]] (`np.around`). */
  def around[T](a: NDArray[T], decimals: Int)(using ev: NumDType[T]): NDArray[T] = Ops.round(a, decimals)
  def around[T](a: NDArray[T])(using ev: NumDType[T]): NDArray[T] = Ops.round(a, 0)

  // ------------------------------------------------------------------ predicates

  /** Test for NaN (`np.isnan`). */
  def isnan[T](x: NDArray[T]): NDArray[Boolean] =
    mapPred(x, "isnan")(b = _ => false, i = _ => false, f = _.isNaN, c = _.isNaN)
  def isnan(x: Double): Boolean = x.isNaN
  def isnan(x: Complex): Boolean = x.isNaN

  /** Test for positive or negative infinity (`np.isinf`). */
  def isinf[T](x: NDArray[T]): NDArray[Boolean] =
    mapPred(x, "isinf")(b = _ => false, i = _ => false, f = _.isInfinite, c = _.isInfinite)
  def isinf(x: Double): Boolean = x.isInfinite
  def isinf(x: Complex): Boolean = x.isInfinite

  /** Test for finiteness (not infinity and not NaN) (`np.isfinite`). */
  def isfinite[T](x: NDArray[T]): NDArray[Boolean] =
    mapPred(x, "isfinite")(b = _ => true, i = _ => true, f = v => !v.isNaN && !v.isInfinite, c = _.isFinite)
  def isfinite(x: Double): Boolean = !x.isNaN && !x.isInfinite
  def isfinite(x: Complex): Boolean = x.isFinite

  /** Test for positive infinity; complex input is an error (`np.isposinf`). */
  def isposinf[T](x: NDArray[T]): NDArray[Boolean] =
    mapPred(x, "isposinf")(b = _ => false, i = _ => false, f = _ == Double.PositiveInfinity)
  def isposinf(x: Double): Boolean = x == Double.PositiveInfinity
  /** Test for negative infinity (`np.isneginf`). */
  def isneginf[T](x: NDArray[T]): NDArray[Boolean] =
    mapPred(x, "isneginf")(b = _ => false, i = _ => false, f = _ == Double.NegativeInfinity)
  def isneginf(x: Double): Boolean = x == Double.NegativeInfinity

  /** True where the sign bit is set, including `-0.0` (`np.signbit`). */
  def signbit[T](x: NDArray[T]): NDArray[Boolean] =
    mapPred(x, "signbit")(b = _ => false, i = _ < 0L, f = MathK.signbit)
  def signbit(x: Double): Boolean = MathK.signbit(x)

  /** Truth-value NOT (`np.logical_not`). */
  def logical_not[T](x: NDArray[T]): NDArray[Boolean] =
    mapPred(x, "logical_not")(b = v => !v, i = _ == 0L, f = _ == 0.0, c = z => z.re == 0.0 && z.im == 0.0)
  def logical_not(x: Boolean): Boolean = !x

  // ------------------------------------------------------------------ bitwise

  /** Bitwise NOT of integers, logical NOT of bools (`np.bitwise_not`). */
  def bitwise_not[T](x: NDArray[T])(using ev: BitOps[T]): NDArray[T] = mapSame(x, "bitwise_not")(b = v => !v, i = v => ~v)
  /** Alias of [[bitwise_not]] (`np.invert`). */
  def invert[T](x: NDArray[T])(using ev: BitOps[T]): NDArray[T] = bitwise_not(x)
  /** Alias of [[bitwise_not]] (`np.bitwise_invert`). */
  def bitwise_invert[T](x: NDArray[T])(using ev: BitOps[T]): NDArray[T] = bitwise_not(x)
  /** Number of 1-bits in the absolute value (`np.bitwise_count`; NumPy returns uint8, here int8). */
  def bitwise_count[T](x: NDArray[T])(using d: IntDType[T]): NDArray[Byte] =
    x.map(v => java.lang.Long.bitCount(math.abs(d.toLong(v))).toByte)

  // ------------------------------------------------------------------ floating point internals

  /** Distance to the nearest adjacent float, signed like `x` (`np.spacing`). */
  def spacing[T](x: NDArray[T])(using t: ToFloat[T]): NDArray[t.Out] =
    (t.dtype: DType[?]) match
      case DType.Float32 => x.asType(using DType.Float32).map(MathK.spacingF).asInstanceOf[NDArray[t.Out]]
      case _ => mapFloat(x, t.dtype, "spacing")(MathK.spacing)
  def spacing(x: Double): Double = MathK.spacing(x)
