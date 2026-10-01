package com.github.kmizu.numscala

/** A one-dimensional polynomial — `numpy.poly1d`. Coefficients are stored highest power
  * first, with leading zeros trimmed.
  *
  * {{{
  * val p = np.poly1d(np.array(1.0, 2.0, 3.0))   // x**2 + 2x + 3
  * p(0.5)                                         // 4.25
  * p * p; p.deriv(); p.integ(); p.roots
  * println(p)
  * //    2
  * // 1 x + 2 x + 3
  * }}}
  */
final class Poly1d[T] private (val coeffs: NDArray[T], val variable: String)(using val numDType: NumDType[T]):
  import PolyLegacy.*

  private def raw: Array[T] = coeffs.toArray
  private def make[U](a: Array[U])(using u: NumDType[U]): Poly1d[U] = Poly1d(arr(a), variable)

  /** The coefficients (alias of `coeffs`, NumPy `c`). */
  def c: NDArray[T] = coeffs
  /** The coefficients (NumPy `coef`). */
  def coef: NDArray[T] = coeffs
  /** The coefficients (NumPy `coefficients`). */
  def coefficients: NDArray[T] = coeffs
  /** The degree of the polynomial. */
  def order: Int = coeffs.size - 1
  /** Alias of [[order]]. */
  def o: Int = order
  /** `len(p)`: the order. */
  def length: Int = order
  /** The roots of the polynomial (always complex). */
  def roots: NDArray[Complex] = PolyLegacy.roots(coeffs)
  /** Alias of [[roots]]. */
  def r: NDArray[Complex] = roots
  /** `p[k]`: the coefficient of `x**k` (0 when `k` is out of range). */
  def get(k: Int): T = if k > order || k < 0 then numDType.zero else coeffs(order - k)
  /** Iterates over the coefficients. */
  def iterator: Iterator[T] = raw.iterator

  /** Evaluates the polynomial at a scalar. */
  def apply(x: T): T = horner(raw, x, numDType)

  /** Evaluates the polynomial element-wise. */
  def apply[U](x: NDArray[U])(using p: NumPromote[T, U]): NDArray[p.Out] = np.polyval(coeffs, x)

  /** Composition `p(q(x))`. */
  def apply(q: Poly1d[T]): Poly1d[T] =
    var y = Poly1d(arr(Array(numDType.zero)(using numDType.classTag)), variable)
    raw.foreach(pv => y = y * q + pv)
    y

  def +[U](o: Poly1d[U])(using p: NumPromote[T, U]): Poly1d[p.Out] =
    val d = p.dtype
    make(add(cast(coeffs, d), cast(o.coeffs, d), d, sub = false))(using d)
  def -[U](o: Poly1d[U])(using p: NumPromote[T, U]): Poly1d[p.Out] =
    val d = p.dtype
    make(add(cast(coeffs, d), cast(o.coeffs, d), d, sub = true))(using d)
  def *[U](o: Poly1d[U])(using p: NumPromote[T, U]): Poly1d[p.Out] =
    val d = p.dtype
    make(convolve(cast(coeffs, d), cast(o.coeffs, d), d))(using d)
  def +(s: T): Poly1d[T] = make(add(raw, Array(s)(using numDType.classTag), numDType, sub = false))
  def -(s: T): Poly1d[T] = make(add(raw, Array(s)(using numDType.classTag), numDType, sub = true))
  def *(s: T): Poly1d[T] = make(raw.map(numDType.times(_, s))(using numDType.classTag))
  def unary_- : Poly1d[T] = make(raw.map(numDType.negate)(using numDType.classTag))

  /** Polynomial division: `(quotient, remainder)` (NumPy `p / q` with a polynomial divisor). */
  def /[U](o: Poly1d[U])(using p: DivPromote[T, U]): (Poly1d[p.Out], Poly1d[p.Out]) =
    val d = p.dtype
    val (q, rem) = div(cast(coeffs, d), cast(o.coeffs, d), d)
    (make(q)(using d), make(rem)(using d))

  /** Division of every coefficient by a scalar. */
  def /(s: T)(using t: ToInexact[T]): Poly1d[t.Out] =
    val d = t.dtype
    val sv = d.castFrom(numDType, s)
    make(cast(coeffs, d).map(d.div(_, sv))(using d.classTag))(using d)

  /** Integer power. */
  def **(n: Int): Poly1d[T] =
    if n < 0 then throw new IllegalArgumentException("Power to non-negative integers only.")
    var res = Array(numDType.one)(using numDType.classTag)
    for _ <- 0 until n do res = convolve(raw, res, numDType)
    make(res)

  /** `m`-th derivative. */
  def deriv(m: Int = 1): Poly1d[T] = make(der(raw, m, numDType))

  /** `m`-th antiderivative with integration constants `k` (NumPy `integ`). */
  def integ(m: Int = 1, k: Seq[Double] = Seq(0.0))(using t: ToInexact[T]): Poly1d[t.Out] =
    val d = t.dtype
    val kk = k.map(d.fromDouble).toArray(using d.classTag)
    make(int(cast(coeffs, d), m, kk, d))(using d)

  override def equals(other: Any): Boolean = other match
    case p: Poly1d[?] =>
      p.coeffs.size == coeffs.size &&
      raw.indices.forall(i => numDType.toComplex(raw(i)) == p.numDType.asInstanceOf[NumDType[Any]].toComplex(p.coeffs.flatGet(i)))
    case _ => false

  override def hashCode: Int = raw.toSeq.map(numDType.toComplex).hashCode

  /** NumPy's `repr`: `poly1d([1., 2., 3.])`. */
  def repr: String =
    val r = Format.repr(coeffs)
    "poly1d(" + r.substring(6, r.length - 1) + ")"

  /** NumPy's pretty two-line form (`str(p)`). */
  override def toString: String =
    val all = raw
    val first = all.indexWhere(x => numDType.toComplex(x) != Complex.Zero)
    val cs = if first < 0 then Array.empty[Complex] else all.drop(first).map(numDType.toComplex)
    val n = cs.length - 1
    def fmt(q: Double): String =
      val s = pyG(q)
      if s.endsWith(".0000") then s.dropRight(5) else s
    var thestr = "0"
    var k = 0
    while k < cs.length do
      val coeff = cs(k)
      val coefstr =
        if coeff.im == 0.0 then fmt(coeff.re)
        else if coeff.re == 0.0 then s"${fmt(coeff.im)}j"
        else s"(${fmt(coeff.re)} + ${fmt(coeff.im)}j)"
      val power = n - k
      val newstr =
        if power == 0 then
          if coefstr != "0" then coefstr else if k == 0 then "0" else ""
        else if power == 1 then
          if coefstr == "0" then "" else if coefstr == "b" then variable else s"$coefstr $variable"
        else if coefstr == "0" then ""
        else if coefstr == "b" then s"$variable**$power"
        else s"$coefstr $variable**$power"
      if k > 0 then
        if newstr != "" then
          if newstr.startsWith("-") then thestr = s"$thestr - ${newstr.substring(1)}"
          else thestr = s"$thestr + $newstr"
      else thestr = newstr
      k += 1
    raisePower(thestr)

object Poly1d:
  /** Builds a polynomial from coefficients (highest power first); leading zeros are trimmed. */
  def apply[T](c: NDArray[T], variable: String = "x")(using d: NumDType[T]): Poly1d[T] =
    val a = PolyLegacy.atleast1d(c)
    if a.ndim > 1 then throw new IllegalArgumentException("Polynomial must be 1d only.")
    val vals = a.toArray
    val first = vals.indexWhere(x => d.toComplex(x) != Complex.Zero)
    val trimmed = if first < 0 then Array(d.zero)(using d.classTag) else vals.drop(first)
    new Poly1d(NDArray.fromArray(trimmed, Array(trimmed.length)), variable)

  /** A polynomial with the given roots (NumPy `poly1d(r, True)`). */
  def fromRoots[T](r: NDArray[T], variable: String = "x")(using t: ToInexact[T]): Poly1d[t.Out] =
    given InexactDType[t.Out] = t.dtype
    Poly1d(PolyLegacy.poly(r, t.dtype), variable)
