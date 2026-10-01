package numscala

/** A commutative "ring" used to run a basis' Clenshaw recurrence on something other than
  * scalars (series multiplication, composition, conversion to the power basis).
  */
private[numscala] trait PolyRing[A]:
  def const(c: Double): A
  def add(a: A, b: A): A
  def sub(a: A, b: A): A
  def scale(a: A, s: Double): A
  def div(a: A, s: Double): A
  def mulx(a: A): A

/** Series algorithms of one polynomial basis (`numpy.polynomial.<kind>`), on 1-D coefficient
  * arrays ordered from low to high degree.
  */
private[numscala] abstract class PolyBasis(
    val prefix: String,
    val className: String,
    val basisName: String,
    val defaultDomain: Array[Double],
    val defaultWindow: Array[Double]
):
  import PolyBasis.*

  // ---------------------------------------------------------------- per-basis primitives

  /** Multiplication by `x`. */
  protected def mulxRaw(c: Array[Double]): Array[Double]
  /** Scalar evaluation, mirroring NumPy's `<kind>val`. */
  def valScalar(x: Double, c: Array[Double]): Double
  /** Clenshaw/Horner over a ring. */
  def clenshaw[A](c: Array[Double], r: PolyRing[A]): A
  /** One differentiation step on coefficient rows (length n >= 2 -> n-1). Rows may be mutated. */
  protected def derStep(c: Array[Array[Double]]): Array[Array[Double]]
  /** One integration step (length n -> n+1), without the integration constant. */
  protected def intStep(c: Array[Array[Double]]): Array[Array[Double]]
  /** First Vandermonde column `P_1(x)` and the recurrence `P_i` from `P_{i-1}`, `P_{i-2}`. */
  protected def vander1(x: Double): Double
  protected def vanderNext(i: Int, x: Double, p1: Double, p2: Double): Double
  /** The companion matrix of NumPy's `<kind>companion` for `len(c) >= 3`. */
  protected def companionRaw(c: Array[Double]): Array[Array[Double]]
  /** The single root / 1x1 companion of a degree-1 series. */
  protected def linearRoot(c: Array[Double]): Double
  /** `<kind>line(off, scl)`. */
  def line(off: Double, scl: Double): Array[Double]

  // ---------------------------------------------------------------- arithmetic

  def mulx(c0: Array[Double]): Array[Double] =
    val c = asSeries(c0)
    if c.length == 1 && c(0) == 0.0 then c else mulxRaw(c)

  def add(c1: Array[Double], c2: Array[Double]): Array[Double] = trimseq(padOp(asSeries(c1), asSeries(c2), _ + _))
  def sub(c1: Array[Double], c2: Array[Double]): Array[Double] = trimseq(padOp(asSeries(c1), asSeries(c2), _ - _))

  def mul(c1: Array[Double], c2: Array[Double]): Array[Double] =
    val a = asSeries(c1)
    val b = asSeries(c2)
    val (c, xs) = if a.length > b.length then (b, a) else (a, b)
    trimseq(clenshaw(c, seriesRing(xs)))

  /** Ring for computing `c * xs` with this basis' own multiplication by x. */
  private def seriesRing(xs: Array[Double]): PolyRing[Array[Double]] = new PolyRing[Array[Double]]:
    def const(c: Double): Array[Double] = xs.map(_ * c)
    def add(a: Array[Double], b: Array[Double]): Array[Double] = PolyBasis.this.add(a, b)
    def sub(a: Array[Double], b: Array[Double]): Array[Double] = PolyBasis.this.sub(a, b)
    def scale(a: Array[Double], s: Double): Array[Double] = a.map(_ * s)
    def div(a: Array[Double], s: Double): Array[Double] = a.map(_ / s)
    def mulx(a: Array[Double]): Array[Double] = PolyBasis.this.mulx(a)

  def div(c1: Array[Double], c2: Array[Double]): (Array[Double], Array[Double]) =
    val a = asSeries(c1)
    val b = asSeries(c2)
    if b.last == 0.0 then throw new ArithmeticException("division by zero")
    val lc1 = a.length
    val lc2 = b.length
    if lc1 < lc2 then (Array(a(0) * 0.0), a)
    else if lc2 == 1 then (a.map(_ / b.last), Array(a(0) * 0.0))
    else
      val quo = new Array[Double](lc1 - lc2 + 1)
      var rem = a
      var i = lc1 - lc2
      while i >= 0 do
        val p = mul(Array.fill(i)(0.0) :+ 1.0, b)
        val q = rem.last / p.last
        rem = Array.tabulate(rem.length - 1)(j => rem(j) - q * p(j))
        quo(i) = q
        i -= 1
      (quo, trimseq(rem))

  def pow(c0: Array[Double], power: Int, maxpower: Int): Array[Double] =
    val c = asSeries(c0)
    if power < 0 then throw new IllegalArgumentException("Power must be a non-negative integer.")
    else if maxpower >= 0 && power > maxpower then throw new IllegalArgumentException("Power is too large")
    else if power == 0 then Array(1.0)
    else if power == 1 then c
    else
      var prd = c
      for _ <- 2 to power do prd = mul(prd, c)
      prd

  // ---------------------------------------------------------------- calculus

  def der(c: Array[Array[Double]], m: Int, scl: Double): Array[Array[Double]] =
    if m < 0 then throw new IllegalArgumentException("The order of derivation must be non-negative")
    if m == 0 then return c.map(_.clone())
    var cur = c.map(_.clone())
    val n0 = cur.length
    if m >= n0 then return Array(cur(0).map(_ * 0.0))
    for _ <- 0 until m do
      cur = cur.map(_.map(_ * scl))
      cur = derStep(cur)
    cur

  def integ(c: Array[Array[Double]], m: Int, k0: Seq[Double], lbnd: Double, scl: Double): Array[Array[Double]] =
    if m < 0 then throw new IllegalArgumentException("The order of integration must be non-negative")
    if k0.length > m then throw new IllegalArgumentException("Too many integration constants")
    if m == 0 then return c.map(_.clone())
    val k = k0 ++ Seq.fill(m - k0.length)(0.0)
    var cur = c.map(_.clone())
    for i <- 0 until m do
      val n = cur.length
      cur = cur.map(_.map(_ * scl))
      if n == 1 && cur(0).forall(_ == 0.0) then cur(0) = cur(0).map(_ + k(i))
      else
        val tmp = intStep(cur)
        val cols = tmp(0).length
        for r <- 0 until cols do
          val col = Array.tabulate(tmp.length)(j => tmp(j)(r))
          tmp(0)(r) += k(i) - valScalar(lbnd, col)
        cur = tmp
    cur

  // ---------------------------------------------------------------- construction / roots

  def fromroots(roots: Array[Double]): Array[Double] =
    if roots.isEmpty then Array(1.0)
    else
      var p: IndexedSeq[Array[Double]] = roots.sorted.toIndexedSeq.map(r => line(-r, 1.0))
      var n = p.length
      while n > 1 do
        val m = n / 2
        val rr = n % 2
        val tmp = Array.tabulate(m)(i => mul(p(i), p(i + m)))
        if rr == 1 then tmp(0) = mul(tmp(0), p.last)
        p = tmp.toIndexedSeq
        n = m
      p(0)

  def companion(c0: Array[Double]): Array[Array[Double]] =
    val c = asSeries(c0)
    if c.length < 2 then throw new IllegalArgumentException("Series must have maximum degree of at least 1.")
    if c.length == 2 then Array(Array(linearRoot(c))) else companionRaw(c)

  def roots(c0: Array[Double]): Array[Complex] =
    val c = asSeries(c0)
    if c.length < 2 then Array.empty
    else if c.length == 2 then Array(Complex(linearRoot(c), 0.0))
    else
      val m = companionRaw(c)
      val n = m.length
      val rot = Array.tabulate(n, n)((i, j) => m(n - 1 - i)(n - 1 - j))
      val r = PolyLinAlg.eigvalsReal(rot)
      r.sortInPlace()(using Complex.ordering)
      r

  // ---------------------------------------------------------------- Vandermonde / fitting

  def vanderRow(x: Double, deg: Int, out: Array[Double], off: Int): Unit =
    out(off) = 1.0
    if deg > 0 then out(off + 1) = vander1(x)
    var i = 2
    while i <= deg do
      out(off + i) = vanderNext(i, x, out(off + i - 1), out(off + i - 2))
      i += 1

  def vander(x: NDArray[Double], deg: Int): NDArray[Double] =
    if deg < 0 then throw new IllegalArgumentException("deg must be non-negative")
    val xs = x.toArray
    val out = new Array[Double](xs.length * (deg + 1))
    var i = 0
    while i < xs.length do
      vanderRow(xs(i), deg, out, i * (deg + 1))
      i += 1
    NDArray.fromArray(out, x.shapeArr :+ (deg + 1))

  /** NumPy's `_fit`. Returns (coef: deg+1 [x nrhs], resid, rank, sv, rcond). */
  def fit(x0: NDArray[?], y0: NDArray[?], degs: Seq[Int], rcond0: Double, w0: NDArray[?] | Null)
      : (NDArray[Double], NDArray[Double], Int, NDArray[Double], Double) =
    if degs.isEmpty then throw new IllegalArgumentException("deg must be an int or non-empty 1-D array of int")
    if degs.min < 0 then throw new IllegalArgumentException("expected deg >= 0")
    if x0.ndim != 1 then throw new IllegalArgumentException("expected 1D vector for x")
    if x0.size == 0 then throw new IllegalArgumentException("expected non-empty vector for x")
    if y0.ndim < 1 || y0.ndim > 2 then throw new IllegalArgumentException("expected 1D or 2D array for y")
    if x0.shapeArr(0) != y0.shapeArr(0) then throw new IllegalArgumentException("expected x and y to have same length")
    val x = toD(x0).toArray
    val m = x.length
    val nrhs = if y0.ndim == 1 then 1 else y0.shapeArr(1)
    val yv = toD(y0).toArray
    val sorted = degs.sorted.toArray
    val lmax = sorted.last
    val cols = if degs.length == 1 then (0 to lmax).toArray else sorted
    val full = new Array[Double](lmax + 1)
    val lhs = Array.ofDim[Double](m, cols.length)
    for i <- 0 until m do
      vanderRow(x(i), lmax, full, 0)
      for j <- cols.indices do lhs(i)(j) = full(cols(j))
    val rhs = Array.tabulate(m, nrhs)((i, j) => yv(i * nrhs + j))
    if w0 != null then
      val w = toD(w0.nn)
      if w.ndim != 1 then throw new IllegalArgumentException("expected 1D vector for w")
      if w.size != m then throw new IllegalArgumentException("expected x and w to have same length")
      val wv = w.toArray
      for i <- 0 until m do
        for j <- cols.indices do lhs(i)(j) *= wv(i)
        for j <- 0 until nrhs do rhs(i)(j) *= wv(i)
    val rcond = if rcond0 < 0 || rcond0.isNaN then m * java.lang.Math.ulp(1.0) else rcond0
    val scl = cols.indices.map { j =>
      val s = math.sqrt((0 until m).map(i => lhs(i)(j) * lhs(i)(j)).sum)
      if s == 0.0 then 1.0 else s
    }.toArray
    for i <- 0 until m; j <- cols.indices do lhs(i)(j) /= scl(j)
    val res = PolyLinAlg.lstsq(lhs, m, cols.length, rhs, nrhs, rcond)
    val cc = Array.ofDim[Double](lmax + 1, nrhs)
    for j <- cols.indices; r <- 0 until nrhs do cc(cols(j))(r) = res.x(j)(r) / scl(j)
    val coef =
      if y0.ndim == 1 then NDArray.fromArray(cc.map(_(0)), Array(lmax + 1))
      else NDArray.fromArray(cc.flatten, Array(lmax + 1, nrhs))
    (coef, NDArray.fromArray(res.resid), res.rank, NDArray.fromArray(res.s), rcond)

  // ---------------------------------------------------------------- conversion

  /** Converts this basis' series to power-series coefficients (`cheb2poly`, ...). */
  def toPower(c0: Array[Double]): Array[Double] =
    val c = asSeries(c0)
    clenshaw(c, composeRing(PolyBasis.Power, None))

  /** Converts power-series coefficients to this basis (`poly2cheb`, ...). */
  def fromPower(pol0: Array[Double]): Array[Double] =
    val pol = asSeries(pol0)
    var res = Array(0.0)
    var i = pol.length - 1
    while i >= 0 do
      res = add(mulx(res), Array(pol(i)))
      i -= 1
    res

  /** Evaluates this series at a series `arg` expressed in basis `target` (composition). */
  def evalAtSeries(c: Array[Double], target: PolyBasis, arg: Array[Double]): Array[Double] =
    clenshaw(asSeries(c), composeRing(target, Some(arg)))

  // ---------------------------------------------------------------- Gauss quadrature

  def gauss(deg: Int): (Array[Double], Array[Double]) =
    throw new UnsupportedOperationException(s"${prefix}gauss is not defined")

  def weight(x: Double): Double = throw new UnsupportedOperationException(s"${prefix}weight is not defined")

  /** Eigenvalues (ascending) of the symmetric tridiagonal companion of the degree-`deg` basis polynomial. */
  protected def symCompanionEig(deg: Int): Array[Double] =
    val c = Array.fill(deg)(0.0) :+ 1.0
    val m = companion(c)
    val n = m.length
    PolyLinAlg.eigvalshTridiag(Array.tabulate(n)(i => m(i)(i)), Array.tabulate(math.max(n - 1, 0))(i => m(i + 1)(i)))

private[numscala] object PolyBasis:

  def toD(a: NDArray[?]): NDArray[Double] =
    val d = a.dtype.asInstanceOf[DType[Any]]
    if d.isComplex then throw new IllegalArgumentException("complex coefficients/arguments are not supported")
    if d eq DType.Float64 then a.asInstanceOf[NDArray[Double]]
    else a.asInstanceOf[NDArray[Any]].map(d.toDouble)

  /** NumPy `as_series` for one 1-D series with trimming of trailing zeros. */
  def asSeries(c: Array[Double], trim: Boolean = true): Array[Double] =
    if c.isEmpty then throw new IllegalArgumentException("Coefficient array is empty")
    if trim then trimseq(c) else c

  def seriesOf(a: NDArray[?], trim: Boolean = true): Array[Double] =
    val r = if a.ndim == 0 then a.reshape(1) else a
    if r.ndim != 1 then throw new IllegalArgumentException("Coefficient array is not 1-d")
    asSeries(toD(r).toArray, trim)

  def trimseq(c: Array[Double]): Array[Double] =
    if c.isEmpty || c.last != 0.0 then c
    else
      var i = c.length - 1
      while i > 0 && c(i) == 0.0 do i -= 1
      c.take(i + 1)

  def trimcoef(c0: Array[Double], tol: Double): Array[Double] =
    if tol < 0 then throw new IllegalArgumentException("tol must be non-negative")
    val c = asSeries(c0)
    val ind = c.indices.filter(i => math.abs(c(i)) > tol)
    if ind.isEmpty then Array(c(0) * 0.0) else c.take(ind.last + 1)

  def padOp(a: Array[Double], b: Array[Double], f: (Double, Double) => Double): Array[Double] =
    Array.tabulate(math.max(a.length, b.length)) { i =>
      f(if i < a.length then a(i) else 0.0, if i < b.length then b(i) else 0.0)
    }

  /** Ring for evaluating at a series in `target`'s basis (`arg = None`: at `x` itself). */
  def composeRing(target: PolyBasis, arg: Option[Array[Double]]): PolyRing[Array[Double]] = new PolyRing[Array[Double]]:
    def const(c: Double): Array[Double] = Array(c)
    def add(a: Array[Double], b: Array[Double]): Array[Double] = target.add(a, b)
    def sub(a: Array[Double], b: Array[Double]): Array[Double] = target.sub(a, b)
    def scale(a: Array[Double], s: Double): Array[Double] = a.map(_ * s)
    def div(a: Array[Double], s: Double): Array[Double] = a.map(_ / s)
    def mulx(a: Array[Double]): Array[Double] = arg match
      case None => target.mulx(a)
      case Some(x) => target.mul(a, x)

  def mapparms(oldD: Array[Double], newD: Array[Double]): (Double, Double) =
    val oldlen = oldD(1) - oldD(0)
    val newlen = newD(1) - newD(0)
    val off = (oldD(1) * newD(0) - oldD(0) * newD(1)) / oldlen
    val scl = newlen / oldlen
    (off, scl)

  private[numscala] def Sym: Array[Double] = Array(-1.0, 1.0)

  // ================================================================ power series

  object Power extends PolyBasis("poly", "Polynomial", "x", Sym, Sym):
    protected def mulxRaw(c: Array[Double]): Array[Double] = 0.0 +: c
    def valScalar(x: Double, c: Array[Double]): Double =
      var c0 = c.last + x * 0.0
      var i = c.length - 2
      while i >= 0 do
        c0 = c(i) + c0 * x
        i -= 1
      c0
    def clenshaw[A](c: Array[Double], r: PolyRing[A]): A =
      var c0 = r.const(c.last)
      var i = c.length - 2
      while i >= 0 do
        c0 = r.add(r.const(c(i)), r.mulx(c0))
        i -= 1
      c0
    override def mul(c1: Array[Double], c2: Array[Double]): Array[Double] =
      val a = asSeries(c1)
      val b = asSeries(c2)
      trimseq(PolyLegacy.convolve(a, b, DType.Float64))
    override def div(c1: Array[Double], c2: Array[Double]): (Array[Double], Array[Double]) =
      val a = asSeries(c1).clone()
      val b = asSeries(c2)
      if b.last == 0.0 then throw new ArithmeticException("division by zero")
      val lc1 = a.length
      val lc2 = b.length
      if lc1 < lc2 then (Array(a(0) * 0.0), a)
      else if lc2 == 1 then (a.map(_ / b.last), Array(a(0) * 0.0))
      else
        val dlen = lc1 - lc2
        val scl = b.last
        val bb = b.dropRight(1).map(_ / scl)
        var i = dlen
        var j = lc1 - 1
        while i >= 0 do
          val f = a(j)
          for t <- 0 until lc2 - 1 do a(i + t) -= bb(t) * f
          i -= 1
          j -= 1
        (a.drop(j + 1).map(_ / scl), trimseq(a.take(j + 1)))
    override def pow(c0: Array[Double], power: Int, maxpower: Int): Array[Double] =
      val c = asSeries(c0)
      if power < 0 then throw new IllegalArgumentException("Power must be a non-negative integer.")
      else if maxpower >= 0 && power > maxpower then throw new IllegalArgumentException("Power is too large")
      else if power == 0 then Array(1.0)
      else if power == 1 then c
      else
        var prd = c
        for _ <- 2 to power do prd = PolyLegacy.convolve(prd, c, DType.Float64)
        prd
    protected def derStep(c: Array[Array[Double]]): Array[Array[Double]] =
      val n = c.length - 1
      Array.tabulate(n)(j => c(j + 1).map(_ * (j + 1)))
    protected def intStep(c: Array[Array[Double]]): Array[Array[Double]] =
      val n = c.length
      Array.tabulate(n + 1)(j => if j == 0 then c(0).map(_ * 0.0) else c(j - 1).map(_ / j))
    protected def vander1(x: Double): Double = x
    protected def vanderNext(i: Int, x: Double, p1: Double, p2: Double): Double = p1 * x
    protected def companionRaw(c: Array[Double]): Array[Array[Double]] =
      val n = c.length - 1
      val mat = Array.ofDim[Double](n, n)
      for i <- 1 until n do mat(i)(i - 1) = 1.0
      for i <- 0 until n do mat(i)(n - 1) -= c(i) / c(n)
      mat
    protected def linearRoot(c: Array[Double]): Double = -c(0) / c(1)
    def line(off: Double, scl: Double): Array[Double] = if scl != 0 then Array(off, scl) else Array(off)

  // ================================================================ Chebyshev

  object Cheb extends PolyBasis("cheb", "Chebyshev", "T", Sym, Sym):
    protected def mulxRaw(c: Array[Double]): Array[Double] =
      val prd = new Array[Double](c.length + 1)
      prd(0) = c(0) * 0.0
      prd(1) = c(0)
      if c.length > 1 then
        for i <- 1 until c.length do
          val tmp = c(i) / 2
          prd(i + 1) = tmp
          prd(i - 1) += tmp
      prd
    def valScalar(x: Double, c: Array[Double]): Double =
      if c.length == 1 then c(0) + 0.0 * x
      else if c.length == 2 then c(0) + c(1) * x
      else
        val x2 = 2 * x
        var c0 = c(c.length - 2)
        var c1 = c(c.length - 1)
        var i = 3
        while i <= c.length do
          val tmp = c0
          c0 = c(c.length - i) - c1
          c1 = tmp + c1 * x2
          i += 1
        c0 + c1 * x
    def clenshaw[A](c: Array[Double], r: PolyRing[A]): A =
      if c.length == 1 then r.const(c(0))
      else
        var c0 = r.const(c(c.length - 2))
        var c1 = r.const(c(c.length - 1))
        var i = 3
        while i <= c.length do
          val tmp = c0
          c0 = r.sub(r.const(c(c.length - i)), c1)
          c1 = r.add(tmp, r.scale(r.mulx(c1), 2.0))
          i += 1
        r.add(c0, r.mulx(c1))
    protected def derStep(c: Array[Array[Double]]): Array[Array[Double]] =
      val n = c.length - 1
      val der = new Array[Array[Double]](n)
      var j = n
      while j > 2 do
        der(j - 1) = c(j).map(_ * (2 * j))
        val add = c(j).map(v => (j * v) / (j - 2))
        c(j - 2) = c(j - 2).indices.map(r => c(j - 2)(r) + add(r)).toArray
        j -= 1
      if n > 1 then der(1) = c(2).map(_ * 4)
      der(0) = c(1).clone()
      der
    protected def intStep(c: Array[Array[Double]]): Array[Array[Double]] =
      val n = c.length
      val tmp = Array.fill(n + 1)(new Array[Double](c(0).length))
      tmp(0) = c(0).map(_ * 0.0)
      tmp(1) = c(0).clone()
      if n > 1 then tmp(2) = c(1).map(_ / 4)
      for j <- 2 until n do
        tmp(j + 1) = c(j).map(_ / (2 * (j + 1)))
        tmp(j - 1) = tmp(j - 1).indices.map(r => tmp(j - 1)(r) - c(j)(r) / (2 * (j - 1))).toArray
      tmp
    protected def vander1(x: Double): Double = x
    protected def vanderNext(i: Int, x: Double, p1: Double, p2: Double): Double = p1 * (2 * x) - p2
    protected def companionRaw(c: Array[Double]): Array[Array[Double]] =
      val n = c.length - 1
      val mat = Array.ofDim[Double](n, n)
      val scl = Array.tabulate(n)(i => if i == 0 then 1.0 else math.sqrt(0.5))
      for i <- 0 until n - 1 do
        val v = if i == 0 then math.sqrt(0.5) else 0.5
        mat(i)(i + 1) = v
        mat(i + 1)(i) = v
      for i <- 0 until n do mat(i)(n - 1) -= (c(i) / c(n)) * (scl(i) / scl(n - 1)) * 0.5
      mat
    protected def linearRoot(c: Array[Double]): Double = -c(0) / c(1)
    def line(off: Double, scl: Double): Array[Double] = if scl != 0 then Array(off, scl) else Array(off)
    override def gauss(deg: Int): (Array[Double], Array[Double]) =
      if deg <= 0 then throw new IllegalArgumentException("deg must be a positive integer")
      val x = Array.tabulate(deg)(i => math.cos(math.Pi * (2 * i + 1) / (2.0 * deg)))
      (x, Array.fill(deg)(math.Pi / deg))
    override def weight(x: Double): Double = 1.0 / (math.sqrt(1.0 + x) * math.sqrt(1.0 - x))

  // ================================================================ Legendre

  object Leg extends PolyBasis("leg", "Legendre", "P", Sym, Sym):
    protected def mulxRaw(c: Array[Double]): Array[Double] =
      val prd = new Array[Double](c.length + 1)
      prd(0) = c(0) * 0.0
      prd(1) = c(0)
      for i <- 1 until c.length do
        val j = i + 1
        val k = i - 1
        val s = i + j
        prd(j) = (c(i) * j) / s
        prd(k) += (c(i) * i) / s
      prd
    def valScalar(x: Double, c: Array[Double]): Double =
      if c.length == 1 then c(0) + 0.0 * x
      else if c.length == 2 then c(0) + c(1) * x
      else
        var nd = c.length
        var c0 = c(c.length - 2)
        var c1 = c(c.length - 1)
        var i = 3
        while i <= c.length do
          val tmp = c0
          nd -= 1
          c0 = c(c.length - i) - (c1 * (nd - 1)) / nd
          c1 = tmp + (c1 * x * (2 * nd - 1)) / nd
          i += 1
        c0 + c1 * x
    def clenshaw[A](c: Array[Double], r: PolyRing[A]): A =
      if c.length == 1 then r.const(c(0))
      else
        var nd = c.length
        var c0 = r.const(c(c.length - 2))
        var c1 = r.const(c(c.length - 1))
        var i = 3
        while i <= c.length do
          val tmp = c0
          nd -= 1
          c0 = r.sub(r.const(c(c.length - i)), r.div(r.scale(c1, nd - 1), nd))
          c1 = r.add(tmp, r.div(r.scale(r.mulx(c1), 2 * nd - 1), nd))
          i += 1
        r.add(c0, r.mulx(c1))
    protected def derStep(c: Array[Array[Double]]): Array[Array[Double]] =
      val n = c.length - 1
      val der = new Array[Array[Double]](n)
      var j = n
      while j > 2 do
        der(j - 1) = c(j).map(_ * (2 * j - 1))
        c(j - 2) = c(j - 2).indices.map(r => c(j - 2)(r) + c(j)(r)).toArray
        j -= 1
      if n > 1 then der(1) = c(2).map(_ * 3)
      der(0) = c(1).clone()
      der
    protected def intStep(c: Array[Array[Double]]): Array[Array[Double]] =
      val n = c.length
      val tmp = Array.fill(n + 1)(new Array[Double](c(0).length))
      tmp(0) = c(0).map(_ * 0.0)
      tmp(1) = c(0).clone()
      if n > 1 then tmp(2) = c(1).map(_ / 3)
      for j <- 2 until n do
        val t = c(j).map(_ / (2 * j + 1))
        tmp(j + 1) = t
        tmp(j - 1) = tmp(j - 1).indices.map(r => tmp(j - 1)(r) - t(r)).toArray
      tmp
    protected def vander1(x: Double): Double = x
    protected def vanderNext(i: Int, x: Double, p1: Double, p2: Double): Double =
      (p1 * x * (2 * i - 1) - p2 * (i - 1)) / i
    protected def companionRaw(c: Array[Double]): Array[Array[Double]] =
      val n = c.length - 1
      val mat = Array.ofDim[Double](n, n)
      val scl = Array.tabulate(n)(i => 1.0 / math.sqrt(2.0 * i + 1))
      for i <- 0 until n - 1 do
        val v = (i + 1) * scl(i) * scl(i + 1)
        mat(i)(i + 1) = v
        mat(i + 1)(i) = v
      for i <- 0 until n do mat(i)(n - 1) -= (c(i) / c(n)) * (scl(i) / scl(n - 1)) * (n.toDouble / (2 * n - 1))
      mat
    protected def linearRoot(c: Array[Double]): Double = -c(0) / c(1)
    def line(off: Double, scl: Double): Array[Double] = if scl != 0 then Array(off, scl) else Array(off)
    override def gauss(deg: Int): (Array[Double], Array[Double]) =
      if deg <= 0 then throw new IllegalArgumentException("deg must be a positive integer")
      val c = Array.fill(deg)(0.0) :+ 1.0
      val x = symCompanionEig(deg)
      val dc = der(c.map(Array(_)), 1, 1.0).map(_(0))
      val dy = x.map(valScalar(_, c))
      var df = x.map(valScalar(_, dc))
      for i <- x.indices do x(i) -= dy(i) / df(i)
      var fm = x.map(valScalar(_, c.drop(1)))
      val fmax = fm.map(math.abs).max
      fm = fm.map(_ / fmax)
      val dmax = df.map(math.abs).max
      df = df.map(_ / dmax)
      symmetrize(x, Array.tabulate(deg)(i => 1.0 / (fm(i) * df(i))), 2.0)
    override def weight(x: Double): Double = x * 0.0 + 1.0

  /** NumPy's final Gauss step: symmetrize nodes/weights and normalize weights to `total`. */
  private[numscala] def symmetrize(x: Array[Double], w0: Array[Double], total: Double): (Array[Double], Array[Double]) =
    val n = x.length
    val w = Array.tabulate(n)(i => (w0(i) + w0(n - 1 - i)) / 2)
    val xs = Array.tabulate(n)(i => (x(i) - x(n - 1 - i)) / 2)
    val s = w.sum
    (xs, w.map(_ * (total / s)))

  def Herm: PolyBasis = PolyBasisHermite.Herm
  def HermE: PolyBasis = PolyBasisHermite.HermE
  def Lag: PolyBasis = PolyBasisHermite.Lag

  lazy val all: Seq[PolyBasis] = Seq(Power, Cheb, Leg, Herm, HermE, Lag)
