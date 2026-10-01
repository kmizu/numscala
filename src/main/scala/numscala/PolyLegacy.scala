package numscala

/** Result of `np.polyfit(..., full = true)`: coefficients (highest power first), the sum of
  * squared residuals (empty unless the fit is full rank and over-determined), the effective
  * rank of the scaled Vandermonde matrix, its singular values and the `rcond` used.
  */
final case class PolyfitResult(
    coef: NDArray[Double],
    residuals: NDArray[Double],
    rank: Int,
    singularValues: NDArray[Double],
    rcond: Double
)

/** Implementation of the legacy `np.poly*` routines (coefficients ordered highest power first). */
private[numscala] object PolyLegacy:

  def atleast1d[T](a: NDArray[T]): NDArray[T] = if a.ndim == 0 then a.reshape(1) else a

  def require1d[T](a: NDArray[T], what: String = "Input"): NDArray[T] =
    val r = atleast1d(a)
    if r.ndim != 1 then throw new IllegalArgumentException(s"$what must be a rank-1 array.")
    r

  def cast[T, U](a: NDArray[T], u: DType[U]): Array[U] =
    val src = a.toArray
    if a.dtype eq u then src.asInstanceOf[Array[U]] else
      val out = u.newArray(src.length)
      var i = 0
      while i < src.length do
        out(i) = u.castFrom(a.dtype, src(i))
        i += 1
      out

  def arr[U](data: Array[U])(using u: DType[U]): NDArray[U] = NDArray.fromArray(data, Array(data.length))

  // ------------------------------------------------------------------ arithmetic

  def add[U](a: Array[U], b: Array[U], d: NumDType[U], sub: Boolean): Array[U] =
    val n = math.max(a.length, b.length)
    val out = d.newArray(n)
    val oa = n - a.length
    val ob = n - b.length
    var i = 0
    while i < n do
      val x = if i >= oa then a(i - oa) else d.zero
      val y = if i >= ob then b(i - ob) else d.zero
      out(i) = if sub then d.minus(x, y) else d.plus(x, y)
      i += 1
    out

  def convolve[U](a: Array[U], b: Array[U], d: NumDType[U]): Array[U] =
    if a.isEmpty || b.isEmpty then throw new IllegalArgumentException("v cannot be empty")
    val out = d.newArray(a.length + b.length - 1)
    var k = 0
    while k < out.length do
      out(k) = d.zero
      k += 1
    var i = 0
    while i < a.length do
      var j = 0
      while j < b.length do
        out(i + j) = d.plus(out(i + j), d.times(a(i), b(j)))
        j += 1
      i += 1
    out

  /** `polydiv`: returns (quotient, remainder) in the inexact dtype `d`. */
  def div[U](u: Array[U], v: Array[U], d: InexactDType[U]): (Array[U], Array[U]) =
    val m = u.length - 1
    val n = v.length - 1
    val scale = d.div(d.one, v(0))
    val q = d.newArray(math.max(m - n + 1, 1))
    for i <- q.indices do q(i) = d.zero
    var r = u.clone()
    var k = 0
    while k < m - n + 1 do
      val dd = d.times(scale, r(k))
      q(k) = dd
      var j = 0
      while j <= n do
        r(k + j) = d.minus(r(k + j), d.times(dd, v(j)))
        j += 1
      k += 1
    // np.allclose(r[0], 0, rtol=1e-14): |r0| <= 1e-8
    while r.length > 1 && d.toComplex(r(0)).abs <= 1e-8 do r = r.drop(1)
    (q, r)

  def der[U](p: Array[U], m: Int, d: NumDType[U]): Array[U] =
    if m < 0 then throw new IllegalArgumentException("Order of derivative must be positive (see polyint)")
    var c = p
    var k = 0
    while k < m do
      val n = c.length - 1
      val y = d.newArray(math.max(n, 0))
      var i = 0
      while i < n do
        y(i) = d.times(c(i), d.fromInt(n - i))
        i += 1
      c = y
      k += 1
    c

  def int[U](p: Array[U], m: Int, k0: Array[U], d: InexactDType[U]): Array[U] =
    if m < 0 then throw new IllegalArgumentException("Order of integral must be positive (see polyder)")
    val k: Array[U] =
      if k0.length == 1 && m > 1 then Array.fill(m)(k0(0))(using d.classTag)
      else k0
    if k.length < m then throw new IllegalArgumentException("k must be a scalar or a rank-1 array of length 1 or >m.")
    var c = p
    var step = 0
    while step < m do
      val n = c.length
      val y = d.newArray(n + 1)
      var i = 0
      while i < n do
        y(i) = d.div(c(i), d.fromInt(n - i))
        i += 1
      y(n) = k(step)
      c = y
      step += 1
    c

  def horner[U](p: Array[U], x: U, d: NumDType[U]): U =
    var y = d.zero
    var i = 0
    while i < p.length do
      y = d.plus(d.times(y, x), p(i))
      i += 1
    y

  /** Polynomial coefficients (highest first) with the given complex roots. */
  def fromRootsC(roots: Array[Complex]): Array[Complex] =
    var a = Array(Complex.One)
    roots.foreach(z => a = convolve(a, Array(Complex.One, -z), DType.Complex128))
    a

  def fromRootsD(roots: Array[Double]): Array[Double] =
    var a = Array(1.0)
    roots.foreach(z => a = convolve(a, Array(1.0, -z), DType.Float64))
    a

  // ------------------------------------------------------------------ roots

  /** `np.roots` for coefficients already converted to complex / double. */
  def roots[T](p0: NDArray[T]): NDArray[Complex] =
    val p = require1d(p0)
    val dt = p.dtype
    val vals = p.toArray
    val nz = vals.indices.filter(i => dt.toComplex(vals(i)) != Complex.Zero)
    if nz.isEmpty then return arr(Array.empty[Complex])
    val trailing = vals.length - nz.last - 1
    val core = vals.slice(nz.head, nz.last + 1)
    val n = core.length
    val rts: Array[Complex] =
      if n <= 1 then Array.empty
      else if dt.isComplex then
        val c = core.map(dt.toComplex)
        val a = Array.tabulate(n - 1, n - 1) { (i, j) =>
          if i == 0 then -(c(j + 1) / c(0)) else if i == j + 1 then Complex.One else Complex.Zero
        }
        PolyLinAlg.eigvalsComplex(a)
      else
        val c = core.map(dt.toDouble)
        val a = Array.tabulate(n - 1, n - 1) { (i, j) =>
          if i == 0 then -c(j + 1) / c(0) else if i == j + 1 then 1.0 else 0.0
        }
        PolyLinAlg.eigvalsReal(a)
    arr(rts ++ Array.fill(trailing)(Complex.Zero))

  /** `np.poly`. */
  def poly[T, O](a0: NDArray[T], od: InexactDType[O]): NDArray[O] =
    val a = atleast1d(a0)
    val sh = a.shapeArr
    given DType[O] = od
    given scala.reflect.ClassTag[O] = od.classTag
    if sh.length == 2 && sh(0) == sh(1) && sh(0) != 0 then
      val n = sh(0)
      val eig =
        if a.dtype.isComplex then PolyLinAlg.eigvalsComplex(Array.tabulate(n, n)((i, j) => a.dtype.toComplex(a(i, j))))
        else PolyLinAlg.eigvalsReal(Array.tabulate(n, n)((i, j) => a.dtype.toDouble(a(i, j))))
      val c = fromRootsC(eig)
      arr(c.map(z => if od.isComplex then od.fromComplex(z) else od.fromDouble(z.re)))
    else if sh.length == 1 then
      if od.isComplex then arr(fromRootsC(a.toArray.map(a.dtype.toComplex)).map(od.fromComplex))
      else arr(fromRootsD(a.toArray.map(a.dtype.toDouble)).map(od.fromDouble))
    else throw new IllegalArgumentException("input must be 1d or non-empty square 2d array.")

  // ------------------------------------------------------------------ fitting

  final case class FitCore(
      coef: NDArray[Double],
      resid: NDArray[Double],
      rank: Int,
      sv: NDArray[Double],
      rcond: Double,
      lhs: Array[Array[Double]],
      scale: Array[Double]
  )

  def fit[A, B](x0: NDArray[A], y0: NDArray[B], deg: Int, rcond0: Double, w0: NDArray[?] | Null): FitCore =
    if deg < 0 then throw new IllegalArgumentException("expected deg >= 0")
    val order = deg + 1
    if x0.ndim != 1 then throw new IllegalArgumentException("expected 1D vector for x")
    if x0.size == 0 then throw new IllegalArgumentException("expected non-empty vector for x")
    if y0.ndim < 1 || y0.ndim > 2 then throw new IllegalArgumentException("expected 1D or 2D array for y")
    if x0.shapeArr(0) != y0.shapeArr(0) then throw new IllegalArgumentException("expected x and y to have same length")
    if y0.dtype.isComplex || x0.dtype.isComplex then
      throw new IllegalArgumentException("complex data is not supported by polyfit")
    val x = x0.toArray.map(x0.dtype.toDouble)
    val m = x.length
    val nrhs = if y0.ndim == 1 then 1 else y0.shapeArr(1)
    val yArr = y0.toArray.map(y0.dtype.toDouble)
    val rhs = Array.tabulate(m, nrhs)((i, j) => yArr(i * nrhs + j))
    val rcond = if rcond0 < 0 || rcond0.isNaN then m * java.lang.Math.ulp(1.0) else rcond0
    val lhs = Array.ofDim[Double](m, order)
    // vander via repeated multiplication like np.vander
    for i <- 0 until m do
      var acc = 1.0
      var j = order - 1
      while j >= 0 do
        lhs(i)(j) = acc
        acc *= x(i)
        j -= 1
    if w0 != null then
      val w = w0.nn
      if w.ndim != 1 then throw new IllegalArgumentException("expected a 1-d array for weights")
      if w.shapeArr(0) != m then throw new IllegalArgumentException("expected w and y to have the same length")
      val wv = w.toArray.map(w.dtype.asInstanceOf[DType[Any]].toDouble)
      for i <- 0 until m do
        for j <- 0 until order do lhs(i)(j) *= wv(i)
        for j <- 0 until nrhs do rhs(i)(j) *= wv(i)
    val scale = Array.tabulate(order)(j => math.sqrt((0 until m).map(i => lhs(i)(j) * lhs(i)(j)).sum))
    for i <- 0 until m; j <- 0 until order do lhs(i)(j) /= scale(j)
    val res = PolyLinAlg.lstsq(lhs, m, order, rhs, nrhs, rcond)
    val c = Array.tabulate(order, nrhs)((j, r) => res.x(j)(r) / scale(j))
    val coef =
      if y0.ndim == 1 then NDArray.fromArray(c.map(_(0)), Array(order))
      else NDArray.fromArray(c.flatten, Array(order, nrhs))
    FitCore(
      coef,
      NDArray.fromArray(res.resid, Array(res.resid.length)),
      res.rank,
      NDArray.fromArray(res.s, Array(res.s.length)),
      rcond,
      lhs,
      scale
    )

  def fitCov(f: FitCore, n: Int, order: Int, ndimY: Int, unscaled: Boolean): (NDArray[Double], NDArray[Double]) =
    val lhs = f.lhs
    val m = lhs.length
    val ata = Array.tabulate(order, order)((i, j) => (0 until m).map(r => lhs(r)(i) * lhs(r)(j)).sum)
    val vbase = PolyLinAlg.inv(ata)
    for i <- 0 until order; j <- 0 until order do vbase(i)(j) /= f.scale(i) * f.scale(j)
    val fac: Array[Double] =
      if unscaled then Array(1.0)
      else
        if n <= order then
          throw new IllegalArgumentException(
            "the number of data points must exceed order to scale the covariance matrix"
          )
        f.resid.toArray.map(_ / (n - order))
    if ndimY == 1 then
      val s = if fac.isEmpty then Double.NaN else fac(0)
      (f.coef, NDArray.fromArray(vbase.flatten.map(_ * s), Array(order, order)))
    else
      val k = fac.length
      val out = new Array[Double](order * order * k)
      for i <- 0 until order; j <- 0 until order; r <- 0 until k do out((i * order + j) * k + r) = vbase(i)(j) * fac(r)
      (f.coef, NDArray.fromArray(out, Array(order, order, k)))

  // ------------------------------------------------------------------ formatting

  /** Python's `'%.{prec}g' % x`. */
  def pyG(x: Double, prec: Int = 4): String =
    if x.isNaN then "nan"
    else if x.isInfinite then (if x > 0 then "inf" else "-inf")
    else if x == 0.0 then (if 1.0 / x < 0 then "-0" else "0")
    else
      val bd = new java.math.BigDecimal(x).round(new java.math.MathContext(prec, java.math.RoundingMode.HALF_EVEN))
      val exp = bd.precision - bd.scale - 1
      if exp >= -4 && exp < prec then
        val s = bd.setScale(math.max(prec - 1 - exp, 0), java.math.RoundingMode.HALF_EVEN).toPlainString
        if s.contains('.') then s.reverse.dropWhile(_ == '0').reverse.stripSuffix(".") else s
      else
        val digits = bd.unscaledValue.abs.toString.padTo(prec, '0')
        val mant0 = digits.head.toString + "." + digits.tail
        val mant = mant0.reverse.dropWhile(_ == '0').reverse.stripSuffix(".")
        val es = (if exp < 0 then "-" else "+") + f"${math.abs(exp)}%02d"
        (if x < 0 then "-" else "") + mant + "e" + es

  /** NumPy's `_raise_power`: moves `**n` exponents to a line above. */
  def raisePower(astr: String, wrap: Int = 70): String =
    val pat = """\*\*([0-9]*)""".r
    var n = 0
    var line1 = ""
    var line2 = ""
    val output = new StringBuilder(" ")
    var done = false
    while !done do
      pat.findFirstMatchIn(astr.substring(n)) match
        case None => done = true
        case Some(mat) =>
          val start = n + mat.start
          val end = n + mat.end
          val power = mat.group(1)
          val partstr = astr.substring(n, start)
          n = end
          val toadd2 = partstr + " " * (power.length - 1)
          val toadd1 = " " * (partstr.length - 1) + power
          if line2.length + toadd2.length > wrap || line1.length + toadd1.length > wrap then
            output ++= line1 + "\n" + line2 + "\n "
            line1 = toadd1
            line2 = toadd2
          else
            line2 += partstr + " " * (power.length - 1)
            line1 += " " * (partstr.length - 1) + power
    output ++= line1 + "\n" + line2
    output.toString + astr.substring(n)
