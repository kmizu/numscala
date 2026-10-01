package com.github.kmizu.numscala

import PolyBasis.{seriesOf, toD}

/** NDArray-level implementations shared by the functional `numpy.polynomial.<kind>` modules. */
private[numscala] object PolyFunctional:

  private def arr(a: Array[Double]): NDArray[Double] = NDArray.fromArray(a, Array(a.length))

  private def coefs(c: NDArray[?]): NDArray[Double] =
    val d = toD(c)
    val r = if d.ndim == 0 then d.reshape(1) else d
    if r.size == 0 then throw new IllegalArgumentException("Coefficient array is empty")
    r

  // ---------------------------------------------------------------- evaluation

  def valArr(b: PolyBasis, x: NDArray[?], c: NDArray[?], tensor: Boolean): NDArray[Double] =
    val cd = coefs(c)
    val xd = toD(x)
    if cd.ndim == 1 then
      val cs = cd.toArray
      xd.map(b.valScalar(_, cs))
    else
      val n = cd.shapeArr(0)
      val rest = cd.shapeArr.tail
      val nr = Shape.size(rest)
      val flat = cd.toArray
      val cols = Array.tabulate(nr)(r => Array.tabulate(n)(k => flat(k * nr + r)))
      if tensor then
        val xs = xd.toArray
        val out = new Array[Double](nr * xs.length)
        for r <- 0 until nr; i <- xs.indices do out(r * xs.length + i) = b.valScalar(xs(i), cols(r))
        NDArray.fromArray(out, rest ++ xd.shapeArr)
      else
        val idx = NDArray.fromArray(Array.tabulate(nr)(identity), rest)
        NDArray.zipMap(idx, xd)((r, v) => b.valScalar(v, cols(r)))

  def valScalar(b: PolyBasis, x: Double, c: NDArray[?]): Double =
    val cd = coefs(c)
    if cd.ndim != 1 then throw new IllegalArgumentException("a scalar result needs 1-D coefficients; pass x as an array")
    b.valScalar(x, cd.toArray)

  private def sameShape(xs: Seq[NDArray[?]]): Unit =
    if xs.exists(a => !java.util.Arrays.equals(a.shapeArr, xs.head.shapeArr)) then
      throw new IllegalArgumentException(
        if xs.length == 3 then "x, y, z are incompatible" else if xs.length == 2 then "x, y are incompatible"
        else "ordinates are incompatible"
      )

  def valNd(b: PolyBasis, c: NDArray[?], xs: NDArray[?]*): NDArray[Double] =
    sameShape(xs)
    var r = valArr(b, xs.head, c, tensor = true)
    xs.tail.foreach(x => r = valArr(b, x, r, tensor = false))
    r

  def gridNd(b: PolyBasis, c: NDArray[?], xs: NDArray[?]*): NDArray[Double] =
    var r: NDArray[?] = c
    xs.foreach(x => r = valArr(b, x, r, tensor = true))
    r.asInstanceOf[NDArray[Double]]

  // ---------------------------------------------------------------- Vandermonde

  def vander(b: PolyBasis, x: NDArray[?], deg: Int): NDArray[Double] = b.vander(toD(x), deg)

  def vanderNd(b: PolyBasis, xs: Seq[NDArray[?]], degs: Seq[Int]): NDArray[Double] =
    if xs.length != degs.length then
      throw new IllegalArgumentException(s"Expected ${xs.length} dimensions of sample points, got ${degs.length}")
    sameShape(xs)
    val vs = xs.zip(degs).map((x, d) => b.vander(toD(x), d))
    val npts = xs.head.size
    val sizes = degs.map(_ + 1)
    val total = sizes.product
    val out = new Array[Double](npts * total)
    val vdata = vs.map(_.toArray)
    for p <- 0 until npts do
      var k = 0
      while k < total do
        var rem = k
        var prod = 1.0
        var dim = sizes.length - 1
        while dim >= 0 do
          val i = rem % sizes(dim)
          rem /= sizes(dim)
          prod *= vdata(dim)(p * sizes(dim) + i)
          dim -= 1
        out(p * total + k) = prod
        k += 1
    NDArray.fromArray(out, xs.head.shapeArr :+ total)

  // ---------------------------------------------------------------- calculus along an axis

  private def withRows(c: NDArray[?], axis: Int)(f: Array[Array[Double]] => Array[Array[Double]]): NDArray[Double] =
    val cd = coefs(c)
    val ax = Shape.normAxis(axis, cd.ndim)
    val moved = cd.moveaxis(ax, 0)
    val rest = moved.shapeArr.tail
    val nr = Shape.size(rest)
    val flat = moved.toArray
    val rows = Array.tabulate(moved.shapeArr(0))(k => Array.tabulate(nr)(r => flat(k * nr + r)))
    val res = f(rows)
    NDArray.fromArray(res.flatten, res.length +: rest).moveaxis(0, ax).copy()

  def der(b: PolyBasis, c: NDArray[?], m: Int, scl: Double, axis: Int): NDArray[Double] =
    withRows(c, axis)(rows => b.der(rows, m, scl))

  def integ(b: PolyBasis, c: NDArray[?], m: Int, k: Seq[Double], lbnd: Double, scl: Double, axis: Int): NDArray[Double] =
    withRows(c, axis)(rows => b.integ(rows, m, k, lbnd, scl))

  // ---------------------------------------------------------------- 1-D series algebra

  def add(b: PolyBasis, c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = arr(b.add(seriesOf(c1), seriesOf(c2)))
  def sub(b: PolyBasis, c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = arr(b.sub(seriesOf(c1), seriesOf(c2)))
  def mul(b: PolyBasis, c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = arr(b.mul(seriesOf(c1), seriesOf(c2)))
  def mulx(b: PolyBasis, c: NDArray[?]): NDArray[Double] = arr(b.mulx(seriesOf(c)))
  def div(b: PolyBasis, c1: NDArray[?], c2: NDArray[?]): (NDArray[Double], NDArray[Double]) =
    val (q, r) = b.div(seriesOf(c1), seriesOf(c2))
    (arr(q), arr(r))
  def pow(b: PolyBasis, c: NDArray[?], p: Int, maxpower: Int): NDArray[Double] = arr(b.pow(seriesOf(c), p, maxpower))

  def roots(b: PolyBasis, c: NDArray[?]): NDArray[Complex] =
    val r = b.roots(seriesOf(c))
    NDArray.fromArray(r, Array(r.length))

  def companion(b: PolyBasis, c: NDArray[?]): NDArray[Double] =
    val m = b.companion(seriesOf(c))
    NDArray.fromArray(m.flatten, Array(m.length, m.length))

  def fromroots(b: PolyBasis, roots: NDArray[?]): NDArray[Double] =
    val r = toD(roots)
    if r.ndim > 1 then throw new IllegalArgumentException("Coefficient array is not 1-d")
    arr(b.fromroots(r.toArray))

  def line(b: PolyBasis, off: Double, scl: Double): NDArray[Double] = arr(b.line(off, scl))

  def trimcoef(c: NDArray[?], tol: Double): NDArray[Double] = arr(PolyBasis.trimcoef(seriesOf(c), tol))

  def toPower(b: PolyBasis, c: NDArray[?]): NDArray[Double] = arr(b.toPower(seriesOf(c)))
  def fromPower(b: PolyBasis, c: NDArray[?]): NDArray[Double] = arr(b.fromPower(seriesOf(c)))

  def gauss(b: PolyBasis, deg: Int): (NDArray[Double], NDArray[Double]) =
    val (x, w) = b.gauss(deg)
    (arr(x), arr(w))

  def weight(b: PolyBasis, x: NDArray[?]): NDArray[Double] = toD(x).map(b.weight)

  def fit(b: PolyBasis, x: NDArray[?], y: NDArray[?], deg: Int | Seq[Int], rcond: Double, w: NDArray[?] | Null)
      : (NDArray[Double], PolyFitInfo) =
    val degs = deg match
      case i: Int => Seq(i)
      case s: Seq[?] => s.asInstanceOf[Seq[Int]]
    val (coef, resid, rank, sv, rc) = b.fit(x, y, degs, rcond, w)
    (coef, PolyFitInfo(resid, rank, sv, rc))

  // ---------------------------------------------------------------- extras

  def valfromroots(x: NDArray[?], r: NDArray[?], tensor: Boolean): NDArray[Double] =
    val rd0 = toD(r)
    val rd = if rd0.ndim == 0 then rd0.reshape(1) else rd0
    val xd = toD(x)
    val n = rd.shapeArr(0)
    if rd.ndim == 1 then
      val rs = rd.toArray
      xd.map { v =>
        var p = 1.0
        rs.foreach(z => p *= v - z)
        p
      }
    else
      val rest = rd.shapeArr.tail
      val nr = Shape.size(rest)
      val flat = rd.toArray
      def prod(v: Double, col: Int): Double =
        var p = 1.0
        for k <- 0 until n do p *= v - flat(k * nr + col)
        p
      if tensor then
        val xs = xd.toArray
        val out = new Array[Double](nr * xs.length)
        for c <- 0 until nr; i <- xs.indices do out(c * xs.length + i) = prod(xs(i), c)
        NDArray.fromArray(out, rest ++ xd.shapeArr)
      else
        val idx = NDArray.fromArray(Array.tabulate(nr)(identity), rest)
        NDArray.zipMap(idx, xd)((c, v) => prod(v, c))

  def chebpts1(npts: Int): NDArray[Double] =
    if npts < 1 then throw new IllegalArgumentException("npts must be >= 1")
    arr(Array.tabulate(npts)(i => math.sin(0.5 * math.Pi / npts * (-npts + 1 + 2 * i))))

  def chebpts2(npts: Int): NDArray[Double] =
    if npts < 2 then throw new IllegalArgumentException("npts must be >= 2")
    np.linspace(-math.Pi, 0.0, npts).map(math.cos)

  def chebinterpolate(func: Double => Double, deg: Int): NDArray[Double] =
    if deg < 0 then throw new IllegalArgumentException("expected deg >= 0")
    val order = deg + 1
    val xcheb = chebpts1(order).toArray
    val yfunc = xcheb.map(func)
    val row = new Array[Double](order)
    val c = new Array[Double](order)
    for i <- 0 until order do
      PolyBasis.Cheb.vanderRow(xcheb(i), deg, row, 0)
      for j <- 0 until order do c(j) += row(j) * yfunc(i)
    c(0) /= order
    for j <- 1 until order do c(j) /= 0.5 * order
    arr(c)
