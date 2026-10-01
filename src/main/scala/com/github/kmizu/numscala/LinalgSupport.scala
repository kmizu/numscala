package com.github.kmizu.numscala

/** Conversions between `NDArray`s and the dense kernel buffers, plus stacked-matrix iteration. */
private[numscala] object LinalgSupport:

  def assertStacked2d(a: NDArray[?]): Unit =
    if a.ndim < 2 then
      throw new LinAlgError(s"${a.ndim}-dimensional array given. Array must be at least two-dimensional")

  def assertStackedSquare(a: NDArray[?]): Unit =
    assertStacked2d(a)
    if a.shapeArr(a.ndim - 1) != a.shapeArr(a.ndim - 2) then
      throw new LinAlgError("Last 2 dimensions of the array must be square")

  def assertFinite(a: NDArray[?]): Unit =
    val d = a.dtype.asInstanceOf[DType[Any]]
    a.asInstanceOf[NDArray[Any]].foreach { x =>
      val bad = d.kind match
        case 'c' => !d.toComplex(x).isFinite
        case 'f' => { val v = d.toDouble(x); v.isNaN || v.isInfinite }
        case _ => false
      if bad then throw new LinAlgError("Array must not contain infs or NaNs")
    }

  /** Elements in C order as doubles (real part for complex input). */
  def realData(a: NDArray[?]): Array[Double] =
    a.asInstanceOf[NDArray[Any]].asType(using DType.Float64.asInstanceOf[DType[Any]]).toArray.asInstanceOf[Array[Double]]

  /** Elements in C order as (re, im) buffers. */
  def complexData(a: NDArray[?]): (Array[Double], Array[Double]) =
    val c = a.asInstanceOf[NDArray[Any]].asType(using DType.Complex128.asInstanceOf[DType[Any]]).toArray.asInstanceOf[Array[Complex]]
    val re = new Array[Double](c.length)
    val im = new Array[Double](c.length)
    var i = 0
    while i < c.length do
      re(i) = c(i).re
      im(i) = c(i).im
      i += 1
    (re, im)

  def batchShape(a: NDArray[?]): Array[Int] = a.shapeArr.dropRight(2)

  def wrapD(data: Array[Double], shape: Array[Int]): NDArray[Double] = NDArray.fromArray(data, shape)

  def wrapC(re: Array[Double], im: Array[Double], shape: Array[Int]): NDArray[Complex] =
    val out = new Array[Complex](re.length)
    var i = 0
    while i < re.length do
      out(i) = Complex(re(i), im(i))
      i += 1
    NDArray.fromArray(out, shape)

  def wrapZ(z: ZMat): NDArray[Complex] = wrapC(z.re, z.im, Array(z.m, z.n))

  /** Applies `f` to every `m x n` matrix of a real stack; each result has `tail` shape. */
  def mapD(a: NDArray[?], tail: Array[Int])(f: (Array[Double], Int, Int) => Array[Double]): NDArray[Double] =
    val m = a.shapeArr(a.ndim - 2)
    val n = a.shapeArr(a.ndim - 1)
    val bs = batchShape(a)
    val nb = Shape.size(bs)
    val data = realData(a)
    val tl = Shape.size(tail)
    val out = new Array[Double](nb * tl)
    var b = 0
    while b < nb do
      val r = f(java.util.Arrays.copyOfRange(data, b * m * n, (b + 1) * m * n), m, n)
      System.arraycopy(r, 0, out, b * tl, tl)
      b += 1
    wrapD(out, bs ++ tail)

  /** Complex counterpart of [[mapD]]; `f` returns `(re, im)`. */
  def mapZ(a: NDArray[?], tail: Array[Int])(f: ZMat => (Array[Double], Array[Double])): NDArray[Complex] =
    val m = a.shapeArr(a.ndim - 2)
    val n = a.shapeArr(a.ndim - 1)
    val bs = batchShape(a)
    val nb = Shape.size(bs)
    val (re, im) = complexData(a)
    val tl = Shape.size(tail)
    val ore = new Array[Double](nb * tl)
    val oim = new Array[Double](nb * tl)
    var b = 0
    while b < nb do
      val z = ZMat(m, n, java.util.Arrays.copyOfRange(re, b * m * n, (b + 1) * m * n),
        java.util.Arrays.copyOfRange(im, b * m * n, (b + 1) * m * n))
      val (rr, ri) = f(z)
      System.arraycopy(rr, 0, ore, b * tl, tl)
      System.arraycopy(ri, 0, oim, b * tl, tl)
      b += 1
    wrapC(ore, oim, bs ++ tail)

  /** Iterates over the matrices of a stack, calling `f(batchIndex, matrix)`. */
  def foreachD(a: NDArray[?])(f: (Int, Array[Double]) => Unit): Unit =
    val m = a.shapeArr(a.ndim - 2)
    val n = a.shapeArr(a.ndim - 1)
    val nb = Shape.size(batchShape(a))
    val data = realData(a)
    var b = 0
    while b < nb do
      f(b, java.util.Arrays.copyOfRange(data, b * m * n, (b + 1) * m * n))
      b += 1

  def foreachZ(a: NDArray[?])(f: (Int, ZMat) => Unit): Unit =
    val m = a.shapeArr(a.ndim - 2)
    val n = a.shapeArr(a.ndim - 1)
    val nb = Shape.size(batchShape(a))
    val (re, im) = complexData(a)
    var b = 0
    while b < nb do
      f(b, ZMat(m, n, java.util.Arrays.copyOfRange(re, b * m * n, (b + 1) * m * n),
        java.util.Arrays.copyOfRange(im, b * m * n, (b + 1) * m * n)))
      b += 1

  /** Symmetrises from one triangle (`lower` = use the lower triangle), real. */
  def symmetrizeD(a: Array[Double], n: Int, lower: Boolean): Unit =
    var i = 0
    while i < n do
      var j = i + 1
      while j < n do
        if lower then a(i * n + j) = a(j * n + i) else a(j * n + i) = a(i * n + j)
        j += 1
      i += 1

  /** Makes `z` Hermitian from one triangle (diagonal imaginary parts are dropped). */
  def hermitianize(z: ZMat, lower: Boolean): Unit =
    val n = z.n
    var i = 0
    while i < n do
      z.im(i * n + i) = 0.0
      var j = i + 1
      while j < n do
        if lower then
          z.re(i * n + j) = z.re(j * n + i)
          z.im(i * n + j) = -z.im(j * n + i)
        else
          z.re(j * n + i) = z.re(i * n + j)
          z.im(j * n + i) = -z.im(i * n + j)
        j += 1
      i += 1

  /** Lower triangle of `a` read as is; with `upper`, the lower triangle is taken as conj(upper)^T. */
  def triangleSource(z: ZMat, upper: Boolean): ZMat =
    if !upper then z
    else
      val n = z.n
      val out = ZMat.zeros(n, n)
      var i = 0
      while i < n do
        var j = 0
        while j <= i do
          out.re(i * n + j) = z.re(j * n + i)
          out.im(i * n + j) = -z.im(j * n + i)
          j += 1
        i += 1
      out

  def checkUplo(uplo: String): Boolean = uplo match
    case "L" | "l" => true
    case "U" | "u" => false
    case _ => throw new IllegalArgumentException("UPLO argument must be 'L' or 'U'")

  /** Stable argsort, descending. */
  def argsortDesc(s: Array[Double]): Array[Int] =
    s.indices.sortWith((i, j) => s(i) > s(j) || (s(i) == s(j) && i < j)).toArray
