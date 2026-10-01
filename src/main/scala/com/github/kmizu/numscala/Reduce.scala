package com.github.kmizu.numscala

/** Lane-based iteration: the building block of every reduction and of every
  * operation that works "along an axis".
  */
object Lanes:
  /** Reduces over `axis` (one or more axes). For each output element, `f` receives a
    * buffer filled with the lane's elements (in C order of the reduced axes) and its length.
    * The buffer is reused between calls.
    */
  def reduce[T, U](a: NDArray[T], axis: Axis, keepdims: Boolean)(f: (Array[T], Int) => U)(using
      u: DType[U]
  ): NDArray[U] =
    val ax = normAxes(axis, a.ndim)
    reduceAxes(a, ax, keepdims)(f)

  def reduceAxes[T, U](a: NDArray[T], ax: Array[Int], keepdims: Boolean)(f: (Array[T], Int) => U)(using
      u: DType[U]
  ): NDArray[U] =
    val axSet = ax.toSet
    val keep = (0 until a.ndim).filterNot(axSet).toArray
    val outShape = keep.map(a.shapeArr(_))
    val outStrides = keep.map(a.stridesArr(_))
    val inShape = ax.map(a.shapeArr(_))
    val inStrides = ax.map(a.stridesArr(_))
    val n = Shape.size(inShape)
    val buf = a.dtype.newArray(n)
    val res = u.newArray(Shape.size(outShape))
    val data = a.data
    var k = 0
    Strided.foreach1(outShape, outStrides, a.offset) { base =>
      var j = 0
      Strided.foreach1(inShape, inStrides, base) { o =>
        buf(j) = data(o)
        j += 1
      }
      res(k) = f(buf, n)
      k += 1
    }
    // like NumPy, reducing over an empty axis raises for reductions without an identity
    // (`min`, `argmax`, ...) even when the result itself is empty
    if n == 0 && res.length == 0 then f(buf, 0)
    val finalShape =
      if keepdims then (0 until a.ndim).map(i => if axSet(i) then 1 else a.shapeArr(i)).toArray
      else outShape
    NDArray.fromArray(res, finalShape)

  /** Reduces all elements to a single value. */
  def reduceAll[T, U](a: NDArray[T])(f: (Array[T], Int) => U): U =
    val arr = a.toArray
    f(arr, arr.length)

  /** Transforms each lane along `axis` into a lane of length `outLen` (same length by default).
    * `f(in, inLen, out)` reads `in(0 until inLen)` and must fill `out(0 until outLen)`.
    */
  def transform[T, U](a: NDArray[T], axis: Int, outLen: Int = -1)(f: (Array[T], Int, Array[U]) => Unit)(using
      u: DType[U]
  ): NDArray[U] =
    if a.ndim == 0 then
      val in = a.toArray
      val out = u.newArray(if outLen < 0 then 1 else outLen)
      f(in, 1, out)
      return NDArray.fromArray(out, Array(out.length))
    val ax = Shape.normAxis(axis, a.ndim)
    val n = a.shapeArr(ax)
    val m = if outLen < 0 then n else outLen
    val outShape = a.shapeArr.clone()
    outShape(ax) = m
    val outStrides = Shape.cStrides(outShape)
    val keep = (0 until a.ndim).filter(_ != ax).toArray
    val laneShape = keep.map(a.shapeArr(_))
    val inLaneStrides = keep.map(a.stridesArr(_))
    val outLaneStrides = keep.map(outStrides(_))
    val res = u.newArray(Shape.size(outShape))
    val inBuf = a.dtype.newArray(n)
    val outBuf = u.newArray(m)
    val data = a.data
    val sIn = a.stridesArr(ax)
    val sOut = outStrides(ax)
    Strided.foreach2(laneShape, inLaneStrides, a.offset, outLaneStrides, 0) { (bi, bo) =>
      var j = 0
      while j < n do
        inBuf(j) = data(bi + j * sIn)
        j += 1
      f(inBuf, n, outBuf)
      j = 0
      while j < m do
        res(bo + j * sOut) = outBuf(j)
        j += 1
    }
    NDArray.fromArray(res, outShape)

/** Reductions (`sum`, `mean`, `max`, ...). */
private[numscala] object Reduce:

  /** NumPy-style pairwise summation of doubles (good accuracy, matches NumPy closely). */
  def pairwiseSum(a: Array[Double], start: Int, n: Int): Double =
    if n < 8 then
      var s = 0.0
      var i = 0
      while i < n do
        s += a(start + i)
        i += 1
      s
    else if n <= 128 then
      val r = new Array[Double](8)
      var i = 0
      while i < 8 do
        r(i) = a(start + i)
        i += 1
      i = 8
      while i < n - (n % 8) do
        var j = 0
        while j < 8 do
          r(j) += a(start + i + j)
          j += 1
        i += 8
      var res = ((r(0) + r(1)) + (r(2) + r(3))) + ((r(4) + r(5)) + (r(6) + r(7)))
      while i < n do
        res += a(start + i)
        i += 1
      res
    else
      var n2 = n / 2
      n2 -= n2 % 8
      pairwiseSum(a, start, n2) + pairwiseSum(a, start + n2, n - n2)

  def sumBuf[T, U](src: DType[T], buf: Array[T], n: Int, d: NumDType[U]): U =
    d match
      case DType.Float64 if src eq DType.Float64 =>
        pairwiseSum(buf.asInstanceOf[Array[Double]], 0, n).asInstanceOf[U]
      case f: FloatDType[U] =>
        val tmp = new Array[Double](n)
        var i = 0
        while i < n do
          tmp(i) = src.toDouble(buf(i))
          i += 1
        f.fromDouble(pairwiseSum(tmp, 0, n))
      case c: ComplexDType =>
        val re = new Array[Double](n)
        val im = new Array[Double](n)
        var i = 0
        while i < n do
          val z = src.toComplex(buf(i))
          re(i) = z.re
          im(i) = z.im
          i += 1
        Complex(pairwiseSum(re, 0, n), pairwiseSum(im, 0, n)).asInstanceOf[U]
      case _ =>
        var acc = d.zero
        var i = 0
        while i < n do
          acc = d.plus(acc, d.castFrom(src, buf(i)))
          i += 1
        acc

  def prodBuf[T, U](src: DType[T], buf: Array[T], n: Int, d: NumDType[U]): U =
    var acc = d.one
    var i = 0
    while i < n do
      acc = d.times(acc, d.castFrom(src, buf(i)))
      i += 1
    acc

  def sumAll[T, U](a: NDArray[T], d: NumDType[U]): U = Lanes.reduceAll(a)((b, n) => sumBuf(a.dtype, b, n, d))
  def sum[T, U](a: NDArray[T], axis: Axis, keepdims: Boolean, d: NumDType[U]): NDArray[U] =
    Lanes.reduce(a, axis, keepdims)((b, n) => sumBuf(a.dtype, b, n, d))(using d)
  def prodAll[T, U](a: NDArray[T], d: NumDType[U]): U = Lanes.reduceAll(a)((b, n) => prodBuf(a.dtype, b, n, d))
  def prod[T, U](a: NDArray[T], axis: Axis, keepdims: Boolean, d: NumDType[U]): NDArray[U] =
    Lanes.reduce(a, axis, keepdims)((b, n) => prodBuf(a.dtype, b, n, d))(using d)

  def meanBuf[T, U](src: DType[T], buf: Array[T], n: Int, d: InexactDType[U]): U =
    if n == 0 then d.nan else d.div(sumBuf(src, buf, n, d), d.fromLong(n.toLong))
  def meanAll[T, U](a: NDArray[T], d: InexactDType[U]): U = Lanes.reduceAll(a)((b, n) => meanBuf(a.dtype, b, n, d))
  def mean[T, U](a: NDArray[T], axis: Axis, keepdims: Boolean, d: InexactDType[U]): NDArray[U] =
    Lanes.reduce(a, axis, keepdims)((b, n) => meanBuf(a.dtype, b, n, d))(using d)

  def varBuf[T, U](src: DType[T], buf: Array[T], n: Int, ddof: Int, d: FloatDType[U]): U =
    val dof = n - ddof
    if n == 0 || dof <= 0 then d.nan
    else if src.isComplex then
      var sr = 0.0
      var si = 0.0
      var i = 0
      while i < n do
        val z = src.toComplex(buf(i))
        sr += z.re
        si += z.im
        i += 1
      val mr = sr / n
      val mi = si / n
      val dev = new Array[Double](n)
      i = 0
      while i < n do
        val z = src.toComplex(buf(i))
        val dr = z.re - mr
        val di = z.im - mi
        dev(i) = dr * dr + di * di
        i += 1
      d.fromDouble(pairwiseSum(dev, 0, n) / dof)
    else
      val x = new Array[Double](n)
      var i = 0
      while i < n do
        x(i) = src.toDouble(buf(i))
        i += 1
      val m = pairwiseSum(x, 0, n) / n
      i = 0
      while i < n do
        val dv = x(i) - m
        x(i) = dv * dv
        i += 1
      d.fromDouble(pairwiseSum(x, 0, n) / dof)

  def varAll[T, U](a: NDArray[T], ddof: Int, d: FloatDType[U]): U =
    Lanes.reduceAll(a)((b, n) => varBuf(a.dtype, b, n, ddof, d))
  def variance[T, U](a: NDArray[T], axis: Axis, ddof: Int, keepdims: Boolean, d: FloatDType[U]): NDArray[U] =
    Lanes.reduce(a, axis, keepdims)((b, n) => varBuf(a.dtype, b, n, ddof, d))(using d)

  private def emptyErr(op: String) =
    new IllegalArgumentException(s"zero-size array to reduction operation $op which has no identity")

  /** Max/min with NaN propagation (the first NaN wins). */
  def extremeBuf[T](d: DType[T], buf: Array[T], n: Int, wantMax: Boolean): T =
    if n == 0 then throw emptyErr(if wantMax then "maximum" else "minimum")
    var best = buf(0)
    var i = 1
    if d.isNaN(best) then return best
    while i < n do
      val x = buf(i)
      if d.isNaN(x) then return x
      val c = d.compare(x, best)
      if (wantMax && c > 0) || (!wantMax && c < 0) then best = x
      i += 1
    best

  def argExtremeBuf[T](d: DType[T], buf: Array[T], n: Int, wantMax: Boolean): Int =
    if n == 0 then throw new IllegalArgumentException(s"attempt to get ${if wantMax then "argmax" else "argmin"} of an empty sequence")
    var bi = 0
    if d.isNaN(buf(0)) then return 0
    var i = 1
    while i < n do
      val x = buf(i)
      if d.isNaN(x) then return i
      val c = d.compare(x, buf(bi))
      if (wantMax && c > 0) || (!wantMax && c < 0) then bi = i
      i += 1
    bi

  def maxAll[T](a: NDArray[T]): T = Lanes.reduceAll(a)((b, n) => extremeBuf(a.dtype, b, n, true))
  def minAll[T](a: NDArray[T]): T = Lanes.reduceAll(a)((b, n) => extremeBuf(a.dtype, b, n, false))
  def max[T](a: NDArray[T], axis: Axis, keepdims: Boolean): NDArray[T] =
    Lanes.reduce(a, axis, keepdims)((b, n) => extremeBuf(a.dtype, b, n, true))(using a.dtype)
  def min[T](a: NDArray[T], axis: Axis, keepdims: Boolean): NDArray[T] =
    Lanes.reduce(a, axis, keepdims)((b, n) => extremeBuf(a.dtype, b, n, false))(using a.dtype)
  def argmaxAll[T](a: NDArray[T]): Int = Lanes.reduceAll(a)((b, n) => argExtremeBuf(a.dtype, b, n, true))
  def argminAll[T](a: NDArray[T]): Int = Lanes.reduceAll(a)((b, n) => argExtremeBuf(a.dtype, b, n, false))
  def argmax[T](a: NDArray[T], axis: Int, keepdims: Boolean): NDArray[Int] =
    Lanes.reduce(a, axis, keepdims)((b, n) => argExtremeBuf(a.dtype, b, n, true))(using DType.Int32)
  def argmin[T](a: NDArray[T], axis: Int, keepdims: Boolean): NDArray[Int] =
    Lanes.reduce(a, axis, keepdims)((b, n) => argExtremeBuf(a.dtype, b, n, false))(using DType.Int32)

  private def allBuf[T](d: DType[T], b: Array[T], n: Int): Boolean =
    var i = 0
    while i < n do
      if !d.toBoolean(b(i)) then return false
      i += 1
    true
  private def anyBuf[T](d: DType[T], b: Array[T], n: Int): Boolean =
    var i = 0
    while i < n do
      if d.toBoolean(b(i)) then return true
      i += 1
    false

  def allAll[T](a: NDArray[T]): Boolean = Lanes.reduceAll(a)((b, n) => allBuf(a.dtype, b, n))
  def anyAll[T](a: NDArray[T]): Boolean = Lanes.reduceAll(a)((b, n) => anyBuf(a.dtype, b, n))
  def all[T](a: NDArray[T], axis: Axis, keepdims: Boolean): NDArray[Boolean] =
    Lanes.reduce(a, axis, keepdims)((b, n) => allBuf(a.dtype, b, n))(using DType.Bool)
  def any[T](a: NDArray[T], axis: Axis, keepdims: Boolean): NDArray[Boolean] =
    Lanes.reduce(a, axis, keepdims)((b, n) => anyBuf(a.dtype, b, n))(using DType.Bool)

  /** Running accumulation along an axis (`cumsum`, `cumprod`, ufunc `accumulate`). */
  def cumulate[T, U](a: NDArray[T], axis: Int, d: DType[U])(f: (U, U) => U): NDArray[U] =
    val src = a.dtype
    Lanes.transform(a, axis) { (in: Array[T], n: Int, out: Array[U]) =>
      if n > 0 then
        var acc = d.castFrom(src, in(0))
        out(0) = acc
        var i = 1
        while i < n do
          acc = f(acc, d.castFrom(src, in(i)))
          out(i) = acc
          i += 1
    }(using d)
