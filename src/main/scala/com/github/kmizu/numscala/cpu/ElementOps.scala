package com.github.kmizu.numscala.cpu

/** Scalar elementwise kernels and row reductions (K3). Arguments are validated by [[F32Kernels]]. */
private[cpu] object ElementOps:

  def copy(src: MatrixF32, dst: MatrixF32): Unit =
    var i = 0
    while i < src.rows do
      System.arraycopy(src.data, src.offset + i * src.rowStride, dst.data, dst.offset + i * dst.rowStride, src.cols)
      i += 1

  def fill(dst: MatrixF32, value: Float): Unit =
    var i = 0
    while i < dst.rows do
      val o = dst.offset + i * dst.rowStride
      java.util.Arrays.fill(dst.data, o, o + dst.cols, value)
      i += 1

  def axpy(alpha: Float, x: MatrixF32, y: MatrixF32): Unit =
    val xd = x.data; val yd = y.data
    var i = 0
    while i < x.rows do
      val xi = x.offset + i * x.rowStride
      val yi = y.offset + i * y.rowStride
      var j = 0
      while j < x.cols do
        yd(yi + j) += alpha * xd(xi + j)
        j += 1
      i += 1

  def mul(x: MatrixF32, y: MatrixF32, out: MatrixF32): Unit =
    val xd = x.data; val yd = y.data; val od = out.data
    var i = 0
    while i < x.rows do
      val xi = x.offset + i * x.rowStride
      val yi = y.offset + i * y.rowStride
      val oi = out.offset + i * out.rowStride
      var j = 0
      while j < x.cols do
        od(oi + j) = xd(xi + j) * yd(yi + j)
        j += 1
      i += 1

  /** Overflow-free logistic function: `exp` is only ever applied to a non-positive argument. */
  inline def sigmoid1(x: Float): Float =
    if x >= 0f then (1.0 / (1.0 + math.exp(-x.toDouble))).toFloat
    else
      val e = math.exp(x.toDouble) // also NaN for NaN input
      (e / (1.0 + e)).toFloat

  /** `x * sigmoid(x)` with `silu(-inf) = -0` (the limit) instead of `-inf * 0 = NaN`. */
  inline def silu1(x: Float): Float =
    if x == Float.NegativeInfinity then -0f
    else if x >= 0f then (x / (1.0 + math.exp(-x.toDouble))).toFloat
    else
      val e = math.exp(x.toDouble)
      (x * e / (1.0 + e)).toFloat

  def sigmoid(x: MatrixF32, out: MatrixF32): Unit =
    val xd = x.data; val od = out.data
    var i = 0
    while i < x.rows do
      val xi = x.offset + i * x.rowStride
      val oi = out.offset + i * out.rowStride
      var j = 0
      while j < x.cols do
        od(oi + j) = sigmoid1(xd(xi + j))
        j += 1
      i += 1

  def silu(x: MatrixF32, out: MatrixF32): Unit =
    val xd = x.data; val od = out.data
    var i = 0
    while i < x.rows do
      val xi = x.offset + i * x.rowStride
      val oi = out.offset + i * out.rowStride
      var j = 0
      while j < x.cols do
        od(oi + j) = silu1(xd(xi + j))
        j += 1
      i += 1

  /** Max-shifted log-sum-exp per row (accumulated in Double). */
  def rowLogSumExp(x: MatrixF32, out: Array[Float], outOffset: Int): Unit =
    val xd = x.data
    var i = 0
    while i < x.rows do
      val xi = x.offset + i * x.rowStride
      var mx = Float.NegativeInfinity
      var nan = false
      var j = 0
      while j < x.cols do
        val v = xd(xi + j)
        if v != v then nan = true
        else if v > mx then mx = v
        j += 1
      // computed into a local first: loops inside `out(..) = <expr>` would run with a non-empty operand stack
      var res = if nan then Float.NaN else mx // NaN row, +inf present, or all -inf / empty
      if !nan && mx != Float.PositiveInfinity && mx != Float.NegativeInfinity then
        var s = 0.0
        j = 0
        while j < x.cols do
          s += math.exp((xd(xi + j) - mx).toDouble)
          j += 1
        res = (mx + math.log(s)).toFloat
      out(outOffset + i) = res
      i += 1

  /** Sum of squares per row (accumulated in Double). */
  def rowSumSquares(x: MatrixF32, out: Array[Float], outOffset: Int): Unit =
    val xd = x.data
    var i = 0
    while i < x.rows do
      val xi = x.offset + i * x.rowStride
      var s = 0.0
      var j = 0
      while j < x.cols do
        val v = xd(xi + j).toDouble
        s += v * v
        j += 1
      out(outOffset + i) = s.toFloat
      i += 1
