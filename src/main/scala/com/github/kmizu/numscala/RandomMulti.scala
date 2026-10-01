package com.github.kmizu.numscala.random

import com.github.kmizu.numscala.*

/** Multivariate samplers: multinomial, multivariate hypergeometric, Dirichlet, multivariate normal. */
private[numscala] object RandomMulti:
  private def outShape(size: SizeArg, tail: Int): Array[Int] =
    val shp = RCommon.sizeShape(size)
    if shp == null then Array(tail) else RCommon.checkShape(shp) :+ tail

  private def checkPvals(pix: Array[Double], d: Int, ndim: Int): Unit =
    var off = 0
    while off < pix.length do
      if RCommon.kahanSum(pix, off, d - 1) > 1.0 + 1e-12 then
        throw new IllegalArgumentException(s"sum(pvals${if ndim == 1 then "[:-1]" else "[...,:-1]"}) > 1.0")
      off += d

  /** `multinomial(n, pvals, size)`; the legacy variant only accepts a scalar `n` and 1-d `pvals`. */
  def multinomial(bg: BitGenerator, bs: BinomialState, n: Param, pvals: Param, size: SizeArg, legacy: Boolean): NDArray[Long] =
    val parr = RCommon.toD(pvals)
    val ndim = parr.ndim
    if legacy && ndim != 1 then
      if ndim == 0 then throw new IllegalArgumentException("pvals must be a 1-d sequence")
      throw new IllegalArgumentException("object too deep for desired array")
    val d = if ndim >= 1 then parr.shapeArr(ndim - 1) else 0
    if d == 0 && !legacy then
      throw new IllegalArgumentException(
        "pvals must have at least 1 dimension and the last dimension of pvals must be greater than 0.")
    val pix = parr.toArray
    RCommon.checkArr(pix, "pvals", Cons.Bounded01)
    if d > 0 then checkPvals(pix, d, ndim)
    val on = RCommon.toL(n)
    if !legacy && (on.ndim != 0 || ndim > 1) then
      RCommon.checkArr(on.toArray.map(_.toDouble), "n", Cons.NonNegative)
      val offsetsShape = parr.shapeArr.dropRight(1)
      val offsets = NDArray.fromArray(Array.tabulate(Shape.size(offsetsShape))(i => (i * d).toLong), offsetsShape)
      val (itShape, fl) =
        val sz = RCommon.sizeShape(size)
        val base = RCommon.broadcastAll(null, Seq(on, offsets))
        if sz != null then
          val joint = Shape.broadcast(on.shapeArr, offsetsShape, sz)
          if !joint.sameElements(sz) then
            throw new IllegalArgumentException(
              s"Output size ${Shape.str(sz)} is not compatible with broadcast dimensions of inputs ${Shape.str(base._1)}.")
          RCommon.broadcastAll(size, Seq(on, offsets))
        else base
      val cnt = Shape.size(itShape)
      val out = new Array[Long](cnt * d)
      var i = 0
      while i < cnt do
        Dist.multinomial(bg, fl(0)(i), out, i * d, pix, fl(1)(i).toInt, d, bs)
        i += 1
      NDArray.fromArray(out, itShape :+ d)
    else
      val shape = outShape(size, d)
      val out = new Array[Long](Shape.size(shape))
      val ni = on.item
      RCommon.check(ni.toDouble, "n", Cons.NonNegative)
      if d > 0 then
        var off = 0
        while off < out.length do
          Dist.multinomial(bg, ni, out, off, pix, 0, d, bs)
          off += d
      NDArray.fromArray(out, shape)

  /** `Generator.multivariate_hypergeometric(colors, nsample, size, method)`. */
  def mvhg(bg: BitGenerator, colors: Param, nsample: Long, size: SizeArg, method: String): NDArray[Long] =
    if method != "count" && method != "marginals" then
      throw new IllegalArgumentException("method must be \"count\" or \"marginals\".")
    if nsample < 0 then throw new IllegalArgumentException("nsample must be nonnegative.")
    val ca = colors match
      case a: NDArray[?] if a.dtype.isFloating => null
      case _ => RCommon.toL(colors)
    if ca == null || ca.ndim != 1 || ca.toArray.exists(_ < 0) then
      throw new IllegalArgumentException(
        s"colors must be a one-dimensional sequence of nonnegative integers not exceeding ${Long.MaxValue}.")
    val cols = ca.toArray
    var total = 0L
    for c <- cols do
      if c > Long.MaxValue - total then
        throw new IllegalArgumentException(
          s"sum(colors) must not exceed the maximum value of a 64 bit signed integer (${Long.MaxValue})")
      total += c
    if method == "marginals" && total >= 1000000000L then
      throw new IllegalArgumentException("When method is \"marginals\", sum(colors) must be less than 1000000000.")
    if method == "count" && total > Int.MaxValue then
      throw new IllegalArgumentException(s"When method is 'count', sum(colors) must not exceed ${Int.MaxValue}")
    if nsample > total then throw new IllegalArgumentException("nsample > sum(colors)")
    val shape = outShape(size, cols.length)
    val out = new Array[Long](Shape.size(shape))
    if cols.nonEmpty then
      val nv = out.length / cols.length
      if method == "count" then Dist.mvhgCount(bg, total, cols, nsample, nv, out)
      else Dist.mvhgMarginals(bg, total, cols, nsample, nv, out)
    NDArray.fromArray(out, shape)

  /** `dirichlet(alpha, size)` for the Generator (with the small-alpha stick-breaking path). */
  def dirichlet(bg: BitGenerator, alpha: Param, size: SizeArg): NDArray[Double] =
    val aa = RCommon.toD(alpha)
    if aa.ndim != 1 then throw new IllegalArgumentException("object of too small depth for desired array")
    val a = aa.toArray
    val k = a.length
    if a.exists(_ < 0) then throw new IllegalArgumentException("alpha < 0")
    val shape = outShape(size, k)
    val out = new Array[Double](Shape.size(shape))
    if k > 0 && a.max < 0.1 then
      val csum = new Array[Double](k)
      var s = 0.0
      var j = k - 1
      while j >= 0 do { s += a(j); csum(j) = s; j -= 1 }
      if s > 0 then
        var i = 0
        while i < out.length do
          var acc = 1.0
          var jj = 0
          var stop = false
          while jj < k - 1 && !stop do
            val v = Dist.beta(bg, a(jj), csum(jj + 1))
            out(i + jj) = acc * v
            acc *= (1.0 - v)
            if csum(jj + 1) == 0 then stop = true
            jj += 1
          out(i + k - 1) = acc
          i += k
    else
      var i = 0
      while i < out.length do
        var acc = 0.0
        var jj = 0
        while jj < k do
          out(i + jj) = Dist.standardGamma(bg, a(jj))
          acc += out(i + jj)
          jj += 1
        val inv = 1.0 / acc
        jj = 0
        while jj < k do { out(i + jj) *= inv; jj += 1 }
        i += k
    NDArray.fromArray(out, shape)

  /** Legacy `RandomState.dirichlet(alpha, size)`. */
  def dirichletLegacy(s: AugState, alpha: Param, size: SizeArg): NDArray[Double] =
    val aa = RCommon.toD(alpha)
    if aa.ndim != 1 then throw new IllegalArgumentException("object of too small depth for desired array")
    val a = aa.toArray
    val k = a.length
    if a.exists(_ <= 0) then throw new IllegalArgumentException("alpha <= 0")
    val shape = outShape(size, k)
    val out = new Array[Double](Shape.size(shape))
    var i = 0
    while i < out.length do
      var acc = 0.0
      var jj = 0
      while jj < k do
        out(i + jj) = LegacyDist.standardGamma(s, a(jj))
        acc += out(i + jj)
        jj += 1
      val inv = 1 / acc
      jj = 0
      while jj < k do { out(i + jj) *= inv; jj += 1 }
      i += k
    NDArray.fromArray(out, shape)

  /** `multivariate_normal`: `x = mean + z @ factor.T` with `factor` from svd/eigh/cholesky of `cov`. */
  def multivariateNormal(mean: Param, cov: Param, size: SizeArg, checkValid: String, tol: Double, method: String,
      stdNormal: Array[Int] => NDArray[Double], legacy: Boolean): NDArray[Double] =
    if !Set("svd", "eigh", "cholesky").contains(method) then
      throw new IllegalArgumentException("method must be one of {'eigh', 'svd', 'cholesky'}")
    val m = RCommon.toD(mean)
    val c = RCommon.toD(cov)
    if m.ndim != 1 then throw new IllegalArgumentException("mean must be 1 dimensional")
    if c.ndim != 2 || c.shapeArr(0) != c.shapeArr(1) then
      throw new IllegalArgumentException("cov must be 2 dimensional and square")
    val n = m.shapeArr(0)
    if n != c.shapeArr(0) then throw new IllegalArgumentException("mean and cov must have same length")
    val sz = RCommon.sizeShape(size)
    val shape = if sz == null then Array.empty[Int] else sz
    val finalShape = shape :+ n
    val z = stdNormal(finalShape).toArray
    val cm = Array.tabulate(n, n)((i, j) => c(i, j))
    val factor: Array[Array[Double]] = method match
      case "cholesky" => RandomLinalg.cholesky(cm)
      case _ =>
        val (w0, v0) = RandomLinalg.symEig(cm)
        // numpy.linalg.eigh returns ascending eigenvalues, svd descending singular values
        val (w, v) = if method == "eigh" then (w0.reverse, v0.map(_.reverse)) else (w0, v0)
        if checkValid != "ignore" then
          if checkValid != "warn" && checkValid != "raise" then
            throw new IllegalArgumentException("check_valid must equal 'warn', 'raise', or 'ignore'")
          val psd =
            if method == "eigh" then !w.exists(_ < -tol)
            else
              // svd of a symmetric matrix: s = |w|, vh = sign(w) * v^T; check vh.T * s @ vh ~= cov
              (0 until n).forall(i => (0 until n).forall { j =>
                var acc = 0.0
                for k <- 0 until n do acc += v(i)(k) * math.abs(w(k)) * v(j)(k)
                math.abs(acc - cm(i)(j)) <= tol + tol * math.abs(cm(i)(j))
              })
          if !psd then
            if checkValid == "warn" then
              System.err.println("RuntimeWarning: covariance is not symmetric positive-semidefinite.")
            else throw new IllegalArgumentException("covariance is not symmetric positive-semidefinite.")
        Array.tabulate(n, n)((i, j) => v(i)(j) * math.sqrt(math.abs(w(j))))
    val rows = z.length / math.max(n, 1)
    val out = new Array[Double](z.length)
    val mv = m.toArray
    var r = 0
    while r < rows do
      var i = 0
      while i < n do
        var acc = 0.0
        var j = 0
        while j < n do { acc += z(r * n + j) * factor(i)(j); j += 1 }
        out(r * n + i) = if legacy then acc + mv(i) else mv(i) + acc
        i += 1
      r += 1
    NDArray.fromArray(out, finalShape)

/** Small dense symmetric linear algebra used by `multivariate_normal`. */
private[numscala] object RandomLinalg:
  /** Lower Cholesky factor (`numpy.linalg.cholesky`). */
  def cholesky(a: Array[Array[Double]]): Array[Array[Double]] =
    val n = a.length
    val l = Array.ofDim[Double](n, n)
    for j <- 0 until n do
      var s = a(j)(j)
      for k <- 0 until j do s -= l(j)(k) * l(j)(k)
      if !(s > 0) then throw new LinAlgError("Matrix is not positive definite")
      val d = math.sqrt(s)
      l(j)(j) = d
      for i <- j + 1 until n do
        var t = a(i)(j)
        for k <- 0 until j do t -= l(i)(k) * l(j)(k)
        l(i)(j) = t / d
    l

  /** Cyclic Jacobi eigen-decomposition of a symmetric matrix. Returns eigenvalues sorted in
    * descending order and the matching eigenvectors as columns (`v(i)(k)` = component `i` of
    * vector `k`), with each vector's largest-magnitude component made positive.
    */
  def symEig(a0: Array[Array[Double]]): (Array[Double], Array[Array[Double]]) =
    val n = a0.length
    val a = Array.tabulate(n, n)((i, j) => 0.5 * (a0(i)(j) + a0(j)(i)))
    val v = Array.tabulate(n, n)((i, j) => if i == j then 1.0 else 0.0)
    var sweep = 0
    var off = Double.MaxValue
    var norm = 0.0
    for i <- 0 until n; j <- 0 until n do norm += a(i)(j) * a(i)(j)
    val thresh = norm * 1e-32
    while sweep < 100 && off > thresh do
      off = 0.0
      for p <- 0 until n; q <- p + 1 until n do off += a(p)(q) * a(p)(q)
      if off > thresh then
        for p <- 0 until n; q <- p + 1 until n if a(p)(q) != 0.0 do
          val theta = (a(q)(q) - a(p)(p)) / (2 * a(p)(q))
          val t = math.signum(theta) match
            case 0.0 => 1.0
            case s => s / (math.abs(theta) + math.sqrt(theta * theta + 1))
          val cs = 1 / math.sqrt(t * t + 1)
          val sn = t * cs
          for k <- 0 until n do
            val akp = a(k)(p); val akq = a(k)(q)
            a(k)(p) = cs * akp - sn * akq
            a(k)(q) = sn * akp + cs * akq
          for k <- 0 until n do
            val apk = a(p)(k); val aqk = a(q)(k)
            a(p)(k) = cs * apk - sn * aqk
            a(q)(k) = sn * apk + cs * aqk
          for k <- 0 until n do
            val vkp = v(k)(p); val vkq = v(k)(q)
            v(k)(p) = cs * vkp - sn * vkq
            v(k)(q) = sn * vkp + cs * vkq
      sweep += 1
    val order = (0 until n).sortBy(i => -a(i)(i)).toArray
    val w = order.map(i => a(i)(i))
    val vec = Array.tabulate(n, n)((i, k) => v(i)(order(k)))
    for k <- 0 until n do
      val big = (0 until n).maxBy(i => math.abs(vec(i)(k)))
      if vec(big)(k) < 0 then for i <- 0 until n do vec(i)(k) = -vec(i)(k)
    (w, vec)
