package com.github.kmizu.numscala

import NpReduceImpl.*

/** Statistics and numerical calculus: `np.diff`, `np.gradient`, `np.trapezoid`, `np.cov`,
  * `np.corrcoef`, `np.correlate`, `np.convolve`, `np.interp`, plus histograms (see [[NpStatsHist]]).
  */
trait NpStats extends NpStatsHist:

  // ---------------------------------------------------------------- diff

  /** `n`-th discrete difference along `axis` (`np.diff(a, n, axis, prepend, append)`).
    * Boolean arrays use `!=` like NumPy; a 0-d `prepend`/`append` is broadcast along `axis`.
    */
  def diff[T](
      a: NDArray[T],
      n: Int = 1,
      axis: Int = -1,
      prepend: NDArray[T] | Null = null,
      append: NDArray[T] | Null = null
  ): NDArray[T] =
    if n == 0 then return a
    if n < 0 then throw new IllegalArgumentException(s"order must be non-negative but got $n")
    if a.ndim == 0 then throw new IllegalArgumentException("diff requires input that is at least one dimensional")
    val ax = Shape.normAxis(axis, a.ndim)
    var src = a
    if prepend != null then src = concat(edgeArray(prepend, a, ax), src, ax)
    if append != null then src = concat(src, edgeArray(append, a, ax), ax)
    val d = a.dtype
    val op: (T, T) => T = (d: DType[?]) match
      case DType.Bool => (x, y) => (x != y).asInstanceOf[T]
      case nd: NumDType[?] => nd.asInstanceOf[NumDType[T]].minus
      case _ => throw new IllegalArgumentException(s"diff is not supported for dtype ${d.name}")
    val len = src.shapeArr(ax)
    val outLen = math.max(0, len - n)
    Lanes.transform(src, ax, outLen) { (in: Array[T], l: Int, out: Array[T]) =>
      if outLen > 0 then
        val w = in.clone()
        var k = 0
        while k < n do
          var i = 0
          while i < l - k - 1 do
            w(i) = op(w(i + 1), w(i))
            i += 1
          k += 1
        System.arraycopy(w, 0, out, 0, outLen)
    }(using d)

  /** Differences between consecutive elements of the flattened array (`np.ediff1d`). */
  def ediff1d[T](ary: NDArray[T], to_end: NDArray[T] | Null = null, to_begin: NDArray[T] | Null = null)(using
      d: NumDType[T]
  ): NDArray[T] =
    given scala.reflect.ClassTag[T] = d.classTag
    val x = ary.toArray
    val mid = if x.length < 2 then Array.empty[T] else Array.tabulate(x.length - 1)(i => d.minus(x(i + 1), x(i)))
    val b = if to_begin == null then Array.empty[T] else to_begin.toArray
    val e = if to_end == null then Array.empty[T] else to_end.toArray
    val out = b ++ mid ++ e
    NDArray.fromArray(out, Array(out.length))

  // ---------------------------------------------------------------- gradient

  /** Gradient by central differences (`np.gradient(f, *varargs, axis, edge_order)`).
    *
    * `spacing` is empty (unit spacing), a single scalar for all axes, or one entry per axis,
    * each a scalar step or a 1-d coordinate array. Returns one array per differentiated axis
    * (a single-element `Seq` for 1-d input).
    */
  def gradient[T](
      f: NDArray[T],
      spacing: Seq[Double | NDArray[?]] = Nil,
      axis: Axis | Null = null,
      edge_order: Int = 1
  )(using t: ToInexact[T]): Seq[NDArray[t.Out]] =
    val od = t.dtype
    val axes: Array[Int] =
      if axis == null then Array.tabulate(f.ndim)(identity)
      else
        val raw: Seq[Int] = axis match
          case i: Int => Seq(i)
          case s: Seq[?] => s.asInstanceOf[Seq[Int]]
        raw.map(Shape.normAxis(_, f.ndim)).toArray
    if axes.distinct.length != axes.length then throw new IllegalArgumentException("duplicate value in 'axis'")
    val nAx = axes.length
    val dxs: Seq[Double | NDArray[?]] =
      if spacing.isEmpty then Seq.fill(nAx)(1.0)
      else if spacing.length == 1 && (spacing.head match
          case _: Double => true
          case a: NDArray[?] => a.ndim == 0
        )
      then Seq.fill(nAx)(spacing.head)
      else if spacing.length == nAx then spacing
      else throw new IllegalArgumentException("invalid number of arguments")
    if edge_order > 2 then throw new IllegalArgumentException("'edge_order' greater than 2 not supported")
    if edge_order < 1 then throw new IllegalArgumentException("'edge_order' must be 1 or 2")
    // per-axis spacing: Left(uniform dx) or Right(differences)
    val steps: Seq[Either[Double, Array[Double]]] = axes.indices.map { i =>
      val n = f.shapeArr(axes(i))
      dxs(i) match
        case d: Double => Left(d)
        case arr: NDArray[?] =>
          if arr.ndim == 0 then Left(doublesOf(arr, "spacing")(0))
          else if arr.ndim != 1 then throw new IllegalArgumentException("distances must be either scalars or 1d")
          else
            if arr.shapeArr(0) != n then
              throw new IllegalArgumentException("when 1d, distances must match the length of the corresponding dimension")
            val c = doublesOf(arr, "spacing")
            val d = Array.tabulate(math.max(0, c.length - 1))(j => c(j + 1) - c(j))
            if d.nonEmpty && d.forall(_ == d(0)) then Left(d(0)) else Right(d)
    }
    val isComplex = od.isComplex
    val parts: Seq[NDArray[Double]] =
      if isComplex then
        val z = f.asType(using DType.Complex128)
        Seq(z.map(_.re)(using DType.Float64), z.map(_.im)(using DType.Float64))
      else Seq(f.asType(using DType.Float64))
    axes.indices.map { i =>
      val ax = axes(i)
      if f.shapeArr(ax) < edge_order + 1 then
        throw new IllegalArgumentException(
          "Shape of array too small to calculate a numerical gradient, at least (edge_order + 1) elements are required."
        )
      val res = parts.map(p =>
        Lanes.transform(p, ax) { (in: Array[Double], n: Int, out: Array[Double]) =>
          gradLane(in, n, out, steps(i), edge_order)
        }(using DType.Float64)
      )
      if isComplex then
        NDArray.zipMap(res(0), res(1))((re, im) => Complex(re, im))(using DType.Complex128).asInstanceOf[NDArray[t.Out]]
      else res(0).asType(using od)
    }.toSeq

  private def gradLane(f: Array[Double], n: Int, out: Array[Double], step: Either[Double, Array[Double]], edge: Int): Unit =
    step match
      case Left(dx) =>
        var i = 1
        while i < n - 1 do
          out(i) = (f(i + 1) - f(i - 1)) / (2.0 * dx)
          i += 1
        if edge == 1 then
          out(0) = (f(1) - f(0)) / dx
          out(n - 1) = (f(n - 1) - f(n - 2)) / dx
        else
          var a = -1.5 / dx
          var b = 2.0 / dx
          var c = -0.5 / dx
          out(0) = a * f(0) + b * f(1) + c * f(2)
          a = 0.5 / dx
          b = -2.0 / dx
          c = 1.5 / dx
          out(n - 1) = a * f(n - 3) + b * f(n - 2) + c * f(n - 1)
      case Right(d) =>
        var i = 1
        while i < n - 1 do
          val dx1 = d(i - 1)
          val dx2 = d(i)
          val a = -dx2 / (dx1 * (dx1 + dx2))
          val b = (dx2 - dx1) / (dx1 * dx2)
          val c = dx1 / (dx2 * (dx1 + dx2))
          out(i) = a * f(i - 1) + b * f(i) + c * f(i + 1)
          i += 1
        if edge == 1 then
          out(0) = (f(1) - f(0)) / d(0)
          out(n - 1) = (f(n - 1) - f(n - 2)) / d(n - 2)
        else
          var dx1 = d(0)
          var dx2 = d(1)
          var a = -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))
          var b = (dx1 + dx2) / (dx1 * dx2)
          var c = -dx1 / (dx2 * (dx1 + dx2))
          out(0) = a * f(0) + b * f(1) + c * f(2)
          dx1 = d(n - 3)
          dx2 = d(n - 2)
          a = dx2 / (dx1 * (dx1 + dx2))
          b = -(dx2 + dx1) / (dx1 * dx2)
          c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2))
          out(n - 1) = a * f(n - 3) + b * f(n - 2) + c * f(n - 1)

  // ---------------------------------------------------------------- trapezoid

  /** Trapezoidal integral along `axis` (`np.trapezoid(y, x, dx, axis)`); the result drops `axis`
    * (a 0-d array for 1-d `y`; use `.item` for the scalar). `x` is 1-d or broadcastable to `y`.
    */
  def trapezoid[T](y: NDArray[T], x: NDArray[?] | Null = null, dx: Double = 1.0, axis: Int = -1)(using
      t: ToInexact[T]
  ): NDArray[t.Out] =
    trapImpl(y, x, dx, axis, t.dtype)

  /** Deprecated alias of [[trapezoid]] (`np.trapz`). */
  def trapz[T](y: NDArray[T], x: NDArray[?] | Null = null, dx: Double = 1.0, axis: Int = -1)(using
      t: ToInexact[T]
  ): NDArray[t.Out] =
    trapImpl(y, x, dx, axis, t.dtype)

  private def trapImpl[T, U](y: NDArray[T], x: NDArray[?] | Null, dx: Double, axis: Int, od: InexactDType[U]): NDArray[U] =
    if y.ndim == 0 then throw new IndexOutOfBoundsException(s"axis $axis is out of bounds for array of dimension 0")
    val ax = Shape.normAxis(axis, y.ndim)
    val n = y.shapeArr(ax)
    // per-lane differences of x (or None for uniform spacing)
    val xLanes: Option[(Array[Double], Boolean)] =
      if x == null then None
      else if x.ndim == 0 then throw new IllegalArgumentException("diff requires input that is at least one dimensional")
      else if x.ndim == 1 then
        if x.shapeArr(0) != n then
          throw new IllegalArgumentException(
            s"operands could not be broadcast together with shapes (${x.shapeArr(0) - 1},) (${math.max(0, n - 1)},)"
          )
        val c = doublesOf(x, "x")
        Some((Array.tabulate(math.max(0, n - 1))(j => c(j + 1) - c(j)), false))
      else
        if x.dtype.isComplex then throw new IllegalArgumentException("x must be real, got complex")
        val xb = x.asInstanceOf[NDArray[Any]].asType(using DType.Float64).broadcastTo(y.shapeArr.toSeq*)
        val (g, _, _) = gather(xb, Array(ax))
        Some((g, true))
    val isComplex = od.isComplex
    val parts: Seq[NDArray[Double]] =
      if isComplex then
        val z = y.asType(using DType.Complex128)
        Seq(z.map(_.re)(using DType.Float64), z.map(_.im)(using DType.Float64))
      else Seq(y.asType(using DType.Float64))
    val res = parts.map { p =>
      val (buf, m, l) = gather(p, Array(ax))
      val out = new Array[Double](m)
      val tmp = new Array[Double](math.max(0, l - 1))
      var r = 0
      while r < m do
        var j = 0
        while j < l - 1 do
          val d = xLanes match
            case None => dx
            case Some((xs, false)) => xs(j)
            case Some((xs, true)) => xs(r * l + j + 1) - xs(r * l + j)
          tmp(j) = d * (buf(r * l + j + 1) + buf(r * l + j)) / 2.0
          j += 1
        out(r) = Reduce.pairwiseSum(tmp, 0, tmp.length)
        r += 1
      out
    }
    val outShape = reducedShape(y.shapeArr, Array(ax), false)
    if isComplex then
      NDArray
        .fromArray(Array.tabulate(res(0).length)(i => Complex(res(0)(i), res(1)(i))), outShape)(using DType.Complex128)
        .asInstanceOf[NDArray[U]]
    else NDArray.fromArray(res(0).map(od.fromDouble)(using od.classTag), outShape)(using od)

  // ---------------------------------------------------------------- cov / corrcoef

  /** Covariance matrix (`np.cov(m, y, rowvar, bias, ddof, fweights, aweights)`).
    * Rows of `m` are variables (columns if `rowvar = false`); a single variable gives a 0-d array.
    */
  def cov[T](
      m: NDArray[T],
      y: NDArray[?] | Null = null,
      rowvar: Boolean = true,
      bias: Boolean = false,
      ddof: Int | Null = null,
      fweights: NDArray[?] | Null = null,
      aweights: NDArray[?] | Null = null
  )(using p: DivPromote[T, Double]): NDArray[p.Out] =
    val od = p.dtype
    if m.ndim > 2 then throw new IllegalArgumentException("m has more than 2 dimensions")
    def asMatrix(a: NDArray[?]): (Array[p.Out], Int, Int) =
      val ad = a.asInstanceOf[NDArray[Any]]
      if !od.isComplex && ad.dtype.isComplex then throw new IllegalArgumentException("cannot cast complex y to a real result")
      val x2: NDArray[Any] =
        if ad.ndim == 0 then ad.reshape(1, 1) else if ad.ndim == 1 then ad.reshape(1, ad.shapeArr(0)) else ad
      val x3 = if !rowvar && x2.shapeArr(0) != 1 then x2.transpose() else x2
      (x3.asType(using od).toArray, x3.shapeArr(0), x3.shapeArr(1))
    var (xs, nv, ns) = asMatrix(m)
    if nv == 0 then return NDArray.zerosOf(od, Array(0, 0))
    if y != null then
      if y.ndim > 2 then throw new IllegalArgumentException("y has more than 2 dimensions")
      val (ys, nvy, nsy) = asMatrix(y)
      if nsy != ns then
        throw new IllegalArgumentException(
          s"all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size $ns and the array at index 1 has size $nsy"
        )
      xs = (xs ++ ys)(using od.classTag)
      nv += nvy
    val dd: Int = if ddof == null then (if bias then 0 else 1) else ddof.asInstanceOf[Int]
    def checkW(w: NDArray[?], name: String): Array[Double] =
      if w.ndim > 1 then throw new IllegalStateException(s"cannot handle multidimensional $name")
      if w.ndim == 0 || w.shapeArr(0) != ns then
        throw new IllegalStateException(s"incompatible numbers of samples and $name")
      val v = doublesOf(w, name)
      if name == "fweights" && v.exists(x => x != math.rint(x)) then
        throw new IllegalArgumentException("fweights must be integer")
      if v.exists(_ < 0) then throw new IllegalArgumentException(s"$name cannot be negative")
      v
    val fw = if fweights == null then null else checkW(fweights, "fweights")
    val aw = if aweights == null then null else checkW(aweights, "aweights")
    val w: Array[Double] | Null =
      if fw == null then aw
      else if aw == null then fw
      else Array.tabulate(ns)(i => fw(i) * aw(i))
    // weighted means
    val avg = od.newArray(nv)
    var wSum = ns.toDouble
    if w == null then
      var r = 0
      while r < nv do
        avg(r) = Reduce.meanBuf(od, xs.slice(r * ns, (r + 1) * ns), ns, od)
        r += 1
    else
      val wv = w.asInstanceOf[Array[Double]]
      wSum = Reduce.pairwiseSum(wv, 0, ns)
      if wSum == 0.0 then throw new ArithmeticException("Weights sum to zero, can't be normalized")
      var r = 0
      val tmp = od.newArray(ns)
      while r < nv do
        var k = 0
        while k < ns do
          tmp(k) = od.times(xs(r * ns + k), od.fromDouble(wv(k)))
          k += 1
        avg(r) = od.div(Reduce.sumBuf(od, tmp, ns, od), od.fromDouble(wSum))
        r += 1
    var fact: Double =
      if w == null then (ns - dd).toDouble
      else if dd == 0 then wSum
      else if aw == null then wSum - dd
      else
        val wv = w.asInstanceOf[Array[Double]]
        val awv = aw.asInstanceOf[Array[Double]]
        wSum - dd * Reduce.pairwiseSum(Array.tabulate(ns)(i => wv(i) * awv(i)), 0, ns) / wSum
    if fact <= 0 then fact = 0.0
    val xc = od.newArray(nv * ns)
    var i = 0
    while i < nv * ns do
      xc(i) = od.minus(xs(i), avg(i / ns))
      i += 1
    val conj: p.Out => p.Out =
      if od.isComplex then z => z.asInstanceOf[Complex].conj.asInstanceOf[p.Out] else identity
    val xt = od.newArray(nv * ns)
    i = 0
    while i < nv * ns do
      xt(i) = conj(if w == null then xc(i) else od.times(xc(i), od.fromDouble(w.asInstanceOf[Array[Double]](i % ns))))
      i += 1
    val scale = od.fromDouble(1.0 / fact)
    val c = od.newArray(nv * nv)
    val acc = od.newArray(ns)
    var a = 0
    while a < nv do
      var b = 0
      while b < nv do
        var k = 0
        while k < ns do
          acc(k) = od.times(xc(a * ns + k), xt(b * ns + k))
          k += 1
        c(a * nv + b) = od.times(Reduce.sumBuf(od, acc, ns, od), scale)
        b += 1
      a += 1
    if nv == 1 then NDArray.fromArray(c, Array.emptyIntArray)(using od)
    else NDArray.fromArray(c, Array(nv, nv))(using od)

  /** Pearson correlation coefficients (`np.corrcoef(x, y, rowvar)`), clipped to `[-1, 1]`. */
  def corrcoef[T](x: NDArray[T], y: NDArray[?] | Null = null, rowvar: Boolean = true)(using
      p: DivPromote[T, Double]
  ): NDArray[p.Out] =
    val od = p.dtype
    val c = cov(x, y, rowvar)
    if c.ndim == 0 then return c.map(v => od.div(v, v))(using od)
    val nv = c.shapeArr(0)
    val cv = c.toArray
    val sd = Array.tabulate(nv)(i => math.sqrt(od.toComplex(cv(i * nv + i)).re))
    val out = od.newArray(nv * nv)
    var i = 0
    while i < nv do
      var j = 0
      while j < nv do
        val v = od.div(od.div(cv(i * nv + j), od.fromDouble(sd(i))), od.fromDouble(sd(j)))
        out(i * nv + j) =
          if od.isComplex then
            val z = v.asInstanceOf[Complex]
            Complex(clip1(z.re), clip1(z.im)).asInstanceOf[p.Out]
          else od.fromDouble(clip1(od.toDouble(v)))
        j += 1
      i += 1
    NDArray.fromArray(out, Array(nv, nv))(using od)

  private def clip1(v: Double): Double = if v.isNaN then v else math.max(-1.0, math.min(1.0, v))

  // ---------------------------------------------------------------- correlate / convolve

  /** Cross-correlation of two 1-d sequences (`np.correlate(a, v, mode)`); `v` is conjugated. */
  def correlate[A, B](a: NDArray[A], v: NDArray[B], mode: String = "valid")(using p: NumPromote[A, B]): NDArray[p.Out] =
    val (x, y) = conv1dArgs(a, v, p.dtype)
    val n = x.length
    val m = y.length
    val od = p.dtype
    val yc =
      if od.isComplex then y.map(z => z.asInstanceOf[Complex].conj.asInstanceOf[p.Out])(using od.classTag) else y
    val full = corrFull(x, yc, od)
    val (start, len) = mode match
      case "full" => (0, n + m - 1)
      case "same" =>
        if n >= m then ((m - 1) - m / 2, n) else (n - 1 - ((n - 1) - n / 2), m)
      case "valid" =>
        if n >= m then (m - 1, n - m + 1) else (n - 1, m - n + 1)
      case other => throw badMode(other)
    NDArray.fromArray(full.slice(start, start + len), Array(len))(using od)

  /** Discrete linear convolution of two 1-d sequences (`np.convolve(a, v, mode)`). */
  def convolve[A, B](a: NDArray[A], v: NDArray[B], mode: String = "full")(using p: NumPromote[A, B]): NDArray[p.Out] =
    val (x0, y0) = conv1dArgs(a, v, p.dtype)
    val (x, y) = if y0.length > x0.length then (y0, x0) else (x0, y0)
    val n = x.length
    val m = y.length
    val od = p.dtype
    val full = corrFull(x, y.reverse, od)
    val (start, len) = mode match
      case "full" => (0, n + m - 1)
      case "same" => ((m - 1) - m / 2, n)
      case "valid" => (m - 1, n - m + 1)
      case other => throw badMode(other)
    NDArray.fromArray(full.slice(start, start + len), Array(len))(using od)

  private def badMode(m: String) =
    new IllegalArgumentException(s"mode must be one of 'valid', 'same', or 'full' (got '$m')")

  private def conv1dArgs[A, B, U](a: NDArray[A], v: NDArray[B], od: NumDType[U]): (Array[U], Array[U]) =
    if a.ndim > 1 || v.ndim > 1 then throw new IllegalArgumentException("object too deep for desired array")
    val x = a.asType(using od).toArray
    val y = v.asType(using od).toArray
    if x.isEmpty then throw new IllegalArgumentException("a cannot be empty")
    if y.isEmpty then throw new IllegalArgumentException("v cannot be empty")
    (x, y)

  /** Full correlation `z(i) = sum_j x(j + i - (m - 1)) * y(j)`. */
  private def corrFull[U](x: Array[U], y: Array[U], od: NumDType[U]): Array[U] =
    val n = x.length
    val m = y.length
    val out = od.newArray(n + m - 1)
    (od: DType[?]) match
      case DType.Float64 =>
        val xd = x.asInstanceOf[Array[Double]]
        val yd = y.asInstanceOf[Array[Double]]
        val o = out.asInstanceOf[Array[Double]]
        var i = 0
        while i < n + m - 1 do
          val k = i - (m - 1)
          var s = 0.0
          var j = math.max(0, -k)
          val hi = math.min(m - 1, n - 1 - k)
          while j <= hi do
            s += xd(j + k) * yd(j)
            j += 1
          o(i) = s
          i += 1
      case _ =>
        var i = 0
        while i < n + m - 1 do
          val k = i - (m - 1)
          var s = od.zero
          var j = math.max(0, -k)
          val hi = math.min(m - 1, n - 1 - k)
          while j <= hi do
            s = od.plus(s, od.times(x(j + k), y(j)))
            j += 1
          out(i) = s
          i += 1
    out

  // ---------------------------------------------------------------- interp

  /** One-dimensional piecewise-linear interpolation (`np.interp(x, xp, fp, left, right, period)`).
    * `xp` must be increasing (unless `period` is given); complex `fp` gives a complex result.
    */
  def interp[F](
      x: NDArray[?],
      xp: NDArray[?],
      fp: NDArray[F],
      left: Double | Complex | Null = null,
      right: Double | Complex | Null = null,
      period: Double | Null = null
  )(using p: DivPromote[F, Double]): NDArray[p.Out] =
    val od = p.dtype
    val xs = doublesOf(x, "x")
    if xp.ndim != 1 || fp.ndim != 1 then throw new IllegalArgumentException("Data points must be 1-D sequences")
    var xpv = doublesOf(xp, "xp")
    val fpc = fp.asType(using DType.Complex128).toArray
    if xpv.length != fpc.length then throw new IllegalArgumentException("fp and xp are not of the same length.")
    if xpv.isEmpty then throw new IllegalArgumentException("array of sample points is empty")
    var fre = fpc.map(_.re)
    var fim = fpc.map(_.im)
    var xq = xs
    var lv: Complex | Null = left match
      case null => null
      case d: Double => Complex(d, 0.0)
      case c: Complex => c
    var rv: Complex | Null = right match
      case null => null
      case d: Double => Complex(d, 0.0)
      case c: Complex => c
    if period != null then
      val per0 = period.asInstanceOf[Double]
      if per0 == 0 then throw new IllegalArgumentException("period must be a non-zero value")
      val per = math.abs(per0)
      lv = null
      rv = null
      xq = xs.map(pyMod(_, per))
      val xpm = xpv.map(pyMod(_, per))
      val order = xpm.indices.sortWith((i, j) => xpm(i) < xpm(j)).toArray
      val sx = order.map(xpm(_))
      val sre = order.map(fre(_))
      val sim = order.map(fim(_))
      val k = sx.length
      xpv = (sx(k - 1) - per) +: sx :+ (sx(0) + per)
      fre = sre(k - 1) +: sre :+ sre(0)
      fim = sim(k - 1) +: sim :+ sim(0)
    val lre = if lv == null then fre(0) else lv.asInstanceOf[Complex].re
    val lim = if lv == null then fim(0) else lv.asInstanceOf[Complex].im
    val rre = if rv == null then fre(fre.length - 1) else rv.asInstanceOf[Complex].re
    val rim = if rv == null then fim(fim.length - 1) else rv.asInstanceOf[Complex].im
    val re = xq.map(v => interp1(v, xpv, fre, lre, rre))
    if od.isComplex then
      val im = xq.map(v => interp1(v, xpv, fim, lim, rim))
      NDArray
        .fromArray(Array.tabulate(re.length)(i => Complex(re(i), im(i))), x.shapeArr.clone())(using DType.Complex128)
        .asInstanceOf[NDArray[p.Out]]
    else NDArray.fromArray(re, x.shapeArr.clone())(using DType.Float64).asInstanceOf[NDArray[p.Out]]

  /** Interpolates a single point (`np.interp(x, xp, fp)` with scalar `x`). */
  def interp[F](x: Double, xp: NDArray[?], fp: NDArray[F])(using p: DivPromote[F, Double]): p.Out =
    interp(NDArray.scalar(x), xp, fp, null, null, null).item

  private def interp1(x: Double, xp: Array[Double], fp: Array[Double], lval: Double, rval: Double): Double =
    val len = xp.length
    if len == 1 then
      if x < xp(0) then lval else if x > xp(0) then rval else fp(0)
    else if x.isNaN then Double.NaN
    else if x > xp(len - 1) then rval
    else if x < xp(0) then lval
    else
      val j = searchSorted(xp, x, right = true) - 1
      if j >= len - 1 then fp(len - 1)
      else if xp(j) == x then fp(j)
      else
        val slope = (fp(j + 1) - fp(j)) / (xp(j + 1) - xp(j))
        var r = slope * (x - xp(j)) + fp(j)
        if r.isNaN then
          r = slope * (x - xp(j + 1)) + fp(j + 1)
          if r.isNaN && fp(j) == fp(j + 1) then r = fp(j)
        r
