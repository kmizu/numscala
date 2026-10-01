package numscala

/** Discrete Fourier transforms — `numpy.fft`.
  *
  * Every transform accepts real or complex input of any numeric dtype. Conventions follow
  * NumPy: `n` (or `s`) pads with zeros or truncates the input along the transformed axes,
  * `n = -1` / `s = Nil` mean "use the input length" (`None`), `axes = Nil` means "the default
  * axes" (`None`), and `norm` is one of `"backward"` (default), `"ortho"` or `"forward"`.
  *
  * {{{
  * np.fft.fft(np.array(1.0, 2.0, 3.0, 4.0))        // [10+0j, -2+2j, -2+0j, -2-2j]
  * np.fft.irfft(np.fft.rfft(x), n = x.shape(0))      // round trip
  * np.fft.fftshift(np.fft.fftfreq(8))
  * }}}
  */
object FFT:

  // ------------------------------------------------------------------ helpers

  private def checkNorm(norm: String): Unit = norm match
    case "backward" | "ortho" | "forward" => ()
    case other =>
      throw new IllegalArgumentException(
        s"""Invalid norm value $other; should be "backward", "ortho" or "forward"."""
      )

  /** Scale factor applied after an unnormalized transform of length `n`. */
  private def factor(norm: String, n: Int, inverse: Boolean): Double =
    checkNorm(norm)
    norm match
      case "ortho" => 1.0 / math.sqrt(n.toDouble)
      case "backward" => if inverse then 1.0 / n else 1.0
      case _ => if inverse then 1.0 else 1.0 / n

  private def swapNorm(norm: String): String =
    checkNorm(norm)
    norm match
      case "backward" => "forward"
      case "forward" => "backward"
      case o => o

  private def toComplex[T](a: NDArray[T])(using d: NumDType[T]): NDArray[Complex] =
    if d.isComplex then a.asInstanceOf[NDArray[Complex]] else a.map(d.toComplex)

  private def toReal[T](a: NDArray[T])(using d: NumDType[T]): NDArray[Double] =
    if d eq DType.Float64 then a.asInstanceOf[NDArray[Double]] else a.map(d.toDouble)

  private def checkAxis(a: NDArray[?], axis: Int): Int =
    if a.ndim == 0 then throw new IllegalArgumentException(s"axis $axis is out of bounds for array of dimension 0")
    Shape.normAxis(axis, a.ndim)

  private def checkN(n: Int): Unit =
    if n < 1 then throw new IllegalArgumentException(s"Invalid number of FFT data points ($n) specified.")

  /** Complex-to-complex transform along one axis. */
  private[numscala] def c2c(a: NDArray[Complex], n0: Int, axis: Int, inverse: Boolean, norm: String): NDArray[Complex] =
    val ax = checkAxis(a, axis)
    val n = if n0 == -1 then a.shapeArr(ax) else n0
    checkN(n)
    val fct = factor(norm, n, inverse)
    val plan = FFTPlan.get(n)
    val re = new Array[Double](n)
    val im = new Array[Double](n)
    Lanes.transform[Complex, Complex](a, ax, n) { (in, len, out) =>
      var i = 0
      val k = math.min(len, n)
      while i < k do
        val z = in(i)
        re(i) = z.re
        im(i) = z.im
        i += 1
      while i < n do
        re(i) = 0.0
        im(i) = 0.0
        i += 1
      if inverse then plan.backward(re, im) else plan.forward(re, im)
      i = 0
      while i < n do
        out(i) = Complex(re(i) * fct, im(i) * fct)
        i += 1
    }

  /** Real-to-complex transform along one axis (first `n/2+1` outputs). */
  private def r2c(a: NDArray[Double], n0: Int, axis: Int, norm: String): NDArray[Complex] =
    val ax = checkAxis(a, axis)
    val n = if n0 == -1 then a.shapeArr(ax) else n0
    checkN(n)
    val fct = factor(norm, n, false)
    val plan = FFTPlan.get(n)
    val re = new Array[Double](n)
    val im = new Array[Double](n)
    val outLen = n / 2 + 1
    Lanes.transform[Double, Complex](a, ax, outLen) { (in, len, out) =>
      var i = 0
      val k = math.min(len, n)
      while i < k do
        re(i) = in(i)
        im(i) = 0.0
        i += 1
      while i < n do
        re(i) = 0.0
        im(i) = 0.0
        i += 1
      plan.forward(re, im)
      i = 0
      while i < outLen do
        out(i) = Complex(re(i) * fct, im(i) * fct)
        i += 1
    }

  /** Hermitian complex-to-real inverse transform along one axis producing `n` real points. */
  private def c2r(a: NDArray[Complex], n0: Int, axis: Int, norm: String): NDArray[Double] =
    val ax = checkAxis(a, axis)
    val n = if n0 == -1 then 2 * (a.shapeArr(ax) - 1) else n0
    checkN(n)
    val fct = factor(norm, n, true)
    val plan = FFTPlan.get(n)
    val re = new Array[Double](n)
    val im = new Array[Double](n)
    val half = n / 2 + 1
    Lanes.transform[Complex, Double](a, ax, n) { (in, len, out) =>
      java.util.Arrays.fill(re, 0.0)
      java.util.Arrays.fill(im, 0.0)
      val k = math.min(len, half)
      var i = 0
      while i < k do
        re(i) = in(i).re
        im(i) = in(i).im
        i += 1
      // Hermitian completion
      i = 1
      while i < (n + 1) / 2 do
        re(n - i) = re(i)
        im(n - i) = -im(i)
        i += 1
      // pocketfft ignores the imaginary parts of the DC and Nyquist terms
      im(0) = 0.0
      if n % 2 == 0 then im(n / 2) = 0.0
      plan.backward(re, im)
      i = 0
      while i < n do
        out(i) = re(i) * fct
        i += 1
    }

  /** NumPy's `_cook_nd_args`: resolves (s, axes) for the n-dimensional transforms. */
  private def cookNd(a: NDArray[?], s: Seq[Int], axes: Seq[Int], invreal: Boolean): (Array[Int], Array[Int]) =
    val shapeless = s.isEmpty
    val ax: Array[Int] =
      if axes.isEmpty then
        val k = if shapeless then a.ndim else s.length
        (-k until 0).toArray
      else axes.toArray
    if a.ndim == 0 && ax.nonEmpty then throw new IllegalArgumentException("axis -1 is out of bounds for array of dimension 0")
    val axn = ax.map(Shape.normAxis(_, a.ndim))
    val ss: Array[Int] =
      if shapeless then axn.map(a.shapeArr(_))
      else s.toArray
    if ss.length != axn.length then throw new IllegalArgumentException("Shape and axes have different lengths.")
    // NumPy >= 2: -1 in `s` means "use the input length along that axis"
    for i <- ss.indices if ss(i) == -1 do
      ss(i) = if invreal && i == ss.length - 1 then (a.shapeArr(axn(i)) - 1) * 2 else a.shapeArr(axn(i))
    if invreal && shapeless && axn.nonEmpty then ss(ss.length - 1) = (a.shapeArr(axn.last) - 1) * 2
    (ss, axn)

  // ------------------------------------------------------------------ 1-D

  /** `numpy.fft.fft`: 1-D discrete Fourier transform along `axis`. */
  def fft[T](a: NDArray[T], n: Int = -1, axis: Int = -1, norm: String = "backward")(using
      d: NumDType[T]
  ): NDArray[Complex] = c2c(toComplex(a), n, axis, false, norm)

  /** `numpy.fft.ifft`: 1-D inverse discrete Fourier transform along `axis`. */
  def ifft[T](a: NDArray[T], n: Int = -1, axis: Int = -1, norm: String = "backward")(using
      d: NumDType[T]
  ): NDArray[Complex] = c2c(toComplex(a), n, axis, true, norm)

  /** `numpy.fft.rfft`: 1-D transform of real input, returning the `n/2+1` non-negative frequencies.
    * Complex input is rejected at compile time (NumPy raises `TypeError`).
    */
  def rfft[T](a: NDArray[T], n: Int = -1, axis: Int = -1, norm: String = "backward")(using
      d: RealDType[T]
  ): NDArray[Complex] = r2c(toReal(a), n, axis, norm)

  /** `numpy.fft.irfft`: inverse of [[rfft]]; the output has `n` (default `2*(m-1)`) real points. */
  def irfft[T](a: NDArray[T], n: Int = -1, axis: Int = -1, norm: String = "backward")(using
      d: NumDType[T]
  ): NDArray[Double] = c2r(toComplex(a), n, axis, norm)

  /** `numpy.fft.hfft`: FFT of a Hermitian-symmetric signal given by its first half; real output. */
  def hfft[T](a: NDArray[T], n: Int = -1, axis: Int = -1, norm: String = "backward")(using
      d: NumDType[T]
  ): NDArray[Double] =
    c2r(toComplex(a).map(_.conj), n, axis, swapNorm(norm))

  /** `numpy.fft.ihfft`: inverse of [[hfft]] for real input (`n/2+1` complex outputs). */
  def ihfft[T](a: NDArray[T], n: Int = -1, axis: Int = -1, norm: String = "backward")(using
      d: RealDType[T]
  ): NDArray[Complex] =
    r2c(toReal(a), n, axis, swapNorm(norm)).map(_.conj)

  // ------------------------------------------------------------------ n-D

  /** `numpy.fft.fftn`: n-dimensional FFT over `axes` (default: all axes, or the last `len(s)`). */
  def fftn[T](a: NDArray[T], s: Seq[Int] = Nil, axes: Seq[Int] = Nil, norm: String = "backward")(using
      d: NumDType[T]
  ): NDArray[Complex] = rawFftNd(toComplex(a), s, axes, false, norm)

  /** `numpy.fft.ifftn`: n-dimensional inverse FFT. */
  def ifftn[T](a: NDArray[T], s: Seq[Int] = Nil, axes: Seq[Int] = Nil, norm: String = "backward")(using
      d: NumDType[T]
  ): NDArray[Complex] = rawFftNd(toComplex(a), s, axes, true, norm)

  /** `numpy.fft.fft2`: 2-D FFT over `axes` (default the last two). */
  def fft2[T](a: NDArray[T], s: Seq[Int] = Nil, axes: Seq[Int] = Seq(-2, -1), norm: String = "backward")(using
      d: NumDType[T]
  ): NDArray[Complex] = rawFftNd(toComplex(a), s, axes, false, norm)

  /** `numpy.fft.ifft2`: 2-D inverse FFT over `axes` (default the last two). */
  def ifft2[T](a: NDArray[T], s: Seq[Int] = Nil, axes: Seq[Int] = Seq(-2, -1), norm: String = "backward")(using
      d: NumDType[T]
  ): NDArray[Complex] = rawFftNd(toComplex(a), s, axes, true, norm)

  private def rawFftNd(a: NDArray[Complex], s: Seq[Int], axes: Seq[Int], inverse: Boolean, norm: String): NDArray[Complex] =
    checkNorm(norm)
    val (ss, ax) = cookNd(a, s, axes, false)
    var r = a
    var i = ax.length - 1
    while i >= 0 do
      r = c2c(r, ss(i), ax(i), inverse, norm)
      i -= 1
    if ax.isEmpty then r.copy() else r

  /** `numpy.fft.rfftn`: n-dimensional FFT of real input (last axis halved). */
  def rfftn[T](a: NDArray[T], s: Seq[Int] = Nil, axes: Seq[Int] = Nil, norm: String = "backward")(using
      d: RealDType[T]
  ): NDArray[Complex] =
    val ra = toReal(a)
    val (ss, ax) = cookNd(ra, s, axes, false)
    if ax.isEmpty then throw new IllegalArgumentException("at least 1 axis must be transformed")
    var r = r2c(ra, ss.last, ax.last, norm)
    var i = 0
    while i < ax.length - 1 do
      r = c2c(r, ss(i), ax(i), false, norm)
      i += 1
    r

  /** `numpy.fft.irfftn`: inverse of [[rfftn]]. */
  def irfftn[T](a: NDArray[T], s: Seq[Int] = Nil, axes: Seq[Int] = Nil, norm: String = "backward")(using
      d: NumDType[T]
  ): NDArray[Double] =
    val ca = toComplex(a)
    val (ss, ax) = cookNd(ca, s, axes, true)
    if ax.isEmpty then throw new IllegalArgumentException("at least 1 axis must be transformed")
    var r = ca
    var i = 0
    while i < ax.length - 1 do
      r = c2c(r, ss(i), ax(i), true, norm)
      i += 1
    c2r(r, ss.last, ax.last, norm)

  /** `numpy.fft.rfft2`: 2-D FFT of real input. */
  def rfft2[T](a: NDArray[T], s: Seq[Int] = Nil, axes: Seq[Int] = Seq(-2, -1), norm: String = "backward")(using
      d: RealDType[T]
  ): NDArray[Complex] = rfftn(a, s, axes, norm)

  /** `numpy.fft.irfft2`: inverse of [[rfft2]]. */
  def irfft2[T](a: NDArray[T], s: Seq[Int] = Nil, axes: Seq[Int] = Seq(-2, -1), norm: String = "backward")(using
      d: NumDType[T]
  ): NDArray[Double] = irfftn(a, s, axes, norm)

  // ------------------------------------------------------------------ helpers (public)

  /** `numpy.fft.fftfreq`: sample frequencies `[0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n)`. */
  def fftfreq(n: Int, d: Double = 1.0): NDArray[Double] =
    if n < 1 then throw new IllegalArgumentException("n should be an integer greater than 0")
    val v = 1.0 / (n * d)
    val out = new Array[Double](n)
    val nPos = (n - 1) / 2 + 1
    var i = 0
    while i < nPos do
      out(i) = i * v
      i += 1
    var k = -(n / 2)
    while i < n do
      out(i) = k * v
      i += 1
      k += 1
    NDArray.fromArray(out, Array(n))

  /** `numpy.fft.rfftfreq`: sample frequencies `[0, 1, ..., n/2] / (d*n)` for [[rfft]]. */
  def rfftfreq(n: Int, d: Double = 1.0): NDArray[Double] =
    if n < 1 then throw new IllegalArgumentException("n should be an integer greater than 0")
    val v = 1.0 / (n * d)
    val m = n / 2 + 1
    NDArray.fromArray(Array.tabulate(m)(i => i * v), Array(m))

  /** `numpy.fft.fftshift`: moves the zero-frequency term to the centre (`axes = Nil`: all axes). */
  def fftshift[T](x: NDArray[T], axes: Seq[Int] = Nil): NDArray[T] = shift(x, axes, inverse = false)

  /** `numpy.fft.fftshift` along a single axis. */
  def fftshift[T](x: NDArray[T], axes: Int): NDArray[T] = shift(x, Seq(axes), inverse = false)

  /** `numpy.fft.ifftshift`: inverse of [[fftshift]] (`axes = Nil`: all axes). */
  def ifftshift[T](x: NDArray[T], axes: Seq[Int] = Nil): NDArray[T] = shift(x, axes, inverse = true)

  /** `numpy.fft.ifftshift` along a single axis. */
  def ifftshift[T](x: NDArray[T], axes: Int): NDArray[T] = shift(x, Seq(axes), inverse = true)

  private def shift[T](x: NDArray[T], axes: Seq[Int], inverse: Boolean): NDArray[T] =
    val ax = if axes.isEmpty then (0 until x.ndim).toArray else axes.map(Shape.normAxis(_, x.ndim)).toArray
    val shifts = new Array[Int](x.ndim)
    ax.foreach { a =>
      val dim = x.shapeArr(a)
      shifts(a) += (if inverse then -(dim / 2) else dim / 2)
    }
    roll(x, shifts)

  /** Rolls every axis `i` by `shifts(i)` (NumPy `roll` semantics). */
  private def roll[T](x: NDArray[T], shifts: Array[Int]): NDArray[T] =
    val shape = x.shapeArr
    val nd = shape.length
    val src = x.toArray
    if nd == 0 then return NDArray.fromArray(src, Array.emptyIntArray)(using x.dtype)
    val out = x.dtype.newArray(src.length)
    val strides = Shape.cStrides(shape)
    val sh = Array.tabulate(nd)(i => if shape(i) == 0 then 0 else Math.floorMod(shifts(i), shape(i)))
    val idx = new Array[Int](nd)
    var flat = 0
    while flat < src.length do
      var dst = 0
      var i = 0
      while i < nd do
        var j = idx(i) + sh(i)
        if j >= shape(i) then j -= shape(i)
        dst += j * strides(i)
        i += 1
      out(dst) = src(flat)
      // increment index
      var k = nd - 1
      var carry = true
      while carry && k >= 0 do
        idx(k) += 1
        if idx(k) < shape(k) then carry = false
        else
          idx(k) = 0
          k -= 1
      flat += 1
    NDArray.fromArray(out, shape.clone())(using x.dtype)
