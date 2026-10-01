package com.github.kmizu.numscala

/** Implementation entry points of `np.polyfit` (public only because `polyfit` is inline). */
object PolyFitImpl:
  def plain[A, B](x: NDArray[A], y: NDArray[B], deg: Int, rcond: Double, w: NDArray[?] | Null): NDArray[Double] =
    PolyLegacy.fit(x, y, deg, rcond, w).coef
  def full[A, B](x: NDArray[A], y: NDArray[B], deg: Int, rcond: Double, w: NDArray[?] | Null): PolyfitResult =
    val f = PolyLegacy.fit(x, y, deg, rcond, w)
    PolyfitResult(f.coef, f.resid, f.rank, f.sv, f.rcond)
  def cov[A, B](x: NDArray[A], y: NDArray[B], deg: Int, rcond: Double, w: NDArray[?] | Null, unscaled: Boolean)
      : (NDArray[Double], NDArray[Double]) =
    val f = PolyLegacy.fit(x, y, deg, rcond, w)
    PolyLegacy.fitCov(f, x.size, deg + 1, y.ndim, unscaled)

/** Legacy polynomial routines of the `np` namespace (`np.poly`, `np.polyval`, `np.polyfit`,
  * `np.roots`, ..., `np.poly1d`). Coefficient arrays are ordered from the highest power down.
  */
trait NpPoly:
  import PolyLegacy.*

  /** The `numpy.poly1d` class. */
  type poly1d[T] = Poly1d[T]

  /** `numpy.poly1d(c)`: a polynomial object from coefficients (highest power first). */
  def poly1d[T](c_or_r: NDArray[T], variable: String = "x")(using d: NumDType[T]): Poly1d[T] =
    Poly1d(c_or_r, variable)

  /** `numpy.poly1d(c_or_r, r)`: with `r = true` the argument holds the roots. */
  def poly1d[T](c_or_r: NDArray[T], r: Boolean)(using t: ToInexact[T]): Poly1d[t.Out] =
    given InexactDType[t.Out] = t.dtype
    if r then Poly1d.fromRoots(c_or_r)
    else Poly1d(NDArray.fromArray(cast(atleast1d(c_or_r), t.dtype), Array(atleast1d(c_or_r).size)))

  /** `numpy.poly`: coefficients of the monic polynomial with the given roots, or the
    * characteristic polynomial of a square matrix.
    */
  def poly[T](seq_of_zeros: NDArray[T])(using t: ToInexact[T]): NDArray[t.Out] =
    PolyLegacy.poly(seq_of_zeros, t.dtype)

  /** `numpy.polyval`: evaluates `p` (highest power first) at every element of `x` (Horner). */
  def polyval[T, U](p: NDArray[T], x: NDArray[U])(using pr: NumPromote[T, U]): NDArray[pr.Out] =
    val d = pr.dtype
    val pc = cast(atleast1d(p), d)
    x.map(xv => horner(pc, d.castFrom(x.dtype, xv), d))(using d)

  /** `numpy.polyval` at a scalar point. */
  def polyval[T, U](p: NDArray[T], x: U)(using dx: NumDType[U], pr: NumPromote[T, U]): pr.Out =
    val d = pr.dtype
    horner(cast(atleast1d(p), d), d.castFrom(dx, x), d)

  /** `numpy.polyval` with a polynomial argument: the composition `p(x)`. */
  def polyval[T](p: Poly1d[T], x: Poly1d[T]): Poly1d[T] = p(x)

  /** `numpy.polyfit`: least-squares polynomial fit of degree `deg` (coefficients highest first).
    *
    * `rcond < 0` means NumPy's default `len(x)*eps`; `w` holds optional weights. With
    * `full = true` the result is a [[PolyfitResult]]; with `cov = true` (or `"unscaled"`) it is
    * the pair `(coef, covariance)`; otherwise the coefficient array. `y` may be 2-D (one fit per column).
    */
  transparent inline def polyfit[A, B](
      x: NDArray[A],
      y: NDArray[B],
      deg: Int,
      rcond: Double = -1.0,
      inline full: Boolean = false,
      w: NDArray[?] | Null = null,
      inline cov: Boolean | String = false
  ) =
    inline if full then PolyFitImpl.full(x, y, deg, rcond, w)
    else
      inline cov match
        case false => PolyFitImpl.plain(x, y, deg, rcond, w)
        case true => PolyFitImpl.cov(x, y, deg, rcond, w, false)
        case "unscaled" => PolyFitImpl.cov(x, y, deg, rcond, w, true)

  /** `numpy.roots`: the roots of a polynomial (eigenvalues of its companion matrix), as complex numbers. */
  def roots[T](p: NDArray[T])(using d: NumDType[T]): NDArray[Complex] = PolyLegacy.roots(p)

  /** `numpy.polyadd`. */
  def polyadd[T, U](a1: NDArray[T], a2: NDArray[U])(using p: NumPromote[T, U]): NDArray[p.Out] =
    given NumDType[p.Out] = p.dtype
    arr(add(cast(atleast1d(a1), p.dtype), cast(atleast1d(a2), p.dtype), p.dtype, sub = false))

  /** `numpy.polyadd` of two `poly1d` objects. */
  def polyadd[T, U](a1: Poly1d[T], a2: Poly1d[U])(using p: NumPromote[T, U]): Poly1d[p.Out] = a1 + a2

  /** `numpy.polysub`. */
  def polysub[T, U](a1: NDArray[T], a2: NDArray[U])(using p: NumPromote[T, U]): NDArray[p.Out] =
    given NumDType[p.Out] = p.dtype
    arr(add(cast(atleast1d(a1), p.dtype), cast(atleast1d(a2), p.dtype), p.dtype, sub = true))

  /** `numpy.polysub` of two `poly1d` objects. */
  def polysub[T, U](a1: Poly1d[T], a2: Poly1d[U])(using p: NumPromote[T, U]): Poly1d[p.Out] = a1 - a2

  /** `numpy.polymul`. */
  def polymul[T, U](a1: NDArray[T], a2: NDArray[U])(using p: NumPromote[T, U]): NDArray[p.Out] =
    given NumDType[p.Out] = p.dtype
    arr(convolve(cast(atleast1d(a1), p.dtype), cast(atleast1d(a2), p.dtype), p.dtype))

  /** `numpy.polymul` of two `poly1d` objects. */
  def polymul[T, U](a1: Poly1d[T], a2: Poly1d[U])(using p: NumPromote[T, U]): Poly1d[p.Out] = a1 * a2

  /** `numpy.polydiv`: `(quotient, remainder)`. */
  def polydiv[T, U](u: NDArray[T], v: NDArray[U])(using p: DivPromote[T, U]): (NDArray[p.Out], NDArray[p.Out]) =
    given InexactDType[p.Out] = p.dtype
    val (q, r) = div(cast(atleast1d(u), p.dtype), cast(atleast1d(v), p.dtype), p.dtype)
    (arr(q), arr(r))

  /** `numpy.polydiv` of two `poly1d` objects. */
  def polydiv[T, U](u: Poly1d[T], v: Poly1d[U])(using p: DivPromote[T, U]): (Poly1d[p.Out], Poly1d[p.Out]) = u / v

  /** `numpy.polyder`: the `m`-th derivative. */
  def polyder[T](p: NDArray[T], m: Int = 1)(using d: NumDType[T]): NDArray[T] =
    arr(der(atleast1d(p).toArray, m, d))

  /** `numpy.polyint`: the `m`-th antiderivative; `k` holds the integration constants
    * (empty: zeros; one value: used for every step).
    */
  def polyint[T](p: NDArray[T], m: Int = 1, k: Seq[Double] = Nil)(using t: ToInexact[T]): NDArray[t.Out] =
    val d = t.dtype
    given InexactDType[t.Out] = d
    val kk = (if k.isEmpty then Seq.fill(m)(0.0) else k).map(d.fromDouble).toArray(using d.classTag)
    arr(int(cast(atleast1d(p), d), m, kk, d))
