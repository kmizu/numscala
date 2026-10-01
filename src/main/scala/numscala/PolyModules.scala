// GENERATED (num-scala polynomial module generator) -- functional numpy.polynomial modules.
package numscala

/** `numpy.polynomial.polynomial`: functions on Polynomial series (coefficients lowest degree first). */
object PolyPowerModule:
  /** The `Polynomial` class. */
  val Polynomial: numscala.Polynomial.type = numscala.Polynomial
  /** The `Polynomial` class. */
  type Polynomial = numscala.Polynomial
  /** `numpy.polynomial.polynomial.polydomain`. */
  def polydomain: NDArray[Double] = NDArray.fromArray(Array(-1.0, 1.0))
  /** `numpy.polynomial.polynomial.polyzero`. */
  def polyzero: NDArray[Double] = NDArray.fromArray(Array(0.0))
  /** `numpy.polynomial.polynomial.polyone`. */
  def polyone: NDArray[Double] = NDArray.fromArray(Array(1.0))
  /** `numpy.polynomial.polynomial.polyx`. */
  def polyx: NDArray[Double] = NDArray.fromArray(Array(0.0, 1.0))

  /** `numpy.polynomial.polynomial.polyval`: evaluates the series `c` at `x` (multi-dimensional `c` holds several series along axis 0). */
  def polyval(x: NDArray[?], c: NDArray[?], tensor: Boolean = true): NDArray[Double] = PolyFunctional.valArr(PolyBasis.Power, x, c, tensor)
  /** `numpy.polynomial.polynomial.polyval` at a scalar point (1-D `c`). */
  def polyval(x: Double, c: NDArray[?]): Double = PolyFunctional.valScalar(PolyBasis.Power, x, c)
  /** `numpy.polynomial.polynomial.polyval2d`: evaluates a 2-D series at the points `(x, y)`. */
  def polyval2d(x: NDArray[?], y: NDArray[?], c: NDArray[?]): NDArray[Double] = PolyFunctional.valNd(PolyBasis.Power, c, x, y)
  /** `numpy.polynomial.polynomial.polyval3d`: evaluates a 3-D series at the points `(x, y, z)`. */
  def polyval3d(x: NDArray[?], y: NDArray[?], z: NDArray[?], c: NDArray[?]): NDArray[Double] =
    PolyFunctional.valNd(PolyBasis.Power, c, x, y, z)
  /** `numpy.polynomial.polynomial.polygrid2d`: evaluates a 2-D series on the Cartesian product of `x` and `y`. */
  def polygrid2d(x: NDArray[?], y: NDArray[?], c: NDArray[?]): NDArray[Double] = PolyFunctional.gridNd(PolyBasis.Power, c, x, y)
  /** `numpy.polynomial.polynomial.polygrid3d`: evaluates a 3-D series on the Cartesian product of `x`, `y` and `z`. */
  def polygrid3d(x: NDArray[?], y: NDArray[?], z: NDArray[?], c: NDArray[?]): NDArray[Double] =
    PolyFunctional.gridNd(PolyBasis.Power, c, x, y, z)
  /** `numpy.polynomial.polynomial.polyvander`: pseudo-Vandermonde matrix of degree `deg` (shape `x.shape :+ (deg+1)`). */
  def polyvander(x: NDArray[?], deg: Int): NDArray[Double] = PolyFunctional.vander(PolyBasis.Power, x, deg)
  /** `numpy.polynomial.polynomial.polyvander2d`. */
  def polyvander2d(x: NDArray[?], y: NDArray[?], deg: Seq[Int]): NDArray[Double] = PolyFunctional.vanderNd(PolyBasis.Power, Seq(x, y), deg)
  /** `numpy.polynomial.polynomial.polyvander3d`. */
  def polyvander3d(x: NDArray[?], y: NDArray[?], z: NDArray[?], deg: Seq[Int]): NDArray[Double] =
    PolyFunctional.vanderNd(PolyBasis.Power, Seq(x, y, z), deg)

  /** `numpy.polynomial.polynomial.polyder`: differentiates `m` times along `axis`, multiplying by `scl` each time. */
  def polyder(c: NDArray[?], m: Int = 1, scl: Double = 1.0, axis: Int = 0): NDArray[Double] =
    PolyFunctional.der(PolyBasis.Power, c, m, scl, axis)
  /** `numpy.polynomial.polynomial.polyint`: integrates `m` times along `axis` with constants `k` at `lbnd`, scaling by `scl`. */
  def polyint(c: NDArray[?], m: Int = 1, k: Seq[Double] = Nil, lbnd: Double = 0.0, scl: Double = 1.0, axis: Int = 0)
      : NDArray[Double] = PolyFunctional.integ(PolyBasis.Power, c, m, k, lbnd, scl, axis)
  /** `numpy.polynomial.polynomial.polyadd`. */
  def polyadd(c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = PolyFunctional.add(PolyBasis.Power, c1, c2)
  /** `numpy.polynomial.polynomial.polysub`. */
  def polysub(c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = PolyFunctional.sub(PolyBasis.Power, c1, c2)
  /** `numpy.polynomial.polynomial.polymul`. */
  def polymul(c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = PolyFunctional.mul(PolyBasis.Power, c1, c2)
  /** `numpy.polynomial.polynomial.polymulx`: multiplication by `x`. */
  def polymulx(c: NDArray[?]): NDArray[Double] = PolyFunctional.mulx(PolyBasis.Power, c)
  /** `numpy.polynomial.polynomial.polydiv`: `(quotient, remainder)`. */
  def polydiv(c1: NDArray[?], c2: NDArray[?]): (NDArray[Double], NDArray[Double]) = PolyFunctional.div(PolyBasis.Power, c1, c2)
  /** `numpy.polynomial.polynomial.polypow`. */
  def polypow(c: NDArray[?], pow: Int, maxpower: Int = 16): NDArray[Double] = PolyFunctional.pow(PolyBasis.Power, c, pow, maxpower)
  /** `numpy.polynomial.polynomial.polyroots`: roots of the series (complex, sorted). */
  def polyroots(c: NDArray[?]): NDArray[Complex] = PolyFunctional.roots(PolyBasis.Power, c)
  /** `numpy.polynomial.polynomial.polycompanion`: the (scaled) companion matrix. */
  def polycompanion(c: NDArray[?]): NDArray[Double] = PolyFunctional.companion(PolyBasis.Power, c)
  /** `numpy.polynomial.polynomial.polyfromroots`: the series with the given (real) roots. */
  def polyfromroots(roots: NDArray[?]): NDArray[Double] = PolyFunctional.fromroots(PolyBasis.Power, roots)
  /** `numpy.polynomial.polynomial.polyline`: the series `off + scl*x`. */
  def polyline(off: Double, scl: Double): NDArray[Double] = PolyFunctional.line(PolyBasis.Power, off, scl)
  /** `numpy.polynomial.polynomial.polytrim`: removes small trailing coefficients. */
  def polytrim(c: NDArray[?], tol: Double = 0.0): NDArray[Double] = PolyFunctional.trimcoef(c, tol)
  /** `numpy.polynomial.polynomial.polyfit` with `full=True`: coefficients and fit diagnostics. */
  def polyfitFull(x: NDArray[?], y: NDArray[?], deg: Int | Seq[Int], rcond: Double, w: NDArray[?] | Null)
      : (NDArray[Double], PolyFitInfo) = PolyFunctional.fit(PolyBasis.Power, x, y, deg, rcond, w)
  /** `numpy.polynomial.polynomial.polyfit`: least-squares fit (`rcond < 0`: NumPy default; `full = true` also returns [[PolyFitInfo]]). */
  transparent inline def polyfit(
      x: NDArray[?],
      y: NDArray[?],
      deg: Int | Seq[Int],
      rcond: Double = -1.0,
      inline full: Boolean = false,
      w: NDArray[?] | Null = null
  ) =
    inline if full then polyfitFull(x, y, deg, rcond, w) else polyfitFull(x, y, deg, rcond, w)._1
  /** `numpy.polynomial.polynomial.polyvalfromroots`: evaluates `prod(x - r)`. */
  def polyvalfromroots(x: NDArray[?], r: NDArray[?], tensor: Boolean = true): NDArray[Double] =
    PolyFunctional.valfromroots(x, r, tensor)

/** `numpy.polynomial.chebyshev`: functions on Chebyshev series (coefficients lowest degree first). */
object PolyChebModule:
  /** The `Chebyshev` class. */
  val Chebyshev: numscala.Chebyshev.type = numscala.Chebyshev
  /** The `Chebyshev` class. */
  type Chebyshev = numscala.Chebyshev
  /** `numpy.polynomial.chebyshev.chebdomain`. */
  def chebdomain: NDArray[Double] = NDArray.fromArray(Array(-1.0, 1.0))
  /** `numpy.polynomial.chebyshev.chebzero`. */
  def chebzero: NDArray[Double] = NDArray.fromArray(Array(0.0))
  /** `numpy.polynomial.chebyshev.chebone`. */
  def chebone: NDArray[Double] = NDArray.fromArray(Array(1.0))
  /** `numpy.polynomial.chebyshev.chebx`. */
  def chebx: NDArray[Double] = NDArray.fromArray(Array(0.0, 1.0))

  /** `numpy.polynomial.chebyshev.chebval`: evaluates the series `c` at `x` (multi-dimensional `c` holds several series along axis 0). */
  def chebval(x: NDArray[?], c: NDArray[?], tensor: Boolean = true): NDArray[Double] = PolyFunctional.valArr(PolyBasis.Cheb, x, c, tensor)
  /** `numpy.polynomial.chebyshev.chebval` at a scalar point (1-D `c`). */
  def chebval(x: Double, c: NDArray[?]): Double = PolyFunctional.valScalar(PolyBasis.Cheb, x, c)
  /** `numpy.polynomial.chebyshev.chebval2d`: evaluates a 2-D series at the points `(x, y)`. */
  def chebval2d(x: NDArray[?], y: NDArray[?], c: NDArray[?]): NDArray[Double] = PolyFunctional.valNd(PolyBasis.Cheb, c, x, y)
  /** `numpy.polynomial.chebyshev.chebval3d`: evaluates a 3-D series at the points `(x, y, z)`. */
  def chebval3d(x: NDArray[?], y: NDArray[?], z: NDArray[?], c: NDArray[?]): NDArray[Double] =
    PolyFunctional.valNd(PolyBasis.Cheb, c, x, y, z)
  /** `numpy.polynomial.chebyshev.chebgrid2d`: evaluates a 2-D series on the Cartesian product of `x` and `y`. */
  def chebgrid2d(x: NDArray[?], y: NDArray[?], c: NDArray[?]): NDArray[Double] = PolyFunctional.gridNd(PolyBasis.Cheb, c, x, y)
  /** `numpy.polynomial.chebyshev.chebgrid3d`: evaluates a 3-D series on the Cartesian product of `x`, `y` and `z`. */
  def chebgrid3d(x: NDArray[?], y: NDArray[?], z: NDArray[?], c: NDArray[?]): NDArray[Double] =
    PolyFunctional.gridNd(PolyBasis.Cheb, c, x, y, z)
  /** `numpy.polynomial.chebyshev.chebvander`: pseudo-Vandermonde matrix of degree `deg` (shape `x.shape :+ (deg+1)`). */
  def chebvander(x: NDArray[?], deg: Int): NDArray[Double] = PolyFunctional.vander(PolyBasis.Cheb, x, deg)
  /** `numpy.polynomial.chebyshev.chebvander2d`. */
  def chebvander2d(x: NDArray[?], y: NDArray[?], deg: Seq[Int]): NDArray[Double] = PolyFunctional.vanderNd(PolyBasis.Cheb, Seq(x, y), deg)
  /** `numpy.polynomial.chebyshev.chebvander3d`. */
  def chebvander3d(x: NDArray[?], y: NDArray[?], z: NDArray[?], deg: Seq[Int]): NDArray[Double] =
    PolyFunctional.vanderNd(PolyBasis.Cheb, Seq(x, y, z), deg)

  /** `numpy.polynomial.chebyshev.chebder`: differentiates `m` times along `axis`, multiplying by `scl` each time. */
  def chebder(c: NDArray[?], m: Int = 1, scl: Double = 1.0, axis: Int = 0): NDArray[Double] =
    PolyFunctional.der(PolyBasis.Cheb, c, m, scl, axis)
  /** `numpy.polynomial.chebyshev.chebint`: integrates `m` times along `axis` with constants `k` at `lbnd`, scaling by `scl`. */
  def chebint(c: NDArray[?], m: Int = 1, k: Seq[Double] = Nil, lbnd: Double = 0.0, scl: Double = 1.0, axis: Int = 0)
      : NDArray[Double] = PolyFunctional.integ(PolyBasis.Cheb, c, m, k, lbnd, scl, axis)
  /** `numpy.polynomial.chebyshev.chebadd`. */
  def chebadd(c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = PolyFunctional.add(PolyBasis.Cheb, c1, c2)
  /** `numpy.polynomial.chebyshev.chebsub`. */
  def chebsub(c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = PolyFunctional.sub(PolyBasis.Cheb, c1, c2)
  /** `numpy.polynomial.chebyshev.chebmul`. */
  def chebmul(c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = PolyFunctional.mul(PolyBasis.Cheb, c1, c2)
  /** `numpy.polynomial.chebyshev.chebmulx`: multiplication by `x`. */
  def chebmulx(c: NDArray[?]): NDArray[Double] = PolyFunctional.mulx(PolyBasis.Cheb, c)
  /** `numpy.polynomial.chebyshev.chebdiv`: `(quotient, remainder)`. */
  def chebdiv(c1: NDArray[?], c2: NDArray[?]): (NDArray[Double], NDArray[Double]) = PolyFunctional.div(PolyBasis.Cheb, c1, c2)
  /** `numpy.polynomial.chebyshev.chebpow`. */
  def chebpow(c: NDArray[?], pow: Int, maxpower: Int = 16): NDArray[Double] = PolyFunctional.pow(PolyBasis.Cheb, c, pow, maxpower)
  /** `numpy.polynomial.chebyshev.chebroots`: roots of the series (complex, sorted). */
  def chebroots(c: NDArray[?]): NDArray[Complex] = PolyFunctional.roots(PolyBasis.Cheb, c)
  /** `numpy.polynomial.chebyshev.chebcompanion`: the (scaled) companion matrix. */
  def chebcompanion(c: NDArray[?]): NDArray[Double] = PolyFunctional.companion(PolyBasis.Cheb, c)
  /** `numpy.polynomial.chebyshev.chebfromroots`: the series with the given (real) roots. */
  def chebfromroots(roots: NDArray[?]): NDArray[Double] = PolyFunctional.fromroots(PolyBasis.Cheb, roots)
  /** `numpy.polynomial.chebyshev.chebline`: the series `off + scl*x`. */
  def chebline(off: Double, scl: Double): NDArray[Double] = PolyFunctional.line(PolyBasis.Cheb, off, scl)
  /** `numpy.polynomial.chebyshev.chebtrim`: removes small trailing coefficients. */
  def chebtrim(c: NDArray[?], tol: Double = 0.0): NDArray[Double] = PolyFunctional.trimcoef(c, tol)
  /** `numpy.polynomial.chebyshev.chebfit` with `full=True`: coefficients and fit diagnostics. */
  def chebfitFull(x: NDArray[?], y: NDArray[?], deg: Int | Seq[Int], rcond: Double, w: NDArray[?] | Null)
      : (NDArray[Double], PolyFitInfo) = PolyFunctional.fit(PolyBasis.Cheb, x, y, deg, rcond, w)
  /** `numpy.polynomial.chebyshev.chebfit`: least-squares fit (`rcond < 0`: NumPy default; `full = true` also returns [[PolyFitInfo]]). */
  transparent inline def chebfit(
      x: NDArray[?],
      y: NDArray[?],
      deg: Int | Seq[Int],
      rcond: Double = -1.0,
      inline full: Boolean = false,
      w: NDArray[?] | Null = null
  ) =
    inline if full then chebfitFull(x, y, deg, rcond, w) else chebfitFull(x, y, deg, rcond, w)._1
  /** `numpy.polynomial.chebyshev.cheb2poly`: converts to power-series coefficients. */
  def cheb2poly(c: NDArray[?]): NDArray[Double] = PolyFunctional.toPower(PolyBasis.Cheb, c)
  /** `numpy.polynomial.chebyshev.poly2cheb`: converts power-series coefficients to this basis. */
  def poly2cheb(pol: NDArray[?]): NDArray[Double] = PolyFunctional.fromPower(PolyBasis.Cheb, pol)
  /** `numpy.polynomial.chebyshev.chebgauss`: Gauss quadrature nodes and weights. */
  def chebgauss(deg: Int): (NDArray[Double], NDArray[Double]) = PolyFunctional.gauss(PolyBasis.Cheb, deg)
  /** `numpy.polynomial.chebyshev.chebweight`: the weight function of the basis. */
  def chebweight(x: NDArray[?]): NDArray[Double] = PolyFunctional.weight(PolyBasis.Cheb, x)
  /** `numpy.polynomial.chebyshev.chebpts1`: Chebyshev points of the first kind. */
  def chebpts1(npts: Int): NDArray[Double] = PolyFunctional.chebpts1(npts)
  /** `numpy.polynomial.chebyshev.chebpts2`: Chebyshev points of the second kind. */
  def chebpts2(npts: Int): NDArray[Double] = PolyFunctional.chebpts2(npts)
  /** `numpy.polynomial.chebyshev.chebinterpolate`: interpolates `func` at the Chebyshev points of the first kind. */
  def chebinterpolate(func: Double => Double, deg: Int): NDArray[Double] = PolyFunctional.chebinterpolate(func, deg)

/** `numpy.polynomial.legendre`: functions on Legendre series (coefficients lowest degree first). */
object PolyLegModule:
  /** The `Legendre` class. */
  val Legendre: numscala.Legendre.type = numscala.Legendre
  /** The `Legendre` class. */
  type Legendre = numscala.Legendre
  /** `numpy.polynomial.legendre.legdomain`. */
  def legdomain: NDArray[Double] = NDArray.fromArray(Array(-1.0, 1.0))
  /** `numpy.polynomial.legendre.legzero`. */
  def legzero: NDArray[Double] = NDArray.fromArray(Array(0.0))
  /** `numpy.polynomial.legendre.legone`. */
  def legone: NDArray[Double] = NDArray.fromArray(Array(1.0))
  /** `numpy.polynomial.legendre.legx`. */
  def legx: NDArray[Double] = NDArray.fromArray(Array(0.0, 1.0))

  /** `numpy.polynomial.legendre.legval`: evaluates the series `c` at `x` (multi-dimensional `c` holds several series along axis 0). */
  def legval(x: NDArray[?], c: NDArray[?], tensor: Boolean = true): NDArray[Double] = PolyFunctional.valArr(PolyBasis.Leg, x, c, tensor)
  /** `numpy.polynomial.legendre.legval` at a scalar point (1-D `c`). */
  def legval(x: Double, c: NDArray[?]): Double = PolyFunctional.valScalar(PolyBasis.Leg, x, c)
  /** `numpy.polynomial.legendre.legval2d`: evaluates a 2-D series at the points `(x, y)`. */
  def legval2d(x: NDArray[?], y: NDArray[?], c: NDArray[?]): NDArray[Double] = PolyFunctional.valNd(PolyBasis.Leg, c, x, y)
  /** `numpy.polynomial.legendre.legval3d`: evaluates a 3-D series at the points `(x, y, z)`. */
  def legval3d(x: NDArray[?], y: NDArray[?], z: NDArray[?], c: NDArray[?]): NDArray[Double] =
    PolyFunctional.valNd(PolyBasis.Leg, c, x, y, z)
  /** `numpy.polynomial.legendre.leggrid2d`: evaluates a 2-D series on the Cartesian product of `x` and `y`. */
  def leggrid2d(x: NDArray[?], y: NDArray[?], c: NDArray[?]): NDArray[Double] = PolyFunctional.gridNd(PolyBasis.Leg, c, x, y)
  /** `numpy.polynomial.legendre.leggrid3d`: evaluates a 3-D series on the Cartesian product of `x`, `y` and `z`. */
  def leggrid3d(x: NDArray[?], y: NDArray[?], z: NDArray[?], c: NDArray[?]): NDArray[Double] =
    PolyFunctional.gridNd(PolyBasis.Leg, c, x, y, z)
  /** `numpy.polynomial.legendre.legvander`: pseudo-Vandermonde matrix of degree `deg` (shape `x.shape :+ (deg+1)`). */
  def legvander(x: NDArray[?], deg: Int): NDArray[Double] = PolyFunctional.vander(PolyBasis.Leg, x, deg)
  /** `numpy.polynomial.legendre.legvander2d`. */
  def legvander2d(x: NDArray[?], y: NDArray[?], deg: Seq[Int]): NDArray[Double] = PolyFunctional.vanderNd(PolyBasis.Leg, Seq(x, y), deg)
  /** `numpy.polynomial.legendre.legvander3d`. */
  def legvander3d(x: NDArray[?], y: NDArray[?], z: NDArray[?], deg: Seq[Int]): NDArray[Double] =
    PolyFunctional.vanderNd(PolyBasis.Leg, Seq(x, y, z), deg)

  /** `numpy.polynomial.legendre.legder`: differentiates `m` times along `axis`, multiplying by `scl` each time. */
  def legder(c: NDArray[?], m: Int = 1, scl: Double = 1.0, axis: Int = 0): NDArray[Double] =
    PolyFunctional.der(PolyBasis.Leg, c, m, scl, axis)
  /** `numpy.polynomial.legendre.legint`: integrates `m` times along `axis` with constants `k` at `lbnd`, scaling by `scl`. */
  def legint(c: NDArray[?], m: Int = 1, k: Seq[Double] = Nil, lbnd: Double = 0.0, scl: Double = 1.0, axis: Int = 0)
      : NDArray[Double] = PolyFunctional.integ(PolyBasis.Leg, c, m, k, lbnd, scl, axis)
  /** `numpy.polynomial.legendre.legadd`. */
  def legadd(c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = PolyFunctional.add(PolyBasis.Leg, c1, c2)
  /** `numpy.polynomial.legendre.legsub`. */
  def legsub(c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = PolyFunctional.sub(PolyBasis.Leg, c1, c2)
  /** `numpy.polynomial.legendre.legmul`. */
  def legmul(c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = PolyFunctional.mul(PolyBasis.Leg, c1, c2)
  /** `numpy.polynomial.legendre.legmulx`: multiplication by `x`. */
  def legmulx(c: NDArray[?]): NDArray[Double] = PolyFunctional.mulx(PolyBasis.Leg, c)
  /** `numpy.polynomial.legendre.legdiv`: `(quotient, remainder)`. */
  def legdiv(c1: NDArray[?], c2: NDArray[?]): (NDArray[Double], NDArray[Double]) = PolyFunctional.div(PolyBasis.Leg, c1, c2)
  /** `numpy.polynomial.legendre.legpow`. */
  def legpow(c: NDArray[?], pow: Int, maxpower: Int = 16): NDArray[Double] = PolyFunctional.pow(PolyBasis.Leg, c, pow, maxpower)
  /** `numpy.polynomial.legendre.legroots`: roots of the series (complex, sorted). */
  def legroots(c: NDArray[?]): NDArray[Complex] = PolyFunctional.roots(PolyBasis.Leg, c)
  /** `numpy.polynomial.legendre.legcompanion`: the (scaled) companion matrix. */
  def legcompanion(c: NDArray[?]): NDArray[Double] = PolyFunctional.companion(PolyBasis.Leg, c)
  /** `numpy.polynomial.legendre.legfromroots`: the series with the given (real) roots. */
  def legfromroots(roots: NDArray[?]): NDArray[Double] = PolyFunctional.fromroots(PolyBasis.Leg, roots)
  /** `numpy.polynomial.legendre.legline`: the series `off + scl*x`. */
  def legline(off: Double, scl: Double): NDArray[Double] = PolyFunctional.line(PolyBasis.Leg, off, scl)
  /** `numpy.polynomial.legendre.legtrim`: removes small trailing coefficients. */
  def legtrim(c: NDArray[?], tol: Double = 0.0): NDArray[Double] = PolyFunctional.trimcoef(c, tol)
  /** `numpy.polynomial.legendre.legfit` with `full=True`: coefficients and fit diagnostics. */
  def legfitFull(x: NDArray[?], y: NDArray[?], deg: Int | Seq[Int], rcond: Double, w: NDArray[?] | Null)
      : (NDArray[Double], PolyFitInfo) = PolyFunctional.fit(PolyBasis.Leg, x, y, deg, rcond, w)
  /** `numpy.polynomial.legendre.legfit`: least-squares fit (`rcond < 0`: NumPy default; `full = true` also returns [[PolyFitInfo]]). */
  transparent inline def legfit(
      x: NDArray[?],
      y: NDArray[?],
      deg: Int | Seq[Int],
      rcond: Double = -1.0,
      inline full: Boolean = false,
      w: NDArray[?] | Null = null
  ) =
    inline if full then legfitFull(x, y, deg, rcond, w) else legfitFull(x, y, deg, rcond, w)._1
  /** `numpy.polynomial.legendre.leg2poly`: converts to power-series coefficients. */
  def leg2poly(c: NDArray[?]): NDArray[Double] = PolyFunctional.toPower(PolyBasis.Leg, c)
  /** `numpy.polynomial.legendre.poly2leg`: converts power-series coefficients to this basis. */
  def poly2leg(pol: NDArray[?]): NDArray[Double] = PolyFunctional.fromPower(PolyBasis.Leg, pol)
  /** `numpy.polynomial.legendre.leggauss`: Gauss quadrature nodes and weights. */
  def leggauss(deg: Int): (NDArray[Double], NDArray[Double]) = PolyFunctional.gauss(PolyBasis.Leg, deg)
  /** `numpy.polynomial.legendre.legweight`: the weight function of the basis. */
  def legweight(x: NDArray[?]): NDArray[Double] = PolyFunctional.weight(PolyBasis.Leg, x)

/** `numpy.polynomial.hermite`: functions on Hermite series (coefficients lowest degree first). */
object PolyHermModule:
  /** The `Hermite` class. */
  val Hermite: numscala.Hermite.type = numscala.Hermite
  /** The `Hermite` class. */
  type Hermite = numscala.Hermite
  /** `numpy.polynomial.hermite.hermdomain`. */
  def hermdomain: NDArray[Double] = NDArray.fromArray(Array(-1.0, 1.0))
  /** `numpy.polynomial.hermite.hermzero`. */
  def hermzero: NDArray[Double] = NDArray.fromArray(Array(0.0))
  /** `numpy.polynomial.hermite.hermone`. */
  def hermone: NDArray[Double] = NDArray.fromArray(Array(1.0))
  /** `numpy.polynomial.hermite.hermx`. */
  def hermx: NDArray[Double] = NDArray.fromArray(Array(0.0, 0.5))

  /** `numpy.polynomial.hermite.hermval`: evaluates the series `c` at `x` (multi-dimensional `c` holds several series along axis 0). */
  def hermval(x: NDArray[?], c: NDArray[?], tensor: Boolean = true): NDArray[Double] = PolyFunctional.valArr(PolyBasis.Herm, x, c, tensor)
  /** `numpy.polynomial.hermite.hermval` at a scalar point (1-D `c`). */
  def hermval(x: Double, c: NDArray[?]): Double = PolyFunctional.valScalar(PolyBasis.Herm, x, c)
  /** `numpy.polynomial.hermite.hermval2d`: evaluates a 2-D series at the points `(x, y)`. */
  def hermval2d(x: NDArray[?], y: NDArray[?], c: NDArray[?]): NDArray[Double] = PolyFunctional.valNd(PolyBasis.Herm, c, x, y)
  /** `numpy.polynomial.hermite.hermval3d`: evaluates a 3-D series at the points `(x, y, z)`. */
  def hermval3d(x: NDArray[?], y: NDArray[?], z: NDArray[?], c: NDArray[?]): NDArray[Double] =
    PolyFunctional.valNd(PolyBasis.Herm, c, x, y, z)
  /** `numpy.polynomial.hermite.hermgrid2d`: evaluates a 2-D series on the Cartesian product of `x` and `y`. */
  def hermgrid2d(x: NDArray[?], y: NDArray[?], c: NDArray[?]): NDArray[Double] = PolyFunctional.gridNd(PolyBasis.Herm, c, x, y)
  /** `numpy.polynomial.hermite.hermgrid3d`: evaluates a 3-D series on the Cartesian product of `x`, `y` and `z`. */
  def hermgrid3d(x: NDArray[?], y: NDArray[?], z: NDArray[?], c: NDArray[?]): NDArray[Double] =
    PolyFunctional.gridNd(PolyBasis.Herm, c, x, y, z)
  /** `numpy.polynomial.hermite.hermvander`: pseudo-Vandermonde matrix of degree `deg` (shape `x.shape :+ (deg+1)`). */
  def hermvander(x: NDArray[?], deg: Int): NDArray[Double] = PolyFunctional.vander(PolyBasis.Herm, x, deg)
  /** `numpy.polynomial.hermite.hermvander2d`. */
  def hermvander2d(x: NDArray[?], y: NDArray[?], deg: Seq[Int]): NDArray[Double] = PolyFunctional.vanderNd(PolyBasis.Herm, Seq(x, y), deg)
  /** `numpy.polynomial.hermite.hermvander3d`. */
  def hermvander3d(x: NDArray[?], y: NDArray[?], z: NDArray[?], deg: Seq[Int]): NDArray[Double] =
    PolyFunctional.vanderNd(PolyBasis.Herm, Seq(x, y, z), deg)

  /** `numpy.polynomial.hermite.hermder`: differentiates `m` times along `axis`, multiplying by `scl` each time. */
  def hermder(c: NDArray[?], m: Int = 1, scl: Double = 1.0, axis: Int = 0): NDArray[Double] =
    PolyFunctional.der(PolyBasis.Herm, c, m, scl, axis)
  /** `numpy.polynomial.hermite.hermint`: integrates `m` times along `axis` with constants `k` at `lbnd`, scaling by `scl`. */
  def hermint(c: NDArray[?], m: Int = 1, k: Seq[Double] = Nil, lbnd: Double = 0.0, scl: Double = 1.0, axis: Int = 0)
      : NDArray[Double] = PolyFunctional.integ(PolyBasis.Herm, c, m, k, lbnd, scl, axis)
  /** `numpy.polynomial.hermite.hermadd`. */
  def hermadd(c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = PolyFunctional.add(PolyBasis.Herm, c1, c2)
  /** `numpy.polynomial.hermite.hermsub`. */
  def hermsub(c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = PolyFunctional.sub(PolyBasis.Herm, c1, c2)
  /** `numpy.polynomial.hermite.hermmul`. */
  def hermmul(c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = PolyFunctional.mul(PolyBasis.Herm, c1, c2)
  /** `numpy.polynomial.hermite.hermmulx`: multiplication by `x`. */
  def hermmulx(c: NDArray[?]): NDArray[Double] = PolyFunctional.mulx(PolyBasis.Herm, c)
  /** `numpy.polynomial.hermite.hermdiv`: `(quotient, remainder)`. */
  def hermdiv(c1: NDArray[?], c2: NDArray[?]): (NDArray[Double], NDArray[Double]) = PolyFunctional.div(PolyBasis.Herm, c1, c2)
  /** `numpy.polynomial.hermite.hermpow`. */
  def hermpow(c: NDArray[?], pow: Int, maxpower: Int = 16): NDArray[Double] = PolyFunctional.pow(PolyBasis.Herm, c, pow, maxpower)
  /** `numpy.polynomial.hermite.hermroots`: roots of the series (complex, sorted). */
  def hermroots(c: NDArray[?]): NDArray[Complex] = PolyFunctional.roots(PolyBasis.Herm, c)
  /** `numpy.polynomial.hermite.hermcompanion`: the (scaled) companion matrix. */
  def hermcompanion(c: NDArray[?]): NDArray[Double] = PolyFunctional.companion(PolyBasis.Herm, c)
  /** `numpy.polynomial.hermite.hermfromroots`: the series with the given (real) roots. */
  def hermfromroots(roots: NDArray[?]): NDArray[Double] = PolyFunctional.fromroots(PolyBasis.Herm, roots)
  /** `numpy.polynomial.hermite.hermline`: the series `off + scl*x`. */
  def hermline(off: Double, scl: Double): NDArray[Double] = PolyFunctional.line(PolyBasis.Herm, off, scl)
  /** `numpy.polynomial.hermite.hermtrim`: removes small trailing coefficients. */
  def hermtrim(c: NDArray[?], tol: Double = 0.0): NDArray[Double] = PolyFunctional.trimcoef(c, tol)
  /** `numpy.polynomial.hermite.hermfit` with `full=True`: coefficients and fit diagnostics. */
  def hermfitFull(x: NDArray[?], y: NDArray[?], deg: Int | Seq[Int], rcond: Double, w: NDArray[?] | Null)
      : (NDArray[Double], PolyFitInfo) = PolyFunctional.fit(PolyBasis.Herm, x, y, deg, rcond, w)
  /** `numpy.polynomial.hermite.hermfit`: least-squares fit (`rcond < 0`: NumPy default; `full = true` also returns [[PolyFitInfo]]). */
  transparent inline def hermfit(
      x: NDArray[?],
      y: NDArray[?],
      deg: Int | Seq[Int],
      rcond: Double = -1.0,
      inline full: Boolean = false,
      w: NDArray[?] | Null = null
  ) =
    inline if full then hermfitFull(x, y, deg, rcond, w) else hermfitFull(x, y, deg, rcond, w)._1
  /** `numpy.polynomial.hermite.herm2poly`: converts to power-series coefficients. */
  def herm2poly(c: NDArray[?]): NDArray[Double] = PolyFunctional.toPower(PolyBasis.Herm, c)
  /** `numpy.polynomial.hermite.poly2herm`: converts power-series coefficients to this basis. */
  def poly2herm(pol: NDArray[?]): NDArray[Double] = PolyFunctional.fromPower(PolyBasis.Herm, pol)
  /** `numpy.polynomial.hermite.hermgauss`: Gauss quadrature nodes and weights. */
  def hermgauss(deg: Int): (NDArray[Double], NDArray[Double]) = PolyFunctional.gauss(PolyBasis.Herm, deg)
  /** `numpy.polynomial.hermite.hermweight`: the weight function of the basis. */
  def hermweight(x: NDArray[?]): NDArray[Double] = PolyFunctional.weight(PolyBasis.Herm, x)

/** `numpy.polynomial.hermite_e`: functions on HermiteE series (coefficients lowest degree first). */
object PolyHermEModule:
  /** The `HermiteE` class. */
  val HermiteE: numscala.HermiteE.type = numscala.HermiteE
  /** The `HermiteE` class. */
  type HermiteE = numscala.HermiteE
  /** `numpy.polynomial.hermite_e.hermedomain`. */
  def hermedomain: NDArray[Double] = NDArray.fromArray(Array(-1.0, 1.0))
  /** `numpy.polynomial.hermite_e.hermezero`. */
  def hermezero: NDArray[Double] = NDArray.fromArray(Array(0.0))
  /** `numpy.polynomial.hermite_e.hermeone`. */
  def hermeone: NDArray[Double] = NDArray.fromArray(Array(1.0))
  /** `numpy.polynomial.hermite_e.hermex`. */
  def hermex: NDArray[Double] = NDArray.fromArray(Array(0.0, 1.0))

  /** `numpy.polynomial.hermite_e.hermeval`: evaluates the series `c` at `x` (multi-dimensional `c` holds several series along axis 0). */
  def hermeval(x: NDArray[?], c: NDArray[?], tensor: Boolean = true): NDArray[Double] = PolyFunctional.valArr(PolyBasis.HermE, x, c, tensor)
  /** `numpy.polynomial.hermite_e.hermeval` at a scalar point (1-D `c`). */
  def hermeval(x: Double, c: NDArray[?]): Double = PolyFunctional.valScalar(PolyBasis.HermE, x, c)
  /** `numpy.polynomial.hermite_e.hermeval2d`: evaluates a 2-D series at the points `(x, y)`. */
  def hermeval2d(x: NDArray[?], y: NDArray[?], c: NDArray[?]): NDArray[Double] = PolyFunctional.valNd(PolyBasis.HermE, c, x, y)
  /** `numpy.polynomial.hermite_e.hermeval3d`: evaluates a 3-D series at the points `(x, y, z)`. */
  def hermeval3d(x: NDArray[?], y: NDArray[?], z: NDArray[?], c: NDArray[?]): NDArray[Double] =
    PolyFunctional.valNd(PolyBasis.HermE, c, x, y, z)
  /** `numpy.polynomial.hermite_e.hermegrid2d`: evaluates a 2-D series on the Cartesian product of `x` and `y`. */
  def hermegrid2d(x: NDArray[?], y: NDArray[?], c: NDArray[?]): NDArray[Double] = PolyFunctional.gridNd(PolyBasis.HermE, c, x, y)
  /** `numpy.polynomial.hermite_e.hermegrid3d`: evaluates a 3-D series on the Cartesian product of `x`, `y` and `z`. */
  def hermegrid3d(x: NDArray[?], y: NDArray[?], z: NDArray[?], c: NDArray[?]): NDArray[Double] =
    PolyFunctional.gridNd(PolyBasis.HermE, c, x, y, z)
  /** `numpy.polynomial.hermite_e.hermevander`: pseudo-Vandermonde matrix of degree `deg` (shape `x.shape :+ (deg+1)`). */
  def hermevander(x: NDArray[?], deg: Int): NDArray[Double] = PolyFunctional.vander(PolyBasis.HermE, x, deg)
  /** `numpy.polynomial.hermite_e.hermevander2d`. */
  def hermevander2d(x: NDArray[?], y: NDArray[?], deg: Seq[Int]): NDArray[Double] = PolyFunctional.vanderNd(PolyBasis.HermE, Seq(x, y), deg)
  /** `numpy.polynomial.hermite_e.hermevander3d`. */
  def hermevander3d(x: NDArray[?], y: NDArray[?], z: NDArray[?], deg: Seq[Int]): NDArray[Double] =
    PolyFunctional.vanderNd(PolyBasis.HermE, Seq(x, y, z), deg)

  /** `numpy.polynomial.hermite_e.hermeder`: differentiates `m` times along `axis`, multiplying by `scl` each time. */
  def hermeder(c: NDArray[?], m: Int = 1, scl: Double = 1.0, axis: Int = 0): NDArray[Double] =
    PolyFunctional.der(PolyBasis.HermE, c, m, scl, axis)
  /** `numpy.polynomial.hermite_e.hermeint`: integrates `m` times along `axis` with constants `k` at `lbnd`, scaling by `scl`. */
  def hermeint(c: NDArray[?], m: Int = 1, k: Seq[Double] = Nil, lbnd: Double = 0.0, scl: Double = 1.0, axis: Int = 0)
      : NDArray[Double] = PolyFunctional.integ(PolyBasis.HermE, c, m, k, lbnd, scl, axis)
  /** `numpy.polynomial.hermite_e.hermeadd`. */
  def hermeadd(c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = PolyFunctional.add(PolyBasis.HermE, c1, c2)
  /** `numpy.polynomial.hermite_e.hermesub`. */
  def hermesub(c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = PolyFunctional.sub(PolyBasis.HermE, c1, c2)
  /** `numpy.polynomial.hermite_e.hermemul`. */
  def hermemul(c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = PolyFunctional.mul(PolyBasis.HermE, c1, c2)
  /** `numpy.polynomial.hermite_e.hermemulx`: multiplication by `x`. */
  def hermemulx(c: NDArray[?]): NDArray[Double] = PolyFunctional.mulx(PolyBasis.HermE, c)
  /** `numpy.polynomial.hermite_e.hermediv`: `(quotient, remainder)`. */
  def hermediv(c1: NDArray[?], c2: NDArray[?]): (NDArray[Double], NDArray[Double]) = PolyFunctional.div(PolyBasis.HermE, c1, c2)
  /** `numpy.polynomial.hermite_e.hermepow`. */
  def hermepow(c: NDArray[?], pow: Int, maxpower: Int = 16): NDArray[Double] = PolyFunctional.pow(PolyBasis.HermE, c, pow, maxpower)
  /** `numpy.polynomial.hermite_e.hermeroots`: roots of the series (complex, sorted). */
  def hermeroots(c: NDArray[?]): NDArray[Complex] = PolyFunctional.roots(PolyBasis.HermE, c)
  /** `numpy.polynomial.hermite_e.hermecompanion`: the (scaled) companion matrix. */
  def hermecompanion(c: NDArray[?]): NDArray[Double] = PolyFunctional.companion(PolyBasis.HermE, c)
  /** `numpy.polynomial.hermite_e.hermefromroots`: the series with the given (real) roots. */
  def hermefromroots(roots: NDArray[?]): NDArray[Double] = PolyFunctional.fromroots(PolyBasis.HermE, roots)
  /** `numpy.polynomial.hermite_e.hermeline`: the series `off + scl*x`. */
  def hermeline(off: Double, scl: Double): NDArray[Double] = PolyFunctional.line(PolyBasis.HermE, off, scl)
  /** `numpy.polynomial.hermite_e.hermetrim`: removes small trailing coefficients. */
  def hermetrim(c: NDArray[?], tol: Double = 0.0): NDArray[Double] = PolyFunctional.trimcoef(c, tol)
  /** `numpy.polynomial.hermite_e.hermefit` with `full=True`: coefficients and fit diagnostics. */
  def hermefitFull(x: NDArray[?], y: NDArray[?], deg: Int | Seq[Int], rcond: Double, w: NDArray[?] | Null)
      : (NDArray[Double], PolyFitInfo) = PolyFunctional.fit(PolyBasis.HermE, x, y, deg, rcond, w)
  /** `numpy.polynomial.hermite_e.hermefit`: least-squares fit (`rcond < 0`: NumPy default; `full = true` also returns [[PolyFitInfo]]). */
  transparent inline def hermefit(
      x: NDArray[?],
      y: NDArray[?],
      deg: Int | Seq[Int],
      rcond: Double = -1.0,
      inline full: Boolean = false,
      w: NDArray[?] | Null = null
  ) =
    inline if full then hermefitFull(x, y, deg, rcond, w) else hermefitFull(x, y, deg, rcond, w)._1
  /** `numpy.polynomial.hermite_e.herme2poly`: converts to power-series coefficients. */
  def herme2poly(c: NDArray[?]): NDArray[Double] = PolyFunctional.toPower(PolyBasis.HermE, c)
  /** `numpy.polynomial.hermite_e.poly2herme`: converts power-series coefficients to this basis. */
  def poly2herme(pol: NDArray[?]): NDArray[Double] = PolyFunctional.fromPower(PolyBasis.HermE, pol)
  /** `numpy.polynomial.hermite_e.hermegauss`: Gauss quadrature nodes and weights. */
  def hermegauss(deg: Int): (NDArray[Double], NDArray[Double]) = PolyFunctional.gauss(PolyBasis.HermE, deg)
  /** `numpy.polynomial.hermite_e.hermeweight`: the weight function of the basis. */
  def hermeweight(x: NDArray[?]): NDArray[Double] = PolyFunctional.weight(PolyBasis.HermE, x)

/** `numpy.polynomial.laguerre`: functions on Laguerre series (coefficients lowest degree first). */
object PolyLagModule:
  /** The `Laguerre` class. */
  val Laguerre: numscala.Laguerre.type = numscala.Laguerre
  /** The `Laguerre` class. */
  type Laguerre = numscala.Laguerre
  /** `numpy.polynomial.laguerre.lagdomain`. */
  def lagdomain: NDArray[Double] = NDArray.fromArray(Array(0.0, 1.0))
  /** `numpy.polynomial.laguerre.lagzero`. */
  def lagzero: NDArray[Double] = NDArray.fromArray(Array(0.0))
  /** `numpy.polynomial.laguerre.lagone`. */
  def lagone: NDArray[Double] = NDArray.fromArray(Array(1.0))
  /** `numpy.polynomial.laguerre.lagx`. */
  def lagx: NDArray[Double] = NDArray.fromArray(Array(1.0, -1.0))

  /** `numpy.polynomial.laguerre.lagval`: evaluates the series `c` at `x` (multi-dimensional `c` holds several series along axis 0). */
  def lagval(x: NDArray[?], c: NDArray[?], tensor: Boolean = true): NDArray[Double] = PolyFunctional.valArr(PolyBasis.Lag, x, c, tensor)
  /** `numpy.polynomial.laguerre.lagval` at a scalar point (1-D `c`). */
  def lagval(x: Double, c: NDArray[?]): Double = PolyFunctional.valScalar(PolyBasis.Lag, x, c)
  /** `numpy.polynomial.laguerre.lagval2d`: evaluates a 2-D series at the points `(x, y)`. */
  def lagval2d(x: NDArray[?], y: NDArray[?], c: NDArray[?]): NDArray[Double] = PolyFunctional.valNd(PolyBasis.Lag, c, x, y)
  /** `numpy.polynomial.laguerre.lagval3d`: evaluates a 3-D series at the points `(x, y, z)`. */
  def lagval3d(x: NDArray[?], y: NDArray[?], z: NDArray[?], c: NDArray[?]): NDArray[Double] =
    PolyFunctional.valNd(PolyBasis.Lag, c, x, y, z)
  /** `numpy.polynomial.laguerre.laggrid2d`: evaluates a 2-D series on the Cartesian product of `x` and `y`. */
  def laggrid2d(x: NDArray[?], y: NDArray[?], c: NDArray[?]): NDArray[Double] = PolyFunctional.gridNd(PolyBasis.Lag, c, x, y)
  /** `numpy.polynomial.laguerre.laggrid3d`: evaluates a 3-D series on the Cartesian product of `x`, `y` and `z`. */
  def laggrid3d(x: NDArray[?], y: NDArray[?], z: NDArray[?], c: NDArray[?]): NDArray[Double] =
    PolyFunctional.gridNd(PolyBasis.Lag, c, x, y, z)
  /** `numpy.polynomial.laguerre.lagvander`: pseudo-Vandermonde matrix of degree `deg` (shape `x.shape :+ (deg+1)`). */
  def lagvander(x: NDArray[?], deg: Int): NDArray[Double] = PolyFunctional.vander(PolyBasis.Lag, x, deg)
  /** `numpy.polynomial.laguerre.lagvander2d`. */
  def lagvander2d(x: NDArray[?], y: NDArray[?], deg: Seq[Int]): NDArray[Double] = PolyFunctional.vanderNd(PolyBasis.Lag, Seq(x, y), deg)
  /** `numpy.polynomial.laguerre.lagvander3d`. */
  def lagvander3d(x: NDArray[?], y: NDArray[?], z: NDArray[?], deg: Seq[Int]): NDArray[Double] =
    PolyFunctional.vanderNd(PolyBasis.Lag, Seq(x, y, z), deg)

  /** `numpy.polynomial.laguerre.lagder`: differentiates `m` times along `axis`, multiplying by `scl` each time. */
  def lagder(c: NDArray[?], m: Int = 1, scl: Double = 1.0, axis: Int = 0): NDArray[Double] =
    PolyFunctional.der(PolyBasis.Lag, c, m, scl, axis)
  /** `numpy.polynomial.laguerre.lagint`: integrates `m` times along `axis` with constants `k` at `lbnd`, scaling by `scl`. */
  def lagint(c: NDArray[?], m: Int = 1, k: Seq[Double] = Nil, lbnd: Double = 0.0, scl: Double = 1.0, axis: Int = 0)
      : NDArray[Double] = PolyFunctional.integ(PolyBasis.Lag, c, m, k, lbnd, scl, axis)
  /** `numpy.polynomial.laguerre.lagadd`. */
  def lagadd(c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = PolyFunctional.add(PolyBasis.Lag, c1, c2)
  /** `numpy.polynomial.laguerre.lagsub`. */
  def lagsub(c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = PolyFunctional.sub(PolyBasis.Lag, c1, c2)
  /** `numpy.polynomial.laguerre.lagmul`. */
  def lagmul(c1: NDArray[?], c2: NDArray[?]): NDArray[Double] = PolyFunctional.mul(PolyBasis.Lag, c1, c2)
  /** `numpy.polynomial.laguerre.lagmulx`: multiplication by `x`. */
  def lagmulx(c: NDArray[?]): NDArray[Double] = PolyFunctional.mulx(PolyBasis.Lag, c)
  /** `numpy.polynomial.laguerre.lagdiv`: `(quotient, remainder)`. */
  def lagdiv(c1: NDArray[?], c2: NDArray[?]): (NDArray[Double], NDArray[Double]) = PolyFunctional.div(PolyBasis.Lag, c1, c2)
  /** `numpy.polynomial.laguerre.lagpow`. */
  def lagpow(c: NDArray[?], pow: Int, maxpower: Int = 16): NDArray[Double] = PolyFunctional.pow(PolyBasis.Lag, c, pow, maxpower)
  /** `numpy.polynomial.laguerre.lagroots`: roots of the series (complex, sorted). */
  def lagroots(c: NDArray[?]): NDArray[Complex] = PolyFunctional.roots(PolyBasis.Lag, c)
  /** `numpy.polynomial.laguerre.lagcompanion`: the (scaled) companion matrix. */
  def lagcompanion(c: NDArray[?]): NDArray[Double] = PolyFunctional.companion(PolyBasis.Lag, c)
  /** `numpy.polynomial.laguerre.lagfromroots`: the series with the given (real) roots. */
  def lagfromroots(roots: NDArray[?]): NDArray[Double] = PolyFunctional.fromroots(PolyBasis.Lag, roots)
  /** `numpy.polynomial.laguerre.lagline`: the series `off + scl*x`. */
  def lagline(off: Double, scl: Double): NDArray[Double] = PolyFunctional.line(PolyBasis.Lag, off, scl)
  /** `numpy.polynomial.laguerre.lagtrim`: removes small trailing coefficients. */
  def lagtrim(c: NDArray[?], tol: Double = 0.0): NDArray[Double] = PolyFunctional.trimcoef(c, tol)
  /** `numpy.polynomial.laguerre.lagfit` with `full=True`: coefficients and fit diagnostics. */
  def lagfitFull(x: NDArray[?], y: NDArray[?], deg: Int | Seq[Int], rcond: Double, w: NDArray[?] | Null)
      : (NDArray[Double], PolyFitInfo) = PolyFunctional.fit(PolyBasis.Lag, x, y, deg, rcond, w)
  /** `numpy.polynomial.laguerre.lagfit`: least-squares fit (`rcond < 0`: NumPy default; `full = true` also returns [[PolyFitInfo]]). */
  transparent inline def lagfit(
      x: NDArray[?],
      y: NDArray[?],
      deg: Int | Seq[Int],
      rcond: Double = -1.0,
      inline full: Boolean = false,
      w: NDArray[?] | Null = null
  ) =
    inline if full then lagfitFull(x, y, deg, rcond, w) else lagfitFull(x, y, deg, rcond, w)._1
  /** `numpy.polynomial.laguerre.lag2poly`: converts to power-series coefficients. */
  def lag2poly(c: NDArray[?]): NDArray[Double] = PolyFunctional.toPower(PolyBasis.Lag, c)
  /** `numpy.polynomial.laguerre.poly2lag`: converts power-series coefficients to this basis. */
  def poly2lag(pol: NDArray[?]): NDArray[Double] = PolyFunctional.fromPower(PolyBasis.Lag, pol)
  /** `numpy.polynomial.laguerre.laggauss`: Gauss quadrature nodes and weights. */
  def laggauss(deg: Int): (NDArray[Double], NDArray[Double]) = PolyFunctional.gauss(PolyBasis.Lag, deg)
  /** `numpy.polynomial.laguerre.lagweight`: the weight function of the basis. */
  def lagweight(x: NDArray[?]): NDArray[Double] = PolyFunctional.weight(PolyBasis.Lag, x)
