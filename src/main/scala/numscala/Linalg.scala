package numscala

import LinalgSupport as S

/** Error raised by linear algebra routines (`numpy.linalg.LinAlgError`): singular matrices,
  * matrices that are not positive definite, non-convergence, wrong dimensions.
  */
class LinAlgError(message: String) extends RuntimeException(message)

/** Result dtype of `np.linalg` routines: `float64` for bool / integer / float input
  * (integers are converted like NumPy does) and `complex128` for complex input.
  */
trait LinalgOut[T]:
  type Out
  def dtype: InexactDType[Out]

object LinalgOut:
  type Aux[T, O] = LinalgOut[T] { type Out = O }
  private def make[T, O](d: InexactDType[O]): Aux[T, O] = new LinalgOut[T]:
    type Out = O
    val dtype: InexactDType[O] = d
  given realOut[T](using ev: RealDType[T]): Aux[T, Double] = make(DType.Float64)
  given boolOut: Aux[Boolean, Double] = make(DType.Float64)
  given complexOut: Aux[Complex, Complex] = make(DType.Complex128)

/** Result of `np.linalg.eig`. Eigenvalues and eigenvectors are always complex128 (NumPy returns
  * float64 arrays for real input whose eigenvalues are all real; here the imaginary parts are then
  * exactly zero). `eigenvectors(:, i)` belongs to `eigenvalues(i)`.
  */
final case class EigResult(eigenvalues: NDArray[Complex], eigenvectors: NDArray[Complex])

/** Result of `np.linalg.eigh`: ascending real eigenvalues and orthonormal eigenvectors (columns). */
final case class EighResult[T](eigenvalues: NDArray[Double], eigenvectors: NDArray[T])

/** Result of `np.linalg.svd`: `a == U @ diag(S) @ Vh`. With `compute_uv = false`, `U` and `Vh` are empty. */
final case class SVDResult[T](U: NDArray[T], S: NDArray[Double], Vh: NDArray[T])

/** Result of `np.linalg.qr`. For `mode = "r"`, `Q` is empty; for `mode = "raw"`, `Q` holds `h`
  * (shape `(..., N, M)`) and `R` holds `tau`.
  */
final case class QRResult[T](Q: NDArray[T], R: NDArray[T])

/** Result of `np.linalg.slogdet`: `det == sign * exp(logabsdet)`. */
final case class SlogdetResult[T](sign: NDArray[T], logabsdet: NDArray[Double])

/** Result of `np.linalg.lstsq`. */
final case class LstsqResult[T](x: NDArray[T], residuals: NDArray[Double], rank: Int, s: NDArray[Double])

/** `numpy.linalg`. Routines accept stacks of matrices (`ndim > 2`) wherever NumPy does. Matrix-valued
  * scalars (`det`, `cond`, `matrix_rank`, ...) are returned as arrays (0-d for a single matrix; use `.item`).
  */
object Linalg:
  /** Alias of [[numscala.LinAlgError]] (`np.linalg.LinAlgError`). */
  type LinAlgError = numscala.LinAlgError

  /** A norm order: a number (`1`, `2`, `Double.PositiveInfinity`, ...), `"fro"`, `"nuc"`, or `None`. */
  type NormOrd = Double | Int | String | None.type

  private val F64 = DType.Float64
  private val C128 = DType.Complex128

  private def isC(a: NDArray[?]): Boolean = a.dtype.isComplex

  // ------------------------------------------------------------------ determinants and inverses

  /** Determinant of (a stack of) square matrices (`np.linalg.det`). */
  def det[T](a: NDArray[T])(using l: LinalgOut[T]): NDArray[l.Out] =
    S.assertStackedSquare(a)
    val r: NDArray[?] =
      if isC(a) then
        S.mapZ(a, Array.emptyIntArray) { z =>
          val f = LinalgCplx.lu(z)
          var d = Complex(f.sign.toDouble, 0.0)
          var i = 0
          while i < z.n do
            d = d * Complex(z.re(i * z.n + i), z.im(i * z.n + i))
            i += 1
          (Array(d.re), Array(d.im))
        }
      else S.mapD(a, Array.emptyIntArray)((x, _, n) => Array(LinalgReal.det(x, n)))
    r.asInstanceOf[NDArray[l.Out]]

  /** Sign and natural log of the absolute determinant (`np.linalg.slogdet`). */
  def slogdet[T](a: NDArray[T])(using l: LinalgOut[T]): SlogdetResult[l.Out] =
    S.assertStackedSquare(a)
    val bs = S.batchShape(a)
    val nb = Shape.size(bs)
    val logs = new Array[Double](nb)
    if isC(a) then
      val sr = new Array[Double](nb)
      val si = new Array[Double](nb)
      S.foreachZ(a) { (b, z) =>
        val f = LinalgCplx.lu(z)
        if f.singular then logs(b) = Double.NegativeInfinity
        else
          var s = Complex(f.sign.toDouble, 0.0)
          var lg = 0.0
          var i = 0
          while i < z.n do
            val d = Complex(z.re(i * z.n + i), z.im(i * z.n + i))
            val ad = d.abs
            s = s * (d / ad)
            lg += math.log(ad)
            i += 1
          sr(b) = s.re
          si(b) = s.im
          logs(b) = lg
      }
      SlogdetResult(S.wrapC(sr, si, bs).asInstanceOf[NDArray[l.Out]], S.wrapD(logs, bs))
    else
      val sg = new Array[Double](nb)
      S.foreachD(a) { (b, x) =>
        val n = a.shapeArr(a.ndim - 1)
        val f = LinalgReal.lu(x, n)
        if f.singular then logs(b) = Double.NegativeInfinity
        else
          var s = f.sign.toDouble
          var lg = 0.0
          var i = 0
          while i < n do
            val d = x(i * n + i)
            if d < 0 then s = -s
            lg += math.log(math.abs(d))
            i += 1
          sg(b) = s
          logs(b) = lg
      }
      SlogdetResult(S.wrapD(sg, bs).asInstanceOf[NDArray[l.Out]], S.wrapD(logs, bs))

  /** Inverse of (a stack of) square matrices (`np.linalg.inv`); throws [[LinAlgError]] if singular. */
  def inv[T](a: NDArray[T])(using l: LinalgOut[T]): NDArray[l.Out] =
    S.assertStackedSquare(a)
    val n = a.shapeArr(a.ndim - 1)
    val r: NDArray[?] =
      if isC(a) then
        S.mapZ(a, Array(n, n)) { z =>
          val f = LinalgCplx.lu(z)
          if f.singular then throw new LinAlgError("Singular matrix")
          val x = LinalgCplx.luSolve(f, ZMat.identity(n))
          (x.re, x.im)
        }
      else
        S.mapD(a, Array(n, n)) { (x, _, _) =>
          val f = LinalgReal.lu(x, n)
          if f.singular then throw new LinAlgError("Singular matrix")
          LinalgReal.luSolve(f, LinalgReal.identity(n), n)
        }
    r.asInstanceOf[NDArray[l.Out]]

  /** Moore-Penrose pseudo-inverse via SVD (`np.linalg.pinv`); singular values below
    * `rcond * max(s)` are treated as zero.
    */
  def pinv[T](a: NDArray[T], rcond: Double = 1e-15, hermitian: Boolean = false)(using l: LinalgOut[T]): NDArray[l.Out] =
    S.assertStacked2d(a)
    val m = a.shapeArr(a.ndim - 2)
    val n = a.shapeArr(a.ndim - 1)
    val bs = S.batchShape(a)
    if m == 0 || n == 0 then
      return NDArray.zerosOf(l.dtype, bs ++ Array(n, m))
    val r = svd(a, full_matrices = false, compute_uv = true, hermitian = hermitian)
    val k = math.min(m, n)
    val s = r.S.toArray
    val nb = Shape.size(bs)
    val sinv = new Array[Double](nb * k)
    var b = 0
    while b < nb do
      var mx = 0.0
      var i = 0
      while i < k do
        mx = math.max(mx, s(b * k + i))
        i += 1
      i = 0
      while i < k do
        val v = s(b * k + i)
        sinv(b * k + i) = if v > rcond * mx then 1.0 / v else 0.0
        i += 1
      b += 1
    // pinv = Vh^H diag(sinv) U^H
    val res: NDArray[?] =
      if isC(a) then
        val u = r.U.asInstanceOf[NDArray[Complex]].toArray
        val vh = r.Vh.asInstanceOf[NDArray[Complex]].toArray
        val out = new Array[Complex](nb * n * m)
        b = 0
        while b < nb do
          var i = 0
          while i < n do
            var j = 0
            while j < m do
              var acc = Complex.Zero
              var p = 0
              while p < k do
                val w = sinv(b * k + p)
                if w != 0.0 then acc = acc + vh(b * k * n + p * n + i).conj * u(b * m * k + j * k + p).conj * w
                p += 1
              out(b * n * m + i * m + j) = acc
              j += 1
            i += 1
          b += 1
        NDArray.fromArray(out, bs ++ Array(n, m))
      else
        val u = r.U.asInstanceOf[NDArray[Double]].toArray
        val vh = r.Vh.asInstanceOf[NDArray[Double]].toArray
        val out = new Array[Double](nb * n * m)
        b = 0
        while b < nb do
          var p = 0
          while p < k do
            val w = sinv(b * k + p)
            if w != 0.0 then
              var i = 0
              while i < n do
                val vw = vh(b * k * n + p * n + i) * w
                if vw != 0.0 then
                  var j = 0
                  while j < m do
                    out(b * n * m + i * m + j) += vw * u(b * m * k + j * k + p)
                    j += 1
                i += 1
            p += 1
          b += 1
        S.wrapD(out, bs ++ Array(n, m))
    res.asInstanceOf[NDArray[l.Out]]

  // ------------------------------------------------------------------ solving

  /** Solves `a @ x = b` (`np.linalg.solve`). `b` is a vector when 1-D, otherwise a (stack of)
    * matrices; batch dimensions broadcast. Throws [[LinAlgError]] for singular `a`.
    */
  def solve[T, U](a: NDArray[T], b: NDArray[U])(using p: NumPromote[T, U])(using l: LinalgOut[p.Out]): NDArray[l.Out] =
    S.assertStackedSquare(a)
    val n = a.shapeArr(a.ndim - 1)
    val vec = b.ndim == 1
    val b2 = if vec then b.reshape(b.shapeArr(0), 1) else b
    if b2.ndim < 2 then throw new LinAlgError(s"${b.ndim}-dimensional array given. Array must be at least one-dimensional")
    if b2.shapeArr(b2.ndim - 2) != n then
      throw new IllegalArgumentException(
        s"solve: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (m,m),(m,n)->(m,n) (size ${b2.shapeArr(b2.ndim - 2)} is different from $n)"
      )
    val k = b2.shapeArr(b2.ndim - 1)
    val batch = Shape.broadcast(S.batchShape(a), S.batchShape(b2))
    val ab = a.broadcastTo((batch ++ Array(n, n)).toSeq*)
    val bb = b2.broadcastTo((batch ++ Array(n, k)).toSeq*)
    val nb = Shape.size(batch)
    val complex = l.dtype.isComplex
    val res: NDArray[?] =
      if complex then
        val (ar, ai) = S.complexData(ab)
        val (br, bi) = S.complexData(bb)
        val or = new Array[Double](nb * n * k)
        val oi = new Array[Double](nb * n * k)
        var t = 0
        while t < nb do
          val z = ZMat(n, n, java.util.Arrays.copyOfRange(ar, t * n * n, (t + 1) * n * n), java.util.Arrays.copyOfRange(ai, t * n * n, (t + 1) * n * n))
          val f = LinalgCplx.lu(z)
          if f.singular then throw new LinAlgError("Singular matrix")
          val x = LinalgCplx.luSolve(f, ZMat(n, k, java.util.Arrays.copyOfRange(br, t * n * k, (t + 1) * n * k), java.util.Arrays.copyOfRange(bi, t * n * k, (t + 1) * n * k)))
          System.arraycopy(x.re, 0, or, t * n * k, n * k)
          System.arraycopy(x.im, 0, oi, t * n * k, n * k)
          t += 1
        S.wrapC(or, oi, batch ++ Array(n, k))
      else
        val ad = S.realData(ab)
        val bd = S.realData(bb)
        val out = new Array[Double](nb * n * k)
        var t = 0
        while t < nb do
          val f = LinalgReal.lu(java.util.Arrays.copyOfRange(ad, t * n * n, (t + 1) * n * n), n)
          if f.singular then throw new LinAlgError("Singular matrix")
          val x = LinalgReal.luSolve(f, java.util.Arrays.copyOfRange(bd, t * n * k, (t + 1) * n * k), k)
          System.arraycopy(x, 0, out, t * n * k, n * k)
          t += 1
        S.wrapD(out, batch ++ Array(n, k))
    val shaped = if vec then res.reshape((batch :+ n).toSeq*) else res
    shaped.asInstanceOf[NDArray[l.Out]]

  /** Least-squares solution of `a @ x = b` (`np.linalg.lstsq`), via SVD. `rcond` defaults
    * (NaN) to `eps * max(M, N)`; singular values `<= rcond * max(s)` are ignored.
    */
  def lstsq[T, U](a: NDArray[T], b: NDArray[U], rcond: Double = Double.NaN)(using p: NumPromote[T, U])(using
      l: LinalgOut[p.Out]
  ): LstsqResult[l.Out] =
    if a.ndim != 2 then throw new LinAlgError(s"${a.ndim}-dimensional array given. Array must be two-dimensional")
    val m = a.shapeArr(0)
    val n = a.shapeArr(1)
    val vec = b.ndim == 1
    if b.ndim != 1 && b.ndim != 2 then throw new LinAlgError(s"${b.ndim}-dimensional array given. Array must be two-dimensional")
    val b2 = if vec then b.reshape(b.shapeArr(0), 1) else b
    if b2.shapeArr(0) != m then throw new LinAlgError("Incompatible dimensions")
    val k = b2.shapeArr(1)
    val rc = if rcond.isNaN then LinalgReal.eps * math.max(m, n) else if rcond < 0 then LinalgReal.eps else rcond
    val complex = l.dtype.isComplex
    val dd: NumDType[Any] = (if complex then C128 else F64).asInstanceOf[NumDType[Any]]
    val ac = a.asInstanceOf[NDArray[Any]].asType(using dd)
    val bc = b2.asInstanceOf[NDArray[Any]].asType(using dd)
    val (rU, rS, rVh): (NDArray[Any], NDArray[Double], NDArray[Any]) =
      if complex then
        val r = svd(ac.asInstanceOf[NDArray[Complex]], full_matrices = false)
        (r.U.asInstanceOf[NDArray[Any]], r.S, r.Vh.asInstanceOf[NDArray[Any]])
      else
        val r = svd(ac.asInstanceOf[NDArray[Double]], full_matrices = false)
        (r.U.asInstanceOf[NDArray[Any]], r.S, r.Vh.asInstanceOf[NDArray[Any]])
    val s = rS.toArray
    val q = s.length
    val smax = if q == 0 then 0.0 else s.max
    val rank = s.count(_ > rc * smax)
    val sinv = s.map(v => if v > rc * smax then 1.0 / v else 0.0)
    val sinvArr = S.wrapD(sinv, Array(q, 1)).asType(using dd)
    val utb = LinAlgCore.matmulD(rU.conj.T, bc, dd)
    val scaled = Ops.arith(utb, sinvArr, dd, Arith.Mul)
    val x = LinAlgCore.matmulD(rVh.conj.T, scaled, dd)
    val residuals =
      if rank == n && m > n then
        val diff = Ops.arith(bc, LinAlgCore.matmulD(ac, x, dd), dd, Arith.Sub)
        Reduce.sum(LinalgNorm.sqAbsD(diff), 0, false, F64)
      else S.wrapD(new Array[Double](0), Array(0))
    val xs = if vec then x.reshape(n) else x
    LstsqResult(xs.asInstanceOf[NDArray[l.Out]], residuals, rank, rS)

  /** Solves the tensor equation `tensordot(a, x, x.ndim) == b` (`np.linalg.tensorsolve`). */
  def tensorsolve[T, U](a: NDArray[T], b: NDArray[U], axes: Seq[Int] = Nil)(using p: NumPromote[T, U])(using
      l: LinalgOut[p.Out]
  ): NDArray[l.Out] =
    var a2: NDArray[T] = a
    val an = a.ndim
    if axes.nonEmpty then
      val all = (0 until an).toBuffer
      for k <- axes do
        val kk = Shape.normAxis(k, an)
        all -= kk
        all.insert(all.length, kk)
      a2 = a.transpose(all.toSeq*)
    val oldShape = a2.shapeArr.drop(b.ndim)
    val prod = Shape.size(oldShape)
    if Shape.size(a2.shapeArr.take(b.ndim)) != prod then
      throw new LinAlgError(
        "Input arrays must satisfy the requirement prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])"
      )
    val res = solve(a2.reshape(prod, prod), b.reshape(prod))
    res.reshape(oldShape.toSeq*)

  /** Inverse of an N-dimensional array with respect to `tensordot` (`np.linalg.tensorinv`). */
  def tensorinv[T](a: NDArray[T], ind: Int = 2)(using l: LinalgOut[T]): NDArray[l.Out] =
    if ind <= 0 then throw new IllegalArgumentException("Invalid ind argument.")
    val old = a.shapeArr
    val prod = Shape.size(old.drop(ind))
    if Shape.size(old.take(ind)) != prod then
      throw new LinAlgError("Input arrays must satisfy the requirement prod(a.shape[ind:]) == prod(a.shape[:ind])")
    val ia = inv(a.reshape(prod, prod))
    ia.reshape((old.drop(ind) ++ old.take(ind)).toSeq*)

  // ------------------------------------------------------------------ eigenvalues

  private def eigImpl(a: NDArray[?], wantV: Boolean): EigResult =
    S.assertStackedSquare(a)
    S.assertFinite(a)
    val n = a.shapeArr(a.ndim - 1)
    val bs = S.batchShape(a)
    val nb = Shape.size(bs)
    val wr = new Array[Double](nb * n)
    val wi = new Array[Double](nb * n)
    val vr = if wantV then new Array[Double](nb * n * n) else null
    val vi = if wantV then new Array[Double](nb * n * n) else null
    def store(b: Int, r: LinalgEig.Result): Unit =
      System.arraycopy(r.wr, 0, wr, b * n, n)
      System.arraycopy(r.wi, 0, wi, b * n, n)
      if wantV then
        System.arraycopy(r.v.re, 0, vr, b * n * n, n * n)
        System.arraycopy(r.v.im, 0, vi, b * n * n, n * n)
    if isC(a) then S.foreachZ(a)((b, z) => store(b, LinalgEig.complexEig(z, wantV)))
    else S.foreachD(a)((b, x) => store(b, LinalgEig.realEig(x, n, wantV)))
    EigResult(S.wrapC(wr, wi, bs :+ n), if wantV then S.wrapC(vr, vi, bs ++ Array(n, n)) else null)

  /** Eigenvalues and right eigenvectors of a general square matrix (`np.linalg.eig`).
    * Results are complex128 (see [[EigResult]]); eigenvectors have unit norm.
    */
  def eig[T](a: NDArray[T])(using l: LinalgOut[T]): EigResult = eigImpl(a, true)

  /** Eigenvalues of a general square matrix (`np.linalg.eigvals`), as complex128. */
  def eigvals[T](a: NDArray[T])(using l: LinalgOut[T]): NDArray[Complex] = eigImpl(a, false).eigenvalues

  private def eighImpl(a: NDArray[?], uplo: String, wantV: Boolean): (NDArray[Double], NDArray[?]) =
    S.assertStackedSquare(a)
    val lower = S.checkUplo(uplo)
    val n = a.shapeArr(a.ndim - 1)
    val bs = S.batchShape(a)
    val nb = Shape.size(bs)
    val w = new Array[Double](nb * n)
    if isC(a) then
      val vr = new Array[Double](nb * n * n)
      val vi = new Array[Double](nb * n * n)
      S.foreachZ(a) { (b, z) =>
        S.hermitianize(z, lower)
        val (d, v) = LinalgCplx.hermEig(z, wantV)
        System.arraycopy(d, 0, w, b * n, n)
        if wantV then
          System.arraycopy(v.re, 0, vr, b * n * n, n * n)
          System.arraycopy(v.im, 0, vi, b * n * n, n * n)
      }
      (S.wrapD(w, bs :+ n), if wantV then S.wrapC(vr, vi, bs ++ Array(n, n)) else null)
    else
      val vv = new Array[Double](nb * n * n)
      S.foreachD(a) { (b, x) =>
        S.symmetrizeD(x, n, lower)
        val (d, v) = LinalgReal.symEig(x, n, wantV)
        System.arraycopy(d, 0, w, b * n, n)
        if wantV then System.arraycopy(v, 0, vv, b * n * n, n * n)
      }
      (S.wrapD(w, bs :+ n), if wantV then S.wrapD(vv, bs ++ Array(n, n)) else null)

  /** Eigen-decomposition of a real symmetric / complex Hermitian matrix (`np.linalg.eigh`); only the
    * `UPLO` ("L" or "U") triangle is used. Eigenvalues ascend.
    */
  def eigh[T](a: NDArray[T], UPLO: String = "L")(using l: LinalgOut[T]): EighResult[l.Out] =
    val (w, v) = eighImpl(a, UPLO, true)
    EighResult(w, v.asInstanceOf[NDArray[l.Out]])

  /** Eigenvalues of a symmetric / Hermitian matrix, ascending (`np.linalg.eigvalsh`). */
  def eigvalsh[T](a: NDArray[T], UPLO: String = "L")(using l: LinalgOut[T]): NDArray[Double] =
    eighImpl(a, UPLO, false)._1

  // ------------------------------------------------------------------ SVD

  /** Singular value decomposition `a = U @ diag(S) @ Vh` (`np.linalg.svd`), singular values descending.
    * `full_matrices = false` gives the reduced factors; `hermitian = true` uses `eigh`.
    */
  def svd[T](a: NDArray[T], full_matrices: Boolean = true, compute_uv: Boolean = true, hermitian: Boolean = false)(using
      l: LinalgOut[T]
  ): SVDResult[l.Out] =
    S.assertStacked2d(a)
    val m = a.shapeArr(a.ndim - 2)
    val n = a.shapeArr(a.ndim - 1)
    val k = math.min(m, n)
    val bs = S.batchShape(a)
    val nb = Shape.size(bs)
    if hermitian then return svdHermitian(a, compute_uv).asInstanceOf[SVDResult[l.Out]]
    val uc = if full_matrices then m else k
    val vr = if full_matrices then n else k
    val s = new Array[Double](nb * k)
    val res: SVDResult[?] =
      if isC(a) then
        val ur = if compute_uv then new Array[Double](nb * m * uc) else null
        val ui = if compute_uv then new Array[Double](nb * m * uc) else null
        val hr = if compute_uv then new Array[Double](nb * vr * n) else null
        val hi = if compute_uv then new Array[Double](nb * vr * n) else null
        S.foreachZ(a) { (b, z) =>
          val r = LinalgCplx.svd(z, compute_uv, full_matrices)
          System.arraycopy(r.s, 0, s, b * k, k)
          if compute_uv then
            System.arraycopy(r.u.re, 0, ur, b * m * uc, m * uc)
            System.arraycopy(r.u.im, 0, ui, b * m * uc, m * uc)
            System.arraycopy(r.vh.re, 0, hr, b * vr * n, vr * n)
            System.arraycopy(r.vh.im, 0, hi, b * vr * n, vr * n)
        }
        if compute_uv then
          SVDResult(S.wrapC(ur, ui, bs ++ Array(m, uc)), S.wrapD(s, bs :+ k), S.wrapC(hr, hi, bs ++ Array(vr, n)))
        else SVDResult(NDArray.zerosOf(C128, Array(0)), S.wrapD(s, bs :+ k), NDArray.zerosOf(C128, Array(0)))
      else
        val u = if compute_uv then new Array[Double](nb * m * uc) else null
        val vh = if compute_uv then new Array[Double](nb * vr * n) else null
        S.foreachD(a) { (b, x) =>
          val r = LinalgReal.svd(x, m, n, compute_uv, full_matrices)
          System.arraycopy(r.s, 0, s, b * k, k)
          if compute_uv then
            System.arraycopy(r.u, 0, u, b * m * uc, m * uc)
            System.arraycopy(r.vt, 0, vh, b * vr * n, vr * n)
        }
        if compute_uv then SVDResult(S.wrapD(u, bs ++ Array(m, uc)), S.wrapD(s, bs :+ k), S.wrapD(vh, bs ++ Array(vr, n)))
        else SVDResult(NDArray.zerosOf(F64, Array(0)), S.wrapD(s, bs :+ k), NDArray.zerosOf(F64, Array(0)))
    res.asInstanceOf[SVDResult[l.Out]]

  private def svdHermitian(a: NDArray[?], computeUV: Boolean): SVDResult[?] =
    S.assertStackedSquare(a)
    val n = a.shapeArr(a.ndim - 1)
    val bs = S.batchShape(a)
    val nb = Shape.size(bs)
    val (w, v) = eighImpl(a, "L", computeUV)
    val wd = w.toArray
    val s = new Array[Double](nb * n)
    val orders = Array.tabulate(nb) { b =>
      val abs = Array.tabulate(n)(i => math.abs(wd(b * n + i)))
      val ord = S.argsortDesc(abs)
      var i = 0
      while i < n do
        s(b * n + i) = abs(ord(i))
        i += 1
      ord
    }
    val sArr = S.wrapD(s, bs :+ n)
    if !computeUV then return SVDResult(NDArray.zerosOf(v0dtype(a), Array(0)), sArr, NDArray.zerosOf(v0dtype(a), Array(0)))
    def entry(b: Int, i: Int, j: Int): (Int, Double) =
      val src = orders(b)(j)
      (b * n * n + i * n + src, math.signum(wd(b * n + src)))
    val shape = bs ++ Array(n, n)
    if isC(a) then
      val (vr, vi) = S.complexData(v)
      val ur = new Array[Double](nb * n * n); val ui = new Array[Double](nb * n * n)
      val hr = new Array[Double](nb * n * n); val hi = new Array[Double](nb * n * n)
      for b <- 0 until nb; i <- 0 until n; j <- 0 until n do
        val (o, sg) = entry(b, i, j)
        ur(b * n * n + i * n + j) = vr(o); ui(b * n * n + i * n + j) = vi(o)
        // Vh[j, i] = conj(sgn * u[i, j])
        hr(b * n * n + j * n + i) = sg * vr(o); hi(b * n * n + j * n + i) = -sg * vi(o)
      SVDResult(S.wrapC(ur, ui, shape), sArr, S.wrapC(hr, hi, shape))
    else
      val vd = S.realData(v)
      val u = new Array[Double](nb * n * n)
      val vh = new Array[Double](nb * n * n)
      for b <- 0 until nb; i <- 0 until n; j <- 0 until n do
        val (o, sg) = entry(b, i, j)
        u(b * n * n + i * n + j) = vd(o)
        vh(b * n * n + j * n + i) = sg * vd(o)
      SVDResult(S.wrapD(u, shape), sArr, S.wrapD(vh, shape))

  private def v0dtype(a: NDArray[?]): DType[Any] =
    (if isC(a) then C128 else F64).asInstanceOf[DType[Any]]

  /** Singular values, descending (`np.linalg.svdvals`). */
  def svdvals[T](x: NDArray[T])(using l: LinalgOut[T]): NDArray[Double] =
    svd(x, full_matrices = false, compute_uv = false).S

  // ------------------------------------------------------------------ QR and Cholesky

  /** QR factorisation (`np.linalg.qr`) with `mode` "reduced" (default), "complete", "r" or "raw"
    * (see [[QRResult]]). Follows LAPACK's sign conventions (`geqrf`).
    */
  def qr[T](a: NDArray[T], mode: String = "reduced")(using l: LinalgOut[T]): QRResult[l.Out] =
    if !Set("reduced", "complete", "r", "raw").contains(mode) then
      throw new IllegalArgumentException(s"Unrecognized mode '$mode'")
    S.assertStacked2d(a)
    val m = a.shapeArr(a.ndim - 2)
    val n = a.shapeArr(a.ndim - 1)
    val k = math.min(m, n)
    val bs = S.batchShape(a)
    val (qc, rr) = mode match
      case "complete" => (m, m)
      case _ => (k, k)
    val res: QRResult[?] =
      if isC(a) then
        mode match
          case "raw" =>
            val hT = S.mapZ(a, Array(n, m)) { z =>
              LinalgCplx.geqrf(z)
              val t = ZMat.zeros(n, m)
              var i = 0
              while i < m do
                var j = 0
                while j < n do
                  t.re(j * m + i) = z.re(i * n + j)
                  t.im(j * m + i) = z.im(i * n + j)
                  j += 1
                i += 1
              (t.re, t.im)
            }
            val tau = S.mapZ(a, Array(k)) { z => LinalgCplx.geqrf(z) }
            QRResult(hT, tau)
          case _ =>
            val r = S.mapZ(a, Array(rr, n)) { z =>
              LinalgCplx.geqrf(z)
              val x = LinalgCplx.extractR(z, rr)
              (x.re, x.im)
            }
            val q =
              if mode == "r" then NDArray.zerosOf(C128, Array(0))
              else
                S.mapZ(a, Array(m, qc)) { z =>
                  val (tr, ti) = LinalgCplx.geqrf(z)
                  val x = LinalgCplx.ungqr(z, tr, ti, qc)
                  (x.re, x.im)
                }
            QRResult(q, r)
      else
        mode match
          case "raw" =>
            val hT = S.mapD(a, Array(n, m)) { (x, _, _) =>
              LinalgReal.geqrf(x, m, n)
              LinalgReal.transpose(x, m, n)
            }
            val tau = S.mapD(a, Array(k))((x, _, _) => LinalgReal.geqrf(x, m, n))
            QRResult(hT, tau)
          case _ =>
            val r = S.mapD(a, Array(rr, n)) { (x, _, _) =>
              LinalgReal.geqrf(x, m, n)
              LinalgReal.extractR(x, m, n, rr)
            }
            val q =
              if mode == "r" then NDArray.zerosOf(F64, Array(0))
              else
                S.mapD(a, Array(m, qc)) { (x, _, _) =>
                  val tau = LinalgReal.geqrf(x, m, n)
                  LinalgReal.orgqr(x, m, n, tau, qc)
                }
            QRResult(q, r)
    res.asInstanceOf[QRResult[l.Out]]

  /** Cholesky factor (`np.linalg.cholesky`): lower `L` with `a = L @ L^H`, or with `upper = true`
    * the upper `U = L^H`. Throws [[LinAlgError]] when not positive definite.
    */
  def cholesky[T](a: NDArray[T], upper: Boolean = false)(using l: LinalgOut[T]): NDArray[l.Out] =
    S.assertStackedSquare(a)
    val n = a.shapeArr(a.ndim - 1)
    val r: NDArray[?] =
      if isC(a) then
        S.mapZ(a, Array(n, n)) { z =>
          val lz = LinalgCplx.cholesky(S.triangleSource(z, upper))
          val res = if upper then ZMat.adjoint(lz) else lz
          (res.re, res.im)
        }
      else
        S.mapD(a, Array(n, n)) { (x, _, _) =>
          val src = if upper then LinalgReal.transpose(x, n, n) else x
          val lm = LinalgReal.cholesky(src, n)
          if upper then LinalgReal.transpose(lm, n, n) else lm
        }
    r.asInstanceOf[NDArray[l.Out]]

  // ------------------------------------------------------------------ rank, power, condition

  /** Matrix rank via SVD (`np.linalg.matrix_rank`): number of singular values above `tol`
    * (default NaN: `S.max * max(M, N) * eps`, or `S.max * rtol` when `rtol` is given).
    */
  def matrix_rank[T](A: NDArray[T], tol: Double = Double.NaN, hermitian: Boolean = false, rtol: Double = Double.NaN)(using
      l: LinalgOut[T]
  ): NDArray[Int] =
    if A.ndim < 2 then
      val d = A.dtype.asInstanceOf[DType[Any]]
      var nz = false
      A.asInstanceOf[NDArray[Any]].foreach(x => if !d.equiv(x, d.zero) then nz = true)
      return NDArray.scalar(if nz then 1 else 0)
    val m = A.shapeArr(A.ndim - 2)
    val n = A.shapeArr(A.ndim - 1)
    val s = if hermitian then svd(A, compute_uv = false, hermitian = true).S else svdvals(A)
    val k = s.shapeArr.last
    val sd = s.toArray
    val bs = s.shapeArr.dropRight(1)
    val nb = Shape.size(bs)
    val out = new Array[Int](nb)
    var b = 0
    while b < nb do
      var mx = 0.0
      var i = 0
      while i < k do
        mx = math.max(mx, sd(b * k + i))
        i += 1
      val t =
        if !tol.isNaN then tol
        else if !rtol.isNaN then mx * rtol
        else mx * math.max(m, n) * LinalgReal.eps
      i = 0
      while i < k do
        if sd(b * k + i) > t then out(b) += 1
        i += 1
      b += 1
    NDArray.fromArray(out, bs)

  /** Raises a square matrix (stack) to an integer power (`np.linalg.matrix_power`). Negative powers
    * invert first, which requires a float or complex matrix (NumPy would silently switch to float64).
    */
  def matrix_power[T](a: NDArray[T], n: Int)(using d: NumDType[T]): NDArray[T] =
    S.assertStackedSquare(a)
    val m = a.shapeArr(a.ndim - 1)
    var base: NDArray[T] =
      if n >= 0 then a
      else
        d.kind match
          case 'f' =>
            inv(a.asType(using F64))(using LinalgOut.realOut[Double]).asType(using d)
          case 'c' =>
            inv(a.asInstanceOf[NDArray[Complex]]).asInstanceOf[NDArray[T]]
          case _ =>
            throw new IllegalArgumentException(
              "matrix_power: negative powers of integer matrices need a float matrix (convert with astype[Double])"
            )
    var e = math.abs(n.toLong)
    if e == 0 then
      val eye = NDArray.zerosOf(d, a.shapeArr.clone())
      val nb = Shape.size(S.batchShape(a))
      var b = 0
      while b < nb do
        var i = 0
        while i < m do
          eye.data(b * m * m + i * m + i) = d.one
          i += 1
        b += 1
      return eye
    if e == 1 then return base.copy()
    var result: NDArray[T] | Null = null
    while e > 0 do
      if (e & 1L) == 1L then
        result = if result == null then base else LinAlgCore.matmulD(result.nn, base, d)
      e >>= 1
      if e > 0 then base = LinAlgCore.matmulD(base, base, d)
    result.nn

  /** Condition number (`np.linalg.cond`) in the norm `p` (None/2/-2 via SVD, or 1, -1, inf, -inf, "fro", "nuc"). */
  def cond[T](x: NDArray[T], p: NormOrd = None)(using l: LinalgOut[T]): NDArray[Double] =
    S.assertStacked2d(x)
    if x.size == 0 then throw new LinAlgError("cond is not defined on empty arrays")
    val pv = LinalgNorm.ordOf(p)
    val r: NDArray[Double] = pv match
      case None | Some(Right(2.0)) | Some(Right(-2.0)) =>
        val s = svdvals(x)
        val k = s.shapeArr.last
        val first = s(---, 0)
        val last = s(---, k - 1)
        if pv == Some(Right(-2.0)) then NDArray.zipMap(last, first)(_ / _)
        else NDArray.zipMap(first, last)(_ / _)
      case _ =>
        S.assertStackedSquare(x)
        val n = x.shapeArr.last
        // inverse with NaN for singular matrices (NumPy ignores the error here)
        val ix: NDArray[?] =
          if isC(x) then
            S.mapZ(x, Array(n, n)) { z =>
              val f = LinalgCplx.lu(z)
              if f.singular then (Array.fill(n * n)(Double.NaN), Array.fill(n * n)(Double.NaN))
              else { val q = LinalgCplx.luSolve(f, ZMat.identity(n)); (q.re, q.im) }
            }
          else
            S.mapD(x, Array(n, n)) { (d, _, _) =>
              val f = LinalgReal.lu(d, n)
              if f.singular then Array.fill(n * n)(Double.NaN) else LinalgReal.luSolve(f, LinalgReal.identity(n), n)
            }
        val n1 = LinalgNorm.norm(x, p, Seq(-2, -1), false)
        val n2 = LinalgNorm.norm(ix, p, Seq(-2, -1), false)
        NDArray.zipMap(n1, n2)(_ * _)
    // NaN -> inf unless the input itself had NaNs
    val xNan = LinalgNorm.matrixHasNaN(x)
    val out = r.toArray
    var i = 0
    while i < out.length do
      if out(i).isNaN && !xNan(i) then out(i) = Double.PositiveInfinity
      i += 1
    NDArray.fromArray(out, r.shapeArr)

  // ------------------------------------------------------------------ norms

  /** Frobenius / 2-norm of the flattened array (`np.linalg.norm(x)`). */
  def norm[T](x: NDArray[T])(using d: NumDType[T]): Double = LinalgNorm.norm(x, None, None, false).item

  /** Vector norm (1-D input) or matrix norm (2-D input) of order `ord` (`np.linalg.norm(x, ord)`). */
  def norm[T](x: NDArray[T], ord: NormOrd)(using d: NumDType[T]): Double = LinalgNorm.norm(x, ord, None, false).item

  /** `np.linalg.norm(x, ord, axis, keepdims)`: vector norms along one axis, matrix norms over two axes. */
  def norm[T](x: NDArray[T], ord: NormOrd = None, axis: Axis | None.type = None, keepdims: Boolean = false)(using
      d: NumDType[T]
  ): NDArray[Double] = LinalgNorm.norm(x, ord, axis, keepdims)

  /** Vector norm over `axis` (all elements by default) (`np.linalg.vector_norm`). */
  def vector_norm[T](x: NDArray[T], axis: Axis | None.type = None, keepdims: Boolean = false, ord: NormOrd = 2.0)(using
      d: NumDType[T]
  ): NDArray[Double] = LinalgNorm.vectorNorm(x, axis, keepdims, ord)

  /** Matrix norm over the last two axes (`np.linalg.matrix_norm`), default Frobenius. */
  def matrix_norm[T](x: NDArray[T], keepdims: Boolean = false, ord: NormOrd = "fro")(using d: NumDType[T]): NDArray[Double] =
    S.assertStacked2d(x)
    LinalgNorm.norm(x, ord, Seq(-2, -1), keepdims)

  // ------------------------------------------------------------------ products

  /** Product of a chain of matrices in the cheapest association order (`np.linalg.multi_dot`).
    * The first / last array may be 1-D (treated as row / column vectors).
    */
  def multi_dot[T](arrays: Seq[NDArray[T]])(using d: NumDType[T]): NDArray[T] =
    val n = arrays.length
    if n < 2 then throw new IllegalArgumentException("Expecting at least two arrays.")
    if n == 2 then return LinAlgCore.dotD(arrays(0), arrays(1), d)
    val firstVec = arrays.head.ndim == 1
    val lastVec = arrays.last.ndim == 1
    val ms = arrays.zipWithIndex.map { (a, i) =>
      if i == 0 && firstVec then a.reshape(1, a.shapeArr(0))
      else if i == n - 1 && lastVec then a.reshape(a.shapeArr(0), 1)
      else a
    }
    ms.foreach(a =>
      if a.ndim != 2 then throw new LinAlgError(s"${a.ndim}-dimensional array given. Array must be two-dimensional")
    )
    val dims = (ms.map(_.shapeArr(0)) :+ ms.last.shapeArr(1)).map(_.toLong).toArray
    // matrix-chain dynamic programming
    val cost = Array.ofDim[Long](n, n)
    val split = Array.ofDim[Int](n, n)
    for len <- 1 until n; i <- 0 until n - len do
      val j = i + len
      cost(i)(j) = Long.MaxValue
      for k <- i until j do
        val c = cost(i)(k) + cost(k + 1)(j) + dims(i) * dims(k + 1) * dims(j + 1)
        if c < cost(i)(j) then
          cost(i)(j) = c
          split(i)(j) = k
    def go(i: Int, j: Int): NDArray[T] =
      if i == j then ms(i) else LinAlgCore.matmulD(go(i, split(i)(j)), go(split(i)(j) + 1, j), d)
    val r = go(0, n - 1)
    if firstVec && lastVec then r.reshape()
    else if firstVec then r.reshape(r.shapeArr(1))
    else if lastVec then r.reshape(r.shapeArr(0))
    else r

  /** Matrix product (`np.linalg.matmul`). */
  def matmul[A, B](x1: NDArray[A], x2: NDArray[B])(using p: NumPromote[A, B]): NDArray[p.Out] =
    LinAlgCore.matmul(x1, x2)

  /** Outer product of two 1-D arrays (`np.linalg.outer`). */
  def outer[A, B](x1: NDArray[A], x2: NDArray[B])(using p: NumPromote[A, B]): NDArray[p.Out] =
    if x1.ndim != 1 || x2.ndim != 1 then
      throw new IllegalArgumentException(
        s"Input arrays must be one-dimensional, but they are x1.ndim=${x1.ndim} and x2.ndim=${x2.ndim}."
      )
    LinalgProducts.outer(x1, x2, p.dtype)

  /** Cross product of 3-element vectors along `axis` (`np.linalg.cross`). */
  def cross[A, B](x1: NDArray[A], x2: NDArray[B], axis: Int = -1)(using p: NumPromote[A, B]): NDArray[p.Out] =
    if x1.shapeArr(Shape.normAxis(axis, x1.ndim)) != 3 || x2.shapeArr(Shape.normAxis(axis, x2.ndim)) != 3 then
      throw new IllegalArgumentException(
        s"Both input arrays must be (arrays of) 3-dimensional vectors, but they are ${x1.shapeArr(Shape.normAxis(axis, x1.ndim))} and ${x2.shapeArr(Shape.normAxis(axis, x2.ndim))} dimensional instead."
      )
    LinalgProducts.cross(x1, x2, axis, axis, axis, p.dtype)

  /** Diagonals of the last two axes (`np.linalg.diagonal`). */
  def diagonal[T](x: NDArray[T], offset: Int = 0): NDArray[T] =
    S.assertStacked2d(x)
    Ops.diagonal(x, offset, -2, -1)

  /** Sum along the diagonals of the last two axes (`np.linalg.trace`); 0-d for a single matrix. */
  def trace[T](x: NDArray[T], offset: Int = 0)(using s: SumOf[T]): NDArray[s.Out] =
    S.assertStacked2d(x)
    Reduce.sum(Ops.diagonal(x, offset, -2, -1), -1, false, s.dtype)

  /** Vector dot product along `axis` with complex conjugation of `x1` (`np.linalg.vecdot`). */
  def vecdot[A, B](x1: NDArray[A], x2: NDArray[B], axis: Int = -1)(using p: NumPromote[A, B]): NDArray[p.Out] =
    LinalgProducts.vecdot(x1, x2, axis, p.dtype)

  /** Transposes the last two axes (`np.linalg.matrix_transpose`). */
  def matrix_transpose[T](x: NDArray[T]): NDArray[T] =
    if x.ndim < 2 then
      throw new IllegalArgumentException(s"Input array must be at least 2-dimensional, but it is ${x.ndim}")
    x.swapaxes(-1, -2)
