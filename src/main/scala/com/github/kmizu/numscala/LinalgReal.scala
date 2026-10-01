package com.github.kmizu.numscala

/** Dense real (float64) kernels on row-major `Array[Double]` matrices:
  * LU with partial pivoting, Householder QR (LAPACK conventions), Cholesky,
  * one-sided Jacobi SVD and the symmetric eigensolver (Householder tridiagonalisation + implicit QL).
  */
private[numscala] object LinalgReal:
  val eps: Double = java.lang.Math.ulp(1.0)

  // ------------------------------------------------------------------ helpers

  /** Euclidean norm of `len` elements starting at `off` with stride `st`, with scaling against overflow. */
  def nrm2(a: Array[Double], off: Int, len: Int, st: Int): Double =
    var scale = 0.0
    var ssq = 1.0
    var i = 0
    var o = off
    while i < len do
      val v = a(o)
      if v != 0.0 then
        val av = math.abs(v)
        if scale < av then
          ssq = 1.0 + ssq * (scale / av) * (scale / av)
          scale = av
        else ssq += (av / scale) * (av / scale)
      else if v.isNaN then return Double.NaN
      i += 1
      o += st
    scale * math.sqrt(ssq)

  def transpose(a: Array[Double], m: Int, n: Int): Array[Double] =
    val out = new Array[Double](m * n)
    var i = 0
    while i < m do
      var j = 0
      while j < n do
        out(j * m + i) = a(i * n + j)
        j += 1
      i += 1
    out

  def identity(n: Int): Array[Double] =
    val out = new Array[Double](n * n)
    var i = 0
    while i < n do
      out(i * n + i) = 1.0
      i += 1
    out

  /** `(m x k) * (k x n)`. */
  def gemm(a: Array[Double], b: Array[Double], m: Int, k: Int, n: Int): Array[Double] =
    val out = new Array[Double](m * n)
    var i = 0
    while i < m do
      var p = 0
      while p < k do
        val av = a(i * k + p)
        if av != 0.0 then
          var j = 0
          val bp = p * n
          val oi = i * n
          while j < n do
            out(oi + j) += av * b(bp + j)
            j += 1
        p += 1
      i += 1
    out

  // ------------------------------------------------------------------ LU

  /** Result of an in-place LU factorisation: `P A = L U`; `perm(i)` is the original row now at row `i`. */
  final class LU(val lu: Array[Double], val n: Int, val perm: Array[Int], val sign: Int, val singular: Boolean)

  def lu(a: Array[Double], n: Int): LU =
    val perm = Array.range(0, n)
    var sign = 1
    var singular = false
    var k = 0
    while k < n do
      var p = k
      var mx = math.abs(a(k * n + k))
      var i = k + 1
      while i < n do
        val v = math.abs(a(i * n + k))
        if v > mx then
          mx = v
          p = i
        i += 1
      if p != k then
        var j = 0
        while j < n do
          val t = a(k * n + j)
          a(k * n + j) = a(p * n + j)
          a(p * n + j) = t
          j += 1
        val t = perm(k)
        perm(k) = perm(p)
        perm(p) = t
        sign = -sign
      val pv = a(k * n + k)
      if pv == 0.0 then singular = true
      else
        i = k + 1
        while i < n do
          val f = a(i * n + k) / pv
          a(i * n + k) = f
          if f != 0.0 then
            var j = k + 1
            val ri = i * n
            val rk = k * n
            while j < n do
              a(ri + j) -= f * a(rk + j)
              j += 1
          i += 1
      k += 1
    LU(a, n, perm, sign, singular)

  /** Solves `A X = B` given the LU factors; `b` is `n x nrhs` row-major. */
  def luSolve(f: LU, b: Array[Double], nrhs: Int): Array[Double] =
    val n = f.n
    val a = f.lu
    val x = new Array[Double](n * nrhs)
    var i = 0
    while i < n do
      System.arraycopy(b, f.perm(i) * nrhs, x, i * nrhs, nrhs)
      i += 1
    i = 0
    while i < n do
      var j = 0
      while j < i do
        val l = a(i * n + j)
        if l != 0.0 then
          var c = 0
          while c < nrhs do
            x(i * nrhs + c) -= l * x(j * nrhs + c)
            c += 1
        j += 1
      i += 1
    i = n - 1
    while i >= 0 do
      var j = i + 1
      while j < n do
        val u = a(i * n + j)
        if u != 0.0 then
          var c = 0
          while c < nrhs do
            x(i * nrhs + c) -= u * x(j * nrhs + c)
            c += 1
        j += 1
      val d = a(i * n + i)
      var c = 0
      while c < nrhs do
        x(i * nrhs + c) /= d
        c += 1
      i -= 1
    x

  def det(a: Array[Double], n: Int): Double =
    val f = lu(a, n)
    var d = f.sign.toDouble
    var i = 0
    while i < n do
      d *= f.lu(i * n + i)
      i += 1
    d

  // ------------------------------------------------------------------ QR (LAPACK dgeqrf / dorgqr conventions)

  /** Householder QR in place (`dgeqrf`): on return the upper triangle holds `R`, the part below the
    * diagonal holds the Householder vectors (with implicit unit leading entry); returns `tau`.
    */
  def geqrf(a: Array[Double], m: Int, n: Int): Array[Double] =
    val k = math.min(m, n)
    val tau = new Array[Double](k)
    var j = 0
    while j < k do
      val alpha = a(j * n + j)
      val xnorm = if j + 1 < m then nrm2(a, (j + 1) * n + j, m - j - 1, n) else 0.0
      if xnorm != 0.0 then
        val beta = -math.copySign(math.hypot(alpha, xnorm), alpha)
        tau(j) = (beta - alpha) / beta
        val scal = 1.0 / (alpha - beta)
        var i = j + 1
        while i < m do
          a(i * n + j) *= scal
          i += 1
        a(j * n + j) = beta
        val t = tau(j)
        var c = j + 1
        while c < n do
          var s = a(j * n + c)
          i = j + 1
          while i < m do
            s += a(i * n + j) * a(i * n + c)
            i += 1
          s *= t
          a(j * n + c) -= s
          i = j + 1
          while i < m do
            a(i * n + c) -= s * a(i * n + j)
            i += 1
          c += 1
      j += 1
    tau

  /** Forms the first `qc` columns of `Q = H(0) ... H(k-1)` (`dorgqr`), an `m x qc` matrix. */
  def orgqr(h: Array[Double], m: Int, n: Int, tau: Array[Double], qc: Int): Array[Double] =
    val k = tau.length
    val q = new Array[Double](m * qc)
    var i = 0
    while i < math.min(m, qc) do
      q(i * qc + i) = 1.0
      i += 1
    var j = k - 1
    while j >= 0 do
      val t = tau(j)
      if t != 0.0 then
        var c = j
        while c < qc do
          var s = q(j * qc + c)
          i = j + 1
          while i < m do
            s += h(i * n + j) * q(i * qc + c)
            i += 1
          s *= t
          q(j * qc + c) -= s
          i = j + 1
          while i < m do
            q(i * qc + c) -= s * h(i * n + j)
            i += 1
          c += 1
      j -= 1
    q

  /** Upper-trapezoidal `rows x n` factor taken from the `geqrf` output. */
  def extractR(h: Array[Double], m: Int, n: Int, rows: Int): Array[Double] =
    val r = new Array[Double](rows * n)
    var i = 0
    while i < math.min(rows, m) do
      var j = i
      while j < n do
        r(i * n + j) = h(i * n + j)
        j += 1
      i += 1
    r

  // ------------------------------------------------------------------ Cholesky

  /** Lower Cholesky factor of an SPD matrix, reading only the lower triangle. */
  def cholesky(a: Array[Double], n: Int): Array[Double] =
    val l = new Array[Double](n * n)
    var j = 0
    while j < n do
      var s = a(j * n + j)
      var k = 0
      while k < j do
        s -= l(j * n + k) * l(j * n + k)
        k += 1
      if !(s > 0.0) then throw new LinAlgError("Matrix is not positive definite")
      val d = math.sqrt(s)
      l(j * n + j) = d
      var i = j + 1
      while i < n do
        var t = a(i * n + j)
        k = 0
        while k < j do
          t -= l(i * n + k) * l(j * n + k)
          k += 1
        l(i * n + j) = t / d
        i += 1
      j += 1
    l

  // ------------------------------------------------------------------ SVD (one-sided Jacobi)

  /** `A = U diag(s) Vt`. `u` is `m x ucols`, `vt` is `vrows x n` (both null when not requested). */
  final class SVD(val u: Array[Double], val s: Array[Double], val vt: Array[Double], val ucols: Int, val vrows: Int)

  def svd(a0: Array[Double], m: Int, n: Int, computeUV: Boolean, full: Boolean): SVD =
    if a0.exists(v => v.isNaN || v.isInfinite) then throw new LinAlgError("SVD did not converge")
    var amax = 0.0
    a0.foreach(v => amax = math.max(amax, math.abs(v)))
    if amax > 0.0 && (amax > 1e150 || amax < 1e-150) then
      // scale by a power of two (exact) to avoid overflow / underflow in the Jacobi sums
      val f = java.lang.Math.scalb(1.0, -java.lang.Math.getExponent(amax))
      val r = svdCore(a0.map(_ * f), m, n, computeUV, full)
      SVD(r.u, r.s.map(_ / f), r.vt, r.ucols, r.vrows)
    else svdCore(a0, m, n, computeUV, full)

  private def svdCore(a0: Array[Double], m: Int, n: Int, computeUV: Boolean, full: Boolean): SVD =
    val k = math.min(m, n)
    if k == 0 then
      val uc = if full then m else k
      val vr = if full then n else k
      return SVD(
        if computeUV then (if full then identity(m) else new Array[Double](0)) else null,
        new Array[Double](0),
        if computeUV then (if full then identity(n) else new Array[Double](0)) else null,
        uc,
        vr
      )
    if m < n then
      val r = svdCore(transpose(a0, m, n), n, m, computeUV, full)
      // A^T = U' S V'^T  =>  A = V' S U'^T
      if !computeUV then return SVD(null, r.s, null, 0, 0)
      val u = transpose(r.vt, r.vrows, m) // m x vrows
      val vt = transpose(r.u, n, r.ucols) // ucols x n
      return SVD(u, r.s, vt, r.vrows, r.ucols)
    // m >= n
    if m > n then
      val h = a0.clone()
      val tau = geqrf(h, m, n)
      val rmat = extractR(h, m, n, n)
      val inner = svdCore(rmat, n, n, computeUV, full = true)
      if !computeUV then return SVD(null, inner.s, null, 0, 0)
      val qc = if full then m else n
      val q = orgqr(h, m, n, tau, qc)
      val u = new Array[Double](m * qc)
      // first n columns: Q[:, :n] * U_R ; the rest: Q[:, n:]
      var i = 0
      while i < m do
        var j = 0
        while j < n do
          var acc = 0.0
          var p = 0
          while p < n do
            acc += q(i * qc + p) * inner.u(p * n + j)
            p += 1
          u(i * qc + j) = acc
          j += 1
        while j < qc do
          u(i * qc + j) = q(i * qc + j)
          j += 1
        i += 1
      return SVD(u, inner.s, inner.vt, qc, n)
    // square: Jacobi on columns, stored as rows of g (g = A^T)
    val g = transpose(a0, m, n)
    val v = if computeUV then identity(n) else null
    jacobi(g, n, m, v)
    val s = new Array[Double](n)
    var i = 0
    while i < n do
      s(i) = nrm2(g, i * m, m, 1)
      i += 1
    val order = (0 until n).sortBy(j => -s(j)).toArray
    val ss = order.map(s(_))
    if !computeUV then return SVD(null, ss, null, 0, 0)
    val ut = new Array[Double](n * m) // rows = columns of U
    val vt = new Array[Double](n * n)
    val good = new Array[Boolean](n)
    i = 0
    while i < n do
      val j = order(i)
      System.arraycopy(v, j * n, vt, i * n, n)
      val sv = ss(i)
      if sv > 1e-300 then
        good(i) = true
        var x = 0
        while x < m do
          ut(i * m + x) = g(j * m + x) / sv
          x += 1
      i += 1
    completeRows(ut, n, m, good)
    SVD(transpose(ut, n, m), ss, vt, n, n)

  /** Cyclic one-sided Jacobi: orthogonalises the `nc` rows (length `len`) of `g`; rotations are
    * mirrored onto the rows of `v` (when non-null).
    */
  private def jacobi(g: Array[Double], nc: Int, len: Int, v: Array[Double]): Unit =
    val tol = math.max(len, 8) * eps
    var sweep = 0
    var rotated = true
    while rotated && sweep < 80 do
      rotated = false
      var i = 0
      while i < nc - 1 do
        var j = i + 1
        while j < nc do
          var alpha = 0.0
          var beta = 0.0
          var gamma = 0.0
          val oi = i * len
          val oj = j * len
          var x = 0
          while x < len do
            val gi = g(oi + x)
            val gj = g(oj + x)
            alpha += gi * gi
            beta += gj * gj
            gamma += gi * gj
            x += 1
          if alpha > 0.0 && beta > 0.0 && math.abs(gamma) > tol * math.sqrt(alpha) * math.sqrt(beta) then
            rotated = true
            val zeta = (beta - alpha) / (2.0 * gamma)
            val t =
              if math.abs(zeta) > 1e150 then 0.5 / zeta
              else (if zeta >= 0 then 1.0 else -1.0) / (math.abs(zeta) + math.sqrt(1.0 + zeta * zeta))
            val c = 1.0 / math.sqrt(1.0 + t * t)
            val s = c * t
            x = 0
            while x < len do
              val gi = g(oi + x)
              val gj = g(oj + x)
              g(oi + x) = c * gi - s * gj
              g(oj + x) = s * gi + c * gj
              x += 1
            if v != null then
              val vi = i * nc
              val vj = j * nc
              x = 0
              while x < nc do
                val a = v(vi + x)
                val b = v(vj + x)
                v(vi + x) = c * a - s * b
                v(vj + x) = s * a + c * b
                x += 1
          j += 1
        i += 1
      sweep += 1
    if rotated then throw new LinAlgError("SVD did not converge")

  /** Fills the rows of `q` (`r x len`) not marked `good` so that all rows are orthonormal. */
  def completeRows(q: Array[Double], r: Int, len: Int, good: Array[Boolean]): Unit =
    var cand = 0
    var i = 0
    while i < r do
      if !good(i) then
        var found = false
        while !found do
          if cand >= len then throw new LinAlgError("SVD did not converge")
          val w = new Array[Double](len)
          w(cand) = 1.0
          cand += 1
          var pass = 0
          while pass < 2 do
            var k = 0
            while k < r do
              if good(k) then
                var d = 0.0
                var x = 0
                while x < len do
                  d += q(k * len + x) * w(x)
                  x += 1
                x = 0
                while x < len do
                  w(x) -= d * q(k * len + x)
                  x += 1
              k += 1
            pass += 1
          val nw = nrm2(w, 0, len, 1)
          if nw > 0.5 then
            var x = 0
            while x < len do
              q(i * len + x) = w(x) / nw
              x += 1
            good(i) = true
            found = true
      i += 1

  // ------------------------------------------------------------------ symmetric eigenproblem

  /** Eigen-decomposition of a symmetric matrix (`a` is symmetrised by the caller); returns ascending
    * eigenvalues and (if requested) the eigenvector matrix `V` (columns), row-major `n x n`.
    */
  def symEig(a: Array[Double], n: Int, wantV: Boolean): (Array[Double], Array[Double]) =
    if n == 0 then return (new Array[Double](0), new Array[Double](0))
    val v = a.clone()
    val d = new Array[Double](n)
    val e = new Array[Double](n)
    tred2(v, n, d, e)
    tql2(v, n, d, e, wantV)
    (d, v)

  /** Householder reduction to tridiagonal form (EISPACK tred2 / JAMA); `v` holds the transform on exit. */
  private def tred2(v: Array[Double], n: Int, d: Array[Double], e: Array[Double]): Unit =
    inline def V(i: Int, j: Int): Double = v(i * n + j)
    inline def setV(i: Int, j: Int, x: Double): Unit = v(i * n + j) = x
    var j = 0
    while j < n do
      d(j) = V(n - 1, j)
      j += 1
    var i = n - 1
    while i > 0 do
      var scale = 0.0
      var h = 0.0
      var k = 0
      while k < i do
        scale += math.abs(d(k))
        k += 1
      if scale == 0.0 then
        e(i) = d(i - 1)
        j = 0
        while j < i do
          d(j) = V(i - 1, j)
          setV(i, j, 0.0)
          setV(j, i, 0.0)
          j += 1
      else
        k = 0
        while k < i do
          d(k) /= scale
          h += d(k) * d(k)
          k += 1
        var f = d(i - 1)
        var g = math.sqrt(h)
        if f > 0 then g = -g
        e(i) = scale * g
        h = h - f * g
        d(i - 1) = f - g
        j = 0
        while j < i do
          e(j) = 0.0
          j += 1
        j = 0
        while j < i do
          f = d(j)
          setV(j, i, f)
          g = e(j) + V(j, j) * f
          k = j + 1
          while k <= i - 1 do
            g += V(k, j) * d(k)
            e(k) += V(k, j) * f
            k += 1
          e(j) = g
          j += 1
        f = 0.0
        j = 0
        while j < i do
          e(j) /= h
          f += e(j) * d(j)
          j += 1
        val hh = f / (h + h)
        j = 0
        while j < i do
          e(j) -= hh * d(j)
          j += 1
        j = 0
        while j < i do
          f = d(j)
          g = e(j)
          k = j
          while k <= i - 1 do
            setV(k, j, V(k, j) - (f * e(k) + g * d(k)))
            k += 1
          d(j) = V(i - 1, j)
          setV(i, j, 0.0)
          j += 1
      d(i) = h
      i -= 1
    // accumulate transformations
    i = 0
    while i < n - 1 do
      setV(n - 1, i, V(i, i))
      setV(i, i, 1.0)
      val h = d(i + 1)
      if h != 0.0 then
        var k = 0
        while k <= i do
          d(k) = V(k, i + 1) / h
          k += 1
        j = 0
        while j <= i do
          var g = 0.0
          k = 0
          while k <= i do
            g += V(k, i + 1) * V(k, j)
            k += 1
          k = 0
          while k <= i do
            setV(k, j, V(k, j) - g * d(k))
            k += 1
          j += 1
      var k = 0
      while k <= i do
        setV(k, i + 1, 0.0)
        k += 1
      i += 1
    j = 0
    while j < n do
      d(j) = V(n - 1, j)
      setV(n - 1, j, 0.0)
      j += 1
    setV(n - 1, n - 1, 1.0)
    e(0) = 0.0

  /** Implicit QL on a symmetric tridiagonal matrix (diagonal `d`, sub-diagonal `e(i) = T(i, i-1)`),
    * accumulating rotations into `v` (`n x n`, columns); eigenvalues are sorted ascending.
    */
  def tql2(v: Array[Double], n: Int, d: Array[Double], e: Array[Double], wantV: Boolean): Unit =
    var i = 1
    while i < n do
      e(i - 1) = e(i)
      i += 1
    e(n - 1) = 0.0
    var f = 0.0
    var tst1 = 0.0
    var l = 0
    while l < n do
      tst1 = math.max(tst1, math.abs(d(l)) + math.abs(e(l)))
      var m = l
      while m < n && !(math.abs(e(m)) <= eps * tst1) do m += 1
      if m >= n then m = n - 1
      if m > l then
        var iter = 0
        var going = true
        while going do
          iter += 1
          if iter > 60 then throw new LinAlgError("Eigenvalues did not converge")
          var g = d(l)
          var p = (d(l + 1) - g) / (2.0 * e(l))
          var r = math.hypot(p, 1.0)
          if p < 0 then r = -r
          d(l) = e(l) / (p + r)
          d(l + 1) = e(l) * (p + r)
          val dl1 = d(l + 1)
          var h = g - d(l)
          i = l + 2
          while i < n do
            d(i) -= h
            i += 1
          f += h
          p = d(m)
          var c = 1.0
          var c2 = c
          var c3 = c
          val el1 = e(l + 1)
          var s = 0.0
          var s2 = 0.0
          i = m - 1
          while i >= l do
            c3 = c2
            c2 = c
            s2 = s
            g = c * e(i)
            h = c * p
            r = math.hypot(p, e(i))
            e(i + 1) = s * r
            s = e(i) / r
            c = p / r
            p = c * d(i) - s * g
            d(i + 1) = h + s * (c * g + s * d(i))
            if wantV then
              var k = 0
              while k < n do
                val hv = v(k * n + i + 1)
                v(k * n + i + 1) = s * v(k * n + i) + c * hv
                v(k * n + i) = c * v(k * n + i) - s * hv
                k += 1
            i -= 1
          p = -s * s2 * c3 * el1 * e(l) / dl1
          e(l) = s * p
          d(l) = c * p
          going = math.abs(e(l)) > eps * tst1
      d(l) = d(l) + f
      e(l) = 0.0
      l += 1
    // sort ascending (selection sort keeps the pairing with columns of v)
    i = 0
    while i < n - 1 do
      var k = i
      var p = d(i)
      var j = i + 1
      while j < n do
        if d(j) < p then
          k = j
          p = d(j)
        j += 1
      if k != i then
        d(k) = d(i)
        d(i) = p
        if wantV then
          j = 0
          while j < n do
            val t = v(j * n + i)
            v(j * n + i) = v(j * n + k)
            v(j * n + k) = t
            j += 1
      i += 1
