package numscala

/** A row-major complex matrix stored as separate real / imaginary buffers. */
private[numscala] final class ZMat(val m: Int, val n: Int, val re: Array[Double], val im: Array[Double]):
  def copy(): ZMat = ZMat(m, n, re.clone(), im.clone())

private[numscala] object ZMat:
  def zeros(m: Int, n: Int): ZMat = ZMat(m, n, new Array[Double](m * n), new Array[Double](m * n))
  def identity(n: Int): ZMat =
    val z = zeros(n, n)
    var i = 0
    while i < n do
      z.re(i * n + i) = 1.0
      i += 1
    z

  /** Conjugate transpose. */
  def adjoint(a: ZMat): ZMat =
    val out = zeros(a.n, a.m)
    var i = 0
    while i < a.m do
      var j = 0
      while j < a.n do
        out.re(j * a.m + i) = a.re(i * a.n + j)
        out.im(j * a.m + i) = -a.im(i * a.n + j)
        j += 1
      i += 1
    out

  def mul(a: ZMat, b: ZMat): ZMat =
    val out = zeros(a.m, b.n)
    val k = a.n
    val n = b.n
    var i = 0
    while i < a.m do
      var p = 0
      while p < k do
        val ar = a.re(i * k + p)
        val ai = a.im(i * k + p)
        if ar != 0.0 || ai != 0.0 then
          var j = 0
          while j < n do
            val br = b.re(p * n + j)
            val bi = b.im(p * n + j)
            out.re(i * n + j) += ar * br - ai * bi
            out.im(i * n + j) += ar * bi + ai * br
            j += 1
        p += 1
      i += 1
    out

/** Dense complex128 kernels (LU, QR, Cholesky, Jacobi SVD, Hermitian eigensolver). */
private[numscala] object LinalgCplx:
  import LinalgReal.eps

  /** Norm of a strided complex vector. */
  def nrm2(re: Array[Double], im: Array[Double], off: Int, len: Int, st: Int): Double =
    var scale = 0.0
    var ssq = 1.0
    var i = 0
    var o = off
    while i < len do
      var t = 0
      while t < 2 do
        val v = if t == 0 then re(o) else im(o)
        if v != 0.0 then
          val av = math.abs(v)
          if scale < av then
            ssq = 1.0 + ssq * (scale / av) * (scale / av)
            scale = av
          else ssq += (av / scale) * (av / scale)
        else if v.isNaN then return Double.NaN
        t += 1
      i += 1
      o += st
    scale * math.sqrt(ssq)

  private inline def cabs1(r: Double, i: Double): Double = math.abs(r) + math.abs(i)

  // ------------------------------------------------------------------ LU

  final class LU(val a: ZMat, val perm: Array[Int], val sign: Int, val singular: Boolean)

  def lu(a: ZMat): LU =
    val n = a.n
    val re = a.re
    val im = a.im
    val perm = Array.range(0, n)
    var sign = 1
    var singular = false
    var k = 0
    while k < n do
      var p = k
      var mx = cabs1(re(k * n + k), im(k * n + k))
      var i = k + 1
      while i < n do
        val v = cabs1(re(i * n + k), im(i * n + k))
        if v > mx then
          mx = v
          p = i
        i += 1
      if p != k then
        var j = 0
        while j < n do
          var t = re(k * n + j); re(k * n + j) = re(p * n + j); re(p * n + j) = t
          t = im(k * n + j); im(k * n + j) = im(p * n + j); im(p * n + j) = t
          j += 1
        val t = perm(k); perm(k) = perm(p); perm(p) = t
        sign = -sign
      val pr = re(k * n + k)
      val pi = im(k * n + k)
      if pr == 0.0 && pi == 0.0 then singular = true
      else
        val pc = Complex(pr, pi)
        i = k + 1
        while i < n do
          val f = Complex(re(i * n + k), im(i * n + k)) / pc
          re(i * n + k) = f.re
          im(i * n + k) = f.im
          if f.re != 0.0 || f.im != 0.0 then
            var j = k + 1
            while j < n do
              val ur = re(k * n + j)
              val ui = im(k * n + j)
              re(i * n + j) -= f.re * ur - f.im * ui
              im(i * n + j) -= f.re * ui + f.im * ur
              j += 1
          i += 1
      k += 1
    LU(a, perm, sign, singular)

  def luSolve(f: LU, b: ZMat): ZMat =
    val n = f.a.n
    val nrhs = b.n
    val ar = f.a.re
    val ai = f.a.im
    val x = ZMat.zeros(n, nrhs)
    val xr = x.re
    val xi = x.im
    var i = 0
    while i < n do
      System.arraycopy(b.re, f.perm(i) * nrhs, xr, i * nrhs, nrhs)
      System.arraycopy(b.im, f.perm(i) * nrhs, xi, i * nrhs, nrhs)
      i += 1
    i = 0
    while i < n do
      var j = 0
      while j < i do
        val lr = ar(i * n + j)
        val li = ai(i * n + j)
        if lr != 0.0 || li != 0.0 then
          var c = 0
          while c < nrhs do
            val yr = xr(j * nrhs + c)
            val yi = xi(j * nrhs + c)
            xr(i * nrhs + c) -= lr * yr - li * yi
            xi(i * nrhs + c) -= lr * yi + li * yr
            c += 1
        j += 1
      i += 1
    i = n - 1
    while i >= 0 do
      var j = i + 1
      while j < n do
        val ur = ar(i * n + j)
        val ui = ai(i * n + j)
        if ur != 0.0 || ui != 0.0 then
          var c = 0
          while c < nrhs do
            val yr = xr(j * nrhs + c)
            val yi = xi(j * nrhs + c)
            xr(i * nrhs + c) -= ur * yr - ui * yi
            xi(i * nrhs + c) -= ur * yi + ui * yr
            c += 1
        j += 1
      val d = Complex(ar(i * n + i), ai(i * n + i))
      var c = 0
      while c < nrhs do
        val q = Complex(xr(i * nrhs + c), xi(i * nrhs + c)) / d
        xr(i * nrhs + c) = q.re
        xi(i * nrhs + c) = q.im
        c += 1
      i -= 1
    x

  // ------------------------------------------------------------------ Householder reflector (zlarfg)

  /** Generates `H = I - tau v v^H` with `H^H [alpha; x] = [beta; 0]` (beta real).  The vector
    * `x` (elements `off + t*st`, `t < len`) is overwritten with `v(1:)`; returns `(tauR, tauI, beta)`.
    */
  def larfg(re: Array[Double], im: Array[Double], aOff: Int, off: Int, len: Int, st: Int): (Double, Double, Double) =
    val ar = re(aOff)
    val ai = im(aOff)
    val xnorm = if len > 0 then nrm2(re, im, off, len, st) else 0.0
    if xnorm == 0.0 && ai == 0.0 then (0.0, 0.0, ar)
    else
      val beta = -math.copySign(math.hypot(math.hypot(ar, ai), xnorm), ar)
      val tr = (beta - ar) / beta
      val ti = -ai / beta
      val scal = Complex.One / Complex(ar - beta, ai)
      var t = 0
      var o = off
      while t < len do
        val xr = re(o)
        val xi = im(o)
        re(o) = xr * scal.re - xi * scal.im
        im(o) = xr * scal.im + xi * scal.re
        t += 1
        o += st
      re(aOff) = beta
      im(aOff) = 0.0
      (tr, ti, beta)

  // ------------------------------------------------------------------ QR

  /** `zgeqrf` in place; returns tau as (re, im). */
  def geqrf(a: ZMat): (Array[Double], Array[Double]) =
    val m = a.m
    val n = a.n
    val re = a.re
    val im = a.im
    val k = math.min(m, n)
    val taur = new Array[Double](k)
    val taui = new Array[Double](k)
    var j = 0
    while j < k do
      val (tr, ti, _) = larfg(re, im, j * n + j, (j + 1) * n + j, m - j - 1, n)
      taur(j) = tr
      taui(j) = ti
      if tr != 0.0 || ti != 0.0 then
        // apply H^H = I - conj(tau) v v^H to A(j:m, j+1:n)
        var c = j + 1
        while c < n do
          var sr = re(j * n + c)
          var si = im(j * n + c)
          var i = j + 1
          while i < m do
            val vr = re(i * n + j)
            val vi = im(i * n + j)
            val yr = re(i * n + c)
            val yi = im(i * n + c)
            sr += vr * yr + vi * yi
            si += vr * yi - vi * yr
            i += 1
          // s *= conj(tau)
          val pr = sr * tr + si * ti
          val pi = si * tr - sr * ti
          re(j * n + c) -= pr
          im(j * n + c) -= pi
          i = j + 1
          while i < m do
            val vr = re(i * n + j)
            val vi = im(i * n + j)
            re(i * n + c) -= vr * pr - vi * pi
            im(i * n + c) -= vr * pi + vi * pr
            i += 1
          c += 1
      j += 1
    (taur, taui)

  /** First `qc` columns of `Q = H(0) ... H(k-1)` (`zungqr`). */
  def ungqr(h: ZMat, taur: Array[Double], taui: Array[Double], qc: Int): ZMat =
    val m = h.m
    val n = h.n
    val k = taur.length
    val q = ZMat.zeros(m, qc)
    var i = 0
    while i < math.min(m, qc) do
      q.re(i * qc + i) = 1.0
      i += 1
    var j = k - 1
    while j >= 0 do
      val tr = taur(j)
      val ti = taui(j)
      if tr != 0.0 || ti != 0.0 then
        var c = j
        while c < qc do
          var sr = q.re(j * qc + c)
          var si = q.im(j * qc + c)
          i = j + 1
          while i < m do
            val vr = h.re(i * n + j)
            val vi = h.im(i * n + j)
            val yr = q.re(i * qc + c)
            val yi = q.im(i * qc + c)
            sr += vr * yr + vi * yi
            si += vr * yi - vi * yr
            i += 1
          val pr = sr * tr - si * ti
          val pi = sr * ti + si * tr
          q.re(j * qc + c) -= pr
          q.im(j * qc + c) -= pi
          i = j + 1
          while i < m do
            val vr = h.re(i * n + j)
            val vi = h.im(i * n + j)
            q.re(i * qc + c) -= vr * pr - vi * pi
            q.im(i * qc + c) -= vr * pi + vi * pr
            i += 1
          c += 1
      j -= 1
    q

  def extractR(h: ZMat, rows: Int): ZMat =
    val n = h.n
    val r = ZMat.zeros(rows, n)
    var i = 0
    while i < math.min(rows, h.m) do
      var j = i
      while j < n do
        r.re(i * n + j) = h.re(i * n + j)
        r.im(i * n + j) = h.im(i * n + j)
        j += 1
      i += 1
    r

  // ------------------------------------------------------------------ Cholesky

  def cholesky(a: ZMat): ZMat =
    val n = a.n
    val l = ZMat.zeros(n, n)
    val lr = l.re
    val li = l.im
    var j = 0
    while j < n do
      var s = a.re(j * n + j)
      var k = 0
      while k < j do
        s -= lr(j * n + k) * lr(j * n + k) + li(j * n + k) * li(j * n + k)
        k += 1
      if !(s > 0.0) then throw new LinAlgError("Matrix is not positive definite")
      val d = math.sqrt(s)
      lr(j * n + j) = d
      var i = j + 1
      while i < n do
        var tr = a.re(i * n + j)
        var ti = a.im(i * n + j)
        k = 0
        while k < j do
          // L(i,k) * conj(L(j,k))
          val xr = lr(i * n + k)
          val xi = li(i * n + k)
          val yr = lr(j * n + k)
          val yi = -li(j * n + k)
          tr -= xr * yr - xi * yi
          ti -= xr * yi + xi * yr
          k += 1
        lr(i * n + j) = tr / d
        li(i * n + j) = ti / d
        i += 1
      j += 1
    l

  // ------------------------------------------------------------------ SVD (one-sided Jacobi)

  final class SVD(val u: ZMat, val s: Array[Double], val vh: ZMat)

  def svd(a: ZMat, computeUV: Boolean, full: Boolean): SVD =
    if a.re.exists(v => v.isNaN || v.isInfinite) || a.im.exists(v => v.isNaN || v.isInfinite) then
      throw new LinAlgError("SVD did not converge")
    var amax = 0.0
    a.re.foreach(v => amax = math.max(amax, math.abs(v)))
    a.im.foreach(v => amax = math.max(amax, math.abs(v)))
    if amax > 0.0 && (amax > 1e150 || amax < 1e-150) then
      val f = java.lang.Math.scalb(1.0, -java.lang.Math.getExponent(amax))
      val r = svdCore(ZMat(a.m, a.n, a.re.map(_ * f), a.im.map(_ * f)), computeUV, full)
      SVD(r.u, r.s.map(_ / f), r.vh)
    else svdCore(a, computeUV, full)

  private def svdCore(a: ZMat, computeUV: Boolean, full: Boolean): SVD =
    val m = a.m
    val n = a.n
    val k = math.min(m, n)
    if k == 0 then
      return SVD(
        if computeUV then (if full then ZMat.identity(m) else ZMat.zeros(m, 0)) else null,
        new Array[Double](0),
        if computeUV then (if full then ZMat.identity(n) else ZMat.zeros(0, n)) else null
      )
    if m < n then
      val r = svdCore(ZMat.adjoint(a), computeUV, full)
      // A^H = U' S V'^H  =>  A = V' S U'^H
      if !computeUV then return SVD(null, r.s, null)
      return SVD(ZMat.adjoint(r.vh), r.s, ZMat.adjoint(r.u))
    if m > n then
      val h = a.copy()
      val (tr, ti) = geqrf(h)
      val inner = svdCore(extractR(h, n), computeUV, true)
      if !computeUV then return SVD(null, inner.s, null)
      val qc = if full then m else n
      val q = ungqr(h, tr, ti, qc)
      val u = ZMat.zeros(m, qc)
      val qn = ZMat.zeros(m, n)
      var i = 0
      while i < m do
        System.arraycopy(q.re, i * qc, qn.re, i * n, n)
        System.arraycopy(q.im, i * qc, qn.im, i * n, n)
        i += 1
      val un = ZMat.mul(qn, inner.u)
      i = 0
      while i < m do
        System.arraycopy(un.re, i * n, u.re, i * qc, n)
        System.arraycopy(un.im, i * n, u.im, i * qc, n)
        var j = n
        while j < qc do
          u.re(i * qc + j) = q.re(i * qc + j)
          u.im(i * qc + j) = q.im(i * qc + j)
          j += 1
        i += 1
      return SVD(u, inner.s, inner.vh)
    // square: columns of A as rows of g
    val g = ZMat.adjoint(a) // rows = conj(columns)
    var t = 0
    while t < g.im.length do
      g.im(t) = -g.im(t)
      t += 1
    val v = if computeUV then ZMat.identity(n) else null
    jacobi(g, v)
    val s = Array.tabulate(n)(i => nrm2(g.re, g.im, i * m, m, 1))
    val order = (0 until n).sortBy(j => -s(j)).toArray
    val ss = order.map(s(_))
    if !computeUV then return SVD(null, ss, null)
    val ut = ZMat.zeros(n, m) // rows = columns of U
    val vh = ZMat.zeros(n, n) // rows = conj(columns of V)
    val good = new Array[Boolean](n)
    var i = 0
    while i < n do
      val j = order(i)
      var x = 0
      while x < n do
        vh.re(i * n + x) = v.re(j * n + x)
        vh.im(i * n + x) = -v.im(j * n + x)
        x += 1
      val sv = ss(i)
      if sv > 1e-300 then
        good(i) = true
        x = 0
        while x < m do
          ut.re(i * m + x) = g.re(j * m + x) / sv
          ut.im(i * m + x) = g.im(j * m + x) / sv
          x += 1
      i += 1
    completeRows(ut, good)
    // U = transpose of ut (not conjugate)
    val u = ZMat.zeros(m, n)
    i = 0
    while i < n do
      var x = 0
      while x < m do
        u.re(x * n + i) = ut.re(i * m + x)
        u.im(x * n + i) = ut.im(i * m + x)
        x += 1
      i += 1
    SVD(u, ss, vh)

  /** One-sided complex Jacobi on the rows of `g`; rotations mirrored onto rows of `v`. */
  private def jacobi(g: ZMat, v: ZMat): Unit =
    val nc = g.m
    val len = g.n
    val gr = g.re
    val gi = g.im
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
          var cr = 0.0
          var ci = 0.0
          val oi = i * len
          val oj = j * len
          var x = 0
          while x < len do
            val ar = gr(oi + x)
            val ai = gi(oi + x)
            val br = gr(oj + x)
            val bi = gi(oj + x)
            alpha += ar * ar + ai * ai
            beta += br * br + bi * bi
            // conj(a) * b
            cr += ar * br + ai * bi
            ci += ar * bi - ai * br
            x += 1
          val gam = math.hypot(cr, ci)
          if alpha > 0.0 && beta > 0.0 && gam > tol * math.sqrt(alpha) * math.sqrt(beta) then
            rotated = true
            val er = cr / gam
            val ei = ci / gam
            val zeta = (beta - alpha) / (2.0 * gam)
            val t =
              if math.abs(zeta) > 1e150 then 0.5 / zeta
              else (if zeta >= 0 then 1.0 else -1.0) / (math.abs(zeta) + math.sqrt(1.0 + zeta * zeta))
            val c = 1.0 / math.sqrt(1.0 + t * t)
            val s = c * t
            rot(gr, gi, oi, oj, len, c, s, er, ei)
            if v != null then rot(v.re, v.im, i * nc, j * nc, nc, c, s, er, ei)
          j += 1
        i += 1
      sweep += 1
    if rotated then throw new LinAlgError("SVD did not converge")

  /** `a' = c a - s conj(e) b`, `b' = s e a + c b`. */
  private def rot(re: Array[Double], im: Array[Double], oa: Int, ob: Int, len: Int, c: Double, s: Double, er: Double, ei: Double): Unit =
    var x = 0
    while x < len do
      val ar = re(oa + x)
      val ai = im(oa + x)
      val br = re(ob + x)
      val bi = im(ob + x)
      // conj(e) * b
      val cbr = er * br + ei * bi
      val cbi = er * bi - ei * br
      // e * a
      val ear = er * ar - ei * ai
      val eai = er * ai + ei * ar
      re(oa + x) = c * ar - s * cbr
      im(oa + x) = c * ai - s * cbi
      re(ob + x) = s * ear + c * br
      im(ob + x) = s * eai + c * bi
      x += 1

  /** Completes rows of `q` not marked `good` to an orthonormal set. */
  def completeRows(q: ZMat, good: Array[Boolean]): Unit =
    val r = q.m
    val len = q.n
    var cand = 0
    var i = 0
    while i < r do
      if !good(i) then
        var found = false
        while !found do
          if cand >= len then throw new LinAlgError("SVD did not converge")
          val wr = new Array[Double](len)
          val wi = new Array[Double](len)
          wr(cand) = 1.0
          cand += 1
          var pass = 0
          while pass < 2 do
            var k = 0
            while k < r do
              if good(k) then
                // d = q_k^H w
                var dr = 0.0
                var di = 0.0
                var x = 0
                while x < len do
                  val qr = q.re(k * len + x)
                  val qi = q.im(k * len + x)
                  dr += qr * wr(x) + qi * wi(x)
                  di += qr * wi(x) - qi * wr(x)
                  x += 1
                x = 0
                while x < len do
                  val qr = q.re(k * len + x)
                  val qi = q.im(k * len + x)
                  wr(x) -= dr * qr - di * qi
                  wi(x) -= dr * qi + di * qr
                  x += 1
              k += 1
            pass += 1
          val nw = nrm2(wr, wi, 0, len, 1)
          if nw > 0.5 then
            var x = 0
            while x < len do
              q.re(i * len + x) = wr(x) / nw
              q.im(i * len + x) = wi(x) / nw
              x += 1
            good(i) = true
            found = true
      i += 1

  // ------------------------------------------------------------------ Hermitian eigenproblem

  /** Eigen-decomposition of a Hermitian matrix (full storage, already Hermitian). Returns ascending
    * eigenvalues and the eigenvectors as columns (null when not requested).
    */
  def hermEig(a0: ZMat, wantV: Boolean): (Array[Double], ZMat) =
    val n = a0.n
    if n == 0 then return (new Array[Double](0), ZMat.zeros(0, 0))
    val a = a0.copy()
    val re = a.re
    val im = a.im
    val q = ZMat.identity(n)
    var k = 0
    while k < n - 1 do
      val len = n - k - 2
      val (tr, ti, beta) = larfg(re, im, (k + 1) * n + k, (k + 2) * n + k, len, n)
      if tr != 0.0 || ti != 0.0 then
        val vr = new Array[Double](n)
        val vi = new Array[Double](n)
        vr(k + 1) = 1.0
        var i = k + 2
        while i < n do
          vr(i) = re(i * n + k)
          vi(i) = im(i * n + k)
          i += 1
        // restore column k (it will be overwritten explicitly)
        re((k + 1) * n + k) = beta
        im((k + 1) * n + k) = 0.0
        // left: H^H A on rows k+1.., columns k+1..n-1
        applyLeft(a, vr, vi, k + 1, k + 1, tr, -ti)
        // right: A H on columns k+1.., rows k+1..n-1
        applyRight(a, vr, vi, k + 1, k + 1, tr, ti)
        applyRight(q, vr, vi, k + 1, 0, tr, ti)
      // column k / row k become (beta, 0, ..., 0)
      var i = k + 2
      while i < n do
        re(i * n + k) = 0.0; im(i * n + k) = 0.0
        re(k * n + i) = 0.0; im(k * n + i) = 0.0
        i += 1
      re((k + 1) * n + k) = beta; im((k + 1) * n + k) = 0.0
      re(k * n + k + 1) = beta; im(k * n + k + 1) = 0.0
      k += 1
    val d = Array.tabulate(n)(i => re(i * n + i))
    val e = Array.tabulate(n)(i => if i == 0 then 0.0 else re(i * n + i - 1))
    val z = LinalgReal.identity(n)
    LinalgReal.tql2(z, n, d, e, wantV)
    if !wantV then return (d, null)
    // V = Q Z
    val v = ZMat.zeros(n, n)
    var i = 0
    while i < n do
      var p = 0
      while p < n do
        val qr = q.re(i * n + p)
        val qi = q.im(i * n + p)
        if qr != 0.0 || qi != 0.0 then
          var j = 0
          while j < n do
            val zz = z(p * n + j)
            v.re(i * n + j) += qr * zz
            v.im(i * n + j) += qi * zz
            j += 1
        p += 1
      i += 1
    (d, v)

  /** `A(r0:, c0:) <- (I - t v v^H) A(r0:, c0:)` where `v` is indexed by row (zero below r0). */
  def applyLeft(a: ZMat, vr: Array[Double], vi: Array[Double], r0: Int, c0: Int, tr: Double, ti: Double): Unit =
    val n = a.n
    val m = a.m
    var c = c0
    while c < n do
      var sr = 0.0
      var si = 0.0
      var i = r0
      while i < m do
        val yr = a.re(i * n + c)
        val yi = a.im(i * n + c)
        sr += vr(i) * yr + vi(i) * yi
        si += vr(i) * yi - vi(i) * yr
        i += 1
      val pr = sr * tr - si * ti
      val pi = sr * ti + si * tr
      i = r0
      while i < m do
        a.re(i * n + c) -= vr(i) * pr - vi(i) * pi
        a.im(i * n + c) -= vr(i) * pi + vi(i) * pr
        i += 1
      c += 1

  /** `A(r0:, c0:) <- A(r0:, c0:) (I - t v v^H)` where `v` is indexed by column (zero below c0). */
  def applyRight(a: ZMat, vr: Array[Double], vi: Array[Double], c0: Int, r0: Int, tr: Double, ti: Double): Unit =
    val n = a.n
    var r = r0
    while r < a.m do
      var sr = 0.0
      var si = 0.0
      var j = c0
      while j < n do
        val xr = a.re(r * n + j)
        val xi = a.im(r * n + j)
        sr += xr * vr(j) - xi * vi(j)
        si += xr * vi(j) + xi * vr(j)
        j += 1
      val pr = sr * tr - si * ti
      val pi = sr * ti + si * tr
      j = c0
      while j < n do
        // p * conj(v_j)
        a.re(r * n + j) -= pr * vr(j) + pi * vi(j)
        a.im(r * n + j) -= pi * vr(j) - pr * vi(j)
        j += 1
      r += 1
