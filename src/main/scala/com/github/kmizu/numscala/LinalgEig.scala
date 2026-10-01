package com.github.kmizu.numscala

/** Nonsymmetric eigenproblem: balancing, Hessenberg reduction and shifted QR.
  *
  * Real matrices use the Francis double-shift algorithm (EISPACK `orthes`/`hqr2`), so real
  * eigenvalues come out exactly real and complex ones in exact conjugate pairs (positive
  * imaginary part first), like LAPACK `dgeev`. Complex matrices use a complex Hessenberg
  * reduction followed by single-shift complex QR to Schur form (like `zgeev`).
  * Eigenvectors are normalised to unit Euclidean norm with their largest component real.
  */
private[numscala] object LinalgEig:
  import LinalgReal.eps

  /** Eigenvalues `(re, im)` and eigenvectors (columns of a `ZMat`, null when not requested). */
  final class Result(val wr: Array[Double], val wi: Array[Double], val v: ZMat)

  // ------------------------------------------------------------------ balancing (scaling only)

  /** EISPACK `balanc` without permutations; returns the scale factors. `abs1(i, j)` gives |a_ij|. */
  private def balance(n: Int, abs1: (Int, Int) => Double, scaleRow: (Int, Double) => Unit, scaleCol: (Int, Double) => Unit): Array[Double] =
    val scale = Array.fill(n)(1.0)
    val radix = 2.0
    val sqrdx = 4.0
    var noconv = true
    var iter = 0
    while noconv && iter < 100 do
      noconv = false
      iter += 1
      var i = 0
      while i < n do
        var c = 0.0
        var r = 0.0
        var j = 0
        while j < n do
          if j != i then
            c += abs1(j, i)
            r += abs1(i, j)
          j += 1
        if c != 0.0 && r != 0.0 && !c.isNaN && !r.isNaN && !c.isInfinite && !r.isInfinite then
          var g = r / radix
          var f = 1.0
          val s = c + r
          while c < g do
            f *= radix
            c *= sqrdx
          g = r * radix
          while c > g do
            f /= radix
            c /= sqrdx
          if (c + r) / f < 0.95 * s then
            noconv = true
            scale(i) *= f
            scaleRow(i, 1.0 / f)
            scaleCol(i, f)
        i += 1
    scale

  /** Power-of-two factor bringing a matrix with max-abs `amax` into a safe range (1 if not needed). */
  private def rangeScale(amax: Double): Double =
    if amax > 0.0 && (amax > 1e150 || amax < 1e-150) then java.lang.Math.scalb(1.0, -java.lang.Math.getExponent(amax))
    else 1.0

  // ------------------------------------------------------------------ real: orthes + hqr2

  def realEig(a: Array[Double], n: Int, wantV: Boolean): Result =
    if n == 0 then return Result(new Array[Double](0), new Array[Double](0), ZMat.zeros(0, 0))
    val f = rangeScale(a.foldLeft(0.0)((m, v) => math.max(m, math.abs(v))))
    val h = if f == 1.0 then a.clone() else a.map(_ * f)
    val scale = balance(
      n,
      (i, j) => math.abs(h(i * n + j)),
      (i, f) => { var j = 0; while j < n do { h(i * n + j) *= f; j += 1 } },
      (j, f) => { var i = 0; while i < n do { h(i * n + j) *= f; i += 1 } }
    )
    val v = new Array[Double](n * n)
    orthes(h, v, n)
    val d = new Array[Double](n)
    val e = new Array[Double](n)
    hqr2(h, v, n, d, e)
    if f != 1.0 then
      var i = 0
      while i < n do
        d(i) /= f
        e(i) /= f
        i += 1
    if !wantV then return Result(d, e, null)
    // back-transform the balancing and build complex vectors
    val z = ZMat.zeros(n, n)
    var j = 0
    while j < n do
      if e(j) == 0.0 then
        var i = 0
        while i < n do
          z.re(i * n + j) = v(i * n + j) * scale(i)
          i += 1
        j += 1
      else
        // pair (j, j+1): eigenvector of d(j) + i e(j) is v_j + i v_{j+1}
        var i = 0
        while i < n do
          val xr = v(i * n + j) * scale(i)
          val xi = v(i * n + j + 1) * scale(i)
          z.re(i * n + j) = xr
          z.im(i * n + j) = xi
          z.re(i * n + j + 1) = xr
          z.im(i * n + j + 1) = -xi
          i += 1
        j += 2
    normalizeColumns(z, e)
    Result(d, e, z)

  private def orthes(h: Array[Double], v: Array[Double], n: Int): Unit =
    val ort = new Array[Double](n)
    val low = 0
    val high = n - 1
    var m = low + 1
    while m <= high - 1 do
      var scale = 0.0
      var i = m
      while i <= high do
        scale += math.abs(h(i * n + m - 1))
        i += 1
      if scale != 0.0 then
        var hh = 0.0
        i = high
        while i >= m do
          ort(i) = h(i * n + m - 1) / scale
          hh += ort(i) * ort(i)
          i -= 1
        var g = math.sqrt(hh)
        if ort(m) > 0 then g = -g
        hh = hh - ort(m) * g
        ort(m) = ort(m) - g
        var j = m
        while j < n do
          var f = 0.0
          i = high
          while i >= m do
            f += ort(i) * h(i * n + j)
            i -= 1
          f = f / hh
          i = m
          while i <= high do
            h(i * n + j) -= f * ort(i)
            i += 1
          j += 1
        i = 0
        while i <= high do
          var f = 0.0
          j = high
          while j >= m do
            f += ort(j) * h(i * n + j)
            j -= 1
          f = f / hh
          j = m
          while j <= high do
            h(i * n + j) -= f * ort(j)
            j += 1
          i += 1
        ort(m) = scale * ort(m)
        h(m * n + m - 1) = scale * g
      m += 1
    var i = 0
    while i < n do
      var j = 0
      while j < n do
        v(i * n + j) = if i == j then 1.0 else 0.0
        j += 1
      i += 1
    m = high - 1
    while m >= low + 1 do
      if h(m * n + m - 1) != 0.0 then
        i = m + 1
        while i <= high do
          ort(i) = h(i * n + m - 1)
          i += 1
        var j = m
        while j <= high do
          var g = 0.0
          i = m
          while i <= high do
            g += ort(i) * v(i * n + j)
            i += 1
          g = (g / ort(m)) / h(m * n + m - 1)
          i = m
          while i <= high do
            v(i * n + j) += g * ort(i)
            i += 1
          j += 1
      m -= 1
    // clear the Householder vectors stored below the sub-diagonal
    i = 2
    while i < n do
      var j = 0
      while j < i - 1 do
        h(i * n + j) = 0.0
        j += 1
      i += 1

  private def cdiv(xr: Double, xi: Double, yr: Double, yi: Double): (Double, Double) =
    if math.abs(yr) > math.abs(yi) then
      val r = yi / yr
      val d = yr + r * yi
      ((xr + r * xi) / d, (xi - r * xr) / d)
    else
      val r = yr / yi
      val d = yi + r * yr
      ((r * xr + xi) / d, (r * xi - xr) / d)

  private def hqr2(hm: Array[Double], vm: Array[Double], nn: Int, d: Array[Double], e: Array[Double]): Unit =
    inline def H(i: Int, j: Int): Double = hm(i * nn + j)
    inline def setH(i: Int, j: Int, x: Double): Unit = hm(i * nn + j) = x
    inline def V(i: Int, j: Int): Double = vm(i * nn + j)
    inline def setV(i: Int, j: Int, x: Double): Unit = vm(i * nn + j) = x
    var n = nn - 1
    val low = 0
    val high = nn - 1
    var exshift = 0.0
    var p = 0.0
    var q = 0.0
    var r = 0.0
    var s = 0.0
    var z = 0.0
    var t = 0.0
    var w = 0.0
    var x = 0.0
    var y = 0.0
    var norm = 0.0
    var i = 0
    while i < nn do
      var j = math.max(i - 1, 0)
      while j < nn do
        norm += math.abs(H(i, j))
        j += 1
      i += 1
    var iter = 0
    var totalIter = 0
    while n >= low do
      var l = n
      var searching = true
      while searching && l > low do
        s = math.abs(H(l - 1, l - 1)) + math.abs(H(l, l))
        if s == 0.0 then s = norm
        if math.abs(H(l, l - 1)) < eps * s then searching = false
        else l -= 1
      if l == n then
        setH(n, n, H(n, n) + exshift)
        d(n) = H(n, n)
        e(n) = 0.0
        n -= 1
        iter = 0
      else if l == n - 1 then
        w = H(n, n - 1) * H(n - 1, n)
        p = (H(n - 1, n - 1) - H(n, n)) / 2.0
        q = p * p + w
        z = math.sqrt(math.abs(q))
        setH(n, n, H(n, n) + exshift)
        setH(n - 1, n - 1, H(n - 1, n - 1) + exshift)
        x = H(n, n)
        if q >= 0 then
          z = if p >= 0 then p + z else p - z
          d(n - 1) = x + z
          d(n) = d(n - 1)
          if z != 0.0 then d(n) = x - w / z
          e(n - 1) = 0.0
          e(n) = 0.0
          x = H(n, n - 1)
          s = math.abs(x) + math.abs(z)
          p = x / s
          q = z / s
          r = math.sqrt(p * p + q * q)
          p = p / r
          q = q / r
          var j = n - 1
          while j < nn do
            z = H(n - 1, j)
            setH(n - 1, j, q * z + p * H(n, j))
            setH(n, j, q * H(n, j) - p * z)
            j += 1
          i = 0
          while i <= n do
            z = H(i, n - 1)
            setH(i, n - 1, q * z + p * H(i, n))
            setH(i, n, q * H(i, n) - p * z)
            i += 1
          i = low
          while i <= high do
            z = V(i, n - 1)
            setV(i, n - 1, q * z + p * V(i, n))
            setV(i, n, q * V(i, n) - p * z)
            i += 1
        else
          d(n - 1) = x + p
          d(n) = x + p
          e(n - 1) = z
          e(n) = -z
        n -= 2
        iter = 0
      else
        x = H(n, n)
        y = 0.0
        w = 0.0
        if l < n then
          y = H(n - 1, n - 1)
          w = H(n, n - 1) * H(n - 1, n)
        if iter == 10 then
          exshift += x
          i = low
          while i <= n do
            setH(i, i, H(i, i) - x)
            i += 1
          s = math.abs(H(n, n - 1)) + math.abs(H(n - 1, n - 2))
          x = 0.75 * s
          y = x
          w = -0.4375 * s * s
        if iter == 30 then
          s = (y - x) / 2.0
          s = s * s + w
          if s > 0 then
            s = math.sqrt(s)
            if y < x then s = -s
            s = x - w / ((y - x) / 2.0 + s)
            i = low
            while i <= n do
              setH(i, i, H(i, i) - s)
              i += 1
            exshift += s
            x = 0.964
            y = x
            w = x
        iter += 1
        totalIter += 1
        if totalIter > 100 * nn then throw new LinAlgError("Eigenvalues did not converge")
        var m = n - 2
        var looking = true
        while looking && m >= l do
          z = H(m, m)
          r = x - z
          s = y - z
          p = (r * s - w) / H(m + 1, m) + H(m, m + 1)
          q = H(m + 1, m + 1) - z - r - s
          r = H(m + 2, m + 1)
          s = math.abs(p) + math.abs(q) + math.abs(r)
          p = p / s
          q = q / s
          r = r / s
          if m == l then looking = false
          else if math.abs(H(m, m - 1)) * (math.abs(q) + math.abs(r)) <
              eps * (math.abs(p) * (math.abs(H(m - 1, m - 1)) + math.abs(z) + math.abs(H(m + 1, m + 1))))
          then looking = false
          else m -= 1
        i = m + 2
        while i <= n do
          setH(i, i - 2, 0.0)
          if i > m + 2 then setH(i, i - 3, 0.0)
          i += 1
        var k = m
        while k <= n - 1 do
          val notlast = k != n - 1
          var skip = false
          if k != m then
            p = H(k, k - 1)
            q = H(k + 1, k - 1)
            r = if notlast then H(k + 2, k - 1) else 0.0
            x = math.abs(p) + math.abs(q) + math.abs(r)
            if x == 0.0 then skip = true
            else
              p = p / x
              q = q / x
              r = r / x
          if !skip then
            s = math.sqrt(p * p + q * q + r * r)
            if p < 0 then s = -s
            if s != 0 then
              if k != m then setH(k, k - 1, -s * x)
              else if l != m then setH(k, k - 1, -H(k, k - 1))
              p = p + s
              x = p / s
              y = q / s
              z = r / s
              q = q / p
              r = r / p
              var j = k
              while j < nn do
                p = H(k, j) + q * H(k + 1, j)
                if notlast then
                  p = p + r * H(k + 2, j)
                  setH(k + 2, j, H(k + 2, j) - p * z)
                setH(k, j, H(k, j) - p * x)
                setH(k + 1, j, H(k + 1, j) - p * y)
                j += 1
              i = 0
              while i <= math.min(n, k + 3) do
                p = x * H(i, k) + y * H(i, k + 1)
                if notlast then
                  p = p + z * H(i, k + 2)
                  setH(i, k + 2, H(i, k + 2) - p * r)
                setH(i, k, H(i, k) - p)
                setH(i, k + 1, H(i, k + 1) - p * q)
                i += 1
              i = low
              while i <= high do
                p = x * V(i, k) + y * V(i, k + 1)
                if notlast then
                  p = p + z * V(i, k + 2)
                  setV(i, k + 2, V(i, k + 2) - p * r)
                setV(i, k, V(i, k) - p)
                setV(i, k + 1, V(i, k + 1) - p * q)
                i += 1
          k += 1
    // back-substitution
    if norm == 0.0 then return
    n = nn - 1
    while n >= 0 do
      p = d(n)
      q = e(n)
      if q == 0 then
        var l = n
        setH(n, n, 1.0)
        i = n - 1
        while i >= 0 do
          w = H(i, i) - p
          r = 0.0
          var j = l
          while j <= n do
            r = r + H(i, j) * H(j, n)
            j += 1
          if e(i) < 0.0 then
            z = w
            s = r
          else
            l = i
            if e(i) == 0.0 then
              if w != 0.0 then setH(i, n, -r / w)
              else setH(i, n, -r / (eps * norm))
            else
              x = H(i, i + 1)
              y = H(i + 1, i)
              q = (d(i) - p) * (d(i) - p) + e(i) * e(i)
              t = (x * s - z * r) / q
              setH(i, n, t)
              if math.abs(x) > math.abs(z) then setH(i + 1, n, (-r - w * t) / x)
              else setH(i + 1, n, (-s - y * t) / z)
            t = math.abs(H(i, n))
            if (eps * t) * t > 1 then
              j = i
              while j <= n do
                setH(j, n, H(j, n) / t)
                j += 1
          i -= 1
      else if q < 0 then
        var l = n - 1
        if math.abs(H(n, n - 1)) > math.abs(H(n - 1, n)) then
          setH(n - 1, n - 1, q / H(n, n - 1))
          setH(n - 1, n, -(H(n, n) - p) / H(n, n - 1))
        else
          val (cr, ci) = cdiv(0.0, -H(n - 1, n), H(n - 1, n - 1) - p, q)
          setH(n - 1, n - 1, cr)
          setH(n - 1, n, ci)
        setH(n, n - 1, 0.0)
        setH(n, n, 1.0)
        i = n - 2
        while i >= 0 do
          var ra = 0.0
          var sa = 0.0
          var j = l
          while j <= n do
            ra = ra + H(i, j) * H(j, n - 1)
            sa = sa + H(i, j) * H(j, n)
            j += 1
          w = H(i, i) - p
          if e(i) < 0.0 then
            z = w
            r = ra
            s = sa
          else
            l = i
            if e(i) == 0 then
              val (cr, ci) = cdiv(-ra, -sa, w, q)
              setH(i, n - 1, cr)
              setH(i, n, ci)
            else
              x = H(i, i + 1)
              y = H(i + 1, i)
              var vr = (d(i) - p) * (d(i) - p) + e(i) * e(i) - q * q
              val vi = (d(i) - p) * 2.0 * q
              if vr == 0.0 && vi == 0.0 then
                vr = eps * norm * (math.abs(w) + math.abs(q) + math.abs(x) + math.abs(y) + math.abs(z))
              val (cr, ci) = cdiv(x * r - z * ra + q * sa, x * s - z * sa - q * ra, vr, vi)
              setH(i, n - 1, cr)
              setH(i, n, ci)
              if math.abs(x) > (math.abs(z) + math.abs(q)) then
                setH(i + 1, n - 1, (-ra - w * H(i, n - 1) + q * H(i, n)) / x)
                setH(i + 1, n, (-sa - w * H(i, n) - q * H(i, n - 1)) / x)
              else
                val (c2r, c2i) = cdiv(-r - y * H(i, n - 1), -s - y * H(i, n), z, q)
                setH(i + 1, n - 1, c2r)
                setH(i + 1, n, c2i)
            t = math.max(math.abs(H(i, n - 1)), math.abs(H(i, n)))
            if (eps * t) * t > 1 then
              j = i
              while j <= n do
                setH(j, n - 1, H(j, n - 1) / t)
                setH(j, n, H(j, n) / t)
                j += 1
          i -= 1
      n -= 1
    // back transformation
    var j = nn - 1
    while j >= low do
      i = low
      while i <= high do
        z = 0.0
        var k = low
        while k <= math.min(j, high) do
          z = z + V(i, k) * H(k, j)
          k += 1
        setV(i, j, z)
        i += 1
      j -= 1

  // ------------------------------------------------------------------ complex: Hessenberg + QR

  def complexEig(a0: ZMat, wantV: Boolean): Result =
    val n = a0.n
    if n == 0 then return Result(new Array[Double](0), new Array[Double](0), ZMat.zeros(0, 0))
    val f = rangeScale(math.max(a0.re.foldLeft(0.0)((m, v) => math.max(m, math.abs(v))), a0.im.foldLeft(0.0)((m, v) => math.max(m, math.abs(v)))))
    val h = if f == 1.0 then a0.copy() else ZMat(n, n, a0.re.map(_ * f), a0.im.map(_ * f))
    val hr = h.re
    val hi = h.im
    val scale = balance(
      n,
      (i, j) => math.abs(hr(i * n + j)) + math.abs(hi(i * n + j)),
      (i, f) => { var j = 0; while j < n do { hr(i * n + j) *= f; hi(i * n + j) *= f; j += 1 } },
      (j, f) => { var i = 0; while i < n do { hr(i * n + j) *= f; hi(i * n + j) *= f; i += 1 } }
    )
    val zq = ZMat.identity(n)
    // Hessenberg reduction
    var k = 0
    while k < n - 2 do
      val (tr, ti, beta) = LinalgCplx.larfg(hr, hi, (k + 1) * n + k, (k + 2) * n + k, n - k - 2, n)
      if tr != 0.0 || ti != 0.0 then
        val vr = new Array[Double](n)
        val vi = new Array[Double](n)
        vr(k + 1) = 1.0
        var i = k + 2
        while i < n do
          vr(i) = hr(i * n + k)
          vi(i) = hi(i * n + k)
          hr(i * n + k) = 0.0
          hi(i * n + k) = 0.0
          i += 1
        hr((k + 1) * n + k) = beta
        hi((k + 1) * n + k) = 0.0
        LinalgCplx.applyLeft(h, vr, vi, k + 1, k + 1, tr, -ti)
        LinalgCplx.applyRight(h, vr, vi, k + 1, 0, tr, ti)
        LinalgCplx.applyRight(zq, vr, vi, k + 1, 0, tr, ti)
      k += 1
    schurQR(h, zq)
    val wr = Array.tabulate(n)(i => hr(i * n + i) / f)
    val wi = Array.tabulate(n)(i => hi(i * n + i) / f)
    if !wantV then return Result(wr, wi, null)
    // eigenvectors of the triangular T, then back-transform
    val x = ZMat.zeros(n, n)
    var tnorm = 0.0
    var i = 0
    while i < n do
      var j = i
      while j < n do
        tnorm = math.max(tnorm, math.abs(hr(i * n + j)) + math.abs(hi(i * n + j)))
        j += 1
      i += 1
    val smlnum = java.lang.Double.MIN_NORMAL * (n / eps)
    k = 0
    while k < n do
      val lr = hr(k * n + k)
      val li = hi(k * n + k)
      val smin = math.max(eps * (math.abs(lr) + math.abs(li)), smlnum)
      x.re(k * n + k) = 1.0
      var j = k - 1
      while j >= 0 do
        var sr = 0.0
        var si = 0.0
        var m = j + 1
        while m <= k do
          val tr = hr(j * n + m)
          val ti = hi(j * n + m)
          val yr = x.re(m * n + k)
          val yi = x.im(m * n + k)
          sr += tr * yr - ti * yi
          si += tr * yi + ti * yr
          m += 1
        var dr = hr(j * n + j) - lr
        var di = hi(j * n + j) - li
        if math.abs(dr) + math.abs(di) < smin then
          dr = smin
          di = 0.0
        val q = Complex(-sr, -si) / Complex(dr, di)
        x.re(j * n + k) = q.re
        x.im(j * n + k) = q.im
        val mag = math.abs(q.re) + math.abs(q.im)
        if mag > 1e100 then
          m = j
          while m <= k do
            x.re(m * n + k) /= mag
            x.im(m * n + k) /= mag
            m += 1
        j -= 1
      k += 1
    val v = ZMat.mul(zq, x)
    i = 0
    while i < n do
      var j = 0
      while j < n do
        v.re(i * n + j) *= scale(i)
        v.im(i * n + j) *= scale(i)
        j += 1
      i += 1
    normalizeColumns(v, null)
    Result(wr, wi, v)

  /** Complex Givens rotation `[c s; -conj(s) c] [f; g] = [r; 0]`; returns (c, sr, si). */
  private def givens(fr: Double, fi: Double, gr: Double, gi: Double): (Double, Double, Double) =
    if gr == 0.0 && gi == 0.0 then (1.0, 0.0, 0.0)
    else if fr == 0.0 && fi == 0.0 then
      val ga = math.hypot(gr, gi)
      (0.0, gr / ga, -gi / ga)
    else
      val fa = math.hypot(fr, fi)
      val ga = math.hypot(gr, gi)
      val nrm = math.hypot(fa, ga)
      val c = fa / nrm
      // s = (f/|f|) conj(g) / nrm
      val ur = fr / fa
      val ui = fi / fa
      ((c), (ur * gr + ui * gi) / nrm, (ui * gr - ur * gi) / nrm)

  /** Reduces the upper Hessenberg `h` to upper triangular Schur form, accumulating into `z`. */
  private def schurQR(h: ZMat, z: ZMat): Unit =
    val n = h.n
    val hr = h.re
    val hi = h.im
    inline def a1(i: Int, j: Int): Double = math.abs(hr(i * n + j)) + math.abs(hi(i * n + j))
    var hiIdx = n - 1
    var its = 0
    var total = 0
    while hiIdx > 0 do
      var l = hiIdx
      var searching = true
      while searching && l > 0 do
        var tst = a1(l - 1, l - 1) + a1(l, l)
        if tst == 0.0 then
          var m = 0
          while m <= hiIdx do
            tst += a1(m, math.max(m - 1, 0))
            m += 1
        if a1(l, l - 1) <= eps * tst then
          hr(l * n + l - 1) = 0.0
          hi(l * n + l - 1) = 0.0
          searching = false
        else l -= 1
      if l == hiIdx then
        hiIdx -= 1
        its = 0
      else
        total += 1
        if total > 60 * n then throw new LinAlgError("Eigenvalues did not converge")
        val mu: Complex =
          if its == 10 then Complex(hr(l * n + l), hi(l * n + l)) + Complex(0.75 * math.abs(hr((l + 1) * n + l)), 0.0)
          else if its == 20 then
            Complex(hr(hiIdx * n + hiIdx), hi(hiIdx * n + hiIdx)) + Complex(0.75 * math.abs(hr(hiIdx * n + hiIdx - 1)), 0.0)
          else
            val a = Complex(hr((hiIdx - 1) * n + hiIdx - 1), hi((hiIdx - 1) * n + hiIdx - 1))
            val b = Complex(hr((hiIdx - 1) * n + hiIdx), hi((hiIdx - 1) * n + hiIdx))
            val c = Complex(hr(hiIdx * n + hiIdx - 1), hi(hiIdx * n + hiIdx - 1))
            val d = Complex(hr(hiIdx * n + hiIdx), hi(hiIdx * n + hiIdx))
            val half = (a - d) * 0.5
            val disc = (half * half + b * c).sqrt
            val m1 = d - half + disc // (a+d)/2 + disc  ... computed relative to d
            val e1 = (a + d) * 0.5 + disc
            val e2 = (a + d) * 0.5 - disc
            val _ = m1
            if (e1 - d).abs <= (e2 - d).abs then e1 else e2
        its += 1
        var k = l
        var xr = hr(l * n + l) - mu.re
        var xi = hi(l * n + l) - mu.im
        var yr = hr((l + 1) * n + l)
        var yi = hi((l + 1) * n + l)
        while k < hiIdx do
          if k > l then
            xr = hr(k * n + k - 1)
            xi = hi(k * n + k - 1)
            yr = hr((k + 1) * n + k - 1)
            yi = hi((k + 1) * n + k - 1)
          val (c, sr, si) = givens(xr, xi, yr, yi)
          // rows k, k+1 (columns from max(k-1, l) to n-1)
          var j = if k > l then k - 1 else l
          while j < n do
            val ar = hr(k * n + j)
            val ai = hi(k * n + j)
            val br = hr((k + 1) * n + j)
            val bi = hi((k + 1) * n + j)
            // a' = c a + s b ; b' = -conj(s) a + c b
            hr(k * n + j) = c * ar + (sr * br - si * bi)
            hi(k * n + j) = c * ai + (sr * bi + si * br)
            hr((k + 1) * n + j) = -(sr * ar + si * ai) + c * br
            hi((k + 1) * n + j) = -(sr * ai - si * ar) + c * bi
            j += 1
          if k > l then
            hr((k + 1) * n + k - 1) = 0.0
            hi((k + 1) * n + k - 1) = 0.0
          // columns k, k+1 (rows 0 .. min(k+2, hi))
          val rmax = math.min(k + 2, hiIdx)
          var i = 0
          while i <= rmax do
            colRot(hr, hi, n, i, k, c, sr, si)
            i += 1
          i = 0
          while i < n do
            colRot(z.re, z.im, n, i, k, c, sr, si)
            i += 1
          k += 1

  /** `[a b] <- [a b] G^H` with `G^H = [c -s; conj(s) c]`. */
  private inline def colRot(re: Array[Double], im: Array[Double], n: Int, i: Int, k: Int, c: Double, sr: Double, si: Double): Unit =
    val ar = re(i * n + k)
    val ai = im(i * n + k)
    val br = re(i * n + k + 1)
    val bi = im(i * n + k + 1)
    // a' = a c + b conj(s) ; b' = -a s + b c
    re(i * n + k) = ar * c + (br * sr + bi * si)
    im(i * n + k) = ai * c + (bi * sr - br * si)
    re(i * n + k + 1) = -(ar * sr - ai * si) + br * c
    im(i * n + k + 1) = -(ar * si + ai * sr) + bi * c

  /** Unit 2-norm columns; complex columns are rotated so that their largest component is real.
    * Columns belonging to real eigenvalues of a real matrix (`wiReal(j) == 0`) are left real.
    */
  private def normalizeColumns(v: ZMat, wiReal: Array[Double]): Unit =
    val n = v.n
    val m = v.m
    var j = 0
    while j < n do
      val nr = LinalgCplx.nrm2(v.re, v.im, j, m, n)
      if nr > 0.0 then
        var i = 0
        while i < m do
          v.re(i * n + j) /= nr
          v.im(i * n + j) /= nr
          i += 1
        val realCol = wiReal != null && wiReal(j) == 0.0
        if !realCol then
          var best = -1.0
          var bi = 0
          i = 0
          while i < m do
            val a = v.re(i * n + j) * v.re(i * n + j) + v.im(i * n + j) * v.im(i * n + j)
            if a > best then
              best = a
              bi = i
            i += 1
          val pr = v.re(bi * n + j)
          val pi = v.im(bi * n + j)
          val pa = math.hypot(pr, pi)
          if pa > 0.0 then
            // multiply by conj(p)/|p|
            val cr = pr / pa
            val ci = -pi / pa
            i = 0
            while i < m do
              val xr = v.re(i * n + j)
              val xi = v.im(i * n + j)
              v.re(i * n + j) = xr * cr - xi * ci
              v.im(i * n + j) = xr * ci + xi * cr
              i += 1
            v.im(bi * n + j) = 0.0
      j += 1
