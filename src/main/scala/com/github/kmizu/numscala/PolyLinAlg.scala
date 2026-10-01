package com.github.kmizu.numscala

/** Small dense linear-algebra kernels used by the polynomial modules (eigenvalues of
  * companion matrices, least squares via SVD). Matrices are row-major `Array[Array[_]]`.
  */
private[numscala] object PolyLinAlg:

  private val Eps = java.lang.Math.ulp(1.0)

  // ---------------------------------------------------------------- real eigenvalues

  /** Eigenvalues of a general real square matrix (balance + Hessenberg + Francis QR). */
  def eigvalsReal(a0: Array[Array[Double]]): Array[Complex] =
    val n = a0.length
    if n == 0 then return Array.empty[Complex]
    // 1-based working copy
    val a = Array.ofDim[Double](n + 1, n + 1)
    for i <- 0 until n; j <- 0 until n do a(i + 1)(j + 1) = a0(i)(j)
    if a.exists(_.exists(x => x.isNaN || x.isInfinite)) then
      throw new LinAlgError("Array must not contain infs or NaNs")
    balance(a, n)
    elmhes(a, n)
    for i <- 1 to n; j <- 1 to n if i > j + 1 do a(i)(j) = 0.0
    val wr = new Array[Double](n + 1)
    val wi = new Array[Double](n + 1)
    hqr(a, n, wr, wi)
    Array.tabulate(n)(i => Complex(wr(i + 1), wi(i + 1)))

  private def balance(a: Array[Array[Double]], n: Int): Unit =
    val radix = 2.0
    val sqrdx = radix * radix
    var done = false
    while !done do
      done = true
      var i = 1
      while i <= n do
        var r = 0.0
        var c = 0.0
        var j = 1
        while j <= n do
          if j != i then
            c += math.abs(a(j)(i))
            r += math.abs(a(i)(j))
          j += 1
        if c != 0.0 && r != 0.0 then
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
            done = false
            g = 1.0 / f
            j = 1
            while j <= n do
              a(i)(j) *= g
              j += 1
            j = 1
            while j <= n do
              a(j)(i) *= f
              j += 1
        i += 1

  private def elmhes(a: Array[Array[Double]], n: Int): Unit =
    var m = 2
    while m < n do
      var x = 0.0
      var i = m
      var j = m
      while j <= n do
        if math.abs(a(j)(m - 1)) > math.abs(x) then
          x = a(j)(m - 1)
          i = j
        j += 1
      if i != m then
        j = m - 1
        while j <= n do
          val t = a(i)(j); a(i)(j) = a(m)(j); a(m)(j) = t
          j += 1
        j = 1
        while j <= n do
          val t = a(j)(i); a(j)(i) = a(j)(m); a(j)(m) = t
          j += 1
      if x != 0.0 then
        i = m + 1
        while i <= n do
          var y = a(i)(m - 1)
          if y != 0.0 then
            y /= x
            a(i)(m - 1) = y
            j = m
            while j <= n do
              a(i)(j) -= y * a(m)(j)
              j += 1
            j = 1
            while j <= n do
              a(j)(m) += y * a(j)(i)
              j += 1
          i += 1
      m += 1

  private inline def sign(a: Double, b: Double): Double = if b >= 0.0 then math.abs(a) else -math.abs(a)

  /** Eigenvalues of an upper Hessenberg matrix (1-based), Numerical Recipes `hqr`. */
  private def hqr(a: Array[Array[Double]], n: Int, wr: Array[Double], wi: Array[Double]): Unit =
    var anorm = 0.0
    for i <- 1 to n; j <- math.max(i - 1, 1) to n do anorm += math.abs(a(i)(j))
    var nn = n
    var t = 0.0
    var p, q, r, s, w, x, y, z = 0.0
    while nn >= 1 do
      var its = 0
      var l = 0
      while
        // look for a single small subdiagonal element
        l = nn
        var found = false
        while l >= 2 && !found do
          s = math.abs(a(l - 1)(l - 1)) + math.abs(a(l)(l))
          if s == 0.0 then s = anorm
          if math.abs(a(l)(l - 1)) + s == s then
            a(l)(l - 1) = 0.0
            found = true
          else l -= 1
        x = a(nn)(nn)
        if l == nn then
          wr(nn) = x + t
          wi(nn) = 0.0
          nn -= 1
        else
          y = a(nn - 1)(nn - 1)
          w = a(nn)(nn - 1) * a(nn - 1)(nn)
          if l == nn - 1 then
            p = 0.5 * (y - x)
            q = p * p + w
            z = math.sqrt(math.abs(q))
            x += t
            if q >= 0.0 then
              z = p + sign(z, p)
              wr(nn - 1) = x + z
              wr(nn) = x + z
              if z != 0.0 then wr(nn) = x - w / z
              wi(nn - 1) = 0.0
              wi(nn) = 0.0
            else
              wr(nn - 1) = x + p
              wr(nn) = x + p
              wi(nn - 1) = -z
              wi(nn) = z
            nn -= 2
          else
            if its == 60 * math.max(n, 1) then throw new LinAlgError("Eigenvalues did not converge")
            if its > 0 && its % 10 == 0 then
              // exceptional shift
              t += x
              var i = 1
              while i <= nn do
                a(i)(i) -= x
                i += 1
              s = math.abs(a(nn)(nn - 1)) + math.abs(a(nn - 1)(nn - 2))
              x = 0.75 * s
              y = x
              w = -0.4375 * s * s
            its += 1
            var m = nn - 2
            var stop = false
            while m >= l && !stop do
              z = a(m)(m)
              r = x - z
              s = y - z
              p = (r * s - w) / a(m + 1)(m) + a(m)(m + 1)
              q = a(m + 1)(m + 1) - z - r - s
              r = a(m + 2)(m + 1)
              s = math.abs(p) + math.abs(q) + math.abs(r)
              p /= s
              q /= s
              r /= s
              if m == l then stop = true
              else
                val u = math.abs(a(m)(m - 1)) * (math.abs(q) + math.abs(r))
                val v = math.abs(p) * (math.abs(a(m - 1)(m - 1)) + math.abs(z) + math.abs(a(m + 1)(m + 1)))
                if u + v == v then stop = true
                else m -= 1
            var i = m + 2
            while i <= nn do
              a(i)(i - 2) = 0.0
              if i != m + 2 then a(i)(i - 3) = 0.0
              i += 1
            var k = m
            while k <= nn - 1 do
              var proceed = true
              if k != m then
                p = a(k)(k - 1)
                q = a(k + 1)(k - 1)
                r = 0.0
                if k != nn - 1 then r = a(k + 2)(k - 1)
                x = math.abs(p) + math.abs(q) + math.abs(r)
                if x != 0.0 then
                  p /= x
                  q /= x
                  r /= x
              s = sign(math.sqrt(p * p + q * q + r * r), p)
              if s != 0.0 then
                if k == m then
                  if l != m then a(k)(k - 1) = -a(k)(k - 1)
                else a(k)(k - 1) = -s * x
                p += s
                x = p / s
                y = q / s
                z = r / s
                q /= p
                r /= p
                var j = k
                while j <= nn do
                  p = a(k)(j) + q * a(k + 1)(j)
                  if k != nn - 1 then
                    p += r * a(k + 2)(j)
                    a(k + 2)(j) -= p * z
                  a(k + 1)(j) -= p * y
                  a(k)(j) -= p * x
                  j += 1
                val mmin = if nn < k + 3 then nn else k + 3
                i = l
                while i <= mmin do
                  p = x * a(i)(k) + y * a(i)(k + 1)
                  if k != nn - 1 then
                    p += z * a(i)(k + 2)
                    a(i)(k + 2) -= p * r
                  a(i)(k + 1) -= p * q
                  a(i)(k) -= p
                  i += 1
              k += 1
        l < nn - 1
      do ()

  // ---------------------------------------------------------------- complex eigenvalues

  /** Eigenvalues of a general complex square matrix (balance + Householder Hessenberg + shifted QR). */
  def eigvalsComplex(a0: Array[Array[Complex]]): Array[Complex] =
    val n = a0.length
    if n == 0 then return Array.empty[Complex]
    if a0.exists(_.exists(z => !z.isFinite)) then throw new LinAlgError("Array must not contain infs or NaNs")
    val a = a0.map(_.clone())
    balanceC(a, n)
    hessenbergC(a, n)
    complexQR(a, n)

  private def cabs1(z: Complex): Double = math.abs(z.re) + math.abs(z.im)

  private def balanceC(a: Array[Array[Complex]], n: Int): Unit =
    var done = false
    while !done do
      done = true
      for i <- 0 until n do
        var r = 0.0
        var c = 0.0
        for j <- 0 until n if j != i do
          c += cabs1(a(j)(i))
          r += cabs1(a(i)(j))
        if c != 0.0 && r != 0.0 then
          var g = r / 2.0
          var f = 1.0
          val s = c + r
          while c < g do
            f *= 2.0
            c *= 4.0
          g = r * 2.0
          while c > g do
            f /= 2.0
            c /= 4.0
          if (c + r) / f < 0.95 * s then
            done = false
            val gi = 1.0 / f
            for j <- 0 until n do a(i)(j) = a(i)(j) * gi
            for j <- 0 until n do a(j)(i) = a(j)(i) * f

  private def hessenbergC(a: Array[Array[Complex]], n: Int): Unit =
    for k <- 0 until n - 2 do
      // Householder vector for column k, rows k+1..n-1
      var alpha = 0.0
      for i <- k + 1 until n do alpha += a(i)(k).abs2
      alpha = math.sqrt(alpha)
      if alpha > 0.0 then
        val x0 = a(k + 1)(k)
        val phase = if x0.abs == 0.0 then Complex.One else x0 / x0.abs
        val v = new Array[Complex](n)
        for i <- 0 until n do v(i) = Complex.Zero
        v(k + 1) = x0 + phase * alpha
        for i <- k + 2 until n do v(i) = a(i)(k)
        var vn = 0.0
        for i <- k + 1 until n do vn += v(i).abs2
        if vn > 0.0 then
          // H = I - 2 v v^H / (v^H v); A <- H A H
          for j <- 0 until n do
            var s = Complex.Zero
            for i <- k + 1 until n do s = s + v(i).conj * a(i)(j)
            val f = s * (2.0 / vn)
            for i <- k + 1 until n do a(i)(j) = a(i)(j) - v(i) * f
          for i <- 0 until n do
            var s = Complex.Zero
            for j <- k + 1 until n do s = s + a(i)(j) * v(j)
            val f = s * (2.0 / vn)
            for j <- k + 1 until n do a(i)(j) = a(i)(j) - f * v(j).conj
          for i <- k + 2 until n do a(i)(k) = Complex.Zero

  private def complexQR(h: Array[Array[Complex]], n: Int): Array[Complex] =
    val eig = new Array[Complex](n)
    var hi = n - 1
    var iter = 0
    var total = 0
    val cs = new Array[Complex](n)
    val sn = new Array[Complex](n)
    var norm = 0.0
    for i <- 0 until n; j <- 0 until n do norm = math.max(norm, cabs1(h(i)(j)))
    while hi >= 0 do
      var l = hi
      var found = false
      while l > 0 && !found do
        var s = cabs1(h(l - 1)(l - 1)) + cabs1(h(l)(l))
        if s == 0.0 then s = norm
        if cabs1(h(l)(l - 1)) <= Eps * s then
          h(l)(l - 1) = Complex.Zero
          found = true
        else l -= 1
      if l == hi then
        eig(hi) = h(hi)(hi)
        hi -= 1
        iter = 0
      else
        iter += 1
        total += 1
        if total > 100 * n then throw new LinAlgError("Eigenvalues did not converge")
        val mu =
          if iter % 11 == 10 then h(hi)(hi) + Complex(cabs1(h(hi)(hi - 1)), 0.0) * 0.75
          else
            val a = h(hi - 1)(hi - 1)
            val b = h(hi - 1)(hi)
            val c = h(hi)(hi - 1)
            val d = h(hi)(hi)
            val half = (a - d) * 0.5
            val disc = (half * half + b * c).sqrt
            val m1 = (a + d) * 0.5 + disc
            val m2 = (a + d) * 0.5 - disc
            if (m1 - d).abs <= (m2 - d).abs then m1 else m2
        for i <- l to hi do h(i)(i) = h(i)(i) - mu
        for k <- l until hi do
          val x = h(k)(k)
          val y = h(k + 1)(k)
          val r = math.hypot(x.abs, y.abs)
          val (c, s) = if r == 0.0 then (Complex.One, Complex.Zero) else (x / r, y / r)
          cs(k) = c
          sn(k) = s
          for j <- k to hi do
            val u = h(k)(j)
            val v = h(k + 1)(j)
            h(k)(j) = c.conj * u + s.conj * v
            h(k + 1)(j) = c * v - s * u
        for k <- l until hi do
          val c = cs(k)
          val s = sn(k)
          for i <- l to math.min(k + 1, hi) do
            val u = h(i)(k)
            val v = h(i)(k + 1)
            h(i)(k) = u * c + v * s
            h(i)(k + 1) = v * c.conj - u * s.conj
        for i <- l to hi do h(i)(i) = h(i)(i) + mu
    eig

  // ---------------------------------------------------------------- symmetric tridiagonal

  /** Eigenvalues (ascending) of the symmetric tridiagonal matrix with diagonal `d0`
    * and off-diagonal `e0` (`e0(i)` couples `i` and `i+1`). Implicit QL.
    */
  def eigvalshTridiag(d0: Array[Double], e0: Array[Double]): Array[Double] =
    val n = d0.length
    val d = d0.clone()
    val e = new Array[Double](n)
    for i <- 0 until n - 1 do e(i) = e0(i)
    var l = 0
    while l < n do
      var iter = 0
      var m = l
      while
        m = l
        var stop = false
        while m < n - 1 && !stop do
          val dd = math.abs(d(m)) + math.abs(d(m + 1))
          if math.abs(e(m)) <= Eps * dd then stop = true else m += 1
        if m != l then
          iter += 1
          if iter > 60 then throw new LinAlgError("Eigenvalues did not converge")
          var g = (d(l + 1) - d(l)) / (2.0 * e(l))
          var r = math.hypot(g, 1.0)
          g = d(m) - d(l) + e(l) / (g + sign(r, g))
          var s = 1.0
          var c = 1.0
          var p = 0.0
          var i = m - 1
          var underflow = false
          while i >= l && !underflow do
            val f = s * e(i)
            val b = c * e(i)
            r = math.hypot(f, g)
            e(i + 1) = r
            if r == 0.0 then
              d(i + 1) -= p
              e(m) = 0.0
              underflow = true
            else
              s = f / r
              c = g / r
              g = d(i + 1) - p
              r = (d(i) - g) * s + 2.0 * c * b
              p = s * r
              d(i + 1) = g + p
              g = c * r - b
              i -= 1
          if !underflow then
            d(l) -= p
            e(l) = g
            e(m) = 0.0
        m != l
      do ()
      l += 1
    java.util.Arrays.sort(d)
    d

  // ---------------------------------------------------------------- SVD / least squares

  /** Thin SVD by one-sided Jacobi: returns (U m x k, s (descending), V n x k), k = min(m, n). */
  def svd(a: Array[Array[Double]], m: Int, n: Int): (Array[Array[Double]], Array[Double], Array[Array[Double]]) =
    if m < n then
      val at = Array.tabulate(n, m)((i, j) => a(j)(i))
      val (u, s, v) = svd(at, n, m)
      return (v, s, u)
    // columns stored as arrays for speed
    val u = Array.tabulate(n, m)((j, i) => a(i)(j))
    val v = Array.tabulate(n, n)((j, i) => if i == j then 1.0 else 0.0)
    var sweep = 0
    var rotated = true
    while rotated && sweep < 80 do
      rotated = false
      sweep += 1
      for p <- 0 until n - 1; q <- p + 1 until n do
        val up = u(p)
        val uq = u(q)
        var alpha = 0.0
        var beta = 0.0
        var gamma = 0.0
        var i = 0
        while i < m do
          alpha += up(i) * up(i)
          beta += uq(i) * uq(i)
          gamma += up(i) * uq(i)
          i += 1
        if gamma != 0.0 && math.abs(gamma) > Eps * math.sqrt(alpha * beta) then
          rotated = true
          val zeta = (beta - alpha) / (2.0 * gamma)
          val t = math.signum(zeta) match
            case 0.0 => 1.0 / (1.0 + math.sqrt(1.0 + zeta * zeta))
            case sg => sg / (math.abs(zeta) + math.sqrt(1.0 + zeta * zeta))
          val c = 1.0 / math.sqrt(1.0 + t * t)
          val s = c * t
          i = 0
          while i < m do
            val x = up(i)
            val y = uq(i)
            up(i) = c * x - s * y
            uq(i) = s * x + c * y
            i += 1
          val vp = v(p)
          val vq = v(q)
          i = 0
          while i < n do
            val x = vp(i)
            val y = vq(i)
            vp(i) = c * x - s * y
            vq(i) = s * x + c * y
            i += 1
    val sv = Array.tabulate(n)(j => math.sqrt(u(j).map(x => x * x).sum))
    val order = (0 until n).sortBy(j => -sv(j)).toArray
    val s = order.map(sv(_))
    val uu = Array.tabulate(m, n) { (i, k) =>
      val j = order(k)
      if sv(j) == 0.0 then 0.0 else u(j)(i) / sv(j)
    }
    val vv = Array.tabulate(n, n)((i, k) => v(order(k))(i))
    (uu, s, vv)

  /** Result of [[lstsq]]: solution `x` (n x nrhs), residual sums (empty unless full rank and m > n),
    * effective rank and singular values.
    */
  final case class LstsqResult(x: Array[Array[Double]], resid: Array[Double], rank: Int, s: Array[Double])

  /** Minimum-norm least squares `a x = b` (`numpy.linalg.lstsq`) with singular values below
    * `rcond * max(s)` treated as zero.
    */
  def lstsq(a: Array[Array[Double]], m: Int, n: Int, b: Array[Array[Double]], nrhs: Int, rcond: Double): LstsqResult =
    val (u, s, v) = svd(a, m, n)
    val k = s.length
    val cutoff = rcond * (if k > 0 then s(0) else 0.0)
    var rank = 0
    for i <- 0 until k do if s(i) > cutoff then rank += 1
    val x = Array.ofDim[Double](n, nrhs)
    for r <- 0 until nrhs do
      for i <- 0 until k if s(i) > cutoff do
        var dot = 0.0
        for row <- 0 until m do dot += u(row)(i) * b(row)(r)
        val coef = dot / s(i)
        for j <- 0 until n do x(j)(r) += v(j)(i) * coef
    val resid =
      if rank == n && m > n then
        Array.tabulate(nrhs) { r =>
          var acc = 0.0
          for row <- 0 until m do
            var ax = 0.0
            for j <- 0 until n do ax += a(row)(j) * x(j)(r)
            val d = b(row)(r) - ax
            acc += d * d
          acc
        }
      else Array.emptyDoubleArray
    LstsqResult(x, resid, rank, s)

  /** Matrix inverse by Gauss-Jordan elimination with partial pivoting. */
  def inv(a0: Array[Array[Double]]): Array[Array[Double]] =
    val n = a0.length
    val a = a0.map(_.clone())
    val r = Array.tabulate(n, n)((i, j) => if i == j then 1.0 else 0.0)
    for col <- 0 until n do
      var piv = col
      for i <- col + 1 until n do if math.abs(a(i)(col)) > math.abs(a(piv)(col)) then piv = i
      if a(piv)(col) == 0.0 then throw new LinAlgError("Singular matrix")
      if piv != col then
        val t = a(piv); a(piv) = a(col); a(col) = t
        val t2 = r(piv); r(piv) = r(col); r(col) = t2
      val d = a(col)(col)
      for j <- 0 until n do
        a(col)(j) /= d
        r(col)(j) /= d
      for i <- 0 until n if i != col do
        val f = a(i)(col)
        if f != 0.0 then
          for j <- 0 until n do
            a(i)(j) -= f * a(col)(j)
            r(i)(j) -= f * r(col)(j)
    r
