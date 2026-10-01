package com.github.kmizu.numscala

import PolyBasis.{Sym, symmetrize}

/** The Hermite, Hermite-E and Laguerre bases (split from [[PolyBasis]] for file size). */
private[numscala] object PolyBasisHermite:
  // ================================================================ Hermite (physicists')

  object Herm extends PolyBasis("herm", "Hermite", "H", Sym, Sym):
    protected def mulxRaw(c: Array[Double]): Array[Double] =
      val prd = new Array[Double](c.length + 1)
      prd(0) = c(0) * 0.0
      prd(1) = c(0) / 2
      for i <- 1 until c.length do
        prd(i + 1) = c(i) / 2
        prd(i - 1) += c(i) * i
      prd
    def valScalar(x: Double, c: Array[Double]): Double =
      val x2 = x * 2
      if c.length == 1 then c(0) + 0.0 * x
      else if c.length == 2 then c(0) + c(1) * x2
      else
        var nd = c.length
        var c0 = c(c.length - 2)
        var c1 = c(c.length - 1)
        var i = 3
        while i <= c.length do
          val tmp = c0
          nd -= 1
          c0 = c(c.length - i) - c1 * (2 * (nd - 1))
          c1 = tmp + c1 * x2
          i += 1
        c0 + c1 * x2
    def clenshaw[A](c: Array[Double], r: PolyRing[A]): A =
      if c.length == 1 then r.const(c(0))
      else
        var nd = c.length
        var c0 = r.const(c(c.length - 2))
        var c1 = r.const(c(c.length - 1))
        var i = 3
        while i <= c.length do
          val tmp = c0
          nd -= 1
          c0 = r.sub(r.const(c(c.length - i)), r.scale(c1, 2.0 * (nd - 1)))
          c1 = r.add(tmp, r.scale(r.mulx(c1), 2.0))
          i += 1
        r.add(c0, r.scale(r.mulx(c1), 2.0))
    protected def derStep(c: Array[Array[Double]]): Array[Array[Double]] =
      val n = c.length - 1
      Array.tabulate(n)(j => c(j + 1).map(_ * (2 * (j + 1))))
    protected def intStep(c: Array[Array[Double]]): Array[Array[Double]] =
      val n = c.length
      Array.tabulate(n + 1) { j =>
        if j == 0 then c(0).map(_ * 0.0)
        else if j == 1 then c(0).map(_ / 2)
        else c(j - 1).map(_ / (2 * j))
      }
    protected def vander1(x: Double): Double = x * 2
    protected def vanderNext(i: Int, x: Double, p1: Double, p2: Double): Double = p1 * (x * 2) - p2 * (2 * (i - 1))
    protected def companionRaw(c: Array[Double]): Array[Array[Double]] =
      val n = c.length - 1
      val mat = Array.ofDim[Double](n, n)
      val raw = Array.tabulate(n)(i => if i == 0 then 1.0 else 1.0 / math.sqrt(2.0 * (n - i)))
      val acc = raw.scanLeft(1.0)(_ * _).tail
      val scl = acc.reverse
      for i <- 0 until n - 1 do
        val v = math.sqrt(0.5 * (i + 1))
        mat(i)(i + 1) = v
        mat(i + 1)(i) = v
      for i <- 0 until n do mat(i)(n - 1) -= scl(i) * c(i) / (2.0 * c(n))
      mat
    protected def linearRoot(c: Array[Double]): Double = -0.5 * c(0) / c(1)
    def line(off: Double, scl: Double): Array[Double] = if scl != 0 then Array(off, scl / 2) else Array(off)
    override def gauss(deg: Int): (Array[Double], Array[Double]) =
      if deg <= 0 then throw new IllegalArgumentException("deg must be a positive integer")
      val x = symCompanionEig(deg)
      for i <- x.indices do
        val dy = normedH(x(i), deg)
        val df = normedH(x(i), deg - 1) * math.sqrt(2.0 * deg)
        x(i) -= dy / df
      var fm = x.map(normedH(_, deg - 1))
      val fmax = fm.map(math.abs).max
      fm = fm.map(_ / fmax)
      symmetrize(x, fm.map(f => 1.0 / (f * f)), math.sqrt(math.Pi))
    override def weight(x: Double): Double = math.exp(-(x * x))

  private def normedH(x: Double, n: Int): Double =
    if n == 0 then 1.0 / math.sqrt(math.sqrt(math.Pi))
    else
      var c0 = 0.0
      var c1 = 1.0 / math.sqrt(math.sqrt(math.Pi))
      var nd = n.toDouble
      for _ <- 0 until n - 1 do
        val tmp = c0
        c0 = -c1 * math.sqrt((nd - 1.0) / nd)
        c1 = tmp + c1 * x * math.sqrt(2.0 / nd)
        nd -= 1.0
      c0 + c1 * x * math.sqrt(2.0)

  private def normedHe(x: Double, n: Int): Double =
    if n == 0 then 1.0 / math.sqrt(math.sqrt(2 * math.Pi))
    else
      var c0 = 0.0
      var c1 = 1.0 / math.sqrt(math.sqrt(2 * math.Pi))
      var nd = n.toDouble
      for _ <- 0 until n - 1 do
        val tmp = c0
        c0 = -c1 * math.sqrt((nd - 1.0) / nd)
        c1 = tmp + c1 * x * math.sqrt(1.0 / nd)
        nd -= 1.0
      c0 + c1 * x

  // ================================================================ Hermite (probabilists')

  object HermE extends PolyBasis("herme", "HermiteE", "He", Sym, Sym):
    protected def mulxRaw(c: Array[Double]): Array[Double] =
      val prd = new Array[Double](c.length + 1)
      prd(0) = c(0) * 0.0
      prd(1) = c(0)
      for i <- 1 until c.length do
        prd(i + 1) = c(i)
        prd(i - 1) += c(i) * i
      prd
    def valScalar(x: Double, c: Array[Double]): Double =
      if c.length == 1 then c(0) + 0.0 * x
      else if c.length == 2 then c(0) + c(1) * x
      else
        var nd = c.length
        var c0 = c(c.length - 2)
        var c1 = c(c.length - 1)
        var i = 3
        while i <= c.length do
          val tmp = c0
          nd -= 1
          c0 = c(c.length - i) - c1 * (nd - 1)
          c1 = tmp + c1 * x
          i += 1
        c0 + c1 * x
    def clenshaw[A](c: Array[Double], r: PolyRing[A]): A =
      if c.length == 1 then r.const(c(0))
      else
        var nd = c.length
        var c0 = r.const(c(c.length - 2))
        var c1 = r.const(c(c.length - 1))
        var i = 3
        while i <= c.length do
          val tmp = c0
          nd -= 1
          c0 = r.sub(r.const(c(c.length - i)), r.scale(c1, nd - 1))
          c1 = r.add(tmp, r.mulx(c1))
          i += 1
        r.add(c0, r.mulx(c1))
    protected def derStep(c: Array[Array[Double]]): Array[Array[Double]] =
      val n = c.length - 1
      Array.tabulate(n)(j => c(j + 1).map(_ * (j + 1)))
    protected def intStep(c: Array[Array[Double]]): Array[Array[Double]] =
      val n = c.length
      Array.tabulate(n + 1) { j =>
        if j == 0 then c(0).map(_ * 0.0)
        else if j == 1 then c(0).clone()
        else c(j - 1).map(_ / j)
      }
    protected def vander1(x: Double): Double = x
    protected def vanderNext(i: Int, x: Double, p1: Double, p2: Double): Double = p1 * x - p2 * (i - 1)
    protected def companionRaw(c: Array[Double]): Array[Array[Double]] =
      val n = c.length - 1
      val mat = Array.ofDim[Double](n, n)
      val raw = Array.tabulate(n)(i => if i == 0 then 1.0 else 1.0 / math.sqrt((n - i).toDouble))
      val scl = raw.scanLeft(1.0)(_ * _).tail.reverse
      for i <- 0 until n - 1 do
        val v = math.sqrt((i + 1).toDouble)
        mat(i)(i + 1) = v
        mat(i + 1)(i) = v
      for i <- 0 until n do mat(i)(n - 1) -= scl(i) * c(i) / c(n)
      mat
    protected def linearRoot(c: Array[Double]): Double = -c(0) / c(1)
    def line(off: Double, scl: Double): Array[Double] = if scl != 0 then Array(off, scl) else Array(off)
    override def gauss(deg: Int): (Array[Double], Array[Double]) =
      if deg <= 0 then throw new IllegalArgumentException("deg must be a positive integer")
      val x = symCompanionEig(deg)
      for i <- x.indices do
        val dy = normedHe(x(i), deg)
        val df = normedHe(x(i), deg - 1) * math.sqrt(deg.toDouble)
        x(i) -= dy / df
      var fm = x.map(normedHe(_, deg - 1))
      val fmax = fm.map(math.abs).max
      fm = fm.map(_ / fmax)
      symmetrize(x, fm.map(f => 1.0 / (f * f)), math.sqrt(2 * math.Pi))
    override def weight(x: Double): Double = math.exp(-0.5 * x * x)

  // ================================================================ Laguerre

  object Lag extends PolyBasis("lag", "Laguerre", "L", Array(0.0, 1.0), Array(0.0, 1.0)):
    protected def mulxRaw(c: Array[Double]): Array[Double] =
      val prd = new Array[Double](c.length + 1)
      prd(0) = c(0)
      prd(1) = -c(0)
      for i <- 1 until c.length do
        prd(i + 1) = -c(i) * (i + 1)
        prd(i) += c(i) * (2 * i + 1)
        prd(i - 1) -= c(i) * i
      prd
    def valScalar(x: Double, c: Array[Double]): Double =
      if c.length == 1 then c(0) + 0.0 * x
      else if c.length == 2 then c(0) + c(1) * (1 - x)
      else
        var nd = c.length
        var c0 = c(c.length - 2)
        var c1 = c(c.length - 1)
        var i = 3
        while i <= c.length do
          val tmp = c0
          nd -= 1
          c0 = c(c.length - i) - (c1 * (nd - 1)) / nd
          c1 = tmp + (c1 * ((2 * nd - 1) - x)) / nd
          i += 1
        c0 + c1 * (1 - x)
    def clenshaw[A](c: Array[Double], r: PolyRing[A]): A =
      if c.length == 1 then r.const(c(0))
      else
        var nd = c.length
        var c0 = r.const(c(c.length - 2))
        var c1 = r.const(c(c.length - 1))
        var i = 3
        while i <= c.length do
          val tmp = c0
          nd -= 1
          c0 = r.sub(r.const(c(c.length - i)), r.div(r.scale(c1, nd - 1), nd))
          c1 = r.add(tmp, r.div(r.sub(r.scale(c1, 2 * nd - 1), r.mulx(c1)), nd))
          i += 1
        r.add(c0, r.sub(c1, r.mulx(c1)))
    protected def derStep(c: Array[Array[Double]]): Array[Array[Double]] =
      val n = c.length - 1
      val der = new Array[Array[Double]](n)
      var j = n
      while j > 1 do
        der(j - 1) = c(j).map(-_)
        c(j - 1) = c(j - 1).indices.map(r => c(j - 1)(r) + c(j)(r)).toArray
        j -= 1
      der(0) = c(1).map(-_)
      der
    protected def intStep(c: Array[Array[Double]]): Array[Array[Double]] =
      val n = c.length
      val tmp = new Array[Array[Double]](n + 1)
      tmp(0) = c(0).clone()
      tmp(1) = c(0).map(-_)
      for j <- 1 until n do
        tmp(j) = tmp(j).indices.map(r => tmp(j)(r) + c(j)(r)).toArray
        tmp(j + 1) = c(j).map(-_)
      tmp
    protected def vander1(x: Double): Double = 1 - x
    protected def vanderNext(i: Int, x: Double, p1: Double, p2: Double): Double =
      (p1 * (2 * i - 1 - x) - p2 * (i - 1)) / i
    protected def companionRaw(c: Array[Double]): Array[Array[Double]] =
      val n = c.length - 1
      val mat = Array.ofDim[Double](n, n)
      for i <- 0 until n do mat(i)(i) = 2.0 * i + 1.0
      for i <- 0 until n - 1 do
        mat(i)(i + 1) = -(i + 1).toDouble
        mat(i + 1)(i) = -(i + 1).toDouble
      for i <- 0 until n do mat(i)(n - 1) += (c(i) / c(n)) * n
      mat
    protected def linearRoot(c: Array[Double]): Double = 1 + c(0) / c(1)
    def line(off: Double, scl: Double): Array[Double] = if scl != 0 then Array(off + scl, -scl) else Array(off)
    override def gauss(deg: Int): (Array[Double], Array[Double]) =
      if deg <= 0 then throw new IllegalArgumentException("deg must be a positive integer")
      val c = Array.fill(deg)(0.0) :+ 1.0
      val x = symCompanionEig(deg)
      val dc = der(c.map(Array(_)), 1, 1.0).map(_(0))
      val dy = x.map(valScalar(_, c))
      var df = x.map(valScalar(_, dc))
      for i <- x.indices do x(i) -= dy(i) / df(i)
      var fm = x.map(valScalar(_, c.drop(1)))
      val fmax = fm.map(math.abs).max
      fm = fm.map(_ / fmax)
      val dmax = df.map(math.abs).max
      df = df.map(_ / dmax)
      val w = Array.tabulate(deg)(i => 1.0 / (fm(i) * df(i)))
      val s = w.sum
      (x, w.map(_ / s))
    override def weight(x: Double): Double = math.exp(-x)
