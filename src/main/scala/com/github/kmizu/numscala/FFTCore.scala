package com.github.kmizu.numscala

import scala.collection.concurrent.TrieMap

/** A precomputed 1-D complex DFT of length `n` (forward sign `exp(-2 pi i jk/n)`, unnormalized).
  *
  * Powers of two use an iterative radix-2 algorithm, short lengths a direct DFT with an exact
  * twiddle table, and every other length Bluestein's chirp-z algorithm on a power-of-two grid.
  */
private[numscala] final class FFTPlan(val n: Int):
  require(n >= 1)
  private val isPow2 = (n & (n - 1)) == 0
  private val direct = !isPow2 && n <= 32

  // cos/sin(2 pi k / n), k in [0, n) (or [0, n/2) for radix-2)
  private val (cosT, sinT) =
    if isPow2 || direct then
      val m = if isPow2 then math.max(n / 2, 1) else n
      val c = new Array[Double](m)
      val s = new Array[Double](m)
      var k = 0
      while k < m do
        val (ck, sk) = FFTPlan.cosSin(k.toLong, n.toLong)
        c(k) = ck
        s(k) = sk
        k += 1
      (c, s)
    else (Array.emptyDoubleArray, Array.emptyDoubleArray)

  // Bluestein data
  private val m: Int = if isPow2 || direct then 0 else Integer.highestOneBit(2 * n - 1) << 1
  private val sub: FFTPlan | Null = if m > 0 then FFTPlan.get(m) else null
  private val (wRe, wIm) =
    if m > 0 then
      val re = new Array[Double](n)
      val im = new Array[Double](n)
      val twoN = 2L * n
      var k = 0
      while k < n do
        // exp(-i pi k^2 / n) = exp(-2 pi i (k^2 mod 2n) / (2n))
        val (c, s) = FFTPlan.cosSin((k.toLong * k.toLong) % twoN, twoN)
        re(k) = c
        im(k) = -s
        k += 1
      (re, im)
    else (Array.emptyDoubleArray, Array.emptyDoubleArray)
  private val (bRe, bIm) =
    if m > 0 then
      val re = new Array[Double](m)
      val im = new Array[Double](m)
      re(0) = wRe(0); im(0) = -wIm(0)
      var k = 1
      while k < n do
        re(k) = wRe(k); im(k) = -wIm(k)
        re(m - k) = wRe(k); im(m - k) = -wIm(k)
        k += 1
      sub.nn.forward(re, im)
      (re, im)
    else (Array.emptyDoubleArray, Array.emptyDoubleArray)

  /** In-place forward transform. */
  def forward(re: Array[Double], im: Array[Double]): Unit =
    if n == 1 then ()
    else if isPow2 then radix2(re, im)
    else if direct then dft(re, im)
    else bluestein(re, im)

  /** In-place unnormalized backward transform (sign +). */
  def backward(re: Array[Double], im: Array[Double]): Unit =
    var i = 0
    while i < n do
      im(i) = -im(i)
      i += 1
    forward(re, im)
    i = 0
    while i < n do
      im(i) = -im(i)
      i += 1

  private def dft(re: Array[Double], im: Array[Double]): Unit =
    val oRe = new Array[Double](n)
    val oIm = new Array[Double](n)
    var k = 0
    while k < n do
      var sr = 0.0
      var si = 0.0
      var j = 0
      var idx = 0
      while j < n do
        val c = cosT(idx)
        val s = sinT(idx)
        sr += re(j) * c + im(j) * s
        si += im(j) * c - re(j) * s
        idx += k
        if idx >= n then idx -= n
        j += 1
      oRe(k) = sr
      oIm(k) = si
      k += 1
    System.arraycopy(oRe, 0, re, 0, n)
    System.arraycopy(oIm, 0, im, 0, n)

  private def radix2(re: Array[Double], im: Array[Double]): Unit =
    // bit reversal
    var j = 0
    var i = 0
    while i < n - 1 do
      if i < j then
        val tr = re(i); re(i) = re(j); re(j) = tr
        val ti = im(i); im(i) = im(j); im(j) = ti
      var bit = n >> 1
      while (j & bit) != 0 do
        j ^= bit
        bit >>= 1
      j |= bit
      i += 1
    var len = 2
    while len <= n do
      val half = len >> 1
      val step = n / len
      var start = 0
      while start < n do
        var k = 0
        var t = 0
        while k < half do
          val c = cosT(t)
          val s = sinT(t)
          val a = start + k
          val b = a + half
          val xr = re(b) * c + im(b) * s
          val xi = im(b) * c - re(b) * s
          re(b) = re(a) - xr
          im(b) = im(a) - xi
          re(a) += xr
          im(a) += xi
          k += 1
          t += step
        start += len
      len <<= 1

  private def bluestein(re: Array[Double], im: Array[Double]): Unit =
    val aRe = new Array[Double](m)
    val aIm = new Array[Double](m)
    var k = 0
    while k < n do
      aRe(k) = re(k) * wRe(k) - im(k) * wIm(k)
      aIm(k) = re(k) * wIm(k) + im(k) * wRe(k)
      k += 1
    val s = sub.nn
    s.forward(aRe, aIm)
    k = 0
    while k < m do
      val r = aRe(k) * bRe(k) - aIm(k) * bIm(k)
      val i = aRe(k) * bIm(k) + aIm(k) * bRe(k)
      aRe(k) = r
      aIm(k) = i
      k += 1
    s.backward(aRe, aIm)
    val inv = 1.0 / m
    k = 0
    while k < n do
      val r = aRe(k) * inv
      val i = aIm(k) * inv
      re(k) = r * wRe(k) - i * wIm(k)
      im(k) = r * wIm(k) + i * wRe(k)
      k += 1

private[numscala] object FFTPlan:
  private val cache = TrieMap.empty[Int, FFTPlan]

  def get(n: Int): FFTPlan =
    cache.get(n) match
      case Some(p) => p
      case None =>
        if cache.size > 64 then cache.clear()
        val p = new FFTPlan(n)
        cache.putIfAbsent(n, p)
        p

  /** cos and sin of `2 pi k / n`, computed with octant symmetry for accuracy. */
  def cosSin(k: Long, n: Long): (Double, Double) =
    // reduce to the first octant: angle = 2 pi k / n
    val kk = ((k % n) + n) % n
    // use 8*k vs n comparisons in exact integer arithmetic
    val k8 = 8 * kk
    def ang(num: Long): Double = 2.0 * math.Pi * num.toDouble / n.toDouble
    if k8 <= n then (math.cos(ang(kk)), math.sin(ang(kk)))
    else if k8 <= 2 * n then
      val r = n - 4 * kk // angle = pi/2 - theta', theta' = 2 pi (n/4 - k)/n
      val t = math.Pi * r.toDouble / (2.0 * n.toDouble)
      (math.sin(t), math.cos(t))
    else if k8 <= 4 * n then
      // angle in (pi/4, pi]: cos(a) = -cos(pi - a), sin(a) = sin(pi - a)
      val (c, s) = cosSin(n - 2 * kk, 2 * n)
      (-c, s)
    else
      // angle in (pi, 2pi): mirror
      val (c, s) = cosSin(n - kk, n)
      (c, -s)
