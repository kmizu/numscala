package com.github.kmizu.numscala

class FFTSuite extends munit.FunSuite:

  private def c(re: Double, im: Double = 0.0): Complex = Complex(re, im)

  private def assertC(a: NDArray[Complex], e: Seq[Complex], tol: Double = 1e-12)(using munit.Location): Unit =
    val av = a.toSeq
    assertEquals(av.length, e.length)
    av.zip(e).zipWithIndex.foreach { case ((x, y), i) =>
      val scale = math.max(1.0, y.abs)
      assert((x - y).abs <= tol * scale, s"index $i: $x != $y")
    }

  private def assertD(a: NDArray[Double], e: Seq[Double], tol: Double = 1e-12)(using munit.Location): Unit =
    val av = a.toSeq
    assertEquals(av.length, e.length)
    av.zip(e).zipWithIndex.foreach { case ((x, y), i) =>
      assert(math.abs(x - y) <= tol * math.max(1.0, math.abs(y)), s"index $i: $x != $y")
    }

  /** Naive DFT with exact angles, for reference. */
  private def naive(x: Array[Complex], inverse: Boolean): Array[Complex] =
    val n = x.length
    Array.tabulate(n) { k =>
      var s = Complex.Zero
      for j <- 0 until n do
        val idx = (k.toLong * j) % n
        val ang = (if inverse then 2.0 else -2.0) * math.Pi * idx / n
        s = s + x(j) * Complex(math.cos(ang), math.sin(ang))
      s
    }

  private def signal(n: Int): Array[Complex] =
    Array.tabulate(n)(i => Complex(math.sin(i * 0.37) + math.cos(i * 1.3) * i / 10.0, math.cos(i * 0.71) - 0.2))

  test("fft of a small real vector") {
    val r = np.fft.fft(np.array(1.0, 2.0, 3.0, 4.0))
    assertC(r, Seq(c(10), c(-2, 2), c(-2), c(-2, -2)))
    assertEquals(r.dtype.name, "complex128")
    // integer input
    assertC(np.fft.fft(np.array(1, 2, 3, 4)), Seq(c(10), c(-2, 2), c(-2), c(-2, -2)))
  }

  test("fft matches a naive DFT for many lengths (radix-2, direct and Bluestein)") {
    for n <- Seq(1, 2, 3, 4, 5, 7, 8, 12, 16, 17, 31, 33, 64, 97, 100, 128, 255, 1000, 1024) do
      val x = signal(n)
      val a = NDArray.fromArray(x)
      assertC(np.fft.fft(a), naive(x, inverse = false).toSeq, 1e-11)
      assertC(np.fft.ifft(a), naive(x, inverse = true).map(_ / n).toSeq, 1e-11)
      // round trip
      assertC(np.fft.ifft(np.fft.fft(a)), x.toSeq, 1e-12)
  }

  test("n pads and truncates") {
    assertC(
      np.fft.fft(np.array(1.0, 2.0, 3.0), n = 5),
      Seq(
        c(6.0),
        c(-0.8090169943749475, -3.6654687894677265),
        c(0.30901699437494745, 1.6775990443005142),
        c(0.30901699437494745, -1.6775990443005142),
        c(-0.8090169943749475, 3.6654687894677265)
      )
    )
    assertC(np.fft.fft(np.array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), n = 4, norm = "ortho"), Seq(c(5), c(-1, 1), c(-1), c(-1, -1)))
  }

  test("norm modes") {
    val x = np.array(c(1), c(0, 2), c(3), c(4, -1), c(5))
    assertC(
      np.fft.ifft(x, norm = "forward"),
      Seq(
        c(13.0, 1.0),
        c(-5.6079322736326755, -3.9160168506433983),
        c(-1.1064799995398977, -3.9149207282920546),
        c(-0.6574520229603124, 0.06081876204236991),
        c(-0.6281357038671147, 6.770118816893083)
      )
    )
    val f = np.fft.fft(x, norm = "forward")
    assertC(f, np.fft.fft(x).toSeq.map(_ / 5.0))
    assertC(np.fft.ifft(f, norm = "forward"), x.toSeq)
    val o = np.fft.fft(x, norm = "ortho")
    assertC(np.fft.ifft(o, norm = "ortho"), x.toSeq)
    intercept[IllegalArgumentException](np.fft.fft(x, norm = "bogus"))
    intercept[IllegalArgumentException](np.fft.fft(x, n = 0))
  }

  test("fft along an axis of a 2-D array") {
    val b = np.arange(12.0).reshape(3, 4)
    val r0 = np.fft.fft(b, axis = 0)
    assertEquals(r0.shape, Seq(3, 4))
    val s = math.sqrt(3.0) * 2
    assertC(r0(0, ::), Seq(c(12), c(15), c(18), c(21)))
    assertC(r0(1, ::), Seq.fill(4)(c(-6, s)))
    assertC(np.fft.fftn(b, axes = Seq(0)), r0.toSeq)
    val r1 = np.fft.fft(b)
    assertC(r1(2, ::), Seq(c(38), c(-2, 2), c(-2), c(-2, -2)))
  }

  test("fft2 / fftn / ifftn") {
    val b = np.arange(12.0).reshape(3, 4)
    val t = 13.856406460551018
    val f2 = np.fft.fft2(b)
    assertC(f2, Seq(c(66), c(-6, 6), c(-6), c(-6, -6), c(-24, t), c(0), c(0), c(0), c(-24, -t), c(0), c(0), c(0)))
    assertC(np.fft.fftn(b), f2.toSeq)
    val fs = np.fft.fftn(b, s = Seq(2, 3))
    assertEquals(fs.shape, Seq(2, 3))
    val r3 = 1.7320508075688772
    assertC(fs, Seq(c(18), c(-3, r3), c(-3, -r3), c(-12), c(0), c(0)))
    assertC(np.fft.fftn(b, s = Seq(-1, -1)), f2.toSeq)
    intercept[IllegalArgumentException](np.fft.fftn(b, s = Seq(2, 3), axes = Seq(0)))
    assertC(np.fft.ifftn(np.fft.fftn(b)), b.toSeq.map(c(_)))
    assertC(np.fft.ifft2(f2), b.toSeq.map(c(_)))
    val cube = np.arange(24.0).reshape(2, 3, 4)
    val f3 = np.fft.fftn(cube)
    assertEquals(f3.shape, Seq(2, 3, 4))
    assertC(np.fft.ifftn(f3), cube.toSeq.map(c(_)))
    assertEqualsDouble(f3(0, 0, 0).re, 276.0, 1e-12)
  }

  test("rfft / irfft") {
    assertC(np.fft.rfft(np.array(1.0, 2.0, 3.0, 4.0, 5.0)), Seq(c(15), c(-2.5, 3.4409548011779334), c(-2.5, 0.8122992405822659)))
    val x = np.array(c(1), c(2, 1), c(3, -2), c(4))
    assertD(
      np.fft.irfft(x),
      Seq(2.5, -0.37799153207185376, -0.8660254037844386, -0.16666666666666666, 0.8660254037844386, -0.9553418012614794)
    )
    assertD(
      np.fft.irfft(x, n = 7),
      Seq(2.714285714285714, -0.38755266510443853, -0.5704813298862547, -0.6625996486021465, 0.4788556108570636,
        0.48248746463771075, -1.0549951461876486)
    )
    for n <- Seq(1, 2, 5, 6, 9, 16, 33) do
      val sig = NDArray.fromArray(signal(n).map(_.re))
      val f = np.fft.rfft(sig)
      assertEquals(f.shape, Seq(n / 2 + 1))
      assertC(f, np.fft.fft(sig).toSeq.take(n / 2 + 1), 1e-12)
      assertD(np.fft.irfft(f, n = n), sig.toSeq, 1e-12)
    intercept[IllegalArgumentException](np.fft.irfft(np.array(c(1))))
  }

  test("hfft / ihfft") {
    assertD(np.fft.hfft(np.array(c(1), c(2, 1), c(3, -2))), Seq(8.0, 0.0, 0.0, -4.0))
    assertC(np.fft.ihfft(np.array(1.0, 2.0, 3.0, 4.0, 5.0)), Seq(c(3), c(-0.5, -0.6881909602355867), c(-0.5, -0.1624598481164532)))
    val sig = np.array(1.0, 2.0, 3.0, 4.0, 5.0)
    assertD(np.fft.hfft(np.fft.ihfft(sig), n = 5), sig.toSeq)
  }

  test("rfft2 / rfftn / irfft2 / irfftn") {
    val b = np.arange(12.0).reshape(3, 4)
    val t = 13.856406460551018
    val r = np.fft.rfft2(b)
    assertEquals(r.shape, Seq(3, 3))
    assertC(r, Seq(c(66), c(-6, 6), c(-6), c(-24, t), c(0), c(0), c(-24, -t), c(0), c(0)))
    assertD(np.fft.irfft2(r, s = Seq(3, 4)), b.toSeq)
    assertD(np.fft.irfftn(np.fft.rfftn(b)), b.toSeq)
    val cube = np.arange(30.0).reshape(2, 3, 5)
    val rn = np.fft.rfftn(cube)
    assertEquals(rn.shape, Seq(2, 3, 3))
    assertD(np.fft.irfftn(rn, s = Seq(2, 3, 5)), cube.toSeq, 1e-12)
    assertC(rn, np.fft.fftn(cube)(::, ::, "0:3").toSeq, 1e-12)
  }

  test("fftfreq / rfftfreq") {
    val q = 0.2857142857142857
    assertD(np.fft.fftfreq(7, 0.5), Seq(0.0, q, 2 * q, 0.8571428571428571, -0.8571428571428571, -2 * q, -q))
    assertD(np.fft.fftfreq(8), Seq(0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125))
    assertD(np.fft.rfftfreq(7, 0.5), Seq(0.0, q, 0.5714285714285714, 0.8571428571428571))
    assertD(np.fft.rfftfreq(8), Seq(0.0, 0.125, 0.25, 0.375, 0.5))
    intercept[IllegalArgumentException](np.fft.fftfreq(0))
  }

  test("fftshift / ifftshift") {
    val a = np.arange(10).reshape(2, 5)
    assertEquals(np.fft.fftshift(a).toList, List(8, 9, 5, 6, 7, 3, 4, 0, 1, 2))
    assertEquals(np.fft.ifftshift(a, axes = 1).toList, List(2, 3, 4, 0, 1, 7, 8, 9, 5, 6))
    assertEquals(np.fft.fftshift(np.arange(7)).toList, List(4, 5, 6, 0, 1, 2, 3))
    assertEquals(np.fft.ifftshift(np.arange(7)).toList, List(3, 4, 5, 6, 0, 1, 2))
    assertEquals(np.fft.ifftshift(np.fft.fftshift(np.arange(9))).toList, (0 until 9).toList)
    assertEquals(np.fft.fftshift(np.fft.fftfreq(4)).toList, List(-0.5, -0.25, 0.0, 0.25))
    assertEquals(np.fft.fftshift(a, axes = Seq(0)).toList, List(5, 6, 7, 8, 9, 0, 1, 2, 3, 4))
  }

  test("Parseval and linearity on a long signal") {
    val n = 4099 // prime-ish length -> Bluestein
    val x = signal(n)
    val a = NDArray.fromArray(x)
    val f = np.fft.fft(a)
    val e1 = x.map(_.abs2).sum
    val e2 = f.toSeq.map(_.abs2).sum / n
    assertEqualsDouble(e2, e1, 1e-9 * e1)
    assertC(np.fft.ifft(f), x.toSeq, 1e-12)
  }

  test("errors for 0-d input") {
    intercept[IllegalArgumentException](np.fft.fft(NDArray.scalar(1.0)))
  }
