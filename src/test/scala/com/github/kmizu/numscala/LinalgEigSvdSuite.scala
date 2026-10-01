package com.github.kmizu.numscala

class LinalgEigSvdSuite extends LinalgTestUtil:

  private def sortC(xs: Seq[Complex]): Seq[Complex] = xs.sortBy(c => (math.rint(c.re * 1e8), c.im))

  /** max |A v_i - w_i v_i| over the columns. */
  private def eigResidual(a: NDArray[Complex], r: EigResult): Double =
    val av = a @@ r.eigenvectors
    val vw = r.eigenvectors * r.eigenvalues.reshape(1, -1)
    maxAbsDiffC(av, vw)

  private def toC(a: NDArray[Double]): NDArray[Complex] = a.astype[Complex]

  test("eig: real matrices") {
    val r = np.linalg.eig(B)
    closeC(sortC(r.eigenvalues.toSeq).pipe(np.array(_)), sortC(Seq(Complex(16.707493316124744, 0), Complex(-0.9057401795217589, 0),
      Complex(0.1982468633970087, 0))), 1e-12)
    r.eigenvalues.toSeq.foreach(c => assertEquals(c.im, 0.0))
    r.eigenvectors.toSeq.foreach(c => assertEquals(c.im, 0.0))
    assert(eigResidual(toC(B), r) < 1e-12)
    // unit-norm columns
    val norms = np.linalg.norm(r.eigenvectors, axis = 0)
    close(norms, Seq(1.0, 1.0, 1.0), 1e-14)
    // complex conjugate pair, positive imaginary part first
    val mm = m(Seq(1.0, 2, 0), Seq(-2.0, 1, 3), Seq(0.0, 1, 1))
    val rm = np.linalg.eig(mm)
    closeC(np.array(sortC(rm.eigenvalues.toSeq)), sortC(Seq(Complex(1, 1), Complex(1, -1), Complex(1, 0))), 1e-12)
    val w = rm.eigenvalues.toSeq
    val k = w.indexWhere(_.im > 0)
    assertEquals(w(k + 1), w(k).conj)
    assert(eigResidual(toC(mm), rm) < 1e-12)
    val rot = np.linalg.eigvals(m(Seq(0.0, -1), Seq(1.0, 0)))
    closeC(rot, Seq(Complex(0, 1), Complex(0, -1)), 1e-15)
    // diagonal / triangular / zero
    closeC(np.linalg.eigvals(np.diag(np.array(3.0, 1.0, 2.0))), Seq(Complex(3, 0), Complex(1, 0), Complex(2, 0)), 0)
    closeC(np.linalg.eigvals(np.zeros(2, 2)), Seq(Complex.Zero, Complex.Zero), 0)
    // larger random matrix
    val big = rand(11, 40, 40)
    val rb = np.linalg.eig(big)
    assert(eigResidual(toC(big), rb) < 1e-11)
    val tr = rb.eigenvalues.toSeq.foldLeft(Complex.Zero)(_ + _)
    assert((tr - Complex(np.trace(big), 0)).abs < 1e-11)
    // badly scaled matrix (balancing)
    val bad = m(Seq(1.0, 1e6, 0), Seq(1e-6, 2, 1e5), Seq(0.0, 1e-5, 3))
    val rbad = np.linalg.eig(bad)
    assert(eigResidual(toC(bad), rbad) < 1e-6)
    // defective matrix
    val jd = np.linalg.eig(m(Seq(1.0, 1), Seq(0.0, 1)))
    closeC(jd.eigenvalues, Seq(Complex(1, 0), Complex(1, 0)), 0)
  }

  test("eig: complex, stacked, errors") {
    val r = np.linalg.eig(Z)
    closeC(np.array(sortC(r.eigenvalues.toSeq)),
      sortC(Seq(Complex(0.18802800918855267, 0.4390040158566455), Complex(4.811971990811448, 2.0609959841433545))), 1e-12)
    assert(eigResidual(Z, r) < 1e-13)
    val big = randC(3, 30, 30)
    val rb = np.linalg.eig(big)
    assert(eigResidual(big, rb) < 1e-11)
    // largest component of each eigenvector is real
    val v = rb.eigenvectors
    for j <- 0 until 30 do
      val col = (0 until 30).map(i => v(i, j))
      val big1 = col.maxBy(_.abs)
      assertEquals(big1.im, 0.0)
    val st = rand(5, 3, 4, 4)
    val rs = np.linalg.eig(st)
    assertEquals(rs.eigenvalues.shape, Seq(3, 4))
    assertEquals(rs.eigenvectors.shape, Seq(3, 4, 4))
    for b <- 0 until 3 do
      assert(eigResidual(toC(st(b, ::, ::)), EigResult(rs.eigenvalues(b, ::), rs.eigenvectors(b, ::, ::))) < 1e-12)
    intercept[LinAlgError](np.linalg.eig(C))
    intercept[LinAlgError](np.linalg.eig(m(Seq(1.0, Double.NaN), Seq(0.0, 1))))
    val ei: NDArray[Complex] = np.linalg.eigvals(np.array(Seq(Seq(2, 0), Seq(0, 3))))
    closeC(ei, Seq(Complex(2, 0), Complex(3, 0)), 0)
  }

  test("eigh / eigvalsh") {
    val r = np.linalg.eigh(A)
    close(r.eigenvalues, Seq(1.854897308799576, 3.4760236029181333, 6.669079088282287), 1e-13)
    val v = r.eigenvectors
    assert(maxAbsDiff(v.T @@ v, np.eye(3)) < 1e-14)
    assert(maxAbsDiff(A @@ v, v * r.eigenvalues.reshape(1, 3)) < 1e-13)
    val h = cm(
      Seq(Complex(2, 0), Complex(1, -1), Complex(0, 0)),
      Seq(Complex(1, 1), Complex(3, 0), Complex(0, 2)),
      Seq(Complex(0, 0), Complex(0, -2), Complex(1, 0))
    )
    val rh = np.linalg.eigh(h)
    close(rh.eigenvalues, Seq(-0.4892885718100789, 1.7108314535516893, 4.7784571182583875), 1e-13)
    val vh = rh.eigenvectors
    assert(maxAbsDiffC(adjoint(vh) @@ vh, np.eye[Complex](3)) < 1e-14)
    assert(maxAbsDiffC(h @@ vh, vh * rh.eigenvalues.astype[Complex].reshape(1, 3)) < 1e-13)
    // UPLO: only one triangle is used
    val lowerOnly = m(Seq(2.0, 100), Seq(1.0, 2))
    close(np.linalg.eigvalsh(lowerOnly), Seq(1.0, 3.0), 1e-14)
    close(np.linalg.eigvalsh(lowerOnly, UPLO = "U"), Seq(-98.0, 102.0), 1e-12)
    intercept[IllegalArgumentException](np.linalg.eigvalsh(lowerOnly, UPLO = "X"))
    // bigger symmetric, stacked
    val g = rand(21, 50, 50)
    val sym = g + g.T
    val rs = np.linalg.eigh(sym)
    assert(maxAbsDiff(sym @@ rs.eigenvectors, rs.eigenvectors * rs.eigenvalues.reshape(1, 50)) < 1e-12)
    val ev = rs.eigenvalues.toSeq
    assert(ev.zip(ev.tail).forall((x, y) => x <= y))
    val gc = randC(22, 25, 25)
    val herm = gc + adjoint(gc)
    val rc = np.linalg.eigh(herm)
    assert(maxAbsDiffC(herm @@ rc.eigenvectors, rc.eigenvectors * rc.eigenvalues.astype[Complex].reshape(1, 25)) < 1e-12)
    val st = np.linalg.eigvalsh(np.array(Seq(Seq(Seq(2.0, 0), Seq(0.0, 1)), Seq(Seq(0.0, 1), Seq(1.0, 0)))))
    close(st, Seq(1.0, 2.0, -1.0, 1.0), 1e-15)
    // repeated eigenvalues
    val rep = np.linalg.eigh(np.eye(4) * 3.0)
    close(rep.eigenvalues, Seq(3.0, 3.0, 3.0, 3.0), 0)
    val ii: NDArray[Double] = np.linalg.eigvalsh(np.array(Seq(Seq(2, 1), Seq(1, 2))))
    close(ii, Seq(1.0, 3.0), 1e-14)
  }

  test("svd: values, reconstruction, shapes") {
    close(np.linalg.svd(B, compute_uv = false).S, Seq(17.412505166808597, 0.8751613501104367, 0.19686652111743008), 1e-13)
    close(np.linalg.svdvals(C), Seq(9.525518091565107, 0.514300580658644), 1e-13)
    close(np.linalg.svdvals(Z), Seq(5.92214438511238, 0.42214438511237984), 1e-13)
    for (rows, cols) <- Seq((5, 3), (3, 5), (4, 4), (1, 4), (4, 1)) do
      val a = rand(rows * 10 + cols, rows, cols)
      val k = math.min(rows, cols)
      val f = np.linalg.svd(a)
      assertEquals(f.U.shape, Seq(rows, rows))
      assertEquals(f.Vh.shape, Seq(cols, cols))
      assert(maxAbsDiff(f.U.T @@ f.U, np.eye(rows)) < 1e-13, s"U orth $rows x $cols")
      assert(maxAbsDiff(f.Vh @@ f.Vh.T, np.eye(cols)) < 1e-13, s"V orth $rows x $cols")
      val red = np.linalg.svd(a, full_matrices = false)
      assertEquals(red.U.shape, Seq(rows, k))
      assertEquals(red.Vh.shape, Seq(k, cols))
      val rec = (red.U * red.S.reshape(1, k)) @@ red.Vh
      assert(maxAbsDiff(rec, a) < 1e-13, s"reconstruction $rows x $cols")
      val s = red.S.toSeq
      assert(s.zip(s.tail).forall((x, y) => x >= y))
    // rank deficient: full orthonormal basis still produced
    val ones = np.ones(3, 3)
    val so = np.linalg.svd(ones)
    close(so.S, Seq(3.0, 0, 0), 1e-14)
    assert(maxAbsDiff(so.U.T @@ so.U, np.eye(3)) < 1e-13)
    assert(maxAbsDiff(so.Vh @@ so.Vh.T, np.eye(3)) < 1e-13)
    val zs = np.linalg.svd(np.zeros(2, 3))
    close(zs.S, Seq(0.0, 0.0), 0)
    assert(maxAbsDiff(zs.U.T @@ zs.U, np.eye(2)) < 1e-15)
    // complex
    val zc = randC(31, 6, 4)
    val rz = np.linalg.svd(zc)
    assertEquals(rz.U.shape, Seq(6, 6))
    assert(maxAbsDiffC(adjoint(rz.U) @@ rz.U, np.eye[Complex](6)) < 1e-13)
    val rzr = np.linalg.svd(zc, full_matrices = false)
    val rec = (rzr.U * rzr.S.astype[Complex].reshape(1, 4)) @@ rzr.Vh
    assert(maxAbsDiffC(rec, zc) < 1e-13)
    val zw = randC(32, 3, 5)
    val rw = np.linalg.svd(zw, full_matrices = false)
    assert(maxAbsDiffC((rw.U * rw.S.astype[Complex].reshape(1, 3)) @@ rw.Vh, zw) < 1e-13)
    // stacked
    val st = rand(41, 2, 3, 4, 2)
    val rst = np.linalg.svd(st, full_matrices = false)
    assertEquals(rst.U.shape, Seq(2, 3, 4, 2))
    assertEquals(rst.S.shape, Seq(2, 3, 2))
    assertEquals(rst.Vh.shape, Seq(2, 3, 2, 2))
    // hermitian
    val sym = m(Seq(2.0, -3), Seq(-3.0, 1))
    val hs = np.linalg.svd(sym, hermitian = true)
    close(hs.S, np.linalg.svdvals(sym).toSeq, 1e-13)
    assert(maxAbsDiff((hs.U * hs.S.reshape(1, 2)) @@ hs.Vh, sym) < 1e-13)
    // integer input
    val si: NDArray[Double] = np.linalg.svdvals(np.array(Seq(Seq(3, 0), Seq(0, 4))))
    close(si, Seq(4.0, 3.0), 1e-15)
    // larger, ill-conditioned
    val hilbert = NDArray.tabulate(8, 8)(k => 1.0 / ((k / 8) + (k % 8) + 1))
    val hsv = np.linalg.svdvals(hilbert).toSeq
    assertEqualsDouble(hsv.head, 1.6959389969219496, 1e-13)
    assert(math.abs(hsv.last - 1.1115389793345086e-10) < 1e-16)
  }

  test("edge cases: scaling, tiny / empty matrices, NaN, other dtypes") {
    val big = B * 1e200
    close(np.linalg.svdvals(big), Seq(17.412505166808597e200, 0.8751613501104367e200, 0.19686652111743008e200), 1e-12)
    val tiny = B * 1e-200
    close(np.linalg.svdvals(tiny), Seq(17.412505166808597e-200, 0.8751613501104367e-200, 0.19686652111743008e-200), 1e-12)
    val ev = np.linalg.eigvals(tiny).toSeq.map(_.re).sorted
    close(np.array(ev), Seq(-0.9057401795217589e-200, 0.1982468633970087e-200, 16.707493316124744e-200), 1e-12)
    val evb = np.linalg.eigvals(big.astype[Complex]).toSeq.map(_.re).sorted
    close(np.array(evb), Seq(-0.9057401795217589e200, 0.1982468633970087e200, 16.707493316124744e200), 1e-11)
    intercept[LinAlgError](np.linalg.svd(m(Seq(1.0, Double.NaN), Seq(0.0, 1))))
    val one = np.linalg.eig(m(Seq(5.0)))
    closeC(one.eigenvalues, Seq(Complex(5, 0)), 0)
    closeC(one.eigenvectors, Seq(Complex(1, 0)), 0)
    val empty = np.linalg.eig(np.zeros(0, 0))
    assertEquals(empty.eigenvalues.shape, Seq(0))
    assertEquals(empty.eigenvectors.shape, Seq(0, 0))
    assertEquals(np.linalg.eigh(np.zeros(0, 0)).eigenvalues.shape, Seq(0))
    assertEquals(np.linalg.svd(np.zeros(0, 3)).Vh.shape, Seq(3, 3))
    val f32 = np.array(Seq(Seq(2.0f, 0.0f), Seq(0.0f, 3.0f)))
    val d32: NDArray[Double] = np.linalg.det(f32)
    assertEqualsDouble(d32.item, 6.0, 1e-12)
    val bools = np.array(Seq(Seq(true, false), Seq(true, true)))
    assertEqualsDouble(np.linalg.det(bools).item, 1.0, 0)
    assertEquals(np.linalg.inv(np.array(Seq(Seq(2L, 0L), Seq(0L, 4L)))).toList, List(0.5, 0.0, 0.0, 0.25))
    val zs = np.linalg.svd(Z, compute_uv = false)
    assertEquals(zs.U.size, 0)
    close(zs.S, Seq(5.92214438511238, 0.42214438511237984), 1e-13)
    val zsh = np.linalg.svd(cm(Seq(Complex(2, 0), Complex(0, -1)), Seq(Complex(0, 1), Complex(-3, 0))), hermitian = true)
    val herm = cm(Seq(Complex(2, 0), Complex(0, -1)), Seq(Complex(0, 1), Complex(-3, 0)))
    assert(maxAbsDiffC((zsh.U * zsh.S.astype[Complex].reshape(1, 2)) @@ zsh.Vh, herm) < 1e-13)
    // stacked complex cholesky / det / inv
    val hz = np.array(Seq(Seq(Seq(Complex(4, 0), Complex(1, 1)), Seq(Complex(1, -1), Complex(3, 0)))))
    val lz = np.linalg.cholesky(hz)
    assertEquals(lz.shape, Seq(1, 2, 2))
    assert(maxAbsDiffC(lz(0, ::, ::) @@ adjoint(lz(0, ::, ::)), hz(0, ::, ::)) < 1e-14)
    assert((np.linalg.det(hz).toSeq.head - Complex(10, 0)).abs < 1e-13)
    // upper-triangle eigh for complex input
    val hu = cm(Seq(Complex(2, 0), Complex(1, -1)), Seq(Complex(99, 99), Complex(3, 0)))
    val full = cm(Seq(Complex(2, 0), Complex(1, -1)), Seq(Complex(1, 1), Complex(3, 0)))
    close(np.linalg.eigvalsh(hu, UPLO = "U"), np.linalg.eigvalsh(full).toSeq, 1e-14)
  }

  extension [X](x: X) private def pipe[Y](f: X => Y): Y = f(x)
