package com.github.kmizu.numscala

/** `np.ma` set routines, comparisons of whole arrays, bitwise ufuncs, products,
  * convolution and `polyfit`.
  */
trait MASetOps:
  import MA.toMA
  import UfuncKind.Same

  // ------------------------------------------------------------------ whole-array comparisons

  private def isInfC(c: Complex): Boolean = c.re.isInfinite || c.im.isInfinite

  /** `np.ma.allclose(a, b, masked_equal, rtol, atol)`: true when all unmasked pairs satisfy
    * `|a - b| <= atol + rtol * |b|`; masked pairs count as equal when `masked_equal`, and
    * infinities must match exactly.
    */
  def allclose[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B], masked_equal: Boolean = true, rtol: Double = 1e-5,
      atol: Double = 1e-8): Boolean =
    val (x, y) = (toMA(a), toMA(b))
    val sh = Shape.broadcast(x._data.shapeArr, y._data.shapeArr).toIndexedSeq
    val (dx, dy) = (x.dtype, y.dtype)
    val xd = x._data.broadcastTo(sh*).toArray
    val yd = y._data.broadcastTo(sh*).toArray
    val xm = x.maskArray.broadcastTo(sh*).toArray
    val ym = y.maskArray.broadcastTo(sh*).toArray
    xd.indices.forall { i =>
      val cx = dx.toComplex(xd(i))
      val cy = dy.toComplex(yd(i))
      val masked = xm(i) || ym(i)
      val xinf = !masked && isInfC(cx)
      val yinf = !ym(i) && isInfC(cy)
      if xinf != yinf then false
      else if xinf then cx == cy
      else if masked then masked_equal
      else (cx - cy).abs <= atol + rtol * cy.abs
    }

  /** `np.ma.allequal(a, b, fill_value)`: true when all entries are equal; masked entries count as
    * equal when `fill_value` is true, otherwise any masked entry makes the result false.
    */
  def allequal[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B], fill_value: Boolean = true)(using p: Promote[A, B]): Boolean =
    val (x, y) = (toMA(a), toMA(b))
    val m = MaskedArray.maskOr(x._mask, y._mask)
    val d = Ops.equal(x._data, y._data, p.dtype, true)
    if m == null then d.all()
    else if fill_value then NDArray.zipMap(d, m)(_ || _).all()
    else false

  // ------------------------------------------------------------------ bitwise ufuncs

  private def bitOp[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B], u: Ufunc[Same])(using
      t: UfuncTypes[Same, A, B]
  ): MaskedArray[t.Out] =
    MaskedArray.binaryWith(toMA(a), toMA(b), t.out)((x, y) => u(x, y))

  private def sc[B](b: B, d: DType[B]): MaskedArray[B] = MaskedArray.scalarOf(b)(using d)

  /** `np.ma.bitwise_and(a, b)`: masked where either input is masked. */
  def bitwise_and[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using t: UfuncTypes[Same, A, B]): MaskedArray[t.Out] =
    bitOp(a, b, np.bitwise_and)
  def bitwise_and[A, B](a: MaskedArrayLike[A], b: B)(using db: DType[B], t: UfuncTypes[Same, A, B]): MaskedArray[t.Out] =
    bitOp(a, sc(b, db), np.bitwise_and)
  /** `np.ma.bitwise_or(a, b)`. */
  def bitwise_or[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using t: UfuncTypes[Same, A, B]): MaskedArray[t.Out] =
    bitOp(a, b, np.bitwise_or)
  def bitwise_or[A, B](a: MaskedArrayLike[A], b: B)(using db: DType[B], t: UfuncTypes[Same, A, B]): MaskedArray[t.Out] =
    bitOp(a, sc(b, db), np.bitwise_or)
  /** `np.ma.bitwise_xor(a, b)`. */
  def bitwise_xor[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using t: UfuncTypes[Same, A, B]): MaskedArray[t.Out] =
    bitOp(a, b, np.bitwise_xor)
  def bitwise_xor[A, B](a: MaskedArrayLike[A], b: B)(using db: DType[B], t: UfuncTypes[Same, A, B]): MaskedArray[t.Out] =
    bitOp(a, sc(b, db), np.bitwise_xor)
  /** `np.ma.left_shift(a, b)`. */
  def left_shift[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using t: UfuncTypes[Same, A, B]): MaskedArray[t.Out] =
    bitOp(a, b, np.left_shift)
  def left_shift[A, B](a: MaskedArrayLike[A], b: B)(using db: DType[B], t: UfuncTypes[Same, A, B]): MaskedArray[t.Out] =
    bitOp(a, sc(b, db), np.left_shift)
  /** `np.ma.right_shift(a, b)`. */
  def right_shift[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using t: UfuncTypes[Same, A, B]): MaskedArray[t.Out] =
    bitOp(a, b, np.right_shift)
  def right_shift[A, B](a: MaskedArrayLike[A], b: B)(using db: DType[B], t: UfuncTypes[Same, A, B]): MaskedArray[t.Out] =
    bitOp(a, sc(b, db), np.right_shift)

  // ------------------------------------------------------------------ products

  private def filled0[T](a: MaskedArrayLike[T]): NDArray[T] =
    val m = toMA(a)
    m.filled(m.dtype.zero)

  /** `np.ma.inner(a, b)`: inner product with masked values treated as 0 (unmasked result). */
  def inner[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using p: NumPromote[A, B]): MaskedArray[p.Out] =
    val fa = filled0(a)
    val fb = filled0(b)
    MaskedArray.wrap(np.inner(if fa.ndim == 0 then fa.reshape(1) else fa, if fb.ndim == 0 then fb.reshape(1) else fb))
  /** `np.ma.innerproduct` (alias of [[inner]]). */
  def innerproduct[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using p: NumPromote[A, B]): MaskedArray[p.Out] = inner(a, b)

  /** `np.ma.outer(a, b)`: outer product of the flattened inputs, masked where either factor is masked. */
  def outer[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using p: NumPromote[A, B]): MaskedArray[p.Out] =
    val (x, y) = (toMA(a), toMA(b))
    val d = np.outer(filled0(x).ravel(), filled0(y).ravel())
    if x._mask == null && y._mask == null then MaskedArray.wrap(d)
    else
      val (mx, my) = (x.maskArray.toArray, y.maskArray.toArray)
      val m = NDArray.tabulate(mx.length, my.length)(i => mx(i / my.length) || my(i % my.length))(using DType.Bool)
      new MaskedArray(d, if m.any() then m else null, None, false)
  /** `np.ma.outerproduct` (alias of [[outer]]). */
  def outerproduct[A, B](a: MaskedArrayLike[A], b: MaskedArrayLike[B])(using p: NumPromote[A, B]): MaskedArray[p.Out] = outer(a, b)

  // ------------------------------------------------------------------ convolution

  private def ints(b: NDArray[Boolean]): NDArray[Int] = b.map(x => if x then 1 else 0)(using DType.Int32)

  private def convOrCorr[T](a: MaskedArrayLike[T], v: MaskedArrayLike[T], mode: String, propagate: Boolean, conv: Boolean)(using
      p: NumPromote[T, T]
  ): MaskedArray[p.Out] =
    val (x, y) = (toMA(a), toMA(v))
    def fi(u: NDArray[Int], w: NDArray[Int]): NDArray[Int] = if conv then np.convolve(u, w, mode) else np.correlate(u, w, mode)
    def fd(u: NDArray[T], w: NDArray[T]): NDArray[p.Out] = if conv then np.convolve(u, w, mode) else np.correlate(u, w, mode)
    def ones(m: MaskedArray[?]): NDArray[Int] = NDArray.fillOf(DType.Int32, m._data.shapeArr.clone(), 1)
    if propagate then
      val m1 = fi(ints(x.maskArray), ones(y))
      val m2 = fi(ones(x), ints(y.maskArray))
      new MaskedArray(fd(x._data, y._data), NDArray.zipMap(m1, m2)((i, j) => i > 0 || j > 0)(using DType.Bool), None, false)
    else
      val valid = fi(ints(x.maskArray.map(!_)), ints(y.maskArray.map(!_)))
      new MaskedArray(fd(filled0(x), filled0(y)), valid.map(_ == 0)(using DType.Bool), None, false)

  /** `np.ma.convolve(a, v, mode, propagate_mask)`: with `propagate_mask` an output is masked when
    * any contributing input is masked; otherwise when no pair of unmasked inputs contributes.
    */
  def convolve[T](a: MaskedArrayLike[T], v: MaskedArrayLike[T], mode: String = "full", propagate_mask: Boolean = true)(using
      p: NumPromote[T, T]
  ): MaskedArray[p.Out] = convOrCorr(a, v, mode, propagate_mask, conv = true)

  /** `np.ma.correlate(a, v, mode, propagate_mask)` (see [[convolve]] for the masking rules). */
  def correlate[T](a: MaskedArrayLike[T], v: MaskedArrayLike[T], mode: String = "valid", propagate_mask: Boolean = true)(using
      p: NumPromote[T, T]
  ): MaskedArray[p.Out] = convOrCorr(a, v, mode, propagate_mask, conv = false)

  // ------------------------------------------------------------------ set routines

  /** Masked `==` / `!=` of two elements, NumPy-style: `(value, masked)`. */
  private def cmpAt[T](d: DType[T], v: Array[T], m: Array[Boolean], i: Int, j: Int, eq: Boolean): (Boolean, Boolean) =
    if m(i) || m(j) then ((m(i) == m(j)) == eq, true)
    else (d.equiv(v(i), v(j)) == eq, false)

  private def gather[T](d: DType[T], v: Array[T], m: Array[Boolean], idx: Array[Int], keepMask: Boolean): MaskedArray[T] =
    val out = d.newArray(idx.length)
    var k = 0
    while k < idx.length do
      out(k) = v(idx(k))
      k += 1
    val data = NDArray.fromArray(out, Array(idx.length))(using d)
    new MaskedArray(data, if keepMask then NDArray.fromArray(idx.map(m(_)), Array(idx.length)) else null, None, false)

  /** NumPy's `unique(ar, return_inverse=True)` on a masked array: sorted unique values with one
    * trailing masked entry (if any), and for each input element its index in the result.
    */
  private def uniqueInv[T](a: MaskedArray[T]): (MaskedArray[T], Array[Int]) =
    val ar = a.ravel()
    val d = ar.dtype
    val perm = ar.argsort().toArray
    val v0 = ar._data.toArray
    val m0 = ar.maskArray.toArray
    val v = perm.map(v0(_))(using d.classTag)
    val m = perm.map(m0(_))
    val n = v.length
    val flag = Array.tabulate(n) { i =>
      i == 0 || {
        val (dif, _) = cmpAt(d, v, m, i, i - 1, eq = false)
        if m(i) || m(i - 1) then dif else dif && !(d.isNaN(v(i)) && d.isNaN(v(i - 1)))
      }
    }
    val inv = new Array[Int](n)
    var c = -1
    var i = 0
    while i < n do
      if flag(i) then c += 1
      inv(perm(i)) = c
      i += 1
    (gather(d, v, m, (0 until n).filter(flag(_)).toArray, ar._mask != null), inv)

  private def concat1d[T](a: MaskedArray[T], b: MaskedArray[T]): MaskedArray[T] = MA.concatenate(Seq(a.ravel(), b.ravel()), 0)

  private def in1dImpl[T](ar1: MaskedArray[T], ar2: MaskedArray[T], assumeUnique: Boolean, invert: Boolean): MaskedArray[Boolean] =
    val (u1, rev) = if assumeUnique then (ar1.ravel(), null) else uniqueInv(ar1)
    val u2 = if assumeUnique then ar2.ravel() else uniqueInv(ar2)._1
    val ar = concat1d(u1, u2)
    val d = ar.dtype
    val order = ar.argsort().toArray
    val v0 = ar._data.toArray
    val m0 = ar.maskArray.toArray
    val sv = order.map(v0(_))(using d.classTag)
    val sm = order.map(m0(_))
    val n = order.length
    val flagD = new Array[Boolean](n)
    val flagM = new Array[Boolean](n)
    var i = 0
    while i < n - 1 do
      val (x, mk) = cmpAt(d, sv, sm, i + 1, i, eq = !invert)
      flagD(i) = x
      flagM(i) = mk
      i += 1
    if n > 0 then flagD(n - 1) = invert
    val pos = new Array[Int](n)
    i = 0
    while i < n do
      pos(order(i)) = i
      i += 1
    val sel0 = pos.take(u1.size)
    val sel = if rev == null then sel0 else rev.map(sel0(_))
    val data = NDArray.fromArray(sel.map(flagD(_)), Array(sel.length))
    new MaskedArray(data, if flagM.exists(identity) then NDArray.fromArray(sel.map(flagM(_)), Array(sel.length)) else null, None, false)

  /** `np.ma.in1d(ar1, ar2, assume_unique, invert)`: whether each element of the flattened `ar1`
    * is in `ar2`, following NumPy's masked algorithm (sort-based, masked values compare equal).
    */
  def in1d[T](ar1: MaskedArrayLike[T], ar2: MaskedArrayLike[T], assume_unique: Boolean = false, invert: Boolean = false): MaskedArray[Boolean] =
    in1dImpl(toMA(ar1), toMA(ar2), assume_unique, invert)

  /** `np.ma.isin(element, test_elements, assume_unique, invert)`: [[in1d]] shaped like `element`. */
  def isin[T](element: MaskedArrayLike[T], test_elements: MaskedArrayLike[T], assume_unique: Boolean = false,
      invert: Boolean = false): MaskedArray[Boolean] =
    val e = toMA(element)
    in1dImpl(e, toMA(test_elements), assume_unique, invert).reshape(e.shape*)

  private def sortedConcat[T](ar1: MaskedArrayLike[T], ar2: MaskedArrayLike[T], assumeUnique: Boolean): MaskedArray[T] =
    val (a, b) = (toMA(ar1), toMA(ar2))
    val aux = if assumeUnique then concat1d(a, b) else concat1d(uniqueInv(a)._1, uniqueInv(b)._1)
    MAReduce.sorted(aux, -1, true)

  /** `np.ma.intersect1d(ar1, ar2, assume_unique)`: sorted unique values in both arrays; one
    * masked entry is kept when both inputs contain masked values.
    */
  def intersect1d[T](ar1: MaskedArrayLike[T], ar2: MaskedArrayLike[T], assume_unique: Boolean = false): MaskedArray[T] =
    val aux = sortedConcat(ar1, ar2, assume_unique)
    val d = aux.dtype
    val v = aux._data.toArray
    val m = aux.maskArray.toArray
    val keep = (0 until math.max(v.length - 1, 0)).filter(i => cmpAt(d, v, m, i + 1, i, eq = true)._1).toArray
    gather(d, v, m, keep, aux._mask != null)

  /** `np.ma.union1d(ar1, ar2)`: sorted unique values of either array (masked values collapse to one). */
  def union1d[T](ar1: MaskedArrayLike[T], ar2: MaskedArrayLike[T]): MaskedArray[T] =
    uniqueInv(concat1d(toMA(ar1), toMA(ar2)))._1

  /** `np.ma.setdiff1d(ar1, ar2, assume_unique)`: sorted unique values of `ar1` not in `ar2`. */
  def setdiff1d[T](ar1: MaskedArrayLike[T], ar2: MaskedArrayLike[T], assume_unique: Boolean = false): MaskedArray[T] =
    val a = if assume_unique then toMA(ar1).ravel() else uniqueInv(toMA(ar1))._1
    val b = if assume_unique then toMA(ar2) else uniqueInv(toMA(ar2))._1
    val f = in1dImpl(a, b, assumeUnique = true, invert = true)._data.toArray
    gather(a.dtype, a._data.toArray, a.maskArray.toArray, f.indices.filter(f(_)).toArray, a._mask != null)

  /** `np.ma.setxor1d(ar1, ar2, assume_unique)`: sorted unique values in exactly one of the arrays. */
  def setxor1d[T](ar1: MaskedArrayLike[T], ar2: MaskedArrayLike[T], assume_unique: Boolean = false): MaskedArray[T] =
    val aux = sortedConcat(ar1, ar2, assume_unique)
    val n = aux.size
    if n == 0 then return aux
    val d = aux.dtype
    val auxf = aux.filled().toArray
    val flag = Array.tabulate(n + 1)(i => i == 0 || i == n || !d.equiv(auxf(i), auxf(i - 1)))
    val keep = (0 until n).filter(i => flag(i) == flag(i + 1)).toArray
    gather(d, aux._data.toArray, aux.maskArray.toArray, keep, aux._mask != null)

  // ------------------------------------------------------------------ polyfit

  /** `np.ma.polyfit(x, y, deg, rcond, w)`: least-squares polynomial fit (highest power first)
    * using only the points where none of `x`, `y` (any column, for 2-D `y`) and `w` is masked.
    */
  def polyfit[A, B](x: MaskedArrayLike[A], y: MaskedArrayLike[B], deg: Int, rcond: Double = -1.0,
      w: MaskedArrayLike[Double] | Null = null): NDArray[Double] =
    val (mx, my) = (toMA(x), toMA(y))
    var m = mx._mask
    my.ndim match
      case 1 => m = MaskedArray.maskOr(m, my._mask)
      case 2 => if my._mask != null then m = MaskedArray.maskOr(m, my._mask.any(1))
      case _ => throw new IllegalArgumentException("Expected a 1D or 2D array for y!")
    val mw = if w == null then null else toMA(w.asInstanceOf[MaskedArrayLike[Double]])
    if mw != null then
      if mw.ndim != 1 then throw new IllegalArgumentException("expected a 1-d array for weights")
      if mw.shape(0) != my.shape(0) then throw new IllegalArgumentException("expected w and y to have the same length")
      m = MaskedArray.maskOr(m, mw._mask)
    if m == null then PolyFitImpl.plain(mx._data, my._data, deg, rcond, if mw == null then null else mw._data)
    else
      val keep = m.map(!_)(using DType.Bool)
      PolyFitImpl.plain(np.compress(keep, mx._data), np.compress(keep, my._data, 0), deg, rcond,
        if mw == null then null else np.compress(keep, mw._data))
