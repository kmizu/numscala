package com.github.kmizu.numscala

import NpSortImpl.*

/** Result of `np.unique` with any of `return_index`, `return_inverse`, `return_counts`:
  * each optional output is present only when requested.
  */
final case class UniqueResult[T](
    values: NDArray[T],
    indices: Option[NDArray[Int]],
    inverse: Option[NDArray[Int]],
    counts: Option[NDArray[Int]]
)

/** Result of `np.unique_all`. */
final case class UniqueAllResult[T](
    values: NDArray[T],
    indices: NDArray[Int],
    inverse_indices: NDArray[Int],
    counts: NDArray[Int]
)

/** Result of `np.unique_counts`. */
final case class UniqueCountsResult[T](values: NDArray[T], counts: NDArray[Int])

/** Result of `np.unique_inverse`. */
final case class UniqueInverseResult[T](values: NDArray[T], inverse_indices: NDArray[Int])

/** Set routines (`np.unique`, `np.isin`, `np.intersect1d`, `np.union1d`, ...). */
trait NpSet:

  /** Sorted unique elements of the flattened array (`np.unique(a)`); NaNs are collapsed. */
  def unique[T](a: NDArray[T]): NDArray[T] =
    NpSetImpl.uniqueFlat(a, equalNan = true).values

  /** `np.unique` with optional extra outputs: first-occurrence `indices`, `inverse` (which
    * reconstructs the input: shaped like `a` when `axis = None`, 1-D otherwise) and `counts`.
    * With an `axis`, unique sub-arrays along that axis are returned (sorted lexicographically).
    */
  def unique[T](
      a: NDArray[T],
      return_index: Boolean = false,
      return_inverse: Boolean = false,
      return_counts: Boolean = false,
      axis: Int | None.type = None,
      equal_nan: Boolean = true
  ): UniqueResult[T] =
    val r = axisOpt(axis) match
      case None => NpSetImpl.uniqueFlat(a, equal_nan)
      case Some(ax) => NpSetImpl.uniqueAxis(a, Shape.normAxis(ax, a.ndim), equal_nan)
    UniqueResult(
      r.values,
      Option.when(return_index)(r.indices),
      Option.when(return_inverse)(r.inverse),
      Option.when(return_counts)(r.counts)
    )

  /** Unique values (`np.unique_values`); NaNs are '''not''' collapsed (`equal_nan=False`). */
  def unique_values[T](x: NDArray[T]): NDArray[T] = NpSetImpl.uniqueFlat(x, equalNan = false).values

  /** Unique values and their counts (`np.unique_counts`). */
  def unique_counts[T](x: NDArray[T]): UniqueCountsResult[T] =
    val r = NpSetImpl.uniqueFlat(x, equalNan = false)
    UniqueCountsResult(r.values, r.counts)

  /** Unique values and inverse indices shaped like `x` (`np.unique_inverse`). */
  def unique_inverse[T](x: NDArray[T]): UniqueInverseResult[T] =
    val r = NpSetImpl.uniqueFlat(x, equalNan = false)
    UniqueInverseResult(r.values, r.inverse)

  /** Unique values, first indices, inverse indices (shaped like `x`) and counts (`np.unique_all`). */
  def unique_all[T](x: NDArray[T]): UniqueAllResult[T] =
    val r = NpSetImpl.uniqueFlat(x, equalNan = false)
    UniqueAllResult(r.values, r.indices, r.inverse, r.counts)

  /** Boolean array, shaped like `element`, telling whether each element is in `test_elements`
    * (`np.isin`). NaN is never found. `invert` negates the result.
    */
  def isin[A, B](
      element: NDArray[A],
      test_elements: NDArray[B],
      invert: Boolean = false,
      assume_unique: Boolean = false
  )(using p: Promote[A, B]): NDArray[Boolean] =
    val od = p.dtype
    val sorted = test_elements.asType(using od).toArray
    val cmp = setCompare(od)
    sortInPlace(sorted, cmp)
    val n = sorted.length
    val ed = element.dtype
    val _ = assume_unique
    element.map { e =>
      val v = od.castFrom(ed, e)
      val found =
        !od.isNaN(v) && {
          val i = bisect(n, sorted(_), v, false, cmp)
          i < n && cmp(sorted(i), v) == 0
        }
      found != invert
    }(using DType.Bool)

  /** Flattened membership test (`np.in1d`, deprecated in NumPy in favour of `isin`). */
  def in1d[A, B](ar1: NDArray[A], ar2: NDArray[B], assume_unique: Boolean = false, invert: Boolean = false)(using
      p: Promote[A, B]
  ): NDArray[Boolean] =
    isin(ar1.ravel(), ar2, invert, assume_unique)

  /** Sorted unique values present in both arrays (`np.intersect1d`). */
  def intersect1d[A, B](ar1: NDArray[A], ar2: NDArray[B], assume_unique: Boolean = false)(using
      p: Promote[A, B]
  ): NDArray[p.Out] =
    NpSetImpl.intersect(ar1.asType(using p.dtype), ar2.asType(using p.dtype), assume_unique)._1

  /** `np.intersect1d(..., return_indices=True)`: the intersection plus the indices of its
    * first occurrences in the (flattened) `ar1` and `ar2`.
    */
  def intersect1d[A, B](ar1: NDArray[A], ar2: NDArray[B], assume_unique: Boolean, return_indices: Boolean)(using
      p: Promote[A, B]
  ): (NDArray[p.Out], NDArray[Int], NDArray[Int]) =
    val _ = return_indices
    NpSetImpl.intersect(ar1.asType(using p.dtype), ar2.asType(using p.dtype), assume_unique)

  /** Sorted unique values in either array (`np.union1d`). */
  def union1d[A, B](ar1: NDArray[A], ar2: NDArray[B])(using p: Promote[A, B]): NDArray[p.Out] =
    unique(NpSetImpl.concat(ar1.asType(using p.dtype), ar2.asType(using p.dtype)))

  /** Sorted unique values of `ar1` that are not in `ar2` (`np.setdiff1d`). */
  def setdiff1d[A, B](ar1: NDArray[A], ar2: NDArray[B], assume_unique: Boolean = false)(using
      p: Promote[A, B]
  ): NDArray[A] =
    val a = if assume_unique then ar1.ravel() else unique(ar1)
    val keep = isin(a, ar2, invert = true).toArray
    val v = a.toArray
    NDArray.fromArray(v.indices.filter(keep(_)).map(v(_)).toArray(using a.dtype.classTag))(using a.dtype)

  /** Sorted unique values that are in exactly one of the arrays (`np.setxor1d`). */
  def setxor1d[A, B](ar1: NDArray[A], ar2: NDArray[B], assume_unique: Boolean = false)(using
      p: Promote[A, B]
  ): NDArray[p.Out] =
    val od = p.dtype
    val a = if assume_unique then ar1.asType(using od).ravel() else unique(ar1.asType(using od))
    val b = if assume_unique then ar2.asType(using od).ravel() else unique(ar2.asType(using od))
    val aux = NpSetImpl.concat(a, b).toArray
    val cmp = setCompare(od)
    sortInPlace(aux, cmp)
    val n = aux.length
    def eq(i: Int) = NpSetImpl.eqNoNan(od, cmp, aux(i), aux(i + 1))
    val res = (0 until n).filter(i => (i == 0 || !eq(i - 1)) && (i == n - 1 || !eq(i))).map(aux(_))
    NDArray.fromArray(res.toArray(using od.classTag))(using od)

/** Kernels for [[NpSet]]. */
private[numscala] object NpSetImpl:
  final case class Full[T](values: NDArray[T], indices: NDArray[Int], inverse: NDArray[Int], counts: NDArray[Int])

  /** Equality as `==` (NaN never equal). */
  def eqNoNan[T](d: DType[T], cmp: (T, T) => Int, x: T, y: T): Boolean =
    !d.isNaN(x) && !d.isNaN(y) && cmp(x, y) == 0

  private def same[T](d: DType[T], cmp: (T, T) => Int, equalNan: Boolean, x: T, y: T): Boolean =
    val nx = d.isNaN(x)
    val ny = d.isNaN(y)
    if nx || ny then equalNan && nx && ny else cmp(x, y) == 0

  def concat[T](a: NDArray[T], b: NDArray[T]): NDArray[T] =
    val x = a.toArray
    val y = b.toArray
    val out = a.dtype.newArray(x.length + y.length)
    System.arraycopy(x, 0, out, 0, x.length)
    System.arraycopy(y, 0, out, x.length, y.length)
    NDArray.fromArray(out)(using a.dtype)

  /** Full unique computation on the flattened array; inverse is shaped like `a`. */
  def uniqueFlat[T](a: NDArray[T], equalNan: Boolean): Full[T] =
    val d = a.dtype
    val flat = a.toArray
    val n = flat.length
    val cmp = setCompare(d)
    val perm = argsortBy(flat, cmp)
    val starts = scala.collection.mutable.ArrayBuilder.make[Int]
    val inv = new Array[Int](n)
    var g = -1
    var i = 0
    while i < n do
      if i == 0 || !same(d, cmp, equalNan, flat(perm(i - 1)), flat(perm(i))) then
        starts += i
        g += 1
      inv(perm(i)) = g
      i += 1
    val st = starts.result()
    val k = st.length
    val values = d.newArray(k)
    val idx = new Array[Int](k)
    val counts = new Array[Int](k)
    var j = 0
    while j < k do
      values(j) = flat(perm(st(j)))
      idx(j) = perm(st(j))
      counts(j) = (if j + 1 < k then st(j + 1) else n) - st(j)
      j += 1
    given DType[Int] = DType.Int32
    Full(
      NDArray.fromArray(values)(using d),
      NDArray.fromArray(idx),
      NDArray.fromArray(inv, a.shapeArr.clone()),
      NDArray.fromArray(counts)
    )

  /** Unique sub-arrays along `ax`; inverse is 1-D. */
  def uniqueAxis[T](a: NDArray[T], ax: Int, equalNan: Boolean): Full[T] =
    val d = a.dtype
    val moved = a.moveaxis(ax, 0)
    val n = moved.shapeArr(0)
    val rest = moved.shapeArr.drop(1)
    val rowLen = Shape.size(rest)
    val flat = moved.toArray
    val cmp = setCompare(d)
    def cmpRows(r1: Int, r2: Int): Int =
      var c = 0
      var j = 0
      while c == 0 && j < rowLen do
        c = cmp(flat(r1 * rowLen + j), flat(r2 * rowLen + j))
        j += 1
      c
    def sameRows(r1: Int, r2: Int): Boolean =
      var ok = true
      var j = 0
      while ok && j < rowLen do
        ok = same(d, cmp, equalNan, flat(r1 * rowLen + j), flat(r2 * rowLen + j))
        j += 1
      ok
    val perm = argsortBy(Array.tabulate(n)(identity), cmpRows)
    val starts = scala.collection.mutable.ArrayBuilder.make[Int]
    val inv = new Array[Int](n)
    var g = -1
    var i = 0
    while i < n do
      if i == 0 || !sameRows(perm(i - 1), perm(i)) then
        starts += i
        g += 1
      inv(perm(i)) = g
      i += 1
    val st = starts.result()
    val k = st.length
    val values = d.newArray(k * rowLen)
    val idx = new Array[Int](k)
    val counts = new Array[Int](k)
    var j = 0
    while j < k do
      val r = perm(st(j))
      System.arraycopy(flat, r * rowLen, values, j * rowLen, rowLen)
      idx(j) = r
      counts(j) = (if j + 1 < k then st(j + 1) else n) - st(j)
      j += 1
    given DType[Int] = DType.Int32
    val vals = NDArray.fromArray(values, k +: rest)(using d).moveaxis(0, ax).contiguous
    Full(vals, NDArray.fromArray(idx), NDArray.fromArray(inv), NDArray.fromArray(counts))

  /** NumPy's `intersect1d` algorithm (always computing the indices). */
  def intersect[T](ar1: NDArray[T], ar2: NDArray[T], assumeUnique: Boolean): (NDArray[T], NDArray[Int], NDArray[Int]) =
    val d = ar1.dtype
    val (a, ia) =
      if assumeUnique then (ar1.toArray, Array.tabulate(ar1.size)(identity))
      else
        val f = uniqueFlat(ar1, equalNan = true)
        (f.values.toArray, f.indices.toArray)
    val (b, ib) =
      if assumeUnique then (ar2.toArray, Array.tabulate(ar2.size)(identity))
      else
        val f = uniqueFlat(ar2, equalNan = true)
        (f.values.toArray, f.indices.toArray)
    val aux = concat(NDArray.fromArray(a)(using d), NDArray.fromArray(b)(using d)).toArray
    val cmp = setCompare(d)
    val perm = argsortBy(aux, cmp)
    val vals = scala.collection.mutable.ArrayBuilder.make[T](using d.classTag)
    val i1 = scala.collection.mutable.ArrayBuilder.make[Int]
    val i2 = scala.collection.mutable.ArrayBuilder.make[Int]
    var i = 0
    while i + 1 < perm.length do
      if eqNoNan(d, cmp, aux(perm(i)), aux(perm(i + 1))) then
        vals += aux(perm(i))
        val p0 = perm(i)
        val p1 = perm(i + 1)
        i1 += (if p0 < a.length then ia(p0) else ib(p0 - a.length))
        i2 += (if p1 >= a.length then ib(p1 - a.length) else ia(p1))
      i += 1
    given DType[Int] = DType.Int32
    (NDArray.fromArray(vals.result())(using d), NDArray.fromArray(i1.result()), NDArray.fromArray(i2.result()))
