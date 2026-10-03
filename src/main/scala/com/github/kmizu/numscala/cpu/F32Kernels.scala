package com.github.kmizu.numscala.cpu

/** Float32 CPU kernels (contract NS-CPU-KERNEL-1, K1–K4).
  *
  * Every public method validates all arguments first and only then computes, so a rejected call
  * never leaves an output partially updated. `*Into` methods write into caller-owned outputs.
  * Outputs must not overlap inputs. The only exception is elementwise ops (`axpyInto`, `mulInto`,
  * `sigmoidInto`, `siluInto`, `copyInto`, `affineScanInto` w.r.t. `a`/`b`), where the output may be
  * exactly the same range as an input. Partial overlaps are always rejected. Kernels are
  * single-threaded and create no thread pools. Backends may differ in rounding (FMA, summation
  * order); NaN and Infinity are never skipped by shortcuts.
  */
trait F32Kernels:

  /** Short backend name (`"scalar"`, `"vector25"`, ...). */
  def name: String

  /** One-line description of the backend configuration, for logs and benchmark reports. */
  def describe: String = name

  // ------------------------------------------------------------------ K1: GEMM

  /** `C := alpha * op(A) * op(B) + beta * C` (BLAS `sgemm` semantics).
    *
    * `beta == 0` never reads C (C may hold garbage or NaN). `K == 0` or `alpha == 0` never reads A
    * or B. A `TT` product packs `B` into the workspace, so it needs [[gemmWorkspaceFloats]].
    */
  final def gemmInto(
      a: MatrixF32, transA: Transpose,
      b: MatrixF32, transB: Transpose,
      c: MatrixF32,
      alpha: Float, beta: Float,
      workspace: Workspace
  ): Unit =
    a.validate("gemmInto: A")
    b.validate("gemmInto: B")
    c.validate("gemmInto: C")
    val m = a.logicalRows(transA)
    val k = a.logicalCols(transA)
    val k2 = b.logicalRows(transB)
    val n = b.logicalCols(transB)
    if k != k2 then throw new IllegalArgumentException(s"gemmInto: inner dimensions differ (op(A) is $m x $k, op(B) is $k2 x $n)")
    if c.rows != m || c.cols != n then
      throw new IllegalArgumentException(s"gemmInto: C is ${c.rows} x ${c.cols}, expected $m x $n")
    if MatrixF32.overlaps(c, a) then throw new IllegalArgumentException("gemmInto: C overlaps A")
    if MatrixF32.overlaps(c, b) then throw new IllegalArgumentException("gemmInto: C overlaps B")
    if m == 0 || n == 0 then return
    val readsAB = k != 0 && alpha != 0f
    val need = if readsAB then gemmWorkspaceFloats(m, n, k, transA, transB) else 0
    workspace.require(need, 0, "gemmInto")
    val mk = workspace.mark
    try gemmImpl(a, transA, b, transB, c, m, n, k, alpha, beta, readsAB, workspace)
    finally workspace.release(mk)

  /** `C := op(A) * op(B)` (`alpha = 1`, `beta = 0`). */
  final def gemmInto(a: MatrixF32, transA: Transpose, b: MatrixF32, transB: Transpose, c: MatrixF32, workspace: Workspace): Unit =
    gemmInto(a, transA, b, transB, c, 1f, 0f, workspace)

  /** Workspace floats a `gemmInto` with logical shape `(m x k) * (k x n)` needs on this backend. */
  def gemmWorkspaceFloats(m: Int, n: Int, k: Int, transA: Transpose, transB: Transpose): Int =
    if transA == Transpose.Yes && transB == Transpose.Yes then Math.multiplyExact(k, n) else 0

  /** Backend GEMM on validated arguments. `readsAB == false` means: only apply `beta` to C. */
  protected def gemmImpl(
      a: MatrixF32, transA: Transpose, b: MatrixF32, transB: Transpose, c: MatrixF32,
      m: Int, n: Int, k: Int, alpha: Float, beta: Float, readsAB: Boolean, ws: Workspace
  ): Unit

  // ------------------------------------------------------------------ K2: sparse rows

  /** `out(r, :) = table(ids(idsOffset + r), :)` for `r < count`. Out-of-range IDs throw (no negative wrap-around). */
  final def gatherRowsInto(table: MatrixF32, ids: Array[Int], idsOffset: Int, count: Int, out: MatrixF32): Unit =
    table.validate("gatherRowsInto: table")
    out.validate("gatherRowsInto: out")
    checkIds("gatherRowsInto", ids, idsOffset, count, table.rows)
    if out.rows != count || out.cols != table.cols then
      throw new IllegalArgumentException(s"gatherRowsInto: out is ${out.rows} x ${out.cols}, expected $count x ${table.cols}")
    if MatrixF32.overlaps(out, table) then throw new IllegalArgumentException("gatherRowsInto: out overlaps table")
    gatherImpl(table, ids, idsOffset, count, out)

  /** [[gatherRowsInto]] over all of `ids`. */
  final def gatherRowsInto(table: MatrixF32, ids: Array[Int], out: MatrixF32): Unit =
    gatherRowsInto(table, ids, 0, if ids == null then 0 else ids.length, out)

  /** `dst(ids(idsOffset + r), :) += values(r, :)` in order `r = 0, 1, ...`; repeated IDs accumulate. */
  final def scatterAddRowsInto(dst: MatrixF32, ids: Array[Int], idsOffset: Int, count: Int, values: MatrixF32): Unit =
    dst.validate("scatterAddRowsInto: dst")
    values.validate("scatterAddRowsInto: values")
    checkIds("scatterAddRowsInto", ids, idsOffset, count, dst.rows)
    if values.rows != count || values.cols != dst.cols then
      throw new IllegalArgumentException(
        s"scatterAddRowsInto: values is ${values.rows} x ${values.cols}, expected $count x ${dst.cols}"
      )
    if MatrixF32.overlaps(dst, values) then throw new IllegalArgumentException("scatterAddRowsInto: dst overlaps values")
    scatterAddImpl(dst, ids, idsOffset, count, values)

  /** [[scatterAddRowsInto]] over all of `ids`. */
  final def scatterAddRowsInto(dst: MatrixF32, ids: Array[Int], values: MatrixF32): Unit =
    scatterAddRowsInto(dst, ids, 0, if ids == null then 0 else ids.length, values)

  /** Merges duplicate row IDs: writes the distinct IDs in ascending order to `outIds(outIdsOffset + u)` and,
    * for each, the sum of its `values` rows (added in original occurrence order) to `outValues(u, :)`.
    * Returns the number `u` of distinct IDs. IDs must be non-negative. `outValues` needs at least that many
    * rows and the workspace `count` longs (see [[coalesceWorkspaceLongs]]).
    */
  final def coalesceRowsInto(
      ids: Array[Int], idsOffset: Int, count: Int, values: MatrixF32,
      outIds: Array[Int], outIdsOffset: Int, outValues: MatrixF32,
      workspace: Workspace
  ): Int =
    values.validate("coalesceRowsInto: values")
    outValues.validate("coalesceRowsInto: outValues")
    checkIds("coalesceRowsInto", ids, idsOffset, count, Int.MaxValue)
    if values.rows != count then
      throw new IllegalArgumentException(s"coalesceRowsInto: values has ${values.rows} rows, expected $count")
    if outValues.cols != values.cols then
      throw new IllegalArgumentException(s"coalesceRowsInto: outValues has ${outValues.cols} columns, expected ${values.cols}")
    if outIds == null || outIdsOffset < 0 || outIdsOffset > outIds.length then
      throw new IllegalArgumentException("coalesceRowsInto: invalid outIds/outIdsOffset")
    if (outIds eq ids) && count > 0 && outIds.length > outIdsOffset &&
      outIdsOffset < idsOffset + count && idsOffset < outIds.length
    then throw new IllegalArgumentException("coalesceRowsInto: outIds overlaps ids")
    if MatrixF32.overlaps(outValues, values) then throw new IllegalArgumentException("coalesceRowsInto: outValues overlaps values")
    workspace.require(0, count, "coalesceRowsInto")
    val mk = workspace.mark
    try RowOps.coalesce(ids, idsOffset, count, values, outIds, outIdsOffset, outValues, workspace)
    finally workspace.release(mk)

  /** [[coalesceRowsInto]] over all of `ids`, writing IDs from `outIds(0)`. */
  final def coalesceRowsInto(ids: Array[Int], values: MatrixF32, outIds: Array[Int], outValues: MatrixF32, workspace: Workspace): Int =
    coalesceRowsInto(ids, 0, if ids == null then 0 else ids.length, values, outIds, 0, outValues, workspace)

  /** Workspace longs [[coalesceRowsInto]] needs for `count` IDs. */
  def coalesceWorkspaceLongs(count: Int): Int = count

  protected def gatherImpl(table: MatrixF32, ids: Array[Int], idsOffset: Int, count: Int, out: MatrixF32): Unit
  protected def scatterAddImpl(dst: MatrixF32, ids: Array[Int], idsOffset: Int, count: Int, values: MatrixF32): Unit

  private def checkIds(op: String, ids: Array[Int], idsOffset: Int, count: Int, limit: Int): Unit =
    if ids == null then throw new IllegalArgumentException(s"$op: ids is null")
    if count < 0 || idsOffset < 0 || idsOffset.toLong + count > ids.length then
      throw new IllegalArgumentException(s"$op: ids range [$idsOffset, ${idsOffset.toLong + count}) exceeds ids length ${ids.length}")
    var r = 0
    while r < count do
      val id = ids(idsOffset + r)
      if id < 0 || id >= limit then
        throw new IndexOutOfBoundsException(
          if limit == Int.MaxValue then s"$op: row id $id at position $r is negative"
          else s"$op: row id $id at position $r is out of bounds for $limit rows"
        )
      r += 1

  // ------------------------------------------------------------------ K3: elementwise and row reductions

  /** `dst := src`. */
  final def copyInto(src: MatrixF32, dst: MatrixF32): Unit =
    sameShape("copyInto", src, dst)
    elementwiseAlias("copyInto", dst, src)
    if !MatrixF32.sameRange(src, dst) then copyImpl(src, dst)

  /** Every element of `dst` set to `value`. */
  final def fillInto(dst: MatrixF32, value: Float): Unit =
    dst.validate("fillInto: dst")
    fillImpl(dst, value)

  /** `y := alpha * x + y`. `x` and `y` may be exactly the same range. */
  final def axpyInto(alpha: Float, x: MatrixF32, y: MatrixF32): Unit =
    sameShape("axpyInto", x, y)
    elementwiseAlias("axpyInto", y, x)
    axpyImpl(alpha, x, y)

  /** `out := x * y` elementwise. */
  final def mulInto(x: MatrixF32, y: MatrixF32, out: MatrixF32): Unit =
    sameShape("mulInto", x, out)
    sameShape("mulInto", y, out)
    elementwiseAlias("mulInto", out, x)
    elementwiseAlias("mulInto", out, y)
    mulImpl(x, y, out)

  /** `out := 1 / (1 + exp(-x))`, without overflow for large `|x|`. */
  final def sigmoidInto(x: MatrixF32, out: MatrixF32): Unit =
    sameShape("sigmoidInto", x, out)
    elementwiseAlias("sigmoidInto", out, x)
    sigmoidImpl(x, out)

  /** `out := x * sigmoid(x)` (SiLU); `silu(-inf) = -0`, `silu(+inf) = +inf`. */
  final def siluInto(x: MatrixF32, out: MatrixF32): Unit =
    sameShape("siluInto", x, out)
    elementwiseAlias("siluInto", out, x)
    siluImpl(x, out)

  /** `out(outOffset + i) = log(sum_j exp(x(i, j)))`, max-shifted.
    * Rows with NaN give NaN; rows containing `+inf` give `+inf`; all-`-inf` or empty rows give `-inf`.
    */
  final def rowLogSumExpInto(x: MatrixF32, out: Array[Float], outOffset: Int): Unit =
    rowReduceArgs("rowLogSumExpInto", x, out, outOffset)
    rowLogSumExpImpl(x, out, outOffset)

  /** `out(outOffset + i) = sum_j x(i, j)^2`. */
  final def rowSumSquaresInto(x: MatrixF32, out: Array[Float], outOffset: Int): Unit =
    rowReduceArgs("rowSumSquaresInto", x, out, outOffset)
    rowSumSquaresImpl(x, out, outOffset)

  protected def copyImpl(src: MatrixF32, dst: MatrixF32): Unit
  protected def fillImpl(dst: MatrixF32, value: Float): Unit
  protected def axpyImpl(alpha: Float, x: MatrixF32, y: MatrixF32): Unit
  protected def mulImpl(x: MatrixF32, y: MatrixF32, out: MatrixF32): Unit
  protected def sigmoidImpl(x: MatrixF32, out: MatrixF32): Unit
  protected def siluImpl(x: MatrixF32, out: MatrixF32): Unit
  protected def rowLogSumExpImpl(x: MatrixF32, out: Array[Float], outOffset: Int): Unit
  protected def rowSumSquaresImpl(x: MatrixF32, out: Array[Float], outOffset: Int): Unit

  private def sameShape(op: String, x: MatrixF32, y: MatrixF32): Unit =
    x.validate(s"$op: input")
    y.validate(s"$op: output")
    if x.rows != y.rows || x.cols != y.cols then
      throw new IllegalArgumentException(s"$op: shapes differ (${x.rows} x ${x.cols} vs ${y.rows} x ${y.cols})")

  private def elementwiseAlias(op: String, out: MatrixF32, in: MatrixF32): Unit =
    if MatrixF32.overlaps(out, in) && !MatrixF32.sameRange(out, in) then
      throw new IllegalArgumentException(s"$op: output partially overlaps an input")

  private def rowReduceArgs(op: String, x: MatrixF32, out: Array[Float], outOffset: Int): Unit =
    x.validate(s"$op: x")
    if out == null || outOffset < 0 || outOffset.toLong + x.rows > out.length then
      throw new IllegalArgumentException(s"$op: out needs ${x.rows} elements from offset $outOffset")
    if (out eq x.data) && x.rows > 0 && !x.isEmpty &&
      outOffset < x.endIndex && x.firstIndex < outOffset.toLong + x.rows
    then throw new IllegalArgumentException(s"$op: out overlaps x")

  // ------------------------------------------------------------------ K4: affine scan

  /** Elementwise linear recurrence over `[time, batch, feature]` data.
    *
    * `a`, `b` and `out` are `(T * B) x D` matrices whose row `t * B + j` is batch `j` at step `t`;
    * `initial` is `B x D`. Computes `s(-1) = initial`, `s(t) = a(t) * s(t-1) + b(t)` and writes `s(t)`
    * to `out`. `out` may be exactly `a` or `b`, but must not overlap `initial`.
    */
  final def affineScanInto(a: MatrixF32, b: MatrixF32, initial: MatrixF32, out: MatrixF32): Unit =
    sameShape("affineScanInto", a, out)
    sameShape("affineScanInto", b, out)
    initial.validate("affineScanInto: initial")
    val bsz = initial.rows
    if initial.cols != a.cols then
      throw new IllegalArgumentException(s"affineScanInto: initial has ${initial.cols} features, expected ${a.cols}")
    if (bsz == 0 && a.rows != 0) || (bsz != 0 && a.rows % bsz != 0) then
      throw new IllegalArgumentException(s"affineScanInto: ${a.rows} rows is not a multiple of batch size $bsz")
    elementwiseAlias("affineScanInto", out, a)
    elementwiseAlias("affineScanInto", out, b)
    if MatrixF32.overlaps(out, initial) then throw new IllegalArgumentException("affineScanInto: out overlaps initial")
    val steps = if bsz == 0 then 0 else a.rows / bsz
    affineScanImpl(a, b, initial, out, steps, bsz)

  protected def affineScanImpl(a: MatrixF32, b: MatrixF32, initial: MatrixF32, out: MatrixF32, steps: Int, batch: Int): Unit
