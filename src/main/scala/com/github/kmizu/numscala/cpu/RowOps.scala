package com.github.kmizu.numscala.cpu

/** Scalar sparse-row kernels (K2). Arguments are validated by [[F32Kernels]]. */
private[cpu] object RowOps:

  def gather(table: MatrixF32, ids: Array[Int], idsOffset: Int, count: Int, out: MatrixF32): Unit =
    val d = table.cols
    var r = 0
    while r < count do
      System.arraycopy(table.data, table.offset + ids(idsOffset + r) * table.rowStride, out.data, out.offset + r * out.rowStride, d)
      r += 1

  def scatterAdd(dst: MatrixF32, ids: Array[Int], idsOffset: Int, count: Int, values: MatrixF32): Unit =
    val d = dst.cols
    val dd = dst.data
    val vd = values.data
    var r = 0
    while r < count do
      val di = dst.offset + ids(idsOffset + r) * dst.rowStride
      val vi = values.offset + r * values.rowStride
      var j = 0
      while j < d do
        dd(di + j) += vd(vi + j)
        j += 1
      r += 1

  /** Stable coalesce: keys `(id << 32) | position` sorted in the workspace, then summed per ID. */
  def coalesce(
      ids: Array[Int], idsOffset: Int, count: Int, values: MatrixF32,
      outIds: Array[Int], outIdsOffset: Int, outValues: MatrixF32, ws: Workspace
  ): Int =
    val ko = ws.takeLongs(count)
    val keys = ws.longs
    var r = 0
    while r < count do
      keys(ko + r) = (ids(idsOffset + r).toLong << 32) | r.toLong
      r += 1
    java.util.Arrays.sort(keys, ko, ko + count)
    var unique = 0
    r = 0
    while r < count do
      if r == 0 || (keys(ko + r) >>> 32) != (keys(ko + r - 1) >>> 32) then unique += 1
      r += 1
    // capacity checks before any output is written
    if outValues.rows < unique then
      throw new IllegalArgumentException(s"coalesceRowsInto: outValues has ${outValues.rows} rows, needs $unique")
    if outIds.length - outIdsOffset < unique then
      throw new IllegalArgumentException(s"coalesceRowsInto: outIds has room for ${outIds.length - outIdsOffset} ids, needs $unique")
    val d = values.cols
    val vd = values.data
    val od = outValues.data
    var u = -1
    r = 0
    while r < count do
      val key = keys(ko + r)
      val id = (key >>> 32).toInt
      val src = values.offset + (key & 0xffffffffL).toInt * values.rowStride
      if r == 0 || id != outIds(outIdsOffset + u) then
        u += 1
        outIds(outIdsOffset + u) = id
        System.arraycopy(vd, src, od, outValues.offset + u * outValues.rowStride, d)
      else
        val oi = outValues.offset + u * outValues.rowStride
        var j = 0
        while j < d do
          od(oi + j) += vd(src + j)
          j += 1
      r += 1
    unique
