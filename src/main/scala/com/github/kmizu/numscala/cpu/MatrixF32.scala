package com.github.kmizu.numscala.cpu

/** Whether a GEMM operand is used as stored (`No`) or transposed (`Yes`). */
enum Transpose:
  case No, Yes

/** A row-major Float32 matrix view over `data` (contract NS-CPU-KERNEL-1, K1).
  *
  * Element `(i, j)` lives at `data(offset + i * rowStride + j)`: rows are contiguous, rows may be
  * padded (`rowStride >= cols`). `rows`/`cols` are the physical shape; a transposed use is expressed
  * with [[Transpose]], never by swapping the fields. The descriptor never owns or copies `data`.
  */
final case class MatrixF32(data: Array[Float], offset: Int, rows: Int, cols: Int, rowStride: Int):

  /** True when the matrix has no elements. */
  def isEmpty: Boolean = rows == 0 || cols == 0

  /** Number of logical rows after applying `t`. */
  def logicalRows(t: Transpose): Int = if t == Transpose.No then rows else cols

  /** Number of logical columns after applying `t`. */
  def logicalCols(t: Transpose): Int = if t == Transpose.No then cols else rows

  /** Element `(i, j)` (physical indices, bounds-checked by the JVM only). */
  def apply(i: Int, j: Int): Float = data(offset + i * rowStride + j)

  /** Sets element `(i, j)`. */
  def update(i: Int, j: Int, v: Float): Unit = data(offset + i * rowStride + j) = v

  /** The single-row view of row `i`. */
  def row(i: Int): MatrixF32 =
    if i < 0 || i >= rows then throw new IndexOutOfBoundsException(s"row $i is out of bounds for $rows rows")
    MatrixF32(data, offset + i * rowStride, 1, cols, math.max(cols, 1))

  /** The view of rows `[from, until)`. */
  def rowRange(from: Int, until: Int): MatrixF32 =
    if from < 0 || until < from || until > rows then
      throw new IndexOutOfBoundsException(s"row range [$from, $until) is out of bounds for $rows rows")
    MatrixF32(data, if from == until then offset else offset + from * rowStride, until - from, cols, rowStride)

  /** Copies the elements in row-major order into a new packed array (a copy, never a view). */
  def toArray: Array[Float] =
    val out = new Array[Float](rows * cols)
    var i = 0
    while i < rows do
      System.arraycopy(data, offset + i * rowStride, out, i * cols, cols)
      i += 1
    out

  /** Index of the first element touched, as a Long (only meaningful when non-empty). */
  private[cpu] def firstIndex: Long = offset.toLong

  /** One past the last element touched, computed in Long (only meaningful when non-empty). */
  private[cpu] def endIndex: Long = offset.toLong + (rows - 1).toLong * rowStride + cols

  /** Throws `IllegalArgumentException` unless the descriptor is valid for its buffer. */
  def validate(name: String): Unit =
    if data == null then throw new IllegalArgumentException(s"$name: data is null")
    if rows < 0 || cols < 0 then throw new IllegalArgumentException(s"$name: negative shape ($rows, $cols)")
    if offset < 0 then throw new IllegalArgumentException(s"$name: negative offset $offset")
    if !isEmpty then
      if rowStride <= 0 || rowStride < cols then
        throw new IllegalArgumentException(s"$name: rowStride $rowStride must be positive and >= cols $cols")
      if endIndex > data.length then
        throw new IllegalArgumentException(
          s"$name: view [offset $offset, rows $rows, cols $cols, rowStride $rowStride] exceeds buffer length ${data.length}"
        )
    else if offset > data.length then
      throw new IllegalArgumentException(s"$name: offset $offset exceeds buffer length ${data.length}")

  override def toString: String = s"MatrixF32(rows=$rows, cols=$cols, offset=$offset, rowStride=$rowStride, buffer=${data.length})"

object MatrixF32:
  /** A new zero-filled packed `rows x cols` matrix. */
  def zeros(rows: Int, cols: Int): MatrixF32 =
    if rows < 0 || cols < 0 then throw new IllegalArgumentException(s"negative shape ($rows, $cols)")
    val n = rows.toLong * cols
    if n > Int.MaxValue then throw new IllegalArgumentException(s"matrix ($rows, $cols) is too large")
    MatrixF32(new Array[Float](n.toInt), 0, rows, cols, math.max(cols, 1))

  /** Wraps a packed row-major buffer (no copy). */
  def wrap(data: Array[Float], rows: Int, cols: Int): MatrixF32 = MatrixF32(data, 0, rows, cols, math.max(cols, 1))

  /** A `1 x n` view of `data[offset, offset + n)`. */
  def vector(data: Array[Float], offset: Int, n: Int): MatrixF32 = MatrixF32(data, offset, 1, n, math.max(n, 1))

  /** A `1 x data.length` view of the whole array. */
  def vector(data: Array[Float]): MatrixF32 = vector(data, 0, data.length)

  /** Conservative overlap test: same buffer and intersecting `[first, end)` intervals of non-empty views. */
  private[cpu] def overlaps(x: MatrixF32, y: MatrixF32): Boolean =
    (x.data eq y.data) && !x.isEmpty && !y.isEmpty && x.firstIndex < y.endIndex && y.firstIndex < x.endIndex

  /** True when both descriptors name exactly the same elements in the same layout. */
  private[cpu] def sameRange(x: MatrixF32, y: MatrixF32): Boolean =
    (x.data eq y.data) && x.offset == y.offset && x.rows == y.rows && x.cols == y.cols &&
      (x.rows <= 1 || x.rowStride == y.rowStride)
