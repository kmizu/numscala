package com.github.kmizu.numscala

/** Top-level products in the `np` namespace: `dot`, `vdot`, `inner`, `outer`, `matmul`,
  * `tensordot`, `einsum`, `kron`, `trace`, `cross`, `vecdot`.
  */
trait NpLinalgTop:

  /** Dot product (`np.dot`): matrix product for 2-D, sum product over the last axis of `a` and the
    * second-to-last of `b` for N-D; a 0-d array for two vectors.
    */
  def dot[A, B](a: NDArray[A], b: NDArray[B])(using p: NumPromote[A, B]): NDArray[p.Out] = LinAlgCore.dot(a, b)

  /** Dot product of the flattened arrays, conjugating the first (`np.vdot`). */
  def vdot[A, B](a: NDArray[A], b: NDArray[B])(using p: NumPromote[A, B]): p.Out = LinalgProducts.vdot(a, b, p.dtype)

  /** Inner product over the last axes (`np.inner`). */
  def inner[A, B](a: NDArray[A], b: NDArray[B])(using p: NumPromote[A, B]): NDArray[p.Out] =
    LinalgProducts.inner(a, b, p.dtype)

  /** Outer product of the flattened inputs (`np.outer`). */
  def outer[A, B](a: NDArray[A], b: NDArray[B])(using p: NumPromote[A, B]): NDArray[p.Out] =
    LinalgProducts.outer(a, b, p.dtype)

  /** Matrix product with broadcasting of stacked matrices (`np.matmul`, the `@` operator). */
  def matmul[A, B](a: NDArray[A], b: NDArray[B])(using p: NumPromote[A, B]): NDArray[p.Out] = LinAlgCore.matmul(a, b)

  /** Tensor dot product over the last `axes` axes of `a` and the first `axes` of `b` (`np.tensordot`). */
  def tensordot[A, B](a: NDArray[A], b: NDArray[B], axes: Int = 2)(using p: NumPromote[A, B]): NDArray[p.Out] =
    if axes < 0 then throw new IllegalArgumentException("tensordot: axes must be non-negative")
    if axes > a.ndim || axes > b.ndim then throw new IllegalArgumentException("shape-mismatch for sum")
    LinAlgCore.tensordotD(a, b, (a.ndim - axes until a.ndim), (0 until axes), p.dtype)

  /** Tensor dot product over explicit axis lists (`np.tensordot(a, b, axes=(axesA, axesB))`). */
  def tensordot[A, B](a: NDArray[A], b: NDArray[B], axes: (Seq[Int], Seq[Int]))(using p: NumPromote[A, B]): NDArray[p.Out] =
    LinAlgCore.tensordotD(a, b, axes._1, axes._2, p.dtype)

  /** Einstein summation (`np.einsum`) of one operand: `"ii"` (trace), `"ii->i"` (diagonal),
    * `"ij->ji"`, `"ijk->"` (sum). Subscripts follow NumPy: implicit output (labels appearing once,
    * alphabetically), explicit `->`, repeated labels, and `...` broadcasting.
    */
  def einsum[A](subscripts: String, a: NDArray[A])(using d: NumDType[A]): NDArray[A] =
    LinalgEinsum.einsum(subscripts, Seq(a), d)

  /** Two-operand `np.einsum` (`"ij,jk->ik"`, `"...ij,...jk"`, `"i,i"`) with dtype promotion. */
  def einsum[A, B](subscripts: String, a: NDArray[A], b: NDArray[B])(using p: NumPromote[A, B]): NDArray[p.Out] =
    LinalgEinsum.einsum(subscripts, Seq(a.asType(using p.dtype), b.asType(using p.dtype)), p.dtype)

  /** Three-operand `np.einsum` with dtype promotion. */
  def einsum[A, B, C](subscripts: String, a: NDArray[A], b: NDArray[B], c: NDArray[C])(using p: NumPromote[A, B])(using
      q: NumPromote[p.Out, C]
  ): NDArray[q.Out] =
    LinalgEinsum.einsum(subscripts, Seq(a.asType(using q.dtype), b.asType(using q.dtype), c.asType(using q.dtype)), q.dtype)

  /** `np.einsum` with four or more operands of one dtype. */
  def einsum[T](subscripts: String, a: NDArray[T], b: NDArray[T], c: NDArray[T], d: NDArray[T], more: NDArray[T]*)(using
      dt: NumDType[T]
  ): NDArray[T] =
    LinalgEinsum.einsum(subscripts, Seq(a, b, c, d) ++ more, dt)

  /** `np.einsum` over a sequence of operands of one dtype. */
  def einsum[T](subscripts: String, operands: Seq[NDArray[T]])(using d: NumDType[T]): NDArray[T] =
    LinalgEinsum.einsum(subscripts, operands, d)

  /** Contraction order used by [[einsum]] in `np.einsum_path` format (pairs of operand positions;
    * contracted operands are removed and the intermediate appended), and a textual report.
    */
  def einsum_path(subscripts: String, operands: NDArray[?]*): (Seq[Seq[Int]], String) =
    LinalgEinsum.path(subscripts, operands.map(_.shapeArr))

  /** Kronecker product (`np.kron`). */
  def kron[A, B](a: NDArray[A], b: NDArray[B])(using p: NumPromote[A, B]): NDArray[p.Out] =
    LinalgProducts.kron(a, b, p.dtype)

  /** Sum of the main diagonal of a 2-D array (`np.trace(a)`); small integers accumulate in int64. */
  def trace[T](a: NDArray[T])(using s: SumOf[T]): s.Out =
    if a.ndim != 2 then
      throw new IllegalArgumentException(
        s"trace(a) of a ${a.ndim}-D array: use trace(a, offset, axis1, axis2) for stacked input"
      )
    LinalgProducts.trace(a, 0, 0, 1, s.dtype).item

  /** Sums along diagonals (`np.trace(a, offset, axis1, axis2)`), as an array (0-d for 2-D input). */
  def trace[T](a: NDArray[T], offset: Int, axis1: Int = 0, axis2: Int = 1)(using s: SumOf[T]): NDArray[s.Out] =
    LinalgProducts.trace(a, offset, axis1, axis2, s.dtype)

  /** Cross product of 2- or 3-element vectors (`np.cross`); `axis`, when given, overrides
    * `axisa`, `axisb` and `axisc`. Two 2-vectors give the scalar z component.
    */
  def cross[A, B](a: NDArray[A], b: NDArray[B], axisa: Int = -1, axisb: Int = -1, axisc: Int = -1, axis: Int | None.type = None)(using
      p: NumPromote[A, B]
  ): NDArray[p.Out] =
    axis match
      case ax: Int => LinalgProducts.cross(a, b, ax, ax, ax, p.dtype)
      case _ => LinalgProducts.cross(a, b, axisa, axisb, axisc, p.dtype)

  /** Vector dot product along `axis`, conjugating `x1` (`np.vecdot`). */
  def vecdot[A, B](x1: NDArray[A], x2: NDArray[B], axis: Int = -1)(using p: NumPromote[A, B]): NDArray[p.Out] =
    LinalgProducts.vecdot(x1, x2, axis, p.dtype)
