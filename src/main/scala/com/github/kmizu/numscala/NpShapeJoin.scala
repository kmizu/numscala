package com.github.kmizu.numscala

import scala.annotation.targetName
import NpShapeOps.*

/** Joining and splitting arrays (`np.concatenate`, `np.stack`, `np.block`, `np.split`, ...). */
trait NpShapeJoin:

  // ------------------------------------------------------------------ joining

  /** Joins arrays along an existing axis (`np.concatenate`); `axis = None` flattens them first. */
  def concatenate[T](arrays: Seq[NDArray[T]], axis: Int | None.type = 0): NDArray[T] = axis match
    case None => concat(arrays.map(_.ravel()), 0)
    case i: Int => concat(arrays, i)

  /** `np.concatenate((a, b, ...))` along axis 0. */
  def concatenate[T](a: NDArray[T], b: NDArray[T], rest: NDArray[T]*): NDArray[T] =
    concat((a +: b +: rest).toSeq, 0)

  /** Joins arrays of identical shape along a new axis (`np.stack`). */
  def stack[T](arrays: Seq[NDArray[T]], axis: Int = 0): NDArray[T] =
    if arrays.isEmpty then throw new IllegalArgumentException("need at least one array to stack")
    val sh = arrays.head.shapeArr
    if arrays.exists(a => !java.util.Arrays.equals(a.shapeArr, sh)) then
      throw new IllegalArgumentException("all input arrays must have the same shape")
    val ax = Shape.normAxis(axis, sh.length + 1)
    concat(arrays.map(a => insertOnes(a, Array(ax))), ax)

  /** `np.stack((a, b, ...))` along a new first axis. */
  def stack[T](a: NDArray[T], b: NDArray[T], rest: NDArray[T]*): NDArray[T] = stack((a +: b +: rest).toSeq)

  /** Stacks arrays vertically, row-wise (`np.vstack`). */
  def vstack[T](tup: Seq[NDArray[T]]): NDArray[T] = concat(tup.map(prependOnes(_, 2)), 0)

  /** `np.vstack((a, b, ...))`. */
  def vstack[T](a: NDArray[T], b: NDArray[T], rest: NDArray[T]*): NDArray[T] = vstack((a +: b +: rest).toSeq)

  /** Alias of [[vstack]] (`np.row_stack`). */
  def row_stack[T](tup: Seq[NDArray[T]]): NDArray[T] = vstack(tup)

  /** `np.row_stack((a, b, ...))`. */
  def row_stack[T](a: NDArray[T], b: NDArray[T], rest: NDArray[T]*): NDArray[T] = vstack((a +: b +: rest).toSeq)

  /** Stacks arrays horizontally, column-wise (`np.hstack`). */
  def hstack[T](tup: Seq[NDArray[T]]): NDArray[T] =
    val arrs = tup.map(prependOnes(_, 1))
    if arrs.nonEmpty && arrs.head.ndim == 1 then concat(arrs, 0) else concat(arrs, 1)

  /** `np.hstack((a, b, ...))`. */
  def hstack[T](a: NDArray[T], b: NDArray[T], rest: NDArray[T]*): NDArray[T] = hstack((a +: b +: rest).toSeq)

  /** Stacks arrays depth-wise, along the third axis (`np.dstack`). */
  def dstack[T](tup: Seq[NDArray[T]]): NDArray[T] = concat(tup.map(atleast3(_)), 2)

  /** `np.dstack((a, b, ...))`. */
  def dstack[T](a: NDArray[T], b: NDArray[T], rest: NDArray[T]*): NDArray[T] = dstack((a +: b +: rest).toSeq)

  /** Stacks 1-D arrays as columns of a 2-D array (`np.column_stack`). */
  def column_stack[T](tup: Seq[NDArray[T]]): NDArray[T] =
    concat(
      tup.map { v =>
        if v.ndim < 2 then prependOnes(v, 2).transpose() else v
      },
      1
    )

  /** `np.column_stack((a, b, ...))`. */
  def column_stack[T](a: NDArray[T], b: NDArray[T], rest: NDArray[T]*): NDArray[T] =
    column_stack((a +: b +: rest).toSeq)

  /** Assembles an array from nested blocks (`np.block`); a single array is copied. */
  def block[T](a: NDArray[T]): NDArray[T] = a.copy()

  /** `np.block([a, b, ...])`: concatenation along the last axis. */
  @targetName("block1")
  def block[T](arrays: Seq[NDArray[T]]): NDArray[T] = blockImpl(arrays, 1)

  /** `np.block([[a, b], [c, d]])`: rows along axis -2, blocks within a row along -1. */
  @targetName("block2")
  def block[T](arrays: Seq[Seq[NDArray[T]]]): NDArray[T] = blockImpl(arrays, 2)

  /** `np.block` with three levels of nesting. */
  @targetName("block3")
  def block[T](arrays: Seq[Seq[Seq[NDArray[T]]]]): NDArray[T] = blockImpl(arrays, 3)

  private def blockImpl[T](arrays: Seq[?], listDepth: Int): NDArray[T] =
    def maxNd(x: Any, path: String): Int = x match
      case a: NDArray[?] => a.ndim
      case s: Seq[?] =>
        if s.isEmpty then throw new IllegalArgumentException(s"List at $path cannot be empty")
        s.zipWithIndex.map((e, i) => maxNd(e, s"$path[$i]")).max
      case other => throw new IllegalArgumentException(s"unsupported block element $other")
    val nd = math.max(listDepth, maxNd(arrays, "arrays"))
    def rec(x: Any, depth: Int): NDArray[T] = x match
      case a: NDArray[?] => prependOnes(a.asInstanceOf[NDArray[T]], nd)
      case s: Seq[?] => concat(s.map(rec(_, depth + 1)), -(listDepth - depth))
      case other => throw new IllegalArgumentException(s"unsupported block element $other")
    rec(arrays, 0)

  // ------------------------------------------------------------------ splitting

  /** Splits into equal sections or at the given indices along `axis` (`np.split`); views. */
  def split[T](ary: NDArray[T], indices_or_sections: Int | Seq[Int], axis: Int = 0): Seq[NDArray[T]] =
    indices_or_sections match
      case n: Int =>
        if n <= 0 then throw new IllegalArgumentException("number sections must be larger than 0.")
        val len = ary.shapeArr(Shape.normAxis(axis, ary.ndim))
        if len % n != 0 then throw new IllegalArgumentException("array split does not result in an equal division")
      case _ => ()
    array_split(ary, indices_or_sections, axis)

  /** Like [[split]] but allows unequal sections (`np.array_split`); views. */
  def array_split[T](ary: NDArray[T], indices_or_sections: Int | Seq[Int], axis: Int = 0): Seq[NDArray[T]] =
    val ax = Shape.normAxis(axis, ary.ndim)
    val total = ary.shapeArr(ax)
    val divPoints: Array[Int] = indices_or_sections match
      case n: Int =>
        if n <= 0 then throw new IllegalArgumentException("number sections must be larger than 0.")
        val each = total / n
        val extras = total % n
        val sizes = Array.tabulate(n)(i => if i < extras then each + 1 else each)
        sizes.scanLeft(0)(_ + _)
      case s: Seq[?] => (0 +: s.asInstanceOf[Seq[Int]] :+ total).toArray
    (0 until divPoints.length - 1).map { i =>
      val sl = Index.Slice(Some(divPoints(i)), Some(divPoints(i + 1)))
      val (start, _, len) = sl.resolve(total)
      sliceAxis(ary, ax, start, len)
    }

  /** Splits horizontally (column-wise; axis 1, or 0 for 1-D input) (`np.hsplit`). */
  def hsplit[T](ary: NDArray[T], indices_or_sections: Int | Seq[Int]): Seq[NDArray[T]] =
    if ary.ndim == 0 then throw new IllegalArgumentException("hsplit only works on arrays of 1 or more dimensions")
    split(ary, indices_or_sections, if ary.ndim > 1 then 1 else 0)

  /** Splits vertically (row-wise, axis 0) (`np.vsplit`). */
  def vsplit[T](ary: NDArray[T], indices_or_sections: Int | Seq[Int]): Seq[NDArray[T]] =
    if ary.ndim < 2 then throw new IllegalArgumentException("vsplit only works on arrays of 2 or more dimensions")
    split(ary, indices_or_sections, 0)

  /** Splits along the third axis (`np.dsplit`). */
  def dsplit[T](ary: NDArray[T], indices_or_sections: Int | Seq[Int]): Seq[NDArray[T]] =
    if ary.ndim < 3 then throw new IllegalArgumentException("dsplit only works on arrays of 3 or more dimensions")
    split(ary, indices_or_sections, 2)
