package com.github.kmizu.numscala

/** `numpy.lib`. */
object Lib:
  /** `numpy.lib.stride_tricks`. */
  object stride_tricks:
    /** Creates a view with the given shape and '''byte''' strides (`as_strided`).
      * Like NumPy this is unsafe: strides that leave the buffer fail on access.
      */
    def as_strided[T](x: NDArray[T], shape: Seq[Int], strides: Seq[Int]): NDArray[T] =
      val isz = math.max(x.dtype.itemSize, 1)
      if strides.exists(_ % isz != 0) then
        throw new IllegalArgumentException(s"strides must be multiples of the item size ($isz)")
      as_strided_elems(x, shape, strides.map(_ / isz))

    /** `as_strided` with strides measured in elements. */
    def as_strided_elems[T](x: NDArray[T], shape: Seq[Int], strides: Seq[Int]): NDArray[T] =
      if shape.length != strides.length then throw new IllegalArgumentException("shape and strides must have the same length")
      x.view(shape.toArray, strides.toArray, x.offset)

    /** Sliding windows over all axes (`window_shape` per axis) or the given `axis` list. */
    def sliding_window_view[T](x: NDArray[T], window_shape: Seq[Int], axis: Seq[Int] = Nil): NDArray[T] =
      val axes = if axis.isEmpty then (0 until x.ndim) else axis.map(Shape.normAxis(_, x.ndim))
      if window_shape.length != axes.length then
        throw new IllegalArgumentException("Since axis is `None`, must provide window_shape for all dimensions of `x`")
      if window_shape.exists(_ < 0) then throw new IllegalArgumentException("`window_shape` cannot contain negative values")
      val outShape = x.shapeArr.clone()
      for (ax, w) <- axes.zip(window_shape) do
        if outShape(ax) < w then throw new IllegalArgumentException("window shape cannot be larger than input array shape")
        outShape(ax) -= w - 1
      val shape = outShape ++ window_shape
      val strides = x.stridesArr ++ axes.map(x.stridesArr(_))
      x.view(shape, strides, x.offset)

    def sliding_window_view[T](x: NDArray[T], window: Int, axis: Int): NDArray[T] =
      sliding_window_view(x, Seq(window), Seq(axis))

    def sliding_window_view[T](x: NDArray[T], window: Int): NDArray[T] =
      if x.ndim != 1 then throw new IllegalArgumentException("window_shape must have one entry per dimension")
      sliding_window_view(x, Seq(window), Seq(0))
