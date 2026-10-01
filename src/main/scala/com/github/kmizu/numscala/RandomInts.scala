package com.github.kmizu.numscala.random

import com.github.kmizu.numscala.*

/** Port of NumPy's `_bounded_integers.pyx`: `Generator.integers` (Lemire) and
  * `RandomState.randint` (masked rejection) for int64/int32/int16/int8, unsigned and bool.
  */
private[numscala] object RandomInts:
  private def boundsError(closed: Boolean, lowIsZero: Boolean): IllegalArgumentException =
    val msg =
      if lowIsZero then (if closed then "high < 0" else "high <= 0")
      else (if closed then "low > high" else "low >= high")
    new IllegalArgumentException(msg)

  private case class Spec(name: String, lb: Long, ub: Long)
  private def spec(d: DType[?]): Spec = d.name match
    case "int64" => Spec("int64", Long.MinValue, Long.MaxValue)
    case "int32" => Spec("int32", Int.MinValue.toLong, Int.MaxValue.toLong)
    case "int16" => Spec("int16", Short.MinValue.toLong, Short.MaxValue.toLong)
    case "int8" => Spec("int8", Byte.MinValue.toLong, Byte.MaxValue.toLong)
    case "bool" => Spec("bool", 0L, 1L)
    // unsigned types reuse the same-width kernels (NumPy does the same); uint64 is limited to < 2**63
    case "uint64" => Spec("int64", 0L, Long.MaxValue)
    case "uint32" => Spec("int32", 0L, 0xffffffffL)
    case "uint16" => Spec("int16", 0L, 0xffffL)
    case "uint8" => Spec("int8", 0L, 0xffL)
    case other => throw new IllegalArgumentException(s"Unsupported dtype '$other' for integers")

  /** Samples int64 values in `[low, high)` (or `[low, high]` when `endpoint`). */
  def int64(bg: BitGenerator, low: Param, high: Param, size: SizeArg, masked: Boolean, endpoint: Boolean): NDArray[Long] =
    generic(bg, low, high, size, DType.Int64, masked, endpoint)

  /** The dtype-generic driver. */
  def generic[T](bg: BitGenerator, low: Param, high: Param, size: SizeArg, dtype: DType[T], masked: Boolean,
      endpoint: Boolean): NDArray[T] =
    val sp = spec(dtype)
    val shp = RCommon.sizeShape(size)
    if shp != null && Shape.size(RCommon.checkShape(shp)) == 0 then return NDArray.zerosOf(dtype, shp)
    val lo = RCommon.toL(low)
    val hi = RCommon.toL(high)
    if lo.ndim == 0 && hi.ndim == 0 then
      val l = BigInt(lo.item)
      val h = BigInt(hi.item) - (if endpoint then 0 else 1)
      if l < sp.lb then throw new IllegalArgumentException(s"low is out of bounds for ${sp.name}")
      if h > sp.ub then throw new IllegalArgumentException(s"high is out of bounds for ${sp.name}")
      if l > h then throw boundsError(endpoint, l == 0)
      val rng = (h - l).longValue
      val off = l.longValue
      val shape = if shp == null then Array.empty[Int] else shp
      fill(bg, off, rng, Shape.size(shape), sp.name, masked, dtype, shape)
    else
      val la = lo.broadcastTo(Shape.broadcast(lo.shapeArr, hi.shapeArr)*).toArray
      val ha = hi.broadcastTo(Shape.broadcast(lo.shapeArr, hi.shapeArr)*).toArray
      if la.exists(_ < sp.lb) then throw new IllegalArgumentException(s"low is out of bounds for ${sp.name}")
      if ha.exists(h => BigInt(h) - (if endpoint then 0 else 1) > sp.ub) then
        throw new IllegalArgumentException(s"high is out of bounds for ${sp.name}")
      val bad = la.indices.exists(i => if endpoint then la(i) > ha(i) else la(i) >= ha(i))
      if bad then throw boundsError(endpoint, la.forall(_ == 0))
      val (outShape, fl) = RCommon.broadcastAll(size, Seq(lo, hi))
      val lf = fl(0); val hf = fl(1)
      val n = lf.length
      val isOpen = if endpoint then 0L else 1L
      val buf = new Bounded.Buf
      sp.name match
        case "int64" =>
          val out = Array.tabulate(n) { i =>
            val rng = (hf(i) - isOpen) - lf(i)
            Bounded.uint64(bg, lf(i), rng, Bounded.genMask(rng), masked)
          }
          NDArray.fromArray(out.asInstanceOf[Array[T]], outShape)(using dtype)
        case "int32" =>
          val out = Array.tabulate(n) { i =>
            val rng = ((hf(i) - isOpen) - lf(i)).toInt
            Bounded.uint32(bg, lf(i).toInt, rng, Bounded.genMask(rng & 0xffffffffL).toInt, masked)
          }
          NDArray.fromArray(out.asInstanceOf[Array[T]], outShape)(using dtype)
        case "int16" =>
          val out = Array.tabulate(n) { i =>
            val rng = (((hf(i) - isOpen) - lf(i)) & 0xffff).toInt
            Bounded.uint16(bg, (lf(i) & 0xffff).toInt, rng, Bounded.genMask(rng).toInt, masked, buf).toShort
          }
          NDArray.fromArray(out.asInstanceOf[Array[T]], outShape)(using dtype)
        case "int8" =>
          val out = Array.tabulate(n) { i =>
            val rng = (((hf(i) - isOpen) - lf(i)) & 0xff).toInt
            Bounded.uint8(bg, (lf(i) & 0xff).toInt, rng, Bounded.genMask(rng).toInt, masked, buf).toByte
          }
          NDArray.fromArray(out.asInstanceOf[Array[T]], outShape)(using dtype)
        case _ =>
          val out = Array.tabulate(n) { i =>
            val rng = (((hf(i) - isOpen) - lf(i)) & 0xff).toInt
            Bounded.bool(bg, lf(i) != 0, rng, buf)
          }
          NDArray.fromArray(out.asInstanceOf[Array[T]], outShape)(using dtype)

  private def fill[T](bg: BitGenerator, off: Long, rng: Long, cnt: Int, kind: String, masked: Boolean, dtype: DType[T],
      shape: Array[Int]): NDArray[T] =
    val buf = new Bounded.Buf
    kind match
      case "int64" =>
        val mask = Bounded.genMask(rng)
        val out = new Array[Long](cnt)
        var i = 0
        while i < cnt do { out(i) = Bounded.uint64(bg, off, rng, mask, masked); i += 1 }
        NDArray.fromArray(out.asInstanceOf[Array[T]], shape)(using dtype)
      case "int32" =>
        val r = rng.toInt
        val mask = Bounded.genMask(rng & 0xffffffffL).toInt
        val out = new Array[Int](cnt)
        var i = 0
        while i < cnt do { out(i) = Bounded.uint32(bg, off.toInt, r, mask, masked); i += 1 }
        NDArray.fromArray(out.asInstanceOf[Array[T]], shape)(using dtype)
      case "int16" =>
        val r = (rng & 0xffff).toInt
        val mask = Bounded.genMask(r).toInt
        val out = new Array[Short](cnt)
        var i = 0
        while i < cnt do { out(i) = Bounded.uint16(bg, (off & 0xffff).toInt, r, mask, masked, buf).toShort; i += 1 }
        NDArray.fromArray(out.asInstanceOf[Array[T]], shape)(using dtype)
      case "int8" =>
        val r = (rng & 0xff).toInt
        val mask = Bounded.genMask(r).toInt
        val out = new Array[Byte](cnt)
        var i = 0
        while i < cnt do { out(i) = Bounded.uint8(bg, (off & 0xff).toInt, r, mask, masked, buf).toByte; i += 1 }
        NDArray.fromArray(out.asInstanceOf[Array[T]], shape)(using dtype)
      case _ =>
        val out = new Array[Boolean](cnt)
        var i = 0
        while i < cnt do { out(i) = Bounded.bool(bg, off != 0, rng.toInt, buf); i += 1 }
        NDArray.fromArray(out.asInstanceOf[Array[T]], shape)(using dtype)
