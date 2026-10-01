package com.github.kmizu.numscala.random

import com.github.kmizu.numscala.*

/** A distribution parameter: a scalar or anything array-like (broadcast like NumPy). */
type Param = Double | Int | Long | Float | NDArray[?] | Seq[?] | Array[?]

/** An output `size` argument: a single length or a shape (`null` = not given). */
type SizeArg = Int | Seq[Int] | Null

/** Parameter constraints of NumPy's `_common.pyx` (`CONS_*`). */
private[numscala] enum Cons:
  case NoCons, NonNegative, Positive, PositiveNotNan, Bounded01, BoundedGt01, BoundedLt01, Gt1, Gte1,
    Poisson, LegacyPoisson, LegacyNonNegInboundsLong

private[numscala] object RCommon:
  val LegacyPoissonLamMax: Double = Dist.PoissonLamMax

  def sizeShape(size: SizeArg): Array[Int] | Null = size match
    case null => null
    case i: Int => Array(i)
    case s: Seq[?] => s.asInstanceOf[Seq[Int]].toArray

  def checkShape(shp: Array[Int]): Array[Int] =
    if shp.exists(_ < 0) then throw new IllegalArgumentException("negative dimensions are not allowed")
    shp

  def isScalar(p: Param): Boolean = p match
    case a: NDArray[?] => a.ndim == 0
    case _: Seq[?] | _: Array[?] => false
    case _ => true

  private def nestedShape(x: Any): List[Int] = x match
    case s: Seq[?] => s.length :: (if s.isEmpty then Nil else nestedShape(s.head))
    case a: Array[?] => a.length :: (if a.isEmpty then Nil else nestedShape(a.head))
    case _ => Nil

  private def flatten(x: Any, out: scala.collection.mutable.ArrayBuffer[Any]): Unit = x match
    case s: Seq[?] => s.foreach(flatten(_, out))
    case a: Array[?] => a.foreach(flatten(_, out))
    case v => out += v

  def scalarToDouble(v: Any): Double = v match
    case d: Double => d
    case i: Int => i.toDouble
    case l: Long => l.toDouble
    case f: Float => f.toDouble
    case s: Short => s.toDouble
    case b: Byte => b.toDouble
    case b: Boolean => if b then 1.0 else 0.0
    case b: BigInt => b.toDouble
    case other => throw new IllegalArgumentException(s"cannot interpret $other as a number")

  def scalarToLong(v: Any): Long = v match
    case d: Double => d.toLong
    case f: Float => f.toLong
    case other => RandomUtil.anyToLong(other)

  /** Converts a parameter to a float64 array (0-d for scalars). */
  def toD(p: Any): NDArray[Double] = p match
    case a: NDArray[?] =>
      val src = a.asInstanceOf[NDArray[Any]]
      val d = src.dtype
      src.map(x => d.toDouble(x))(using DType.Float64)
    case _: Seq[?] | _: Array[?] =>
      val buf = scala.collection.mutable.ArrayBuffer.empty[Any]
      flatten(p, buf)
      val shp = nestedShape(p).toArray
      if Shape.size(shp) != buf.length then throw new IllegalArgumentException("inhomogeneous sequence")
      NDArray.fromArray(buf.map(scalarToDouble).toArray, shp)
    case v => NDArray.scalar(scalarToDouble(v))

  /** Converts a parameter to an int64 array (0-d for scalars), truncating floats like NumPy's cast. */
  def toL(p: Any): NDArray[Long] = p match
    case a: NDArray[?] =>
      val src = a.asInstanceOf[NDArray[Any]]
      val d = src.dtype
      src.map(x => d.toLong(x))(using DType.Int64)
    case _: Seq[?] | _: Array[?] =>
      val buf = scala.collection.mutable.ArrayBuffer.empty[Any]
      flatten(p, buf)
      val shp = nestedShape(p).toArray
      NDArray.fromArray(buf.map(scalarToLong).toArray, shp)
    case v => NDArray.scalar(scalarToLong(v))

  def scalarD(p: Param): Double = p match
    case a: NDArray[?] => toD(a).item
    case v => scalarToDouble(v)

  def scalarL(p: Param): Long = p match
    case a: NDArray[?] => toL(a).item
    case v => scalarToLong(v)

  private def err(msg: String) = new IllegalArgumentException(msg)

  /** `check_constraint` for a scalar. */
  def check(v: Double, name: String, c: Cons): Unit = c match
    case Cons.NoCons => ()
    case Cons.NonNegative => if !v.isNaN && (v < 0 || (v == 0 && 1.0 / v < 0)) then throw err(s"$name < 0")
    case Cons.Positive => if v <= 0 then throw err(s"$name <= 0")
    case Cons.PositiveNotNan =>
      if v.isNaN then throw err(s"$name must not be NaN")
      else if v <= 0 then throw err(s"$name <= 0")
    case Cons.Bounded01 => if !(v >= 0) || !(v <= 1) then throw err(s"$name < 0, $name > 1 or $name is NaN")
    case Cons.BoundedGt01 => if !(v > 0) || !(v <= 1) then throw err(s"$name <= 0, $name > 1 or $name contains NaNs")
    case Cons.BoundedLt01 => if !(v >= 0) || !(v < 1) then throw err(s"$name < 0, $name >= 1 or $name is NaN")
    case Cons.Gt1 => if !(v > 1) then throw err(s"$name <= 1 or $name is NaN")
    case Cons.Gte1 => if !(v >= 1) then throw err(s"$name < 1 or $name is NaN")
    case Cons.Poisson =>
      if !(v >= 0) then throw err(s"$name < 0 or $name is NaN")
      else if !(v <= Dist.PoissonLamMax) then throw err(s"$name value too large")
    case Cons.LegacyPoisson =>
      if !(v >= 0) then throw err(s"$name < 0 or $name is NaN")
      else if !(v <= LegacyPoissonLamMax) then throw err(s"$name value too large")
    case Cons.LegacyNonNegInboundsLong =>
      if v < 0 then throw err(s"$name < 0")
      else if v > Long.MaxValue.toDouble then
        throw err(s"$name is out of bounds for long, consider using the new generator API for 64bit integers.")

  /** `check_array_constraint`. */
  def checkArr(vs: Array[Double], name: String, c: Cons): Unit = c match
    case Cons.NoCons => ()
    case Cons.NonNegative =>
      if vs.exists(v => !v.isNaN && (v < 0 || (v == 0 && 1.0 / v < 0))) then throw err(s"$name < 0")
    case Cons.Positive => if vs.exists(_ <= 0) then throw err(s"$name <= 0")
    case Cons.PositiveNotNan =>
      if vs.exists(_.isNaN) then throw err(s"$name must not be NaN")
      else if vs.exists(_ <= 0) then throw err(s"$name <= 0")
    case Cons.Bounded01 =>
      if !vs.forall(v => v >= 0 && v <= 1) then throw err(s"$name < 0, $name > 1 or $name contains NaNs")
    case Cons.BoundedGt01 =>
      if !vs.forall(v => v > 0 && v <= 1) then throw err(s"$name <= 0, $name > 1 or $name contains NaNs")
    case Cons.BoundedLt01 =>
      if !vs.forall(v => v >= 0 && v < 1) then throw err(s"$name < 0, $name >= 1 or $name contains NaNs")
    case Cons.Gt1 => if !vs.forall(_ > 1) then throw err(s"$name <= 1 or $name contains NaNs")
    case Cons.Gte1 => if !vs.forall(_ >= 1) then throw err(s"$name < 1 or $name contains NaNs")
    case Cons.Poisson | Cons.LegacyPoisson =>
      val mx = if c == Cons.Poisson then Dist.PoissonLamMax else LegacyPoissonLamMax
      if !vs.forall(_ <= mx) then throw err(s"$name value too large")
      else if !vs.forall(_ >= 0.0) then throw err(s"$name < 0 or $name contains NaNs")
    case Cons.LegacyNonNegInboundsLong =>
      if !vs.forall(_ >= 0) then throw err(s"$name < 0")

  /** Broadcasts the parameters against each other (and against `size` when given), returning the
    * output shape and each parameter flattened in C order to that shape.
    */
  def broadcastAll[T](size: SizeArg, ps: Seq[NDArray[T]]): (Array[Int], Seq[Array[T]]) =
    val shapes = ps.map(_.shapeArr)
    val bshape =
      try Shape.broadcast(shapes*)
      catch case _: IllegalArgumentException =>
        throw new IllegalArgumentException("shape mismatch: objects cannot be broadcast to a single shape")
    val sz = sizeShape(size)
    val out =
      if sz == null then bshape
      else
        val s = checkShape(sz)
        val joint =
          try Shape.broadcast((s +: shapes)*)
          catch case _: IllegalArgumentException => null
        if joint == null || !joint.sameElements(s) then
          throw new IllegalArgumentException(
            s"Output size ${Shape.str(s)} is not compatible with broadcast dimensions of inputs ${Shape.str(bshape)}."
          )
        s
    (out, ps.map(p => p.broadcastTo(out*).toArray))

  def fillD(shape: Array[Int])(f: => Double): NDArray[Double] =
    val n = Shape.size(shape)
    val out = new Array[Double](n)
    var i = 0
    while i < n do { out(i) = f; i += 1 }
    NDArray.fromArray(out, shape)

  def fillL(shape: Array[Int])(f: => Long): NDArray[Long] =
    val n = Shape.size(shape)
    val out = new Array[Long](n)
    var i = 0
    while i < n do { out(i) = f; i += 1 }
    NDArray.fromArray(out, shape)

  /** NumPy's `cont` for 1–3 parameters: scalar-or-array parameters broadcast to the output. */
  def cont(size: SizeArg, ps: Seq[(Param, String, Cons)])(f: Array[Double] => Double): NDArray[Double] =
    if ps.forall(t => isScalar(t._1)) then
      val vs = ps.map { case (p, n, c) => val v = scalarD(p); check(v, n, c); v }.toArray
      val shp = sizeShape(size)
      if shp == null then NDArray.scalar(f(vs))
      else fillD(checkShape(shp))(f(vs))
    else
      val arrs = ps.map { case (p, n, c) =>
        val a = toD(p)
        checkArr(a.toArray, n, c)
        a
      }
      val (shape, flat) = broadcastAll(size, arrs)
      val n = Shape.size(shape)
      val out = new Array[Double](n)
      val buf = new Array[Double](flat.length)
      var i = 0
      while i < n do
        var k = 0
        while k < buf.length do { buf(k) = flat(k)(i); k += 1 }
        out(i) = f(buf)
        i += 1
      NDArray.fromArray(out, shape)

  /** NumPy's `disc` for float parameters producing int64 samples. */
  def disc(size: SizeArg, ps: Seq[(Param, String, Cons)])(f: Array[Double] => Long): NDArray[Long] =
    if ps.forall(t => isScalar(t._1)) then
      val vs = ps.map { case (p, n, c) => val v = scalarD(p); check(v, n, c); v }.toArray
      val shp = sizeShape(size)
      if shp == null then NDArray.scalar(f(vs))
      else fillL(checkShape(shp))(f(vs))
    else
      val arrs = ps.map { case (p, n, c) =>
        val a = toD(p)
        checkArr(a.toArray, n, c)
        a
      }
      val (shape, flat) = broadcastAll(size, arrs)
      val n = Shape.size(shape)
      val out = new Array[Long](n)
      val buf = new Array[Double](flat.length)
      var i = 0
      while i < n do
        var k = 0
        while k < buf.length do { buf(k) = flat(k)(i); k += 1 }
        out(i) = f(buf)
        i += 1
      NDArray.fromArray(out, shape)

  /** NumPy's `discrete_broadcast_iii`/`disc` for int64 parameters. */
  def discL(size: SizeArg, ps: Seq[(Param, String, Cons)])(f: Array[Long] => Long): NDArray[Long] =
    val arrs = ps.map { case (p, n, c) =>
      val a = toL(p)
      checkArr(a.toArray.map(_.toDouble), n, c)
      a
    }
    if ps.forall(t => isScalar(t._1)) then
      val vs = arrs.map(_.item).toArray
      val shp = sizeShape(size)
      if shp == null then NDArray.scalar(f(vs))
      else fillL(checkShape(shp))(f(vs))
    else
      val (shape, flat) = broadcastAll(size, arrs)
      val n = Shape.size(shape)
      val out = new Array[Long](n)
      val buf = new Array[Long](flat.length)
      var i = 0
      while i < n do
        var k = 0
        while k < buf.length do { buf(k) = flat(k)(i); k += 1 }
        out(i) = f(buf)
        i += 1
      NDArray.fromArray(out, shape)

  /** `kahan_sum` over `n` items starting at `off`. */
  def kahanSum(a: Array[Double], off: Int, n: Int): Double =
    if n <= 0 then 0.0
    else
      var sum = a(off)
      var c = 0.0
      var i = 1
      while i < n do
        val y = a(off + i) - c
        val t = sum + y
        c = (t - sum) - y
        sum = t
        i += 1
      sum
