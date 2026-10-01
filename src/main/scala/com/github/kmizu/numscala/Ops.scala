package com.github.kmizu.numscala

/** Arithmetic operations with specialised kernels. */
enum Arith:
  case Add, Sub, Mul, Div, Pow

  def generic[U](d: NumDType[U]): (U, U) => U = this match
    case Add => d.plus
    case Sub => d.minus
    case Mul => d.times
    case Pow => d.power
    case Div =>
      d match
        case i: InexactDType[U] => i.div
        case r: RealDType[U] @unchecked => r.floorDiv

/** Ordering comparisons with IEEE semantics (any comparison involving NaN is false). */
enum CmpOp:
  case Lt, Le, Gt, Ge

  def test[T](d: DType[T], x: T, y: T): Boolean =
    if d.isNaN(x) || d.isNaN(y) then false
    else
      val c = d.compare(x, y)
      this match
        case Lt => c < 0
        case Le => c <= 0
        case Gt => c > 0
        case Ge => c >= 0

/** Elementwise kernels shared by operators and ufuncs. */
private[numscala] object Ops:

  /** Generic broadcasting binary operation after converting both inputs to `out`. */
  def binary[A, B, U](a: NDArray[A], b: NDArray[B], out: DType[U])(f: (U, U) => U): NDArray[U] =
    val ca = a.asType(using out)
    val cb = b.asType(using out)
    NDArray.zipMap(ca, cb)(f)(using out)

  /** Broadcasting arithmetic with primitive fast paths for float64/int64/int32. */
  def arith[A, B, U](a: NDArray[A], b: NDArray[B], out: NumDType[U], op: Arith): NDArray[U] =
    val ca = a.asType(using out)
    val cb = b.asType(using out)
    (out: DType[?]) match
      case DType.Float64 =>
        arithD(ca.asInstanceOf[NDArray[Double]], cb.asInstanceOf[NDArray[Double]], op).asInstanceOf[NDArray[U]]
      case DType.Int64 =>
        arithL(ca.asInstanceOf[NDArray[Long]], cb.asInstanceOf[NDArray[Long]], op).asInstanceOf[NDArray[U]]
      case DType.Int32 =>
        arithI(ca.asInstanceOf[NDArray[Int]], cb.asInstanceOf[NDArray[Int]], op).asInstanceOf[NDArray[U]]
      case _ => NDArray.zipMap(ca, cb)(op.generic(out))(using out)

  private inline def loopD(a: NDArray[Double], b: NDArray[Double])(inline f: (Double, Double) => Double): NDArray[Double] =
    val sh = Shape.broadcast(a.shapeArr, b.shapeArr)
    val sa = Strided.broadcastStrides(a.shapeArr, a.stridesArr, sh)
    val sb = Strided.broadcastStrides(b.shapeArr, b.stridesArr, sh)
    val res = new Array[Double](Shape.size(sh))
    val da = a.data
    val db = b.data
    var k = 0
    Strided.foreach2(sh, sa, a.offset, sb, b.offset) { (i, j) =>
      res(k) = f(da(i), db(j))
      k += 1
    }
    NDArray.fromArray(res, sh)

  private def arithD(a: NDArray[Double], b: NDArray[Double], op: Arith): NDArray[Double] = op match
    case Arith.Add => loopD(a, b)(_ + _)
    case Arith.Sub => loopD(a, b)(_ - _)
    case Arith.Mul => loopD(a, b)(_ * _)
    case Arith.Div => loopD(a, b)(_ / _)
    case Arith.Pow => loopD(a, b)(FloatDType.pow)

  private inline def loopL(a: NDArray[Long], b: NDArray[Long])(inline f: (Long, Long) => Long): NDArray[Long] =
    val sh = Shape.broadcast(a.shapeArr, b.shapeArr)
    val sa = Strided.broadcastStrides(a.shapeArr, a.stridesArr, sh)
    val sb = Strided.broadcastStrides(b.shapeArr, b.stridesArr, sh)
    val res = new Array[Long](Shape.size(sh))
    val da = a.data
    val db = b.data
    var k = 0
    Strided.foreach2(sh, sa, a.offset, sb, b.offset) { (i, j) =>
      res(k) = f(da(i), db(j))
      k += 1
    }
    NDArray.fromArray(res, sh)

  private def arithL(a: NDArray[Long], b: NDArray[Long], op: Arith): NDArray[Long] = op match
    case Arith.Add => loopL(a, b)(_ + _)
    case Arith.Sub => loopL(a, b)(_ - _)
    case Arith.Mul => loopL(a, b)(_ * _)
    case Arith.Div => loopL(a, b)((x, y) => if y == 0 then 0L else Math.floorDiv(x, y))
    case Arith.Pow => loopL(a, b)(DType.Int64.power)

  private inline def loopI(a: NDArray[Int], b: NDArray[Int])(inline f: (Int, Int) => Int): NDArray[Int] =
    val sh = Shape.broadcast(a.shapeArr, b.shapeArr)
    val sa = Strided.broadcastStrides(a.shapeArr, a.stridesArr, sh)
    val sb = Strided.broadcastStrides(b.shapeArr, b.stridesArr, sh)
    val res = new Array[Int](Shape.size(sh))
    val da = a.data
    val db = b.data
    var k = 0
    Strided.foreach2(sh, sa, a.offset, sb, b.offset) { (i, j) =>
      res(k) = f(da(i), db(j))
      k += 1
    }
    NDArray.fromArray(res, sh)

  private def arithI(a: NDArray[Int], b: NDArray[Int], op: Arith): NDArray[Int] = op match
    case Arith.Add => loopI(a, b)(_ + _)
    case Arith.Sub => loopI(a, b)(_ - _)
    case Arith.Mul => loopI(a, b)(_ * _)
    case Arith.Div => loopI(a, b)((x, y) => if y == 0 then 0 else Math.floorDiv(x, y))
    case Arith.Pow => loopI(a, b)(DType.Int32.power)

  /** In-place broadcasting update `a = f(a, b)`. */
  def inPlace[T](a: NDArray[T], b: NDArray[T])(f: (T, T) => T): Unit =
    val src = if b.data eq a.data then b.copy() else b
    val sb = Strided.broadcastStrides(src.shapeArr, src.stridesArr, a.shapeArr)
    val da = a.data
    val db = src.data
    Strided.foreach2(a.shapeArr, a.stridesArr, a.offset, sb, src.offset)((i, j) => da(i) = f(da(i), db(j)))

  def compare[A, B, U](a: NDArray[A], b: NDArray[B], d: DType[U], op: CmpOp): NDArray[Boolean] =
    val ca = a.asType(using d)
    val cb = b.asType(using d)
    NDArray.zipMap(ca, cb)((x, y) => op.test(d, x, y))(using DType.Bool)

  def equal[A, B, U](a: NDArray[A], b: NDArray[B], d: DType[U], eq: Boolean): NDArray[Boolean] =
    val ca = a.asType(using d)
    val cb = b.asType(using d)
    NDArray.zipMap(ca, cb)((x, y) => d.equiv(x, y) == eq)(using DType.Bool)

  /** Round half to even to `decimals` places (`np.round`). */
  def round[T](a: NDArray[T], decimals: Int)(using d: NumDType[T]): NDArray[T] =
    def rd(x: Double): Double =
      if decimals >= 0 then
        val f = math.pow(10.0, decimals)
        val y = math.rint(x * f) / f
        if y.isNaN && !x.isNaN then x else y
      else
        val f = math.pow(10.0, -decimals)
        math.rint(x / f) * f
    d.kind match
      case 'f' => a.map(x => d.fromDouble(rd(d.toDouble(x))))
      case 'c' => a.map(x => { val c = d.toComplex(x); d.fromComplex(Complex(rd(c.re), rd(c.im))) })
      case _ => if decimals >= 0 then a.copy() else a.map(x => d.fromDouble(rd(d.toDouble(x))))

  /** Diagonal view (`a.diagonal(offset, axis1, axis2)`). */
  def diagonal[T](a: NDArray[T], offset: Int, axis1: Int, axis2: Int): NDArray[T] =
    if a.ndim < 2 then throw new IllegalArgumentException("diag requires an array of at least two dimensions")
    val x = Shape.normAxis(axis1, a.ndim)
    val y = Shape.normAxis(axis2, a.ndim)
    if x == y then throw new IllegalArgumentException("axis1 and axis2 cannot be the same")
    val n1 = a.shapeArr(x)
    val n2 = a.shapeArr(y)
    var off = a.offset
    val len =
      if offset >= 0 then
        off += offset * a.stridesArr(y)
        math.max(0, math.min(n1, n2 - offset))
      else
        off += -offset * a.stridesArr(x)
        math.max(0, math.min(n1 + offset, n2))
    val keep = (0 until a.ndim).filter(i => i != x && i != y)
    val sh = keep.map(a.shapeArr(_)).toArray :+ len
    val st = keep.map(a.stridesArr(_)).toArray :+ (a.stridesArr(x) + a.stridesArr(y))
    a.view(sh, st, if len == 0 then a.offset else off)
