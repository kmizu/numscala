package numscala

/** Helpers for shapes, strides and broadcasting. Strides are measured in elements. */
object Shape:
  def size(shape: Array[Int]): Int =
    var n = 1L
    var i = 0
    while i < shape.length do
      n *= shape(i)
      i += 1
    if n > Int.MaxValue then throw new IllegalArgumentException("array is too big")
    n.toInt

  /** C-order (row-major) strides. */
  def cStrides(shape: Array[Int]): Array[Int] =
    val st = new Array[Int](shape.length)
    var acc = 1
    var i = shape.length - 1
    while i >= 0 do
      st(i) = acc
      acc *= math.max(shape(i), 1)
      i -= 1
    st

  def str(shape: Array[Int]): String =
    if shape.length == 1 then s"(${shape(0)},)" else shape.mkString("(", ", ", ")")
  def str(shape: Seq[Int]): String = str(shape.toArray)

  /** Normalizes a possibly negative axis. */
  def normAxis(axis: Int, ndim: Int): Int =
    val a = if axis < 0 then axis + ndim else axis
    if a < 0 || a >= ndim then
      throw new IndexOutOfBoundsException(s"axis $axis is out of bounds for array of dimension $ndim")
    a

  /** Broadcast shape of several shapes (NumPy broadcasting rules). */
  def broadcast(shapes: Array[Int]*): Array[Int] =
    val nd = if shapes.isEmpty then 0 else shapes.map(_.length).max
    val out = Array.fill(nd)(1)
    for s <- shapes do
      val off = nd - s.length
      var i = 0
      while i < s.length do
        val d = s(i)
        val o = out(off + i)
        if o == 1 then out(off + i) = d
        else if d != 1 && d != o then
          throw new IllegalArgumentException(
            "operands could not be broadcast together with shapes " + shapes.map(str).mkString(" ")
          )
        i += 1
    out

  /** Resolves a single `-1` entry of a requested shape. */
  def resolve(newShape: Array[Int], size: Int): Array[Int] =
    val unknown = newShape.indices.filter(newShape(_) == -1)
    if unknown.length > 1 then throw new IllegalArgumentException("can only specify one unknown dimension")
    if newShape.exists(_ < -1) then throw new IllegalArgumentException("negative dimensions not allowed")
    if unknown.isEmpty then
      if Shape.size(newShape) != size then
        throw new IllegalArgumentException(s"cannot reshape array of size $size into shape ${str(newShape)}")
      newShape
    else
      val known = newShape.filter(_ != -1).foldLeft(1)(_ * _)
      if known == 0 || size % known != 0 then
        throw new IllegalArgumentException(s"cannot reshape array of size $size into shape ${str(newShape)}")
      newShape.map(d => if d == -1 then size / known else d)

/** Iteration over all multi-indices of a shape in C order, tracking element offsets of
  * several strided operands at once.  This is the core loop of every elementwise kernel.
  */
private[numscala] object Strided:
  /** Calls `f(o0)` with the offset of each element of one operand, in C order. */
  inline def foreach1(shape: Array[Int], st0: Array[Int], off0: Int)(inline f: Int => Unit): Unit =
    val nd = shape.length
    val n = Shape.size(shape)
    if n > 0 then
      if nd == 0 then f(off0)
      else
        val last = nd - 1
        val len = shape(last)
        val s0 = st0(last)
        val idx = new Array[Int](nd)
        var base0 = off0
        var count = 0
        while count < n do
          var j = 0
          var o0 = base0
          while j < len do
            f(o0)
            o0 += s0
            j += 1
          count += len
          // advance outer index
          var ax = last - 1
          var carry = true
          while carry && ax >= 0 do
            idx(ax) += 1
            base0 += st0(ax)
            if idx(ax) < shape(ax) then carry = false
            else
              base0 -= st0(ax) * shape(ax)
              idx(ax) = 0
              ax -= 1

  inline def foreach2(shape: Array[Int], st0: Array[Int], off0: Int, st1: Array[Int], off1: Int)(
      inline f: (Int, Int) => Unit
  ): Unit =
    val nd = shape.length
    val n = Shape.size(shape)
    if n > 0 then
      if nd == 0 then f(off0, off1)
      else
        val last = nd - 1
        val len = shape(last)
        val s0 = st0(last)
        val s1 = st1(last)
        val idx = new Array[Int](nd)
        var base0 = off0
        var base1 = off1
        var count = 0
        while count < n do
          var j = 0
          var o0 = base0
          var o1 = base1
          while j < len do
            f(o0, o1)
            o0 += s0
            o1 += s1
            j += 1
          count += len
          var ax = last - 1
          var carry = true
          while carry && ax >= 0 do
            idx(ax) += 1
            base0 += st0(ax)
            base1 += st1(ax)
            if idx(ax) < shape(ax) then carry = false
            else
              base0 -= st0(ax) * shape(ax)
              base1 -= st1(ax) * shape(ax)
              idx(ax) = 0
              ax -= 1

  inline def foreach3(
      shape: Array[Int],
      st0: Array[Int],
      off0: Int,
      st1: Array[Int],
      off1: Int,
      st2: Array[Int],
      off2: Int
  )(inline f: (Int, Int, Int) => Unit): Unit =
    val nd = shape.length
    val n = Shape.size(shape)
    if n > 0 then
      if nd == 0 then f(off0, off1, off2)
      else
        val last = nd - 1
        val len = shape(last)
        val s0 = st0(last)
        val s1 = st1(last)
        val s2 = st2(last)
        val idx = new Array[Int](nd)
        var base0 = off0
        var base1 = off1
        var base2 = off2
        var count = 0
        while count < n do
          var j = 0
          var o0 = base0
          var o1 = base1
          var o2 = base2
          while j < len do
            f(o0, o1, o2)
            o0 += s0
            o1 += s1
            o2 += s2
            j += 1
          count += len
          var ax = last - 1
          var carry = true
          while carry && ax >= 0 do
            idx(ax) += 1
            base0 += st0(ax)
            base1 += st1(ax)
            base2 += st2(ax)
            if idx(ax) < shape(ax) then carry = false
            else
              base0 -= st0(ax) * shape(ax)
              base1 -= st1(ax) * shape(ax)
              base2 -= st2(ax) * shape(ax)
              idx(ax) = 0
              ax -= 1

  /** Strides of an operand broadcast to `target` (zero stride on broadcast axes). */
  def broadcastStrides(shape: Array[Int], strides: Array[Int], target: Array[Int]): Array[Int] =
    val nd = target.length
    val off = nd - shape.length
    if off < 0 then throw new IllegalArgumentException("cannot broadcast to fewer dimensions")
    val out = new Array[Int](nd)
    var i = 0
    while i < shape.length do
      val d = shape(i)
      if d == target(off + i) then out(off + i) = strides(i)
      else if d == 1 then out(off + i) = 0
      else
        throw new IllegalArgumentException(
          s"could not broadcast input array from shape ${Shape.str(shape)} into shape ${Shape.str(target)}"
        )
      i += 1
    out
