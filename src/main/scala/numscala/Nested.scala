package numscala

/** Type class describing "array-like" values that `np.array` accepts: scalars with a
  * [[DType]], and arbitrarily nested `Seq`s / `Array`s / [[NDArray]]s of them.
  */
trait Nested[-A]:
  type Elem
  def dtype: DType[Elem]
  /** Shape of the value (empty for a scalar). */
  def shapeOf(a: A): List[Int]
  /** Writes the elements of `a` (in C order) into `out` starting at `pos`; returns the next position. */
  def write(a: A, out: Array[Elem], pos: Int): Int

object Nested extends LowPriorityNested:
  type Aux[A, E] = Nested[A] { type Elem = E }

  given seq[A](using n: Nested[A]): Nested.Aux[Seq[A], n.Elem] = new Nested[Seq[A]]:
    type Elem = n.Elem
    def dtype: DType[Elem] = n.dtype
    def shapeOf(a: Seq[A]): List[Int] =
      if a.isEmpty then List(0)
      else
        val inner = n.shapeOf(a.head)
        a.foreach(x =>
          if n.shapeOf(x) != inner then
            throw new IllegalArgumentException(
              "setting an array element with a sequence. The requested array has an inhomogeneous shape"
            )
        )
        a.length :: inner
    def write(a: Seq[A], out: Array[Elem], pos: Int): Int =
      var p = pos
      a.foreach(x => p = n.write(x, out, p))
      p

  given array[A](using n: Nested[A]): Nested.Aux[Array[A], n.Elem] = new Nested[Array[A]]:
    type Elem = n.Elem
    def dtype: DType[Elem] = n.dtype
    def shapeOf(a: Array[A]): List[Int] = seq[A].shapeOf(a.toSeq)
    def write(a: Array[A], out: Array[Elem], pos: Int): Int = seq[A].write(a.toSeq, out, pos)

  given ndarray[T](using d: DType[T]): Nested.Aux[NDArray[T], T] = new Nested[NDArray[T]]:
    type Elem = T
    def dtype: DType[T] = d
    def shapeOf(a: NDArray[T]): List[Int] = a.shape.toList
    def write(a: NDArray[T], out: Array[T], pos: Int): Int =
      val src = a.toArray
      System.arraycopy(src, 0, out, pos, src.length)
      pos + src.length

trait LowPriorityNested:
  given scalar[T](using d: DType[T]): Nested.Aux[T, T] = new Nested[T]:
    type Elem = T
    def dtype: DType[T] = d
    def shapeOf(a: T): List[Int] = Nil
    def write(a: T, out: Array[T], pos: Int): Int =
      out(pos) = a
      pos + 1
