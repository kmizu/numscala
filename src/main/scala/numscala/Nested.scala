package numscala

/** Type class describing "array-like" values that `np.array` accepts: scalars with a
  * [[DType]], and arbitrarily nested `Seq`s / `Array`s / [[NDArray]]s of them.
  * `E` is the element type of the resulting array.
  */
trait Nested[-A, E]:
  def dtype: DType[E]
  /** Shape of the value (empty for a scalar). */
  def shapeOf(a: A): List[Int]
  /** Writes the elements of `a` (in C order) into `out` starting at `pos`; returns the next position. */
  def write(a: A, out: Array[E], pos: Int): Int

object Nested extends LowPriorityNested:
  given seq[A, E](using n: Nested[A, E]): Nested[Seq[A], E] with
    def dtype: DType[E] = n.dtype
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
    def write(a: Seq[A], out: Array[E], pos: Int): Int =
      var p = pos
      a.foreach(x => p = n.write(x, out, p))
      p

  given array[A, E](using n: Nested[A, E]): Nested[Array[A], E] with
    def dtype: DType[E] = n.dtype
    def shapeOf(a: Array[A]): List[Int] = seq[A, E].shapeOf(a.toSeq)
    def write(a: Array[A], out: Array[E], pos: Int): Int = seq[A, E].write(a.toSeq, out, pos)

  given ndarray[T](using d: DType[T]): Nested[NDArray[T], T] with
    def dtype: DType[T] = d
    def shapeOf(a: NDArray[T]): List[Int] = a.shape.toList
    def write(a: NDArray[T], out: Array[T], pos: Int): Int =
      val src = a.toArray
      System.arraycopy(src, 0, out, pos, src.length)
      pos + src.length

trait LowPriorityNested:
  given scalar[T](using d: DType[T]): Nested[T, T] with
    def dtype: DType[T] = d
    def shapeOf(a: T): List[Int] = Nil
    def write(a: T, out: Array[T], pos: Int): Int =
      out(pos) = a
      pos + 1
