package numscala

/** One component of an index expression, like an element of a NumPy index tuple. */
sealed trait Index

object Index:
  /** An integer index (removes the axis). */
  final case class At(i: Int) extends Index
  /** A basic slice `start:stop:step` (produces a view). */
  final case class Slice(start: Option[Int], stop: Option[Int], step: Int = 1) extends Index:
    if step == 0 then throw new IllegalArgumentException("slice step cannot be zero")
    /** Resolves this slice against an axis of length `n`: (start, step, length). */
    def resolve(n: Int): (Int, Int, Int) =
      if step > 0 then
        def clamp(v: Int) = if v < 0 then math.max(v + n, 0) else math.min(v, n)
        val s = start.map(clamp).getOrElse(0)
        val e = stop.map(clamp).getOrElse(n)
        (s, step, if e > s then (e - s + step - 1) / step else 0)
      else
        def clamp(v: Int) = if v < 0 then math.max(v + n, -1) else math.min(v, n - 1)
        val s = start.map(clamp).getOrElse(n - 1)
        val e = stop.map(clamp).getOrElse(-1)
        (s, step, if s > e then (s - e - step - 1) / -step else 0)
    override def toString: String =
      start.fold("")(_.toString) + ":" + stop.fold("")(_.toString) + (if step != 1 then ":" + step else "")
  /** `np.newaxis` — inserts an axis of length one. */
  case object NewAxis extends Index
  /** `...` — expands to as many full slices as needed. */
  case object Ellipsis extends Index
  /** Integer-array ("fancy") indexing; produces a copy. */
  final case class Take(indices: NDArray[Int]) extends Index
  /** Boolean-mask indexing; produces a copy. */
  final case class Mask(mask: NDArray[Boolean]) extends Index

  val All: Slice = Slice(None, None, 1)

  /** Parses a Python-like slice string: `":"`, `"1:"`, `"::-1"`, `"2:-1:2"`, `"..."`, `"None"`, `"-3"`. */
  def parse(s: String): Index =
    val t = s.trim
    if t == "..." then Ellipsis
    else if t == "None" || t == "newaxis" then NewAxis
    else if !t.contains(':') then At(t.toInt)
    else
      val parts = t.split(":", -1).map(_.trim)
      if parts.length > 3 then throw new IllegalArgumentException(s"invalid slice '$s'")
      def opt(i: Int): Option[Int] = if i < parts.length && parts(i).nonEmpty then Some(parts(i).toInt) else None
      Slice(opt(0), opt(1), opt(2).getOrElse(1))

  /** Converts a user-supplied index value into an [[Index]]. */
  def from(v: IndexLike): Index = v match
    case i: Int => At(i)
    case ix: Index => ix
    case s: String => parse(s)
    case r: Range =>
      if r.isEmpty then Slice(Some(r.start), Some(r.start), r.step)
      else if r.step > 0 then
        val stop = r.last + 1
        Slice(Some(r.head), if stop == 0 then None else Some(stop), r.step)
      else
        val stop = r.last - 1
        Slice(Some(r.head), if stop == -1 then None else Some(stop), r.step)
    case a: NDArray[?] =>
      a.dtype.kind match
        case 'b' => Mask(a.asInstanceOf[NDArray[Boolean]])
        case 'i' | 'u' => Take(a.asInstanceOf[NDArray[Any]].astypeDyn(DType.Int32).asInstanceOf[NDArray[Int]])
        case _ => throw new IndexOutOfBoundsException("arrays used as indices must be of integer (or boolean) type")
    case arr: Array[?] =>
      arr match
        case b: Array[Boolean] => Mask(NDArray.fromArray(b.clone(), Array(b.length)))
        case i: Array[Int] => Take(NDArray.fromArray(i.clone(), Array(i.length)))
        case l: Array[Long] => Take(NDArray.fromArray(l.map(_.toInt), Array(l.length)))
        case _ => throw new IndexOutOfBoundsException("unsupported index array")
    case seq: Seq[?] =>
      if seq.nonEmpty && seq.forall(_.isInstanceOf[Boolean]) then
        val b = seq.map(_.asInstanceOf[Boolean]).toArray
        Mask(NDArray.fromArray(b, Array(b.length)))
      else
        val i = seq.map {
          case x: Int => x
          case x: Long => x.toInt
          case other => throw new IndexOutOfBoundsException(s"invalid index $other")
        }.toArray
        Take(NDArray.fromArray(i, Array(i.length)))
    case None => NewAxis
    case _: scala.collection.immutable.::.type => All
