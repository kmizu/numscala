package numscala

import scala.collection.mutable

/** Einstein summation (`np.einsum`): subscript parsing, diagonal extraction for repeated labels,
  * broadcasting of `...` dimensions and pairwise contraction (left to right) through `matmul`.
  */
private[numscala] object LinalgEinsum:

  /** Parsed subscripts: labels per operand and for the output. Letters use their char code,
    * broadcast (`...`) dimensions use negative labels `-E .. -1` (left to right).
    */
  final case class Spec(inputs: Seq[Array[Int]], output: Array[Int])

  private def isLabel(c: Char): Boolean = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')

  /** Splits a term into letters and an optional ellipsis position. */
  private def term(t: String, what: String): (Seq[Char], Int) =
    val chars = mutable.ArrayBuffer.empty[Char]
    var ell = -1
    var i = 0
    while i < t.length do
      val c = t(i)
      if c == '.' then
        if t.startsWith("...", i) && ell < 0 then
          ell = chars.length
          i += 3
        else throw new IllegalArgumentException(s"einstein sum subscripts string contains a '.' that is not part of an ellipsis ('...') in $what")
      else if isLabel(c) then
        chars += c
        i += 1
      else if c == ' ' then i += 1
      else throw new IllegalArgumentException(s"invalid subscript '$c' in einstein sum subscripts string, subscripts must be letters")
    (chars.toSeq, ell)

  def parse(subscripts: String, ndims: Seq[Int]): Spec =
    val arrow = subscripts.indexOf("->")
    val lhs = if arrow >= 0 then subscripts.substring(0, arrow) else subscripts
    val rhs = if arrow >= 0 then Some(subscripts.substring(arrow + 2)) else None
    if rhs.exists(_.contains("->")) then throw new IllegalArgumentException("einstein sum subscript string contains multiple '->'")
    val terms = lhs.split(",", -1).toSeq
    if terms.length != ndims.length then
      throw new IllegalArgumentException(
        if terms.length > ndims.length then "more operands provided to einstein sum function than specified in the subscripts string"
        else "fewer operands provided to einstein sum function than specified in the subscripts string"
      )
    val parsed = terms.zipWithIndex.map((t, k) => term(t, s"operand $k"))
    var maxEll = 0
    val inputs = parsed.zip(ndims).zipWithIndex.map { case (((letters, ell), nd), k) =>
      if ell < 0 then
        if letters.length != nd then
          throw new IllegalArgumentException(
            if letters.length > nd then s"einstein sum subscripts string contains too many subscripts for operand $k"
            else s"operand has more dimensions than subscripts given in einstein sum, but no '...' ellipsis provided to broadcast the extra dimensions."
          )
        letters.map(_.toInt).toArray
      else
        val e = nd - letters.length
        if e < 0 then throw new IllegalArgumentException(s"einstein sum subscripts string contains too many subscripts for operand $k")
        maxEll = math.max(maxEll, e)
        (letters.take(ell).map(_.toInt) ++ (-e until 0) ++ letters.drop(ell).map(_.toInt)).toArray
    }
    val output: Array[Int] = rhs match
      case Some(r) =>
        val (letters, ell) = term(r, "the output")
        if ell < 0 && maxEll > 0 then
          throw new IllegalArgumentException(
            "output has more dimensions than subscripts given in einstein sum, but no '...' ellipsis provided to broadcast the extra dimensions."
          )
        val out =
          if ell < 0 then letters.map(_.toInt).toArray
          else (letters.take(ell).map(_.toInt) ++ (-maxEll until 0) ++ letters.drop(ell).map(_.toInt)).toArray
        val all = inputs.flatten.toSet
        letters.foreach { c =>
          if letters.count(_ == c) > 1 then
            throw new IllegalArgumentException(s"einstein sum subscripts string includes output subscript '$c' multiple times")
          if !all.contains(c.toInt) then
            throw new IllegalArgumentException(s"einstein sum subscripts string included output subscript '$c' which never appeared in an input")
        }
        out
      case None =>
        val counts = mutable.Map.empty[Int, Int]
        inputs.foreach(_.foreach(l => if l >= 0 then counts(l) = counts.getOrElse(l, 0) + 1))
        ((-maxEll until 0) ++ counts.filter(_._2 == 1).keys.toSeq.sorted).toArray
    Spec(inputs, output)

  def einsum[T](subscripts: String, operands: Seq[NDArray[T]], d: NumDType[T]): NDArray[T] =
    if operands.isEmpty then throw new IllegalArgumentException("No input operands")
    val spec = parse(subscripts, operands.map(_.ndim))
    // label sizes (size-1 dimensions broadcast)
    val sizes = mutable.Map.empty[Int, Int]
    spec.inputs.zip(operands).foreach { (labels, a) =>
      labels.indices.foreach { i =>
        val l = labels(i)
        val s = a.shapeArr(i)
        sizes.get(l) match
          case None => sizes(l) = s
          case Some(prev) =>
            if prev == 1 then sizes(l) = s
            else if s != 1 && s != prev then
              val name = if l >= 0 then s"'${l.toChar}'" else "'...'"
              throw new IllegalArgumentException(
                s"operands could not be broadcast together with remapped shapes: label $name has sizes $prev and $s"
              )
      }
    }
    // diagonals for repeated labels, then broadcast every dimension to the label size
    var ops: List[(Array[Int], NDArray[T])] = spec.inputs.zip(operands).map { (labels, a0) =>
      val a = a0.asType(using d)
      val uniq = labels.distinct
      val shape = uniq.map(l => a.shapeArr(labels.indexOf(l)))
      uniq.foreach { l =>
        val dims = labels.indices.filter(labels(_) == l).map(a.shapeArr(_))
        if dims.distinct.length > 1 then
          throw new IllegalArgumentException(
            s"dimensions in operand for collapsing index '${l.toChar}' don't match (${dims.mkString(" != ")})"
          )
      }
      val strides = uniq.map(l => labels.indices.filter(labels(_) == l).map(a.stridesArr(_)).sum)
      val v = if uniq.length == labels.length then a else a.view(shape, strides, a.offset)
      (uniq, v.broadcastTo(uniq.map(sizes(_)).toSeq*))
    }.toList
    val out = spec.output
    while ops.length > 1 do
      val (la, a) = ops.head
      val (lb, b) = ops(1)
      val rest = ops.drop(2)
      val needed = (rest.flatMap(_._1) ++ out).toSet
      ops = contract(la, a, lb, b, needed, d) :: rest
    val (labels, res) = ops.head
    // sum labels not in the output, then order as the output
    val sumAxes = labels.indices.filterNot(i => out.contains(labels(i)))
    val (lab2, r2) =
      if sumAxes.isEmpty then (labels, res)
      else (labels.filter(out.contains), Reduce.sum(res, sumAxes, false, d))
    val perm = out.map(l => lab2.indexOf(l))
    if perm.isEmpty then r2 else r2.transpose(perm.toSeq*)

  /** Contracts two operands, keeping the labels in `needed`. */
  private def contract[T](la0: Array[Int], a0: NDArray[T], lb0: Array[Int], b0: NDArray[T], needed: Set[Int], d: NumDType[T]): (Array[Int], NDArray[T]) =
    def sumOut(labels: Array[Int], x: NDArray[T], other: Array[Int]): (Array[Int], NDArray[T]) =
      val drop = labels.indices.filter(i => !needed(labels(i)) && !other.contains(labels(i)))
      if drop.isEmpty then (labels, x)
      else (labels.indices.filterNot(drop.contains).map(labels(_)).toArray, Reduce.sum(x, drop, false, d))
    val (la, a) = sumOut(la0, a0, lb0)
    val (lb, b) = sumOut(lb0, b0, la)
    val batch = la.filter(l => lb.contains(l) && needed(l))
    val summed = la.filter(l => lb.contains(l) && !needed(l))
    val aFree = la.filterNot(lb.contains)
    val bFree = lb.filterNot(la.contains)
    def dim(labels: Array[Int], x: NDArray[T], l: Int) = x.shapeArr(labels.indexOf(l))
    val bDims = batch.map(dim(la, a, _))
    val aDims = aFree.map(dim(la, a, _))
    val bfDims = bFree.map(dim(lb, b, _))
    val k = summed.map(dim(la, a, _)).product
    val at = a.transpose((batch ++ aFree ++ summed).map(la.indexOf(_)).toSeq*).reshape(bDims.product, aDims.product, k)
    val bt = b.transpose((batch ++ summed ++ bFree).map(lb.indexOf(_)).toSeq*).reshape(bDims.product, k, bfDims.product)
    val r = LinAlgCore.matmulD(at, bt, d)
    ((batch ++ aFree ++ bFree), r.reshape((bDims ++ aDims ++ bfDims).toSeq*))

  /** A left-to-right contraction path in NumPy's `einsum_path` format (contracted operands are
    * removed and the intermediate is appended), plus a short report.
    */
  def path(subscripts: String, shapes: Seq[Array[Int]]): (Seq[Seq[Int]], String) =
    val spec = parse(subscripts, shapes.map(_.length))
    val n = shapes.length
    val steps: Seq[Seq[Int]] =
      if n == 1 then Seq(Seq(0))
      else Seq(0, 1) +: (2 until n).map(i => Seq(0, n - i))
    val labels = (spec.inputs.flatten ++ spec.output).distinct
    def name(l: Int) = if l >= 0 then l.toChar.toString else "..."
    val full =
      spec.inputs.map(_.map(name).mkString).mkString(",") + "->" + spec.output.map(name).mkString
    val report =
      s"  Complete contraction:  $full\n" +
        s"         Naive scaling:  ${labels.length}\n" +
        s"     Optimized scaling:  ${labels.length}\n" +
        s"  Contraction path (left to right): ${steps.map(_.mkString("(", ", ", ")")).mkString(", ")}"
    (steps, report)
