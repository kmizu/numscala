package numscala

import DiffData.*

/** Differential tests against NumPy 2.x.
  *
  * The cases in `src/test/resources/difftest/` are generated (randomized shapes, views and
  * dtypes) by `project/difftest/gen_difftest.py`, which records NumPy's results. Every case is
  * replayed here and the value, shape and dtype must match. Operations num-scala cannot express
  * statically (e.g. `bool - bool`, for which there is no promotion typeclass) are skipped; NumPy
  * errors must be errors here too.
  */
class DiffSuite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(300, "s")

  private case object Skip
  private type Res = NDArray[?] | Seq[NDArray[?]] | String | Skip.type

  private def lines(cat: String): Vector[String] =
    val src = scala.io.Source.fromResource(s"difftest/$cat.jsonl")
    try src.getLines().filter(_.nonEmpty).toVector
    finally src.close()

  /** Replays a category; `run` returns the num-scala result and the comparison tolerance. */
  private def replay(cat: String)(run: J => (Res, Tol)): Unit =
    val failures = Vector.newBuilder[String]
    var nFail = 0
    var nSkip = 0
    val ls = lines(cat)
    ls.zipWithIndex.foreach { (line, no) =>
      val c = DiffJson.parse(line).asInstanceOf[J]
      val r = c.get("r").map(_.asInstanceOf[J]).getOrElse(Map.empty)
      def fail(msg: String): Unit =
        nFail += 1
        if nFail <= 25 then failures += s"[$cat:${no + 1}] ${c.get("op").fold("")(o => s"$o: ")}$msg"
      val got: Either[Throwable, (Res, Tol)] =
        try Right(run(c))
        catch case e: (RuntimeException | NotImplementedError | Error) => Left(e)
      (got, r.get("err")) match
        case (Right((Skip, _)), _) => nSkip += 1
        case (Left(_), Some(_)) => ()
        case (Left(e), None) => fail(s"threw ${e.getClass.getSimpleName}: ${e.getMessage}")
        case (Right(_), Some(err)) => fail(s"numpy raised $err but num-scala returned a result")
        case (Right((res, tol)), None) =>
          res match
            case s: String => if s != c("repr") then fail(s"repr got:\n$s\nnumpy:\n${c("repr")}")
            case a: NDArray[?] => check(a, r("ok"), tol).foreach(fail)
            case xs: Seq[?] =>
              val es = r("list").asInstanceOf[Vector[Any]]
              if xs.length != es.length then fail(s"${xs.length} results != numpy ${es.length}")
              else
                xs.zip(es).zipWithIndex.foreach { case ((x, e), i) =>
                  check(x.asInstanceOf[NDArray[?]], e, tol).foreach(m => fail(s"part $i: $m"))
                }
            case Skip => ()
    }
    val msgs = failures.result()
    if nFail > 0 then
      fail(s"$cat: $nFail of ${ls.length} cases differ from NumPy (skipped $nSkip)\n" + msgs.mkString("\n"))

  private def dn(a: NDArray[?]): String = a.dtype.name
  private def axisOrNull(c: J): Int | Null = c.get("axis").flatMap(Option(_)).map(int).orNull
  private def axisOpt(c: J): Int | None.type = c.get("axis").flatMap(Option(_)).map(int).getOrElse(None)

  // ------------------------------------------------------------------ indexing

  test("getitem: basic, advanced and boolean indexing") {
    replay("getitem") { c => (array(c("a")).index(index(c("idx"))), Exact) }
  }

  test("setitem through views") {
    replay("setitem") { c =>
      val base = plain(c("a"))
      val v = applySteps(base, c("a").asInstanceOf[J]("w"))
      if c("scalar") == true then v.set(index(c("idx")))(array(c("val")).item)
      else v.setArray(index(c("idx")))(array(c("val")))
      (base, Exact)
    }
  }

  test("reshape / transpose / ravel") {
    replay("views") { c =>
      val a = array(c("a"))
      val r = c("op") match
        case "reshape" => a.reshape(ints(c("shape"))*)
        case "transpose" => Option(c("axes")).fold(a.T)(ax => a.transpose(ints(ax)*))
        case "ravel" => a.ravel()
      c.get("view").flatMap(Option(_)) match
        case Some(isView: Boolean) =>
          val shares = r.data eq a.data
          if shares != isView then throw new AssertionError(s"view=$shares but numpy view=$isView")
        case _ => ()
      (r, Exact)
    }
  }

  // ------------------------------------------------------------------ elementwise

  test("binary ufuncs and operators: broadcasting and dtype promotion") {
    replay("binary") { c =>
      val a = array(c("a"))
      val b = array(c("b"))
      val key = (dn(a), dn(b))
      val p = DiffTypes.promote(key).asInstanceOf[Promote[Any, Any]]
      val np2 = DiffTypes.numPromote.get(key).map(_.asInstanceOf[NumPromote[Any, Any]])
      val dp = DiffTypes.divPromote.get(key).map(_.asInstanceOf[DivPromote[Any, Any]])
      val arith = np2.map(q => UfuncTypes.arith(using q))
      val cmp = UfuncTypes.cmp(using p)
      // complex products may be computed with FMA by NumPy's SIMD loops (last-bit differences)
      def mag(x: NDArray[?]): Double = (0 until x.size).map(i => x.dtype.asInstanceOf[DType[Any]].toComplex(
        x.asInstanceOf[A].flatGet(i)).abs).filter(v => !v.isNaN && !v.isInfinite).maxOption.getOrElse(0.0)
      val exact = if p.dtype.isComplex then Tol(1e-15, 1e-15 * mag(a) * mag(b)) else Tol(0.0)
      val pow = Tol(1e-13)
      val f32 = if p.dtype eq DType.Float32 then Tol(1e-13) else exact
      def viaOp(r: NDArray[?], op: => NDArray[?]): NDArray[?] =
        // the operator form must agree with the ufunc form
        val o = op
        check(o, plainOf(r), exact).foreach(m => throw new AssertionError(s"operator form differs: $m"))
        r
      c("op") match
        case "add" =>
          val r = np.add(a, b)(using UfuncTypes.sum(using p))
          (np2.fold(r)(q => viaOp(r, a.+(b)(using q))), exact)
        case "subtract" => np2.fold((Skip, exact))(q => (viaOp(np.subtract(a, b)(using arith.get), a.-(b)(using q)), exact))
        case "multiply" =>
          val r = np.multiply(a, b)(using UfuncTypes.sum(using p))
          (np2.fold(r)(q => viaOp(r, a.*(b)(using q))), exact)
        case "true_divide" =>
          dp.fold((Skip, exact))(q => (viaOp(np.true_divide(a, b)(using UfuncTypes.div(using q)), a./(b)(using q)), exact))
        // NumPy computes float32 divmod in float32; num-scala in float64 then rounds (more exact)
        case "floor_divide" => arith.fold((Skip, exact))(t => (np.floor_divide(a, b)(using t), f32))
        case "remainder" => arith.fold((Skip, exact))(t => (np.remainder(a, b)(using t), f32))
        case "fmod" => arith.fold((Skip, exact))(t => (np.fmod(a, b)(using t), exact))
        case "power" => np2.fold((Skip, pow))(q => (viaOp(np.power(a, b)(using arith.get), a.**(b)(using q)), pow))
        case "equal" => (viaOp(np.equal(a, b)(using cmp), a.===(b)(using p)), exact)
        case "not_equal" => (viaOp(np.not_equal(a, b)(using cmp), a.=!=(b)(using p)), exact)
        case "less" => (viaOp(np.less(a, b)(using cmp), a.<(b)(using p)), exact)
        case "less_equal" => (viaOp(np.less_equal(a, b)(using cmp), a.<=(b)(using p)), exact)
        case "greater" => (viaOp(np.greater(a, b)(using cmp), a.>(b)(using p)), exact)
        case "greater_equal" => (viaOp(np.greater_equal(a, b)(using cmp), a.>=(b)(using p)), exact)
    }
  }

  /** Re-encodes an array as the JSON-like map `check` expects (for operator-vs-ufunc checks). */
  private def plainOf(a: NDArray[?]): Any =
    val d = a.dtype.asInstanceOf[DType[Any]]
    val vs = (0 until a.size).toVector.map(a.asInstanceOf[A].flatGet).map { x =>
      d.kind match
        case 'b' => d.toBoolean(x)
        case 'c' => val z = d.toComplex(x); Vector(z.re.toString, z.im.toString)
        case 'f' => d.toDouble(x).toString
        case _ => DiffJson.Num(d.toLong(x).toString)
    }
    Map("d" -> d.name, "s" -> a.shape.map(i => DiffJson.Num(i.toString)).toVector, "v" -> vs)

  test("astype between all dtypes") {
    replay("astype") { c => (array(c("a")).astypeDyn(DiffTypes.dtypes(c("to").asInstanceOf[String])), Exact) }
  }

  // ------------------------------------------------------------------ reductions

  test("reductions along every axis, keepdims, NaNs") {
    replay("reduce") { c =>
      val a = array(c("a"))
      val u = DiffTypes.unary(dn(a))
      val axis: Axis | Null = c.get("axis").flatMap(Option(_)) match
        case Some(v: Vector[?]) => ints(v)
        case Some(n) => int(n)
        case None => null
      lazy val ax: Int | Null = axisOrNull(c)
      val keep = c("keepdims") == true
      val ddof = int(c("ddof"))
      // tolerance scaled by the magnitude of the input (summation order differs from NumPy's)
      val m = if a.size == 0 then 0.0
        else (0 until a.size).map(a.asInstanceOf[A].flatGet).map(x => a.dtype.asInstanceOf[DType[Any]].toComplex(x).abs)
          .filter(x => !x.isNaN && !x.isInfinite).maxOption.getOrElse(0.0)
      val n = math.max(a.size, 1).toDouble
      val rt = if dn(a) == "float32" then 1e-6 else 1e-12
      val sumTol = Tol(rt, rt * m * n)
      val sum = u.sum.asInstanceOf[SumOf[Any]]
      val inex = u.inexact.asInstanceOf[ToInexact[Any]]
      val real = u.real.asInstanceOf[RealOf[Any]]
      c("op") match
        case "sum" => (np.sum(a, axis, keep)(using sum), sumTol)
        case "prod" => (np.prod(a, axis, keep)(using sum), Tol(rt))
        case "mean" => (np.mean(a, axis, keep)(using inex), sumTol)
        case "std" => (np.std(a, axis, ddof, keep)(using real), Tol(1e-9, 1e-9 * m))
        case "var" => (np.`var`(a, axis, ddof, keep)(using real), Tol(1e-9, 1e-9 * m * m))
        case "min" => (np.min(a, axis, keep), Exact)
        case "max" => (np.max(a, axis, keep), Exact)
        case "argmin" => (np.argmin(a, ax, keep), Tol(0.0, idx = true))
        case "argmax" => (np.argmax(a, ax, keep), Tol(0.0, idx = true))
        case "cumsum" => (np.cumsum(a, ax)(using sum), sumTol)
        case "cumprod" => (np.cumprod(a, ax)(using sum), Tol(rt))
        case "nansum" => (np.nansum(a, axis, keep)(using sum), sumTol)
        case "nanmean" => (np.nanmean(a, axis, keep)(using inex), sumTol)
        case "nanmin" => (np.nanmin(a, axis, keep), Exact)
        case "nanmax" => (np.nanmax(a, axis, keep), Exact)
    }
  }

  // ------------------------------------------------------------------ sorting

  test("sort / argsort (stable) / unique / searchsorted") {
    replay("sort") { c =>
      val a = array(c("a"))
      c("op") match
        case "sort" => (np.sort(a, axisOpt(c)), Exact)
        case "argsort" => (np.argsort(a, axisOpt(c), "stable"), Tol(0.0, idx = true))
        case "unique" => (np.unique(a), Exact)
        case "searchsorted" =>
          val v = array(c("v"))
          val p = DiffTypes.promote((dn(a), dn(v))).asInstanceOf[Promote[Any, Any]]
          (np.searchsorted(a, v, c("side").asInstanceOf[String])(using p), Tol(0.0, idx = true))
    }
  }

  // ------------------------------------------------------------------ shape manipulation

  test("concatenate / stack / split / pad / roll / flip") {
    replay("shape") { c =>
      c("op") match
        case "concatenate" =>
          (np.concatenate(c("arrs").asInstanceOf[Vector[Any]].map(array), int(c("axis"))), Exact)
        case "stack" => (np.stack(c("arrs").asInstanceOf[Vector[Any]].map(array), int(c("axis"))), Exact)
        case op @ ("split" | "array_split") =>
          val a = array(c("a"))
          val ios: Int | Seq[Int] = c("ios") match
            case v: Vector[?] => ints(v)
            case n => int(n)
          val r = if op == "split" then np.split(a, ios, int(c("axis"))) else np.array_split(a, ios, int(c("axis")))
          (r, Exact)
        case "pad" =>
          val a = array(c("a"))
          val pw = c("pw").asInstanceOf[Vector[Any]].map(p => ints(p))
          val mode = c("mode").asInstanceOf[String]
          // statistics / ramps are computed in a different order than NumPy (last-bit differences)
          val tol = if Set("mean", "median", "linear_ramp")(mode) then Tol(1e-14) else Exact
          (np.pad(a, pw, mode, constant_values = int(c("cv"))), tol)
        case "roll" =>
          val a = array(c("a"))
          val shift: Int | Seq[Int] = c("shift") match
            case v: Vector[?] => ints(v)
            case n => int(n)
          val axis: Axis | None.type = c("axis") match
            case null => None
            case v: Vector[?] => ints(v)
            case n => int(n)
          (np.roll(a, shift, axis), Exact)
        case "flip" =>
          val a = array(c("a"))
          (np.flip(a, axisOpt(c)), Exact)
    }
  }

  // ------------------------------------------------------------------ products

  test("matmul / dot with promotion") {
    replay("products") { c =>
      val a = array(c("a"))
      val b = array(c("b"))
      DiffTypes.numPromote.get((dn(a), dn(b))) match
        case None => (Skip, Exact)
        case Some(q0) =>
          val q = q0.asInstanceOf[NumPromote[Any, Any]]
          val r = if c("op") == "matmul" then np.matmul(a, b)(using q) else np.dot(a, b)(using q)
          (r, Tol(1e-12))
    }
  }

  // ------------------------------------------------------------------ printing

  test("str / repr exactly like NumPy") {
    replay("print") { c =>
      val a = array(c("a"))
      val s = a.toString
      if s != c("str") then throw new AssertionError(s"str differs:\n$s\nnumpy:\n${c("str")}")
      (a.repr, Tol(0.0))
    }
  }

  // ------------------------------------------------------------------ regressions found by the cases above

  test("regression: advanced indices separated by an (empty) Ellipsis go first") {
    val b = np.arange(60).reshape(5, 3, 4)
    val r = b(::, np.array(0, 1), ---, 0)
    assertEquals(r.shape, Seq(2, 5))
    assertEquals(r.toList, List(0, 12, 24, 36, 48, 4, 16, 28, 40, 52))
    assertEquals(b(---, 0, np.array(0, 1)).shape, Seq(5, 2)) // adjacent: stays in place
    val c = np.zeros[Int](5, 3, 4)
    c.setArray(Seq(::, np.array(0, 1), ---, 0))(np.arange(10).reshape(2, 5))
    assertEquals(c(::, "0:2", 0).toList, List(0, 5, 1, 6, 2, 7, 3, 8, 4, 9))
  }

  test("regression: (-1) ** +-inf == 1 for the operator too") {
    val r = np.array(-1, -1) ** np.array(Double.PositiveInfinity, Double.NegativeInfinity)
    assertEquals(r.toList, List(1.0, 1.0))
  }

  test("regression: complex division is bit-identical to NumPy") {
    val q = (np.array(Complex(1.0, 1.0)) / np.array(Complex(3.0, 0.3))).item
    assertEquals(q, Complex(0.36303630363036304, 0.297029702970297))
  }

  test("regression: 0 ** b is 0 for Re(b) > 0, same kernel for ** and np.power") {
    val z = np.array(Complex(0.0, 0.0), Complex(0.0, 0.0))
    val b = np.array(Complex(3.0, 5.0), Complex(-1.0, -9.0))
    for r <- Seq(z ** b, np.power(z, b)) do
      assertEquals(r.item(0), Complex(0.0, 0.0))
      assert(r.item(1).isNaN)
  }

  test("regression: ravel copies non-contiguous input like NumPy") {
    val x = np.arange(5.0)("::-2")
    val r = x.ravel()
    assert(!(r.data eq x.data))
    assertEquals(r.toList, List(4.0, 2.0, 0.0))
    val c = np.arange(6.0)
    assert(c.ravel().data eq c.data)
  }

  test("regression: reductions without identity raise on an empty axis even for empty results") {
    val e = np.zeros(2, 0, 0)
    intercept[IllegalArgumentException](np.min(e, 2))
    intercept[IllegalArgumentException](np.nanmax(e, 1))
    intercept[IllegalArgumentException](np.argmin(e, 2))
    assertEquals(np.min(e, 0).shape, Seq(0, 0))
    assertEquals(np.sum(e, 2).shape, Seq(2, 0))
  }

  test("regression: scientific notation uses the exact digits beyond the shortest repr") {
    val a = np.arrayOf(Seq(8.5051408e-07, 1.2926322e-05), DType.Float32)
    assertEquals(a.toString, "[8.5051408e-07 1.2926322e-05]")
  }

  test("regression: repr line width, shape= for summarized arrays, float32 exp cutoff") {
    assertEquals(np.arange(1001).astype[UInt8].repr,
      "array([  0,   1,   2, ..., 230, 231, 232], shape=(1001,), dtype=uint8)")
    val f = np.arrayOf(Seq(0.12345670163631439, 1.2234567403793335, 2.3234567642211914, 3.423456907272339,
      4.523456573486328), DType.Float32)
    assertEquals(f.repr, "array([0.1234567, 1.2234567, 2.3234568, 3.423457 , 4.5234566],\n      dtype=float32)")
    assertEquals(np.arrayOf(Seq(1e6, 2.5), DType.Float32).repr, "array([1.0e+06, 2.5e+00], dtype=float32)")
    assertEquals(NDArray.scalar(-86919810.0f).toString, "-8.691981e+07")
    assertEquals(NDArray.scalar(1e6f).toString, "1e+06")
    assertEquals(NDArray.scalar(999999.0f).toString, "999999.0")
  }
