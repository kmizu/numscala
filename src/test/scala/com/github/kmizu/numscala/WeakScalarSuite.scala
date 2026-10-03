package com.github.kmizu.numscala

/** NEP 50 ("weak" Python scalars): an operator between an array and a Scala `Boolean`, `Int`,
  * `Long`, `Double` or `Complex` keeps the array's dtype whenever the scalar's kind fits in it.
  * Expected values are what NumPy 2.4 returns.
  */
class WeakScalarSuite extends munit.FunSuite:
  private val f32 = np.array(0.5f, 1.5f)
  private val i8 = np.arrayOf(Seq(1, 2), DType.Int8)
  private val u8 = np.arrayOf(Seq(1, 2), DType.UInt8)
  private val i32 = np.array(1, 2)
  private val bools = np.array(true, false)

  test("float32 array with a Double scalar stays float32"):
    val r = f32 + 2.0
    assertEquals(r.dtype, DType.Float32)
    assertEquals(r.toSeq, Seq(2.5f, 3.5f))
    val d: Double = 2.0
    assertEquals((f32 * d).dtype, DType.Float32)
    assertEquals((f32 - d).toSeq, Seq(-1.5f, -0.5f))
    assertEquals((f32 / d).dtype, DType.Float32)
    assertEquals((f32 ** 2.0).toSeq, Seq(0.25f, 2.25f))

  test("small int arrays with an Int scalar keep their dtype"):
    val r = i8 + 1
    assertEquals(r.dtype, DType.Int8)
    assertEquals(r.toSeq, Seq[Byte](2, 3))
    val x: Int = 3
    assertEquals((i8 * x).dtype, DType.Int8)
    assertEquals((u8 + x).dtype, DType.UInt8)
    val l: Long = 4L
    assertEquals((i8 - l).toSeq, Seq[Byte](-3, -2))
    assertEquals((i8 ** 2).dtype, DType.Int8)
    assertEquals((i8 `//` 2).dtype, DType.Int8)
    assertEquals((i8 % 2).toSeq, Seq[Byte](1, 0))

  test("an int scalar out of range for the array dtype overflows like NumPy"):
    val e = intercept[ArithmeticException](i8 + 300)
    assertEquals(e.getMessage, "Python integer 300 out of bounds for int8")
    val e2 = intercept[ArithmeticException](u8 + (-1))
    assertEquals(e2.getMessage, "Python integer -1 out of bounds for uint8")

  test("int arrays with a Double scalar become float64"):
    val r = i32 + 2.5
    assertEquals(r.dtype, DType.Float64)
    assertEquals(r.toSeq, Seq(3.5, 4.5))
    assertEquals((i8 * 1.5).dtype, DType.Float64)
    assertEquals((i32 % 2.5).toSeq, Seq(1.0, 2.0))

  test("true division by an int scalar"):
    assertEquals((i32 / 2).dtype, DType.Float64)
    assertEquals((i8 / 2).toSeq, Seq(0.5, 1.0))
    assertEquals((f32 / 2).dtype, DType.Float32)

  test("Int scalar with a float32 array stays float32"):
    val x: Int = 2
    assertEquals((f32 + x).dtype, DType.Float32)

  test("complex scalars make complex arrays"):
    val r = f32 + Complex(0, 1)
    assertEquals(r.dtype, DType.Complex128)
    assertEquals(r.toSeq.head, Complex(0.5, 1))
    val c = np.array(Complex(1, 1))
    assertEquals((c * 2.0).toSeq, Seq(Complex(2, 2)))

  test("bool arrays with numeric scalars"):
    assertEquals((bools + 1).dtype, DType.Int32)
    assertEquals((bools + 1L).dtype, DType.Int64)
    assertEquals((bools + 1.5).toSeq, Seq(2.5, 1.5))
    assertEquals((bools === true).toSeq, Seq(true, false))
    assertEquals((bools & true).toSeq, Seq(true, false))

  test("strong scalars (Float, Byte) follow normal promotion"):
    val i16 = np.arrayOf(Seq(1), DType.Int16)
    assertEquals((i16 + 1.0f).dtype, DType.Float32)
    assertEquals((i32 + 1.0f).dtype, DType.Float64)
    assertEquals((i8 + 1.toByte).dtype, DType.Int8)

  test("comparisons use the array dtype and never overflow"):
    val f = np.array(0.1f)
    assertEquals((f === 0.1).toSeq, Seq(true))
    assertEquals((f < 0.1).toSeq, Seq(false))
    assertEquals((i8 < 300).toSeq, Seq(true, true))
    assertEquals((i8 > 300).toSeq, Seq(false, false))
    assertEquals((u8 === -1).toSeq, Seq(false, false))
    assertEquals((u8 =!= -1).toSeq, Seq(true, true))
    assertEquals((u8 >= -1).toSeq, Seq(true, true))
    assertEquals((i32 >= 2.5).toSeq, Seq(false, false))

  test("bitwise operators with scalars"):
    assertEquals((i8 & 1).toSeq, Seq[Byte](1, 0))
    assertEquals((i8 | 4).dtype, DType.Int8)
    assertEquals((i8 << 2).toSeq, Seq[Byte](4, 8))
    assertEquals((u8 >> 1).toSeq.map(_.toInt), Seq(0, 1))

  test("in-place operators accept scalars that do not change the dtype"):
    val a = np.array(1.0f, 2.0f)
    a += 1.0
    a *= 2
    a /= 4.0
    assertEquals(a.toSeq, Seq(1.0f, 1.5f))
    val b = np.arrayOf(Seq(10, 20), DType.Int8)
    b -= 1
    b `//=` 3
    assertEquals(b.toSeq, Seq[Byte](3, 6))
    assert(compileErrors("val i = np.array(1, 2); i += 2.5").nonEmpty)

  test("scalar on the left"):
    assertEquals((2.0 * f32).dtype, DType.Float32)
    assertEquals((2.0 - f32).toSeq, Seq(1.5f, 0.5f))
    assertEquals((2.5 + i32).dtype, DType.Float64)
    assertEquals((3 * i8).dtype, DType.Int8)
    assertEquals((1 / i32).toSeq, Seq(1.0, 0.5))
    assertEquals((Complex(0, 1) * f32).dtype, DType.Complex128)
    intercept[ArithmeticException](300 + i8)

  test("generic code over an abstract dtype still compiles"):
    def twice[T: NumDType](a: NDArray[T], x: T): NDArray[T] = a + x + x
    assertEquals(twice(i32, 1).toSeq, Seq(3, 4))

  test("ufuncs called with a scalar follow the same rules"):
    assertEquals(np.add(f32, 2.0).dtype, DType.Float32)
    assertEquals(np.multiply(2.0, f32).toSeq, Seq(1.0f, 3.0f))
    assertEquals(np.add(i8, 1).dtype, DType.Int8)
    assertEquals(np.add(i32, 2.5).dtype, DType.Float64)
    assertEquals(np.divide(i8, 2).toSeq, Seq(0.5, 1.0))
    intercept[ArithmeticException](np.add(i8, 300))
    assertEquals(np.less(i8, 300).toSeq, Seq(true, true))
    assertEquals(np.greater(-1, u8).toSeq, Seq(false, false))
    assertEquals(np.equal(np.array("a", "b"), "a").toSeq, Seq(true, false))
