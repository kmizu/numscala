// Hashes the exact output bits of gemm (all layouts, both backends, warm and cold JIT), gather and scan.
import com.github.kmizu.numscala.cpu.*
import com.github.kmizu.numscala.cpu.vector25.VectorF32Kernels
@main def run(): Unit =
  val rnd = new java.util.Random(99)
  def mat(r: Int, c: Int) = MatrixF32.wrap(Array.fill(r * c)(rnd.nextFloat() - 0.5f), r, c)
  var h = 17L
  def mix(a: Array[Float]) = a.foreach(x => h = h * 31 + java.lang.Float.floatToRawIntBits(x))
  val T = Transpose
  for _ <- 0 until 2 do // second pass after JIT warm-up
    for (m, n, k) <- Seq((1, 384, 384), (7, 385, 383), (8, 384, 384), (13, 40, 77), (64, 384, 384), (128, 768, 384), (300, 50, 33));
        (ta, tb) <- Seq(T.No -> T.No, T.No -> T.Yes, T.Yes -> T.No, T.Yes -> T.Yes);
        kern <- Seq[F32Kernels](ScalarF32Kernels, VectorF32Kernels) do
      val a = if ta == T.No then mat(m, k) else mat(k, m)
      val b = if tb == T.No then mat(k, n) else mat(n, k)
      val c = mat(m, n)
      kern.gemmInto(a, ta, b, tb, c, 0.75f, 0.5f, new Workspace(k * n))
      mix(c.toArray)
    for kern <- Seq[F32Kernels](ScalarF32Kernels, VectorF32Kernels) do
      val table = mat(100, 385); val out = MatrixF32.zeros(50, 385)
      kern.gatherRowsInto(table, Array.fill(50)(rnd.nextInt(100)), out); mix(out.toArray)
      val sa = mat(24, 385); val sb = mat(24, 385); val s0 = mat(4, 385); val so = MatrixF32.zeros(24, 385)
      kern.affineScanInto(sa, sb, s0, so); mix(so.toArray)
  println(f"hash=$h%x")
