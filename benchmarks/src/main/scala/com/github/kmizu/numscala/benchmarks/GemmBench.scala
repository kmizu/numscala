package com.github.kmizu.numscala.benchmarks

import com.github.kmizu.numscala.*
import com.github.kmizu.numscala.cpu.*
import com.github.kmizu.numscala.cpu.vector25.VectorF32Kernels
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole
import java.util.concurrent.TimeUnit
import scala.compiletime.uninitialized

/** GEMM shapes of NS-CPU-001 §9.1. `shape` is `MxNxK-XY` with X/Y = N (as stored) or T (transposed).
  * FLOPs per op = 2 * m * n * k (an FMA counts as 2).
  */
@State(Scope.Thread)
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Fork(value = 3, jvmArgsAppend = Array("--add-modules=jdk.incubator.vector"))
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 8, time = 1)
class GemmBench:
  @Param(Array(
    "1x384x384-NN", "1x768x384-NN", "1x128x384-NN", "1x192x384-NN",   // 1-token inference
    "8x384x384-NN", "32x768x384-NN", "128x768x384-NN", "512x384x384-NN", // small-batch projections
    "32x384x768-NT",                                                     // input-side gradient [B,768] @ [384,768]^T
    "384x768x32-TN",                                                     // weight gradient [B,384]^T @ [B,768]
    "8x256x128-NT",                                                      // candidate scores
    "7x385x383-NN"                                                       // tails
  ))
  var shape: String = uninitialized

  private var a: MatrixF32 = uninitialized
  private var b: MatrixF32 = uninitialized
  private var c: MatrixF32 = uninitialized
  private var ta: Transpose = uninitialized
  private var tb: Transpose = uninitialized
  private var ndA: NDArray[Float] = uninitialized
  private var ndB: NDArray[Float] = uninitialized
  private val ws = new Workspace()

  @Setup
  def setup(): Unit =
    val Array(dims, t) = shape.split("-")
    val Array(m, n, k) = dims.split("x")
    val (mm, nn, kk) = (m.toInt, n.toInt, k.toInt)
    ta = if t(0) == 'T' then Transpose.Yes else Transpose.No
    tb = if t(1) == 'T' then Transpose.Yes else Transpose.No
    val rnd = new java.util.Random(1)
    def mat(r: Int, cc: Int) = MatrixF32.wrap(Array.fill(r * cc)(rnd.nextFloat() - 0.5f), r, cc)
    a = if ta == Transpose.No then mat(mm, kk) else mat(kk, mm)
    b = if tb == Transpose.No then mat(kk, nn) else mat(nn, kk)
    c = MatrixF32.zeros(mm, nn)
    def nd(x: MatrixF32, t: Transpose) =
      val v = NDArray.fromArray(x.data, Array(x.rows, x.cols))
      if t == Transpose.No then v else v.T
    ndA = nd(a, ta)
    ndB = nd(b, tb)
    ws.reserveFloats(math.max(ScalarF32Kernels.gemmWorkspaceFloats(mm, nn, kk, ta, tb), VectorF32Kernels.gemmWorkspaceFloats(mm, nn, kk, ta, tb)))

  /** The numscala 0.2.0 float32 matmul: contiguous + toArray copies, then boxed generic loops. */
  @Benchmark
  def legacyNpMatmul(bh: Blackhole): Unit = bh.consume(LegacyMatmul.matmul(ndA, ndB))

  /** `np.matmul` on float32 (adapter + default scalar backend, fresh output each call). */
  @Benchmark
  def npMatmul(bh: Blackhole): Unit = bh.consume(np.matmul(ndA, ndB))

  @Benchmark
  def scalarGemmInto(bh: Blackhole): Unit =
    ScalarF32Kernels.gemmInto(a, ta, b, tb, c, ws)
    bh.consume(c.data)

  @Benchmark
  def vectorGemmInto(bh: Blackhole): Unit =
    VectorF32Kernels.gemmInto(a, ta, b, tb, c, ws)
    bh.consume(c.data)

/** A copy of the pre-NS-CPU-001 float32 matmul path, kept as the benchmark baseline (N0). */
object LegacyMatmul:
  def matmul(a: NDArray[Float], b: NDArray[Float]): NDArray[Float] =
    val d = DType.Float32
    val m = a.shape(0); val k = a.shape(1); val n = b.shape(1)
    val ad = a.contiguous.toArray
    val bd = b.contiguous.toArray
    val out = new Array[Float](m * n)
    var i = 0
    while i < m do
      var j = 0
      while j < n do
        var acc = d.zero
        var p = 0
        while p < k do
          acc = d.plus(acc, d.times(ad(i * k + p), bd(p * n + j)))
          p += 1
        out(i * n + j) = acc
        j += 1
      i += 1
    NDArray.fromArray(out, Array(m, n))
