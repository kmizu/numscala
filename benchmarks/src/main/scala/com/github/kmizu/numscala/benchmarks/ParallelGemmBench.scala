package com.github.kmizu.numscala.benchmarks

import com.github.kmizu.numscala.cpu.*
import com.github.kmizu.numscala.cpu.vector25.VectorF32Kernels
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole
import java.util.concurrent.{ExecutorService, Executors, TimeUnit}
import scala.compiletime.uninitialized

/** Worker-count scaling of [[ParallelF32.gemmInto]] (NS-CPU-001 §7.3: compare 1/6/12/24 workers).
  * One caller-owned fixed pool of `workers - 1` threads plus the calling thread; one Workspace per task.
  */
@State(Scope.Benchmark)
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Fork(value = 3, jvmArgsAppend = Array("--add-modules=jdk.incubator.vector"))
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 8, time = 1)
class ParallelGemmBench:
  @Param(Array("1", "6", "12", "24"))
  var workers: Int = uninitialized

  @Param(Array("128x768x384-NN", "512x384x384-NN", "512x384x768-NT", "384x768x512-TN", "64x384x384-NT", "1x768x384-NN", "1x16384x384-NN"))
  var shape: String = uninitialized

  @Param(Array("vector25"))
  var backend: String = uninitialized

  private var pool: ExecutorService = uninitialized
  private var workspaces: IndexedSeq[Workspace] = uninitialized
  private var kernels: F32Kernels = uninitialized
  private var a: MatrixF32 = uninitialized
  private var b: MatrixF32 = uninitialized
  private var c: MatrixF32 = uninitialized
  private var ta: Transpose = uninitialized
  private var tb: Transpose = uninitialized

  @Setup
  def setup(): Unit =
    val Array(dims, t) = shape.split("-")
    val Array(m, n, k) = dims.split("x").map(_.toInt)
    ta = if t(0) == 'T' then Transpose.Yes else Transpose.No
    tb = if t(1) == 'T' then Transpose.Yes else Transpose.No
    val rnd = new java.util.Random(1)
    def mat(r: Int, cc: Int) = MatrixF32.wrap(Array.fill(r * cc)(rnd.nextFloat() - 0.5f), r, cc)
    a = if ta == Transpose.No then mat(m, k) else mat(k, m)
    b = if tb == Transpose.No then mat(k, n) else mat(n, k)
    c = MatrixF32.zeros(m, n)
    kernels = if backend == "scalar" then ScalarF32Kernels else VectorF32Kernels
    pool = Executors.newFixedThreadPool(math.max(workers - 1, 1))
    workspaces = IndexedSeq.fill(workers)(new Workspace(k * n))

  @TearDown
  def tearDown(): Unit = pool.shutdown()

  @Benchmark
  def parallelGemmInto(bh: Blackhole): Unit =
    ParallelF32.gemmInto(kernels, pool, workspaces, a, ta, b, tb, c, 1f, 0f)
    bh.consume(c.data)
