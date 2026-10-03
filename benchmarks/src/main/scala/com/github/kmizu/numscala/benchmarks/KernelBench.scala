package com.github.kmizu.numscala.benchmarks

import com.github.kmizu.numscala.cpu.*
import com.github.kmizu.numscala.cpu.vector25.VectorF32Kernels
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole
import java.util.concurrent.TimeUnit
import scala.compiletime.uninitialized

/** Sparse rows, scan and reductions (NS-CPU-001 §9.1: random gather, duplicate-heavy scatter, scan). */
@State(Scope.Thread)
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Fork(value = 3, jvmArgsAppend = Array("--add-modules=jdk.incubator.vector"))
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 8, time = 1)
class KernelBench:
  @Param(Array("scalar", "vector25"))
  var backend: String = uninitialized

  private var k: F32Kernels = uninitialized
  private val rnd = new java.util.Random(7)
  private val dim = 384
  private val table = MatrixF32.wrap(Array.fill(32768 * dim)(rnd.nextFloat()), 32768, dim)
  private val randomIds = Array.fill(512)(rnd.nextInt(32768))
  private val dupIds = Array.fill(512)(rnd.nextInt(16))
  private val rows = MatrixF32.zeros(512, dim)
  private val dst = MatrixF32.zeros(32768, dim)
  private val outIds = new Array[Int](512)
  private val outRows = MatrixF32.zeros(512, dim)
  private val ws = new Workspace(0, 512)
  // scan [T=64, B=8, D=384]
  private val sa = MatrixF32.wrap(Array.fill(64 * 8 * dim)(rnd.nextFloat()), 64 * 8, dim)
  private val sb = MatrixF32.wrap(Array.fill(64 * 8 * dim)(rnd.nextFloat() - 0.5f), 64 * 8, dim)
  private val s0 = MatrixF32.zeros(8, dim)
  private val sOut = MatrixF32.zeros(64 * 8, dim)
  // log-sum-exp over 32 rows of 8192 logits
  private val logits = MatrixF32.wrap(Array.fill(32 * 8192)(rnd.nextFloat() * 10f), 32, 8192)
  private val lse = new Array[Float](32)

  @Setup
  def setup(): Unit = k = if backend == "scalar" then ScalarF32Kernels else VectorF32Kernels

  @Benchmark
  def gatherRandom512(bh: Blackhole): Unit =
    k.gatherRowsInto(table, randomIds, rows)
    bh.consume(rows.data)

  @Benchmark
  def scatterAddDuplicates512(bh: Blackhole): Unit =
    k.scatterAddRowsInto(dst, dupIds, rows)
    bh.consume(dst.data)

  @Benchmark
  def coalesceThenScatter512(bh: Blackhole): Unit =
    val u = k.coalesceRowsInto(dupIds, rows, outIds, outRows, ws)
    k.scatterAddRowsInto(dst, outIds, 0, u, outRows.rowRange(0, u))
    bh.consume(dst.data)

  @Benchmark
  def affineScan64x8x384(bh: Blackhole): Unit =
    k.affineScanInto(sa, sb, s0, sOut)
    bh.consume(sOut.data)

  @Benchmark
  def rowLogSumExp32x8192(bh: Blackhole): Unit =
    k.rowLogSumExpInto(logits, lse, 0)
    bh.consume(lse)

  @Benchmark
  def axpy512x384(bh: Blackhole): Unit =
    k.axpyInto(0.001f, rows, outRows)
    bh.consume(outRows.data)
