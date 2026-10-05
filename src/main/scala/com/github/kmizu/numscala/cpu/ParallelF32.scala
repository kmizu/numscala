package com.github.kmizu.numscala.cpu

import java.util.concurrent.{Callable, ExecutionException, ExecutorService, Future}

/** Parallel GEMM on a caller-owned executor (NS-CPU-001 §7.3).
  *
  * numscala never creates threads: the caller passes its own `ExecutorService` and one [[Workspace]]
  * per task. C is cut into a grid of tiles (row bounds on multiples of 4, column bounds on multiples of
  * 16, see [[tileGrid]]) computed with [[F32Kernels.gemmTileInto]], and task `t` computes tiles
  * `t, t + tasks, ...`. Every element of C therefore gets exactly the bits of a single-threaded `gemmInto`
  * on the same backend, whatever the number of tasks or the grid. Do not use this from inside a task of
  * the same bounded executor (the caller waits for the tasks, which could then starve).
  */
object ParallelF32:

  /** Splits `m` rows into at most `tasks` contiguous chunks whose boundaries are multiples of 4; returns `tasks + 1` bounds. */
  def rowBounds(m: Int, tasks: Int): Array[Int] =
    if tasks <= 0 then throw new IllegalArgumentException(s"rowBounds: tasks must be positive, got $tasks")
    val blocks = (m + 3) / 4
    Array.tabulate(tasks + 1)(t => math.min(m, ((blocks.toLong * t / tasks).toInt) * 4))

  /** Splits `n` columns into `parts` contiguous ranges whose boundaries are multiples of [[ColAlign]]; `parts + 1` bounds. */
  def colBounds(n: Int, parts: Int): Array[Int] =
    if parts <= 0 then throw new IllegalArgumentException(s"colBounds: parts must be positive, got $parts")
    val blocks = (n + ColAlign - 1) / ColAlign
    Array.tabulate(parts + 1)(t => math.min(n, ((blocks.toLong * t / parts).toInt) * ColAlign))

  /** Column alignment of tiles (two 256-bit vectors of floats). */
  final val ColAlign = 16

  /** Default minimum product size (`m * n * k` multiply-adds) worth parallelising. Handing work to pool threads
    * costs a roughly fixed 60–90 us per call (measured with a `newFixedThreadPool` on a Ryzen 9 5900X / WSL2),
    * about the time vector25 needs for 4M multiply-adds; smaller products run on the calling thread only.
    */
  final val DefaultMinWork: Long = 1L << 22

  /** The `(rowParts, colParts)` grid used for an `m x n` (inner dimension `k`) product on `tasks` workers.
    *
    * Minimises the critical path (rounds of tiles per worker times the largest tile), then memory traffic
    * (`rowParts` re-reads of B plus `colParts` re-reads of A). A single row block (e.g. a GEMV) is split by
    * columns only. Large `m` with few tasks is split by rows only, as in 0.4.0.
    */
  def tileGrid(m: Int, n: Int, k: Int, tasks: Int): (Int, Int) =
    if tasks <= 0 then throw new IllegalArgumentException(s"tileGrid: tasks must be positive, got $tasks")
    val rowBlocks = math.max(1, (m + 3) / 4)
    val colBlocks = math.max(1, (n + ColAlign - 1) / ColAlign)
    var best = (1, 1)
    var bestWork = Long.MaxValue
    var bestTraffic = Long.MaxValue
    var pr = 1
    while pr <= math.min(tasks, rowBlocks) do
      val pc = math.min(colBlocks, (tasks + pr - 1) / pr)
      val rounds = ((pr.toLong * pc) + tasks - 1) / tasks
      val tileRows = ((rowBlocks + pr - 1) / pr).toLong * 4
      val tileCols = ((colBlocks + pc - 1) / pc).toLong * ColAlign
      val work = rounds * tileRows * tileCols
      val traffic = pr.toLong * n + pc.toLong * m
      if work < bestWork || (work == bestWork && traffic < bestTraffic) then
        best = (pr, pc); bestWork = work; bestTraffic = traffic
      pr += 1
    best

  /** `C := alpha * op(A) * op(B) + beta * C`, tiles of C (see [[tileGrid]]) spread over `workspaces.length` tasks.
    *
    * Task 0 runs on the calling thread, the others on `executor`. Arguments and every workspace's capacity
    * are checked before any task starts. The call returns (or rethrows the first failure) only after all
    * tasks have finished. Each workspace needs `kernels.gemmWorkspaceFloats(...)` of its tiles; for the
    * `A^T B^T` layout that is `k * n` floats each (enough for any tile). Products with fewer than `minWork`
    * multiply-adds are computed on the calling thread with `workspaces(0)` and never touch the executor; the
    * bits are the same either way.
    */
  def gemmInto(
      kernels: F32Kernels, executor: ExecutorService, workspaces: IndexedSeq[Workspace],
      a: MatrixF32, transA: Transpose, b: MatrixF32, transB: Transpose, c: MatrixF32,
      alpha: Float, beta: Float, minWork: Long = DefaultMinWork
  ): Unit =
    kernels.checkGemm(a, transA, b, transB, c)
    if workspaces.isEmpty then throw new IllegalArgumentException("ParallelF32.gemmInto: no workspaces")
    if executor == null then throw new IllegalArgumentException("ParallelF32.gemmInto: executor is null")
    var i = 0
    while i < workspaces.length do
      var j = 0
      while j < i do
        if workspaces(i) eq workspaces(j) then
          throw new IllegalArgumentException(s"ParallelF32.gemmInto: workspaces $j and $i are the same object")
        j += 1
      i += 1
    val m = c.rows; val n = c.cols; val k = a.logicalCols(transA)
    val tasks = if m.toLong * n * k < minWork then 1 else workspaces.length
    val (pr, pc) = tileGrid(m, n, k, tasks)
    val rb = rowBounds(m, pr)
    val cb = colBounds(n, pc)
    val tiles = pr * pc
    // capacity check for every task's largest tile before anything is written
    if k != 0 && alpha != 0f then
      var t = 0
      while t < tasks do
        var need = 0
        var q = t
        while q < tiles do
          val rows = rb(q / pc + 1) - rb(q / pc)
          val cols = cb(q % pc + 1) - cb(q % pc)
          if rows > 0 && cols > 0 then need = math.max(need, kernels.gemmWorkspaceFloats(rows, cols, k, transA, transB))
          q += tasks
        workspaces(t).require(need, 0, s"ParallelF32.gemmInto (task $t)")
        t += 1
    def chunk(t: Int): Unit =
      var q = t
      while q < tiles do
        val r = q / pc; val cc = q % pc
        kernels.gemmTileInto(a, transA, b, transB, c, alpha, beta, rb(r), rb(r + 1), cb(cc), cb(cc + 1), workspaces(t))
        q += tasks
    val futures = new Array[Future[?]](tasks)
    var first: Throwable | Null = null
    var t = 1
    try
      while t < tasks do
        if t < tiles then
          val task = t
          futures(t) = executor.submit((() => { chunk(task); null }): Callable[Null])
        t += 1
    catch case e: Throwable => first = e // e.g. RejectedExecutionException: still wait for what was submitted
    if first == null then
      try chunk(0)
      catch case e: Throwable => first = e
    // wait for every submitted task, even when interrupted: none may still write C after we return
    var interrupted = false
    t = 1
    while t < tasks do
      if futures(t) != null then
        var done = false
        while !done do
          try
            futures(t).get()
            done = true
          catch
            case e: ExecutionException =>
              if first == null then first = e.getCause
              done = true
            case _: InterruptedException => interrupted = true
      t += 1
    if interrupted then Thread.currentThread().interrupt()
    if first != null then throw first.nn
