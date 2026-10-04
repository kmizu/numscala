package com.github.kmizu.numscala.cpu

import java.util.concurrent.{Callable, ExecutionException, ExecutorService, Future}

/** Parallel GEMM on a caller-owned executor (NS-CPU-001 §7.3).
  *
  * numscala never creates threads: the caller passes its own `ExecutorService` and one [[Workspace]]
  * per task. The rows of C are split into contiguous chunks (multiples of 4 rows) computed with
  * [[F32Kernels.gemmRowsInto]]. Every element of C therefore gets exactly the bits of a single-threaded
  * `gemmInto` on the same backend, whatever the number of tasks. Do not use this from inside a task of
  * the same bounded executor (the caller waits for the tasks, which could then starve).
  */
object ParallelF32:

  /** Splits `m` rows into at most `tasks` contiguous chunks whose boundaries are multiples of 4; returns `tasks + 1` bounds. */
  def rowBounds(m: Int, tasks: Int): Array[Int] =
    if tasks <= 0 then throw new IllegalArgumentException(s"rowBounds: tasks must be positive, got $tasks")
    val blocks = (m + 3) / 4
    Array.tabulate(tasks + 1)(t => math.min(m, ((blocks.toLong * t / tasks).toInt) * 4))

  /** `C := alpha * op(A) * op(B) + beta * C`, rows of C split over `workspaces.length` tasks.
    *
    * Task 0 runs on the calling thread, the others on `executor`. Arguments and every workspace's capacity
    * are checked before any task starts. The call returns (or rethrows the first failure) only after all
    * tasks have finished. Each workspace needs `kernels.gemmWorkspaceFloats(...)` of its chunk; for the
    * `A^T B^T` layout that is `k * n` floats each.
    */
  def gemmInto(
      kernels: F32Kernels, executor: ExecutorService, workspaces: IndexedSeq[Workspace],
      a: MatrixF32, transA: Transpose, b: MatrixF32, transB: Transpose, c: MatrixF32,
      alpha: Float, beta: Float
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
    val bounds = rowBounds(m, workspaces.length)
    val tasks = workspaces.length
    // capacity check for every chunk before anything is written
    if k != 0 && alpha != 0f then
      var t = 0
      while t < tasks do
        val rows = bounds(t + 1) - bounds(t)
        if rows > 0 then
          val need = kernels.gemmWorkspaceFloats(rows, n, k, transA, transB)
          workspaces(t).require(need, 0, s"ParallelF32.gemmInto (task $t)")
        t += 1
    def chunk(t: Int): Unit =
      kernels.gemmRowsInto(a, transA, b, transB, c, alpha, beta, bounds(t), bounds(t + 1), workspaces(t))
    val futures = new Array[Future[?]](tasks)
    var first: Throwable | Null = null
    var t = 1
    try
      while t < tasks do
        if bounds(t + 1) > bounds(t) then
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
