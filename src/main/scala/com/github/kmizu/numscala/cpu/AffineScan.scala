package com.github.kmizu.numscala.cpu

/** Elementwise affine scan `s(t) = a(t) * s(t-1) + b(t)` (K4). */
object AffineScan:

  /** Composition of two steps: applying `(a1, b1)` first and then the later `(a2, b2)` is the single step
    * `(a2 * a1, a2 * b1 + b2)`. Returned as `(a, b)`; a parallel scan must respect this order.
    */
  def combine(a1: Float, b1: Float, a2: Float, b2: Float): (Float, Float) = (a2 * a1, a2 * b1 + b2)

  /** Sequential reference scan on validated arguments (see [[F32Kernels.affineScanInto]]). */
  private[cpu] def scan(a: MatrixF32, b: MatrixF32, initial: MatrixF32, out: MatrixF32, steps: Int, batch: Int): Unit =
    val d = a.cols
    val ad = a.data; val bd = b.data; val od = out.data; val id = initial.data
    var t = 0
    while t < steps do
      var j = 0
      while j < batch do
        val row = t * batch + j
        val ai = a.offset + row * a.rowStride
        val bi = b.offset + row * b.rowStride
        val oi = out.offset + row * out.rowStride
        // previous state: the initial row at t == 0, else out's row from step t-1
        val pd = if t == 0 then id else od
        val pi = if t == 0 then initial.offset + j * initial.rowStride else out.offset + (row - batch) * out.rowStride
        var f = 0
        while f < d do
          od(oi + f) = ad(ai + f) * pd(pi + f) + bd(bi + f)
          f += 1
        j += 1
      t += 1
