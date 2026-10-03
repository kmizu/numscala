package com.github.kmizu.numscala.cpu

import java.util.Random

/** A deliberately tiny "model side" written only against [[F32Kernels]] (NS-CPU-001 §8.1, integration).
  *
  * Embedding gather, two projections, a sigmoid gate and SiLU input feeding an affine scan over time,
  * an output projection and a mean softmax cross-entropy. Forward and backward (including the reverse
  * scan) are model code; numscala only supplies the kernels. Layout is `[time, batch, feature]`.
  */
final class TinyModel(k: F32Kernels, val vocab: Int, val dim: Int, val hidden: Int, val steps: Int, val batch: Int):
  private val N = steps * batch
  private val T = Transpose

  final class Params(val emb: MatrixF32, val w1: MatrixF32, val w2: MatrixF32, val wo: MatrixF32):
    def all: Seq[MatrixF32] = Seq(emb, w1, w2, wo)

  def init(seed: Long): Params =
    val rnd = new Random(seed)
    def m(r: Int, c: Int, s: Float) = MatrixF32.wrap(Array.fill(r * c)((rnd.nextFloat() * 2f - 1f) * s), r, c)
    Params(m(vocab, dim, 1f), m(dim, hidden, 0.5f), m(dim, hidden, 0.5f), m(vocab, hidden, 0.5f))

  def zerosLike(p: Params): Params =
    Params(MatrixF32.zeros(vocab, dim), MatrixF32.zeros(dim, hidden), MatrixF32.zeros(dim, hidden), MatrixF32.zeros(vocab, hidden))

  private val ws = new Workspace(0, N)
  private val x = MatrixF32.zeros(N, dim)
  private val z1 = MatrixF32.zeros(N, hidden)
  private val z2 = MatrixF32.zeros(N, hidden)
  private val gate = MatrixF32.zeros(N, hidden)
  private val inp = MatrixF32.zeros(N, hidden)
  private val s = MatrixF32.zeros(N, hidden)
  private val logits = MatrixF32.zeros(N, vocab)
  private val lse = new Array[Float](N)
  private val s0 = MatrixF32.zeros(batch, hidden)

  /** Mean cross-entropy of predicting `targets` (one per row) from `ids`. */
  def loss(p: Params, ids: Array[Int], targets: Array[Int]): Float =
    forward(p, ids)
    var l = 0.0
    for r <- 0 until N do l += lse(r) - logits(r, targets(r))
    (l / N).toFloat

  private def forward(p: Params, ids: Array[Int]): Unit =
    k.gatherRowsInto(p.emb, ids, x)
    k.gemmInto(x, T.No, p.w1, T.No, z1, ws)
    k.gemmInto(x, T.No, p.w2, T.No, z2, ws)
    k.sigmoidInto(z1, gate)
    k.siluInto(z2, inp)
    k.affineScanInto(gate, inp, s0, s)
    k.gemmInto(s, T.No, p.wo, T.Yes, logits, ws)
    k.rowLogSumExpInto(logits, lse, 0)

  /** Loss and gradients (written into `g`, overwritten). */
  def lossAndGrads(p: Params, ids: Array[Int], targets: Array[Int], g: Params): Float =
    val l = loss(p, ids, targets)
    // dlogits = (softmax - onehot) / N
    val dLogits = MatrixF32.zeros(N, vocab)
    for r <- 0 until N; v <- 0 until vocab do
      dLogits(r, v) = ((math.exp(logits(r, v) - lse(r)) - (if v == targets(r) then 1.0 else 0.0)) / N).toFloat
    val dS = MatrixF32.zeros(N, hidden)
    k.gemmInto(dLogits, T.No, p.wo, T.No, dS, ws)
    k.gemmInto(dLogits, T.Yes, s, T.No, g.wo, ws)
    // reverse scan: gS(t) = dS(t) + a(t+1) * gS(t+1), run as a forward scan on time-reversed blocks
    val aRev = MatrixF32.zeros(N, hidden)
    val dRev = MatrixF32.zeros(N, hidden)
    for t <- 0 until steps do
      val dst = (steps - 1 - t) * batch
      k.copyInto(dS.rowRange(t * batch, (t + 1) * batch), dRev.rowRange(dst, dst + batch))
      if t + 1 < steps then k.copyInto(gate.rowRange((t + 1) * batch, (t + 2) * batch), aRev.rowRange(dst, dst + batch))
    val gRev = MatrixF32.zeros(N, hidden)
    k.affineScanInto(aRev, dRev, MatrixF32.zeros(batch, hidden), gRev)
    val gS = MatrixF32.zeros(N, hidden)
    for t <- 0 until steps do
      val src = (steps - 1 - t) * batch
      k.copyInto(gRev.rowRange(src, src + batch), gS.rowRange(t * batch, (t + 1) * batch))
    // d gate = gS * s(t-1); d inp = gS
    val sPrev = MatrixF32.zeros(N, hidden)
    if steps > 1 then k.copyInto(s.rowRange(0, N - batch), sPrev.rowRange(batch, N))
    val dGate = MatrixF32.zeros(N, hidden)
    k.mulInto(gS, sPrev, dGate)
    // through sigmoid: dz1 = dGate * gate * (1 - gate)
    val oneMinus = MatrixF32.zeros(N, hidden)
    k.fillInto(oneMinus, 1f)
    k.axpyInto(-1f, gate, oneMinus)
    val dz1 = MatrixF32.zeros(N, hidden)
    k.mulInto(dGate, gate, dz1)
    k.mulInto(dz1, oneMinus, dz1)
    // through SiLU: silu'(z) = sig(z) * (1 + z * (1 - sig(z)))
    val dz2 = MatrixF32.zeros(N, hidden)
    for r <- 0 until N; h <- 0 until hidden do
      val z = z2(r, h).toDouble
      val sg = 1 / (1 + math.exp(-z))
      dz2(r, h) = (gS(r, h) * sg * (1 + z * (1 - sg))).toFloat
    // projections
    k.gemmInto(x, T.Yes, dz1, T.No, g.w1, ws)
    k.gemmInto(x, T.Yes, dz2, T.No, g.w2, ws)
    val dx = MatrixF32.zeros(N, dim)
    k.gemmInto(dz1, T.No, p.w1, T.Yes, dx, ws)
    k.gemmInto(dz2, T.No, p.w2, T.Yes, dx, 1f, 1f, ws)
    // embedding: coalesce duplicate ids, then scatter-add into a zeroed gradient
    val uIds = new Array[Int](N)
    val uVals = MatrixF32.zeros(N, dim)
    val u = k.coalesceRowsInto(ids, dx, uIds, uVals, ws)
    k.fillInto(g.emb, 0f)
    k.scatterAddRowsInto(g.emb, uIds, 0, u, uVals.rowRange(0, u))
    l

object TinyModel:
  def data(seed: Long, vocab: Int, n: Int): (Array[Int], Array[Int]) =
    val rnd = new Random(seed)
    (Array.fill(n)(rnd.nextInt(vocab)), Array.fill(n)(rnd.nextInt(vocab)))

/** Gradient check of [[TinyModel]] on one backend. */
abstract class TinyModelContract(kernels: F32Kernels) extends munit.FunSuite:

  test(s"tiny model: analytic gradients match finite differences (${kernels.name})") {
    val model = new TinyModel(kernels, vocab = 11, dim = 8, hidden = 6, steps = 5, batch = 3)
    val p = model.init(1)
    val (ids, targets) = TinyModel.data(2, 11, 15)
    val g = model.zerosLike(p)
    val l = model.lossAndGrads(p, ids, targets, g)
    assert(!l.isNaN && l > 0f)
    val rnd = new Random(3)
    val h = 1e-2f
    for (param, grad) <- p.all.zip(g.all); _ <- 0 until 6 do
      val i = rnd.nextInt(param.rows); val j = rnd.nextInt(param.cols)
      val orig = param(i, j)
      param(i, j) = orig + h
      val lp = model.loss(p, ids, targets)
      param(i, j) = orig - h
      val lm = model.loss(p, ids, targets)
      param(i, j) = orig
      val fd = (lp - lm) / (2 * h)
      assert(math.abs(fd - grad(i, j)) <= 2e-3 + 3e-2 * math.abs(fd), s"($i,$j): fd $fd vs analytic ${grad(i, j)}")
  }

  test(s"tiny model: a few SGD steps reduce the loss (${kernels.name})") {
    val model = new TinyModel(kernels, vocab = 11, dim = 8, hidden = 6, steps = 5, batch = 3)
    val p = model.init(4)
    val (ids, targets) = TinyModel.data(5, 11, 15)
    val g = model.zerosLike(p)
    val first = model.lossAndGrads(p, ids, targets, g)
    for _ <- 0 until 30 do
      model.lossAndGrads(p, ids, targets, g)
      p.all.zip(g.all).foreach((w, dw) => kernels.axpyInto(-0.5f, dw, w))
    assert(model.loss(p, ids, targets) < first * 0.8f)
  }

class ScalarTinyModelSuite extends TinyModelContract(ScalarF32Kernels)
