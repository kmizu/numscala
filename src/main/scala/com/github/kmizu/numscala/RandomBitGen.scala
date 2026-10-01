package com.github.kmizu.numscala.random

import com.github.kmizu.numscala.*

/** Unsigned 64/128-bit helpers shared by the bit generators. */
private[numscala] object U64:
  inline def mulHi(a: Long, b: Long): Long =
    Math.multiplyHigh(a, b) + ((a >> 63) & b) + ((b >> 63) & a)
  inline def ltU(a: Long, b: Long): Boolean = java.lang.Long.compareUnsigned(a, b) < 0
  inline def toDouble53(x: Long): Double = (x >>> 11) * (1.0 / 9007199254740992.0)
  val Two64: BigInt = BigInt(1) << 64
  val Mask64: BigInt = Two64 - 1
  def toBig(x: Long): BigInt = if x >= 0 then BigInt(x) else BigInt(x) + Two64
  def fromBig(x: BigInt): Long = (x & Mask64).longValue
  def toBig32(x: Int): BigInt = BigInt(x & 0xffffffffL)

/** Base class of NumPy's bit generators (`numpy.random.BitGenerator`).
  *
  * Bit generators produce raw 64-bit / 32-bit words and doubles; [[Generator]] and
  * [[RandomState]] turn them into distributions.
  */
abstract class BitGenerator:
  /** The seed sequence used to seed this bit generator (null for legacy-seeded MT19937). */
  protected var seedSeqVar: SeedSequence | Null = null
  /** `BitGenerator.seed_seq`. */
  def seed_seq: SeedSequence | Null = seedSeqVar
  /** Name of the bit generator class, as in the `state["bit_generator"]` entry. */
  def name: String

  /** Next raw 64-bit word (unsigned value stored in a `Long`). */
  def nextUInt64(): Long
  /** Next raw 32-bit word (unsigned value stored in an `Int`). */
  def nextUInt32(): Int
  /** Next double in `[0, 1)`. */
  def nextDouble(): Double = U64.toDouble53(nextUInt64())
  /** Next raw output, as returned by `random_raw`. */
  def nextRaw(): Long = nextUInt64()

  /** `BitGenerator.state`: a NumPy-shaped description of the state. */
  def state: Map[String, Any]
  /** Sets the state from a map produced by [[state]]. */
  def state_=(st: Map[String, Any]): Unit

  /** Creates a fresh instance of the same kind (used by `spawn`). */
  protected def make(seed: SeedSequence): BitGenerator

  /** `BitGenerator.spawn(n_children)`: independent child bit generators. */
  def spawn(n_children: Int): Seq[BitGenerator] =
    val ss = seedSeqVar
    if ss == null then throw new UnsupportedOperationException("The underlying SeedSequence does not implement spawning.")
    ss.spawn(n_children).map(make)

  /** `BitGenerator.random_raw()`: one raw value. */
  def random_raw(): Long = nextRaw()
  /** `BitGenerator.random_raw(size)`: raw values (as signed `Long`s holding uint64 bits). */
  def random_raw(size: Int*): NDArray[Long] =
    val shp = size.toArray
    val out = new Array[Long](Shape.size(shp))
    var i = 0
    while i < out.length do { out(i) = nextRaw(); i += 1 }
    NDArray.fromArray(out, shp)

  override def toString: String = name

/** A bit generator that buffers the upper half of a 64-bit draw for 32-bit requests. */
abstract class BufferedBitGenerator extends BitGenerator:
  protected var hasUInt32: Boolean = false
  protected var uinteger: Int = 0
  def nextUInt32(): Int =
    if hasUInt32 then
      hasUInt32 = false
      uinteger
    else
      val n = nextUInt64()
      hasUInt32 = true
      uinteger = (n >>> 32).toInt
      n.toInt
  protected def resetBuffer(): Unit =
    hasUInt32 = false
    uinteger = 0
  protected def bufferState: Map[String, Any] =
    Map("has_uint32" -> (if hasUInt32 then 1 else 0), "uinteger" -> (uinteger & 0xffffffffL))
  protected def setBuffer(st: Map[String, Any]): Unit =
    hasUInt32 = RandomUtil.anyToLong(st.getOrElse("has_uint32", 0)) != 0
    uinteger = RandomUtil.anyToLong(st.getOrElse("uinteger", 0)).toInt
  protected def checkName(st: Map[String, Any]): Unit =
    if st.getOrElse("bit_generator", "") != name then
      throw new IllegalArgumentException(s"state must be for a $name PRNG")

private[numscala] object RandomUtil:
  def anyToLong(v: Any): Long = v match
    case i: Int => i.toLong
    case l: Long => l
    case b: BigInt => b.longValue
    case b: Boolean => if b then 1L else 0L
    case d: Double => d.toLong
    case s: Short => s.toLong
    case b: Byte => b.toLong
    case other => throw new IllegalArgumentException(s"expected an integer, got $other")
  def anyToBig(v: Any): BigInt = v match
    case b: BigInt => b
    case other => BigInt(anyToLong(other))
  def toSeedSeq(seed: Any): SeedSequence = seed match
    case null => SeedSequence()
    case s: SeedSequence => s
    case other => SeedSequence(other)

/** Shared 128-bit LCG machinery of `PCG64` and `PCG64DXSM`. */
abstract class PCG128Base extends BufferedBitGenerator:
  protected var sHi, sLo, incHi, incLo: Long = 0L
  protected def multHi: Long
  protected def multLo: Long

  protected final def stepDefault(): Unit = stepWith(multHi, multLo)

  /** state = state * (mHi:mLo) + inc  (128-bit arithmetic). */
  protected final def stepWith(mHi: Long, mLo: Long): Unit =
    val lo = sLo * mLo
    val hi = U64.mulHi(sLo, mLo) + sHi * mLo + sLo * mHi
    val nlo = lo + incLo
    val carry = if U64.ltU(nlo, incLo) then 1L else 0L
    sLo = nlo
    sHi = hi + incHi + carry

  protected def bigState: BigInt = (U64.toBig(sHi) << 64) + U64.toBig(sLo)
  protected def bigInc: BigInt = (U64.toBig(incHi) << 64) + U64.toBig(incLo)
  protected def setBigState(v: BigInt): Unit =
    sHi = U64.fromBig(v >> 64); sLo = U64.fromBig(v)
  protected def setBigInc(v: BigInt): Unit =
    incHi = U64.fromBig(v >> 64); incLo = U64.fromBig(v)

  protected def advanceMult: BigInt
  // NumPy seeds both PCG64 and PCG64DXSM with the default-multiplier `pcg64_set_seed`.
  private def step(): Unit = stepWith(2549297995355413924L, 4865540595714422341L)

  protected def seedWith(ss: SeedSequence): Unit =
    val v = ss.generate_state_u64(4)
    val initstate = (U64.toBig(v(0)) << 64) + U64.toBig(v(1))
    val initseq = (U64.toBig(v(2)) << 64) + U64.toBig(v(3))
    val mask = (BigInt(1) << 128) - 1
    setBigState(BigInt(0))
    setBigInc(((initseq << 1) | 1) & mask)
    step()
    setBigState((bigState + initstate) & mask)
    step()
    resetBuffer()

  /** `advance(delta)`: jumps the state as if `delta` draws had been made. */
  def advance(delta: BigInt): this.type =
    val mask = (BigInt(1) << 128) - 1
    var d = delta & mask
    var curMult = advanceMult
    var curPlus = bigInc
    var accMult = BigInt(1)
    var accPlus = BigInt(0)
    while d > 0 do
      if d.testBit(0) then
        accMult = (accMult * curMult) & mask
        accPlus = (accPlus * curMult + curPlus) & mask
      curPlus = ((curMult + 1) * curPlus) & mask
      curMult = (curMult * curMult) & mask
      d = d >> 1
    setBigState((accMult * bigState + accPlus) & mask)
    resetBuffer()
    this

  def state: Map[String, Any] =
    Map("bit_generator" -> name, "state" -> Map("state" -> bigState, "inc" -> bigInc)) ++ bufferState
  def state_=(st: Map[String, Any]): Unit =
    checkName(st)
    val inner = st("state").asInstanceOf[Map[String, Any]]
    setBigState(RandomUtil.anyToBig(inner("state")))
    setBigInc(RandomUtil.anyToBig(inner("inc")))
    setBuffer(st)

  protected def copyFrom(o: PCG128Base): Unit =
    sHi = o.sHi; sLo = o.sLo; incHi = o.incHi; incLo = o.incLo
    hasUInt32 = o.hasUInt32; uinteger = o.uinteger

/** `numpy.random.PCG64`: 128-bit LCG with the XSL-RR output function (NumPy's default). */
final class PCG64 private (dummy: Boolean) extends PCG128Base:
  def name = "PCG64"
  protected def multHi: Long = 2549297995355413924L
  protected def multLo: Long = 4865540595714422341L
  protected def advanceMult: BigInt = (U64.toBig(multHi) << 64) + U64.toBig(multLo)
  def nextUInt64(): Long =
    stepDefault()
    java.lang.Long.rotateRight(sHi ^ sLo, (sHi >>> 58).toInt)

  /** Seeds from a [[SeedSequence]], an integer, a sequence of integers or `null` (OS entropy). */
  def this(seed: Any = null) =
    this(true)
    val ss = RandomUtil.toSeedSeq(seed)
    seedSeqVar = ss
    seedWith(ss)
  protected def make(seed: SeedSequence): BitGenerator = new PCG64(seed)
  /** `PCG64.jumped(jumps)`: a copy advanced by `jumps * 0x9e3779b97f4a7c15f39cc0605cedc835` steps. */
  def jumped(jumps: Int = 1): PCG64 =
    val g = new PCG64(true)
    g.copyFrom(this)
    g.seedSeqVar = seedSeqVar
    g.advance(PCG128Base.JumpStep * jumps)
    g

/** `numpy.random.PCG64DXSM`: PCG64 with the DXSM output function and a cheap multiplier. */
final class PCG64DXSM private (dummy: Boolean) extends PCG128Base:
  def name = "PCG64DXSM"
  private val Cheap = 0xda942042e4dd58b5L
  protected def multHi: Long = 0L
  protected def multLo: Long = Cheap
  protected def advanceMult: BigInt = U64.toBig(Cheap)
  def nextUInt64(): Long =
    var hi = sHi
    val lo = sLo | 1L
    hi ^= hi >>> 32
    hi *= Cheap
    hi ^= hi >>> 48
    hi *= lo
    stepDefault()
    hi
  /** Seeds from a [[SeedSequence]], an integer, a sequence of integers or `null` (OS entropy). */
  def this(seed: Any = null) =
    this(true)
    val ss = RandomUtil.toSeedSeq(seed)
    seedSeqVar = ss
    seedWith(ss)
  protected def make(seed: SeedSequence): BitGenerator = new PCG64DXSM(seed)
  /** `PCG64DXSM.jumped(jumps)`. */
  def jumped(jumps: Int = 1): PCG64DXSM =
    val g = new PCG64DXSM(true)
    g.copyFrom(this)
    g.seedSeqVar = seedSeqVar
    g.advance(PCG128Base.JumpStep * jumps)
    g

private[numscala] object PCG128Base:
  val JumpStep: BigInt = BigInt("9e3779b97f4a7c15f39cc0605cedc835", 16)

/** `numpy.random.SFC64`: Chris Doty-Humphrey's Small Fast Chaotic PRNG. */
final class SFC64 private (dummy: Boolean) extends BufferedBitGenerator:
  def name = "SFC64"
  private val s = new Array[Long](4)
  def nextUInt64(): Long =
    val tmp = s(0) + s(1) + s(3)
    s(3) += 1
    s(0) = s(1) ^ (s(1) >>> 11)
    s(1) = s(2) + (s(2) << 3)
    s(2) = java.lang.Long.rotateLeft(s(2), 24) + tmp
    tmp
  /** Seeds from a [[SeedSequence]], an integer, a sequence of integers or `null` (OS entropy). */
  def this(seed: Any = null) =
    this(true)
    val ss = RandomUtil.toSeedSeq(seed)
    seedSeqVar = ss
    val v = ss.generate_state_u64(3)
    s(0) = v(0); s(1) = v(1); s(2) = v(2); s(3) = 1L
    var i = 0
    while i < 12 do { nextUInt64(); i += 1 }
    resetBuffer()
  protected def make(seed: SeedSequence): BitGenerator = new SFC64(seed)
  def state: Map[String, Any] =
    Map("bit_generator" -> name, "state" -> Map("state" -> s.map(U64.toBig).toSeq)) ++ bufferState
  def state_=(st: Map[String, Any]): Unit =
    checkName(st)
    val v = st("state").asInstanceOf[Map[String, Any]]("state").asInstanceOf[Seq[Any]]
    var i = 0
    while i < 4 do { s(i) = U64.fromBig(RandomUtil.anyToBig(v(i))); i += 1 }
    setBuffer(st)

/** `numpy.random.Philox`: the Philox4x64-10 counter-based generator. */
final class Philox private (dummy: Boolean) extends BufferedBitGenerator:
  def name = "Philox"
  private val ctr = new Array[Long](4)
  private val key = new Array[Long](2)
  private val buffer = new Array[Long](4)
  private var bufferPos = 4

  private def round(c: Array[Long], k0: Long, k1: Long): Unit =
    val m0 = 0xD2E7470EE14C6C93L
    val m1 = 0xCA5A826395121157L
    val lo0 = m0 * c(0); val hi0 = U64.mulHi(m0, c(0))
    val lo1 = m1 * c(2); val hi1 = U64.mulHi(m1, c(2))
    val n0 = hi1 ^ c(1) ^ k0
    val n2 = hi0 ^ c(3) ^ k1
    c(0) = n0; c(1) = lo1; c(2) = n2; c(3) = lo0

  def nextUInt64(): Long =
    if bufferPos < 4 then
      val out = buffer(bufferPos)
      bufferPos += 1
      out
    else
      ctr(0) += 1
      if ctr(0) == 0 then
        ctr(1) += 1
        if ctr(1) == 0 then
          ctr(2) += 1
          if ctr(2) == 0 then ctr(3) += 1
      val c = ctr.clone()
      var k0 = key(0); var k1 = key(1)
      var r = 0
      while r < 10 do
        if r > 0 then
          k0 += 0x9E3779B97F4A7C15L
          k1 += 0xBB67AE8584CAA73BL
        round(c, k0, k1)
        r += 1
      System.arraycopy(c, 0, buffer, 0, 4)
      bufferPos = 1
      buffer(0)

  private def resetAll(): Unit =
    resetBuffer()
    bufferPos = 4
    java.util.Arrays.fill(buffer, 0L)

  private def intToWords(v: BigInt, n: Int, what: String, bits: Int): Array[Long] =
    if v < 0 || v >= (BigInt(1) << bits) then
      throw new IllegalArgumentException(s"$what must be positive and less than 2**$bits.")
    Array.tabulate(n)(i => U64.fromBig(v >> (64 * i)))

  /** Seeds from a seed (or a [[SeedSequence]]), with optional explicit 256-bit `counter` and 128-bit `key`. */
  def this(seed: Any = null, counter: BigInt | Null = null, key: BigInt | Null = null) =
    this(true)
    if seed != null && key != null then throw new IllegalArgumentException("seed and key cannot be both used")
    if key != null then
      val k = intToWords(key, 2, "key", 128)
      this.key(0) = k(0); this.key(1) = k(1)
    else
      val ss = RandomUtil.toSeedSeq(seed)
      seedSeqVar = ss
      val k = ss.generate_state_u64(2)
      this.key(0) = k(0); this.key(1) = k(1)
    val c = intToWords(if counter == null then BigInt(0) else counter, 4, "counter", 256)
    System.arraycopy(c, 0, ctr, 0, 4)
    resetAll()
  protected def make(seed: SeedSequence): BitGenerator = new Philox(seed)

  /** `Philox.advance(delta)`: advances the counter as if `delta` draws had been made. */
  def advance(delta: BigInt): this.type =
    val d = delta & ((BigInt(1) << 256) - 1)
    val step = Array.tabulate(4)(i => U64.fromBig(d >> (64 * i)))
    var carry = false
    var i = 0
    while i < 4 do
      if carry then
        ctr(i) += 1
        carry = ctr(i) == 0
      val orig = ctr(i)
      ctr(i) += step(i)
      if U64.ltU(ctr(i), orig) && !carry then carry = true
      i += 1
    resetAll()
    this

  /** `Philox.jumped(jumps)`: a copy advanced by `jumps * 2**128` draws. */
  def jumped(jumps: Int = 1): Philox =
    val g = new Philox(true)
    g.state = state
    g.seedSeqVar = seedSeqVar
    g.advance(BigInt(jumps) << 128)
    g

  def state: Map[String, Any] =
    Map(
      "bit_generator" -> name,
      "state" -> Map("counter" -> ctr.map(U64.toBig).toSeq, "key" -> key.map(U64.toBig).toSeq),
      "buffer" -> buffer.map(U64.toBig).toSeq,
      "buffer_pos" -> bufferPos
    ) ++ bufferState
  def state_=(st: Map[String, Any]): Unit =
    checkName(st)
    val inner = st("state").asInstanceOf[Map[String, Any]]
    val c = inner("counter").asInstanceOf[Seq[Any]]
    val k = inner("key").asInstanceOf[Seq[Any]]
    for i <- 0 until 4 do ctr(i) = U64.fromBig(RandomUtil.anyToBig(c(i)))
    for i <- 0 until 2 do key(i) = U64.fromBig(RandomUtil.anyToBig(k(i)))
    val b = st("buffer").asInstanceOf[Seq[Any]]
    for i <- 0 until 4 do buffer(i) = U64.fromBig(RandomUtil.anyToBig(b(i)))
    bufferPos = RandomUtil.anyToLong(st("buffer_pos")).toInt
    setBuffer(st)
