package numscala.random

import numscala.*

/** `numpy.random.MT19937`: the 32-bit Mersenne Twister (used by the legacy [[RandomState]]). */
final class MT19937 private (dummy: Boolean) extends BitGenerator:
  import MT19937.*
  def name = "MT19937"
  private[numscala] val key = new Array[Int](N)
  private[numscala] var pos: Int = N

  /** Seeds through a [[SeedSequence]] (as `MT19937(seed)` does in NumPy). */
  def this(seed: Any = null) =
    this(true)
    val ss = RandomUtil.toSeedSeq(seed)
    seedSeqVar = ss
    initFromSeedSeq(ss)

  private def initFromSeedSeq(ss: SeedSequence): Unit =
    val v = ss.generateU32(N)
    key(0) = 0x80000000
    var i = 1
    while i < N do { key(i) = v(i); i += 1 }
    pos = N - 1 // NumPy leaves pos at the last loop index (623)

  /** `MT19937._legacy_seeding(seed)`: `init_genrand` for an integer, `init_by_array` for arrays. */
  def _legacy_seeding(seed: Any): Unit =
    seedSeqVar = null
    seed match
      case null =>
        initFromSeedSeq(SeedSequence())
        pos = N - 1
      case (_: Int | _: Long | _: BigInt | _: Short | _: Byte) =>
        val s = RandomUtil.anyToBig(seed)
        if s < 0 || s > BigInt(0xffffffffL) then throw new IllegalArgumentException("Seed must be between 0 and 2**32 - 1")
        mtSeed(s.toLong.toInt)
      case _: Double | _: Float => throw new IllegalArgumentException("Cannot cast scalar from dtype('float64') to dtype('int64') according to the rule 'safe'")
      case other =>
        val vals: Seq[Any] = other match
          case a: NDArray[?] => a.toArray.toSeq
          case a: Array[?] => a.toSeq
          case s: Iterable[?] => s.toSeq
          case _ => throw new IllegalArgumentException(s"invalid seed $other")
        if vals.isEmpty then throw new IllegalArgumentException("Seed must be non-empty")
        val ks = vals.map(RandomUtil.anyToBig)
        if ks.exists(k => k < 0 || k > BigInt(0xffffffffL)) then
          throw new IllegalArgumentException("Seed must be between 0 and 2**32 - 1")
        initByArray(ks.map(_.toLong.toInt).toArray)

  private def mtSeed(seed0: Int): Unit =
    var seed = seed0
    var p = 0
    while p < N do
      key(p) = seed
      seed = 1812433253 * (seed ^ (seed >>> 30)) + p + 1
      p += 1
    pos = N

  private def initGenrand(s: Int): Unit =
    key(0) = s
    var i = 1
    while i < N do
      key(i) = 1812433253 * (key(i - 1) ^ (key(i - 1) >>> 30)) + i
      i += 1
    pos = N

  private def initByArray(initKey: Array[Int]): Unit =
    initGenrand(19650218)
    var i = 1
    var j = 0
    var k = math.max(N, initKey.length)
    while k > 0 do
      key(i) = (key(i) ^ ((key(i - 1) ^ (key(i - 1) >>> 30)) * 1664525)) + initKey(j) + j
      i += 1; j += 1
      if i >= N then { key(0) = key(N - 1); i = 1 }
      if j >= initKey.length then j = 0
      k -= 1
    k = N - 1
    while k > 0 do
      key(i) = (key(i) ^ ((key(i - 1) ^ (key(i - 1) >>> 30)) * 1566083941)) - i
      i += 1
      if i >= N then { key(0) = key(N - 1); i = 1 }
      k -= 1
    key(0) = 0x80000000

  private def gen(): Unit =
    var i = 0
    var y = 0
    while i < N - M do
      y = (key(i) & Upper) | (key(i + 1) & Lower)
      key(i) = key(i + M) ^ (y >>> 1) ^ (-(y & 1) & MatrixA)
      i += 1
    while i < N - 1 do
      y = (key(i) & Upper) | (key(i + 1) & Lower)
      key(i) = key(i + (M - N)) ^ (y >>> 1) ^ (-(y & 1) & MatrixA)
      i += 1
    y = (key(N - 1) & Upper) | (key(0) & Lower)
    key(N - 1) = key(M - 1) ^ (y >>> 1) ^ (-(y & 1) & MatrixA)
    pos = 0

  def nextUInt32(): Int =
    if pos == N then gen()
    var y = key(pos)
    pos += 1
    y ^= y >>> 11
    y ^= (y << 7) & 0x9d2c5680
    y ^= (y << 15) & 0xefc60000
    y ^ (y >>> 18)

  def nextUInt64(): Long =
    val hi = nextUInt32()
    val lo = nextUInt32()
    (hi.toLong << 32) | (lo & 0xffffffffL)

  override def nextDouble(): Double =
    val a = nextUInt32() >>> 5
    val b = nextUInt32() >>> 6
    (a * 67108864.0 + b) / 9007199254740992.0

  override def nextRaw(): Long = nextUInt32() & 0xffffffffL

  protected def make(seed: SeedSequence): BitGenerator = new MT19937(seed)

  /** `state`: `{"bit_generator": "MT19937", "state": {"key": uint32[624], "pos": int}}`. */
  def state: Map[String, Any] =
    Map(
      "bit_generator" -> name,
      "state" -> Map("key" -> NDArray.fromArray(key.map(_ & 0xffffffffL), Array(N)), "pos" -> pos)
    )
  def state_=(st: Map[String, Any]): Unit =
    if st.getOrElse("bit_generator", "") != name then throw new IllegalArgumentException(s"state must be for a $name PRNG")
    val inner = st("state").asInstanceOf[Map[String, Any]]
    setKey(inner("key"))
    pos = RandomUtil.anyToLong(inner("pos")).toInt

  private[numscala] def setKey(k: Any): Unit =
    val vals: Seq[Any] = k match
      case a: NDArray[?] => a.toArray.toSeq
      case a: Array[?] => a.toSeq
      case s: Iterable[?] => s.toSeq
      case other => throw new IllegalArgumentException(s"invalid key $other")
    if vals.length != N then throw new IllegalArgumentException("key must have 624 elements")
    var i = 0
    while i < N do { key(i) = RandomUtil.anyToLong(vals(i)).toInt; i += 1 }

  /** `MT19937.jumped(jumps)`: a copy advanced by `jumps * 2**128` steps (polynomial jump). */
  def jumped(jumps: Int = 1): MT19937 =
    val g = new MT19937(true)
    System.arraycopy(key, 0, g.key, 0, N)
    g.pos = pos
    g.seedSeqVar = seedSeqVar
    var i = 0
    while i < jumps do { g.jumpState(); i += 1 }
    g

  // ---- polynomial jump (Haramoto et al.), following NumPy's mt19937-jump.c ----
  private def genNext(k: Array[Int], st: Array[Int]): Unit =
    val num = st(0)
    if num < N - M then
      val y = (k(num) & Upper) | (k(num + 1) & Lower)
      k(num) = k(num + M) ^ (y >>> 1) ^ (if (y & 1) != 0 then MatrixA else 0)
      st(0) += 1
    else if num < N - 1 then
      val y = (k(num) & Upper) | (k(num + 1) & Lower)
      k(num) = k(num + (M - N)) ^ (y >>> 1) ^ (if (y & 1) != 0 then MatrixA else 0)
      st(0) += 1
    else if num == N - 1 then
      val y = (k(N - 1) & Upper) | (k(0) & Lower)
      k(N - 1) = k(M - 1) ^ (y >>> 1) ^ (if (y & 1) != 0 then MatrixA else 0)
      st(0) = 0

  private def addState(k1: Array[Int], pt1: Int, k2: Array[Int], pt2: Int): Unit =
    var i = 0
    if pt2 - pt1 >= 0 then
      while i < N - pt2 do { k1(i + pt1) ^= k2(i + pt2); i += 1 }
      while i < N - pt1 do { k1(i + pt1) ^= k2(i + (pt2 - N)); i += 1 }
      while i < N do { k1(i + (pt1 - N)) ^= k2(i + (pt2 - N)); i += 1 }
    else
      while i < N - pt1 do { k1(i + pt1) ^= k2(i + pt2); i += 1 }
      while i < N - pt2 do { k1(i + (pt1 - N)) ^= k2(i + pt2); i += 1 }
      while i < N do { k1(i + (pt1 - N)) ^= k2(i + (pt2 - N)); i += 1 }

  private def coef(deg: Int): Boolean = (MT19937JumpPoly.coef(deg >> 5) & (1 << (deg & 0x1f))) != 0

  private def jumpState(): Unit =
    if pos >= N then pos = 0
    var i = 19937 - 1
    while !coef(i) do i -= 1
    val temp = new Array[Int](N)
    val tpos = Array(0)
    if i > 0 then
      System.arraycopy(key, 0, temp, 0, N)
      tpos(0) = pos
      genNext(temp, tpos)
      i -= 1
      while i > 0 do
        if coef(i) then addState(temp, tpos(0), key, pos)
        genNext(temp, tpos)
        i -= 1
      if coef(0) then addState(temp, tpos(0), key, pos)
    else if i == 0 then
      System.arraycopy(key, 0, temp, 0, N)
      tpos(0) = pos
    System.arraycopy(temp, 0, key, 0, N)
    pos = tpos(0)

object MT19937:
  private val N = 624
  private val M = 397
  private val MatrixA = 0x9908b0df
  private val Upper = 0x80000000
  private val Lower = 0x7fffffff
