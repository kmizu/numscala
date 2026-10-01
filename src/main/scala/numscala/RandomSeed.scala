package numscala.random

import numscala.*

/** `numpy.random.SeedSequence`: mixes entropy (an integer or a sequence of integers) into a pool
  * and derives well-distributed seeds for bit generators, exactly as NumPy does.
  *
  * @param entropy   a non-negative `Int`/`Long`/`BigInt`, a decimal or `0x` hex `String`, a
  *                  sequence/array of those, or `null` for fresh OS entropy
  * @param spawn_key extra entropy identifying the position in a spawn tree
  */
final class SeedSequence(
    entropy0: Any = null,
    val spawn_key: Seq[Long] = Seq.empty,
    val pool_size: Int = 4,
    private var nSpawned: Int = 0
):
  import SeedSequence.*
  if pool_size < DefaultPoolSize then
    throw new IllegalArgumentException(s"The size of the entropy pool should be at least $DefaultPoolSize")

  /** The entropy (a `BigInt` when drawn from the OS). */
  val entropy: Any =
    if entropy0 == null then BigInt(pool_size * 32, new java.security.SecureRandom()) else entropy0

  /** Number of children spawned so far. */
  def n_children_spawned: Int = nSpawned

  /** The mixed entropy pool (uint32 words held in `Int`s). */
  val pool: Array[Int] =
    val mixer = new Array[Int](pool_size)
    mixEntropy(mixer, assembledEntropy)
    mixer

  private def assembledEntropy: Array[Int] =
    var run = coerce(entropy)
    val spawn = coerce(spawn_key)
    if spawn.nonEmpty && run.length < pool_size then run = run ++ new Array[Int](pool_size - run.length)
    run ++ spawn

  private def mixEntropy(mixer: Array[Int], ent: Array[Int]): Unit =
    val hc = Array(InitA)
    def hashmix(v0: Int): Int =
      var value = v0 ^ hc(0)
      hc(0) *= MultA
      value *= hc(0)
      value ^ (value >>> XShift)
    def mix(x: Int, y: Int): Int =
      val r = MixMultL * x - MixMultR * y
      r ^ (r >>> XShift)
    for i <- mixer.indices do mixer(i) = hashmix(if i < ent.length then ent(i) else 0)
    for iSrc <- mixer.indices; iDst <- mixer.indices if iSrc != iDst do
      mixer(iDst) = mix(mixer(iDst), hashmix(mixer(iSrc)))
    for iSrc <- mixer.length until ent.length; iDst <- mixer.indices do
      mixer(iDst) = mix(mixer(iDst), hashmix(ent(iSrc)))

  /** `generate_state(n_words)` with `dtype=uint32`: words returned as non-negative `Long`s. */
  def generate_state(n_words: Int): NDArray[Long] =
    NDArray.fromArray(generateU32(n_words).map(_ & 0xffffffffL), Array(n_words))

  /** `generate_state(n_words, dtype)` where `dtype` is `"uint32"` or `"uint64"`; uint64 words are
    * returned as `BigInt`s inside a `Seq` (they may exceed `Long.MaxValue`).
    */
  def generate_state(n_words: Int, dtype: String): Seq[BigInt] = dtype match
    case "uint32" => generateU32(n_words).toSeq.map(U64.toBig32)
    case "uint64" => generate_state_u64(n_words).toSeq.map(U64.toBig)
    case _ => throw new IllegalArgumentException("only support uint32 or uint64")

  private[numscala] def generateU32(n: Int): Array[Int] =
    var hashConst = InitB
    val out = new Array[Int](n)
    var i = 0
    while i < n do
      var v = pool(i % pool.length)
      v ^= hashConst
      hashConst *= MultB
      v *= hashConst
      v ^= v >>> XShift
      out(i) = v
      i += 1
    out

  private[numscala] def generate_state_u64(n: Int): Array[Long] =
    val w = generateU32(2 * n)
    Array.tabulate(n)(i => (w(2 * i) & 0xffffffffL) | (w(2 * i + 1).toLong << 32))

  /** `spawn(n_children)`: child seed sequences with extended spawn keys. */
  def spawn(n_children: Int): Seq[SeedSequence] =
    val out = (nSpawned until nSpawned + n_children).map(i =>
      new SeedSequence(entropy, spawn_key :+ i.toLong, pool_size)
    )
    nSpawned += n_children
    out

  /** `SeedSequence.state`. */
  def state: Map[String, Any] =
    Map("entropy" -> entropy, "spawn_key" -> spawn_key, "pool_size" -> pool_size, "n_children_spawned" -> nSpawned)

  override def toString: String =
    val b = new StringBuilder("SeedSequence(\n")
    b ++= s"    entropy=${entropy},\n"
    if spawn_key.nonEmpty then b ++= s"    spawn_key=(${spawn_key.mkString(", ")}${if spawn_key.length == 1 then "," else ""}),\n"
    if pool_size != DefaultPoolSize then b ++= s"    pool_size=$pool_size,\n"
    if nSpawned != 0 then b ++= s"    n_children_spawned=$nSpawned,\n"
    b ++= ")"
    b.toString

object SeedSequence:
  /** A seed sequence drawing fresh OS entropy. */
  def apply(): SeedSequence = new SeedSequence(null)
  /** A seed sequence from integer(s) (see the class docs for accepted forms). */
  def apply(entropy: Any): SeedSequence = new SeedSequence(entropy)
  /** A seed sequence with an explicit spawn key. */
  def apply(entropy: Any, spawn_key: Seq[Long]): SeedSequence = new SeedSequence(entropy, spawn_key)

  private[numscala] val DefaultPoolSize = 4
  private val InitA = 0x43b0d7e5
  private val MultA = 0x931e8875
  private val InitB = 0x8b51f9dd
  private val MultB = 0x58f38ded
  private val MixMultL = 0xca01f9dd
  private val MixMultR = 0x4973f715
  private val XShift = 16

  private def intWords(n: BigInt): Array[Int] =
    if n < 0 then throw new IllegalArgumentException("expected non-negative integer")
    if n == 0 then Array(0)
    else
      val b = Array.newBuilder[Int]
      var x = n
      while x > 0 do
        b += (x & 0xffffffffL).toInt
        x = x >> 32
      b.result()

  /** NumPy's `_coerce_to_uint32_array`. */
  private[numscala] def coerce(x: Any): Array[Int] = x match
    case null => Array.empty
    case i: Int => intWords(BigInt(i))
    case l: Long => intWords(BigInt(l))
    case s: Short => intWords(BigInt(s.toInt))
    case b: Byte => intWords(BigInt(b.toInt))
    case b: BigInt => intWords(b)
    case _: Double | _: Float => throw new IllegalArgumentException("seed must be integer")
    case s: String =>
      if s.startsWith("0x") then intWords(BigInt(s.drop(2), 16))
      else if s.nonEmpty && s.head.isDigit then intWords(BigInt(s.takeWhile(_.isDigit)))
      else throw new IllegalArgumentException("unrecognized seed string")
    case a: NDArray[?] => a.toArray.toSeq.flatMap(e => coerce(e)).toArray
    case a: Array[?] => a.toSeq.flatMap(e => coerce(e)).toArray
    case s: Iterable[?] => s.toSeq.flatMap(e => coerce(e)).toArray
    case other => throw new IllegalArgumentException(s"SeedSequence expects int or sequence of ints for entropy not $other")
