package com.github.kmizu.numscala.cpu

/** Counters of a [[Workspace]] (contract NS-CPU-KERNEL-1, K5). */
final case class WorkspaceStats(
    allocatedBytes: Long,
    growCount: Int,
    bytesPacked: Long,
    bytesConverted: Long,
    floatCapacity: Int,
    longCapacity: Int
)

/** Scratch memory owned by exactly one worker (contract NS-CPU-KERNEL-1, K5).
  *
  * Not thread-safe: give each worker its own workspace. Capacity only grows through the explicit
  * `reserve*` calls; a kernel that needs more than is reserved fails before it writes any output.
  * Kernels borrow regions for the duration of one call only, so nothing borrowed may be kept across
  * calls. `reset()` returns all borrowed regions without zeroing them. With `debug = true` the first
  * thread that uses the workspace becomes its owner and use from any other thread throws.
  */
final class Workspace(initialFloats: Int = 0, initialLongs: Int = 0, val debug: Boolean = false):
  if initialFloats < 0 || initialLongs < 0 then
    throw new IllegalArgumentException("Workspace: negative initial capacity")

  private[cpu] var floats: Array[Float] = new Array[Float](initialFloats)
  private[cpu] var longs: Array[Long] = new Array[Long](initialLongs)
  private var floatTop = 0
  private var longTop = 0

  private var allocated: Long = initialFloats.toLong * 4 + initialLongs.toLong * 8
  private var grows = 0
  private var packed: Long = 0
  private var converted: Long = 0
  @volatile private var owner: Thread | Null = null

  /** Ensures room for at least `n` Float32 elements in total (grows, never shrinks). */
  def reserveFloats(n: Int): Unit =
    checkThread()
    if n < 0 then throw new IllegalArgumentException(s"reserveFloats: negative size $n")
    if n > floats.length then
      val nf = new Array[Float](n)
      System.arraycopy(floats, 0, nf, 0, floatTop)
      allocated += n.toLong * 4
      grows += 1
      floats = nf

  /** Ensures room for at least `n` Long elements in total. */
  def reserveLongs(n: Int): Unit =
    checkThread()
    if n < 0 then throw new IllegalArgumentException(s"reserveLongs: negative size $n")
    if n > longs.length then
      val nl = new Array[Long](n)
      System.arraycopy(longs, 0, nl, 0, longTop)
      allocated += n.toLong * 8
      grows += 1
      longs = nl

  /** Returns every borrowed region. Contents are left as they are (no zeroing). */
  def reset(): Unit =
    checkThread()
    floatTop = 0
    longTop = 0

  /** Clears the packed/converted counters (allocation counters are kept). */
  def resetCounters(): Unit =
    packed = 0
    converted = 0

  /** Current counters. */
  def stats: WorkspaceStats =
    WorkspaceStats(allocated, grows, packed, converted, floats.length, longs.length)

  /** Total bytes allocated by this workspace since creation. */
  def allocatedBytes: Long = allocated
  /** Number of capacity increases since creation. */
  def growCount: Int = grows
  /** Bytes copied into packed tiles (transposed/strided operands). */
  def bytesPacked: Long = packed
  /** Bytes produced by dtype conversion before a kernel could run. */
  def bytesConverted: Long = converted

  // ------------------------------------------------------------------ kernel-side borrowing

  private[cpu] def checkThread(): Unit =
    if debug then
      val t = Thread.currentThread()
      val o = owner
      if o == null then owner = t
      else if o ne t then
        throw new IllegalStateException(
          s"Workspace is owned by thread '${o.getName}' but was used from '${t.getName}'; give each worker its own Workspace"
        )

  /** Fails (before any output is written) unless the requested amounts can still be borrowed. */
  private[cpu] def require(nFloats: Int, nLongs: Int, op: String): Unit =
    checkThread()
    if floatTop.toLong + nFloats > floats.length || longTop.toLong + nLongs > longs.length then
      throw new IllegalStateException(
        s"$op: workspace too small (needs floats=$nFloats longs=$nLongs free, has " +
          s"floats=${floats.length - floatTop} longs=${longs.length - longTop}); call reserve* first"
      )

  /** Borrow cursor, restored with [[release]]. */
  private[cpu] def mark: Long = (floatTop.toLong << 32) | longTop.toLong

  private[cpu] def release(m: Long): Unit =
    floatTop = (m >>> 32).toInt
    longTop = m.toInt

  private[cpu] def takeFloats(n: Int): Int =
    val o = floatTop
    floatTop += n
    o

  private[cpu] def takeLongs(n: Int): Int =
    val o = longTop
    longTop += n
    o

  private[cpu] def addPacked(bytes: Long): Unit = packed += bytes
  private[cpu] def addConverted(bytes: Long): Unit = converted += bytes
