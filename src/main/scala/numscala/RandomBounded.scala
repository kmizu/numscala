package numscala.random

/** Port of NumPy's bounded-integer samplers (Lemire's method and masked rejection),
  * `random_bounded_uint64` and the buffered 32/16/8-bit and bool variants.
  * All values are unsigned quantities held in signed JVM integers.
  */
private[numscala] object Bounded:
  /** The 32-bit buffer shared by consecutive 16/8/1-bit draws (`bcnt`, `buf`). */
  final class Buf:
    var bcnt: Int = 0
    var buf: Int = 0

  def genMask(max: Long): Long =
    var mask = max
    mask |= mask >>> 1; mask |= mask >>> 2; mask |= mask >>> 4
    mask |= mask >>> 8; mask |= mask >>> 16; mask |= mask >>> 32
    mask

  private def u16(bg: BitGenerator, b: Buf): Int =
    if b.bcnt == 0 then
      b.buf = bg.nextUInt32()
      b.bcnt = 1
    else
      b.buf = b.buf >>> 16
      b.bcnt -= 1
    b.buf & 0xffff

  private def u8(bg: BitGenerator, b: Buf): Int =
    if b.bcnt == 0 then
      b.buf = bg.nextUInt32()
      b.bcnt = 3
    else
      b.buf = b.buf >>> 8
      b.bcnt -= 1
    b.buf & 0xff

  // ---- 64 bit ----
  private def masked64(bg: BitGenerator, rng: Long, mask: Long): Long =
    var v = 0L
    while
      v = bg.nextUInt64() & mask
      U64.ltU(rng, v)
    do ()
    v

  private def lemire64(bg: BitGenerator, rng: Long): Long =
    val rngExcl = rng + 1
    var x = bg.nextUInt64()
    var leftover = x * rngExcl
    if U64.ltU(leftover, rngExcl) then
      val threshold = java.lang.Long.remainderUnsigned(-1L - rng, rngExcl)
      while U64.ltU(leftover, threshold) do
        x = bg.nextUInt64()
        leftover = x * rngExcl
    U64.mulHi(x, rngExcl)

  // ---- 32 bit (rng and results are uint32 in Int) ----
  private def masked32(bg: BitGenerator, rng: Int, mask: Int): Int =
    var v = 0
    while
      v = bg.nextUInt32() & mask
      Integer.compareUnsigned(v, rng) > 0
    do ()
    v

  private def lemire32(bg: BitGenerator, rng: Int): Int =
    val rngExcl = (rng + 1).toLong & 0xffffffffL
    var m = (bg.nextUInt32() & 0xffffffffL) * rngExcl
    var leftover = m & 0xffffffffL
    if leftover < rngExcl then
      val threshold = (0xffffffffL - (rng & 0xffffffffL)) % rngExcl
      while leftover < threshold do
        m = (bg.nextUInt32() & 0xffffffffL) * rngExcl
        leftover = m & 0xffffffffL
    (m >>> 32).toInt

  /** `random_bounded_uint64(off, rng, mask, use_masked)`: `off + U[0, rng]` (wrapping). */
  def uint64(bg: BitGenerator, off: Long, rng: Long, mask: Long, useMasked: Boolean): Long =
    if rng == 0 then off
    else if !U64.ltU(0xffffffffL, rng) then
      if rng == 0xffffffffL then off + (bg.nextUInt32() & 0xffffffffL)
      else if useMasked then off + (masked32(bg, rng.toInt, mask.toInt) & 0xffffffffL)
      else off + (lemire32(bg, rng.toInt) & 0xffffffffL)
    else if rng == -1L then off + bg.nextUInt64()
    else if useMasked then off + masked64(bg, rng, mask)
    else off + lemire64(bg, rng)

  /** `random_buffered_bounded_uint32`. */
  def uint32(bg: BitGenerator, off: Int, rng: Int, mask: Int, useMasked: Boolean): Int =
    if rng == 0 then off
    else if rng == -1 then off + bg.nextUInt32()
    else if useMasked then off + masked32(bg, rng, mask)
    else off + lemire32(bg, rng)

  /** `random_buffered_bounded_uint16` (rng, mask, off are uint16 values in Int). */
  def uint16(bg: BitGenerator, off: Int, rng: Int, mask: Int, useMasked: Boolean, b: Buf): Int =
    val r =
      if rng == 0 then 0
      else if rng == 0xffff then u16(bg, b)
      else if useMasked then
        var v = 0
        while
          v = u16(bg, b) & mask
          v > rng
        do ()
        v
      else
        val rngExcl = rng + 1
        var m = u16(bg, b) * rngExcl
        var leftover = m & 0xffff
        if leftover < rngExcl then
          val threshold = (0xffff - rng) % rngExcl
          while leftover < threshold do
            m = u16(bg, b) * rngExcl
            leftover = m & 0xffff
        m >>> 16
    (off + r) & 0xffff

  /** `random_buffered_bounded_uint8`. */
  def uint8(bg: BitGenerator, off: Int, rng: Int, mask: Int, useMasked: Boolean, b: Buf): Int =
    val r =
      if rng == 0 then 0
      else if rng == 0xff then u8(bg, b)
      else if useMasked then
        var v = 0
        while
          v = u8(bg, b) & mask
          v > rng
        do ()
        v
      else
        val rngExcl = rng + 1
        var m = u8(bg, b) * rngExcl
        var leftover = m & 0xff
        if leftover < rngExcl then
          val threshold = (0xff - rng) % rngExcl
          while leftover < threshold do
            m = u8(bg, b) * rngExcl
            leftover = m & 0xff
        m >>> 8
    (off + r) & 0xff

  /** `random_buffered_bounded_bool`. */
  def bool(bg: BitGenerator, off: Boolean, rng: Int, b: Buf): Boolean =
    if rng == 0 then off
    else
      if b.bcnt == 0 then
        b.buf = bg.nextUInt32()
        b.bcnt = 31
      else
        b.buf = b.buf >>> 1
        b.bcnt -= 1
      (b.buf & 1) != 0
