package com.github.kmizu.numscala

import scala.reflect.ClassTag

/** Unsigned 8-bit integer (`numpy.uint8`), stored as a `Byte`. */
opaque type UInt8 = Byte
object UInt8:
  def apply(v: Long): UInt8 = v.toByte
  def fromRaw(b: Byte): UInt8 = b
  val MinValue: UInt8 = 0.toByte
  val MaxValue: UInt8 = -1.toByte
  given ClassTag[UInt8] = ClassTag.Byte
  extension (x: UInt8)
    def toInt: Int = x & 0xff
    def toLong: Long = (x & 0xff).toLong
    def toDouble: Double = (x & 0xff).toDouble
    def raw: Byte = x

/** Unsigned 16-bit integer (`numpy.uint16`), stored as a `Short`. */
opaque type UInt16 = Short
object UInt16:
  def apply(v: Long): UInt16 = v.toShort
  def fromRaw(s: Short): UInt16 = s
  val MinValue: UInt16 = 0.toShort
  val MaxValue: UInt16 = -1.toShort
  given ClassTag[UInt16] = ClassTag.Short
  extension (x: UInt16)
    def toInt: Int = x & 0xffff
    def toLong: Long = (x & 0xffff).toLong
    def toDouble: Double = (x & 0xffff).toDouble
    def raw: Short = x

/** Unsigned 32-bit integer (`numpy.uint32`), stored as an `Int`. */
opaque type UInt32 = Int
object UInt32:
  def apply(v: Long): UInt32 = v.toInt
  def fromRaw(i: Int): UInt32 = i
  val MinValue: UInt32 = 0
  val MaxValue: UInt32 = -1
  given ClassTag[UInt32] = ClassTag.Int
  extension (x: UInt32)
    def toLong: Long = Integer.toUnsignedLong(x)
    def toDouble: Double = Integer.toUnsignedLong(x).toDouble
    def raw: Int = x

/** Unsigned 64-bit integer (`numpy.uint64`), stored as a `Long` (two's complement bits). */
opaque type UInt64 = Long
object UInt64:
  def apply(v: Long): UInt64 = v
  def fromRaw(l: Long): UInt64 = l
  def parse(s: String): UInt64 = java.lang.Long.parseUnsignedLong(s.trim)
  val MinValue: UInt64 = 0L
  val MaxValue: UInt64 = -1L
  given ClassTag[UInt64] = ClassTag.Long
  extension (x: UInt64)
    /** The raw bits (values >= 2^63 come out negative). */
    def toLongBits: Long = x
    def toDouble: Double =
      if x >= 0 then x.toDouble
      else ((x >>> 1) | (x & 1L)).toDouble * 2.0
    def raw: Long = x
    def unsignedString: String = java.lang.Long.toUnsignedString(x)
