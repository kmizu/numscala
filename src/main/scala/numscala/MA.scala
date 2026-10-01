package numscala

/** Anything `np.ma` functions accept as an array: a plain [[NDArray]] (treated as having
  * no mask) or a [[MaskedArray]].
  */
type MaskedArrayLike[T] = NDArray[T] | MaskedArray[T]

/** `numpy.ma` (masked arrays).
  *
  * {{{
  * val x = np.ma.masked_invalid(np.array(1.0, Double.NaN, 3.0))
  * x.toString          // [1.0 -- 3.0]
  * np.ma.mean(x)       // 2.0
  * np.ma.sqrt(np.ma.array(Seq(4.0, -1.0)))   // [2.0 --]
  * }}}
  */
object MA extends MACreation, MAFunctions, MAMath:
  /** The masked constant (`np.ma.masked`). */
  val masked: MaskedConstant.type = MaskedConstant
  /** The masked constant class instance (`np.ma.masked_singleton`). */
  val masked_singleton: MaskedConstant.type = MaskedConstant
  /** "No mask" (`np.ma.nomask`): a 0-d `False`; compare with `eq`. */
  val nomask: NDArray[Boolean] = NDArray.scalar(false)
  /** The mask dtype (`np.ma.MaskType`). */
  val MaskType: DType[Boolean] = DType.Bool

  /** Views anything array-like as a masked array (no copy). */
  private[numscala] def toMA[T](a: MaskedArrayLike[T]): MaskedArray[T] = a match
    case m: MaskedArray[T @unchecked] => m
    case n: NDArray[T @unchecked] => MaskedArray.wrap(n)

  /** Internal mask (`null` = nomask) of a user-supplied mask value. */
  private[numscala] def internalMask(m: NDArray[Boolean]): NDArray[Boolean] =
    if m == null || (m eq nomask) then null else m
