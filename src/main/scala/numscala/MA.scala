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
object MA extends MACreation, MAFunctions, MAMath, MAExtras, MASetOps:
  /** The masked constant (`np.ma.masked`). */
  val masked: MaskedConstant.type = MaskedConstant
  /** The masked constant class instance (`np.ma.masked_singleton`). */
  val masked_singleton: MaskedConstant.type = MaskedConstant
  /** "No mask" (`np.ma.nomask`): a 0-d `False`; compare with `eq`. */
  val nomask: NDArray[Boolean] = NDArray.scalar(false)
  /** The mask dtype (`np.ma.MaskType`). */
  val MaskType: DType[Boolean] = DType.Bool
  /** The boolean dtype (`np.ma.bool_`). */
  val bool_ : DType[Boolean] = DType.Bool
  /** How masked entries are displayed (`np.ma.masked_print_option`, `--` by default). */
  val masked_print_option: MaskedPrintOption = new MaskedPrintOption("--")

  /** Base class of masked-array errors (`np.ma.MAError`). Extends `IllegalArgumentException`
    * so that code catching the generic error keeps working.
    */
  class MAError(message: String) extends IllegalArgumentException(message)
  /** Mask-related error (`np.ma.MaskError`), e.g. a mask whose shape does not fit the data. */
  class MaskError(message: String) extends MAError(message)

  /** Views anything array-like as a masked array (no copy). */
  private[numscala] def toMA[T](a: MaskedArrayLike[T]): MaskedArray[T] = a match
    case m: MaskedArray[T @unchecked] => m
    case n: NDArray[T @unchecked] => MaskedArray.wrap(n)

  /** Internal mask (`null` = nomask) of a user-supplied mask value. */
  private[numscala] def internalMask(m: NDArray[Boolean]): NDArray[Boolean] =
    if m == null || (m eq nomask) then null else m

/** `np.ma.masked_print_option`: the string used for masked entries in `str`/`repr`. */
final class MaskedPrintOption private[numscala] (private var _display: String):
  private var _enabled: Boolean = true
  /** The display string (`masked_print_option.display()`). */
  def display(): String = _display
  /** Changes the display string (`masked_print_option.set_display(s)`). */
  def set_display(s: String): Unit = _display = s
  /** Whether the option is enabled (`masked_print_option.enabled()`). */
  def enabled(): Boolean = _enabled
  /** Enables or disables the option (`masked_print_option.enable(shrink)`). */
  def enable(shrink: Boolean = true): Unit = _enabled = shrink
  override def toString: String = _display
