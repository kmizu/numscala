package numscala

/** The NumPy namespace.
  *
  * {{{
  * import numscala.*
  * val a = np.arange(6.0).reshape(2, 3)
  * np.sum(a, axis = 0)
  * np.linalg.inv(np.eye(3))
  * np.fft.fft(np.ones(8))
  * val rng = np.random.default_rng(42)
  * }}}
  */
object np
    extends NpCreation,
      NpShape,
      NpMath,
      NpReduce,
      NpStats,
      NpSort,
      NpSet,
      NpIndexing,
      NpLinalgTop,
      NpPoly,
      NpIO,
      NpMisc:
  /** `numpy.linalg`. */
  val linalg: Linalg.type = Linalg
  /** `numpy.fft`. */
  val fft: FFT.type = FFT
  /** `numpy.random`. */
  val random: NpRandom.type = NpRandom
  /** `numpy.testing`. */
  val testing: Testing.type = Testing
  /** `numpy.polynomial`. */
  val polynomial: PolynomialModule.type = PolynomialModule
