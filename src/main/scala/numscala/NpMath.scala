package numscala

/** Mathematical functions and ufuncs of the `np` namespace: binary ufuncs as [[Ufunc]]
  * values (`np.add`, `np.maximum`, ... with `reduce`/`accumulate`/`outer`/`at`/`reduceat`),
  * unary elementwise functions (`np.sqrt`, `np.sin`, ...), rounding, clipping, complex
  * helpers and `np.emath`.
  */
trait NpMath extends NpMathUfuncs, NpMathUnary, NpMathMisc:
  /** `numpy.emath`: functions with complex results outside the real domain. */
  val emath: Emath.type = Emath
