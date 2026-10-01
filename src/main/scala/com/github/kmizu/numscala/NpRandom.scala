package com.github.kmizu.numscala

import com.github.kmizu.numscala.random as R

/** `numpy.random`: `default_rng`, the `Generator`/`RandomState` classes, the bit generators and
  * the legacy module-level functions operating on a global [[R.RandomState]].
  *
  * {{{
  * np.random.seed(0)
  * np.random.rand(3)                      // 0.5488135, 0.71518937, 0.60276338
  * val rng = np.random.default_rng(42)
  * rng.random(3)                          // 0.77395605, 0.43887844, 0.85859792
  * }}}
  */
object NpRandom:
  /** `numpy.random.Generator`. */
  type Generator = R.Generator
  /** `numpy.random.RandomState`. */
  type RandomState = R.RandomState
  /** `numpy.random.BitGenerator`. */
  type BitGenerator = R.BitGenerator
  /** `numpy.random.SeedSequence`. */
  type SeedSequence = R.SeedSequence
  /** `numpy.random.PCG64`. */
  type PCG64 = R.PCG64
  /** `numpy.random.PCG64DXSM`. */
  type PCG64DXSM = R.PCG64DXSM
  /** `numpy.random.MT19937`. */
  type MT19937 = R.MT19937
  /** `numpy.random.Philox`. */
  type Philox = R.Philox
  /** `numpy.random.SFC64`. */
  type SFC64 = R.SFC64

  /** `np.random.Generator(bit_generator)`. */
  def Generator(bit_generator: R.BitGenerator): R.Generator = new R.Generator(bit_generator)
  /** `np.random.RandomState(seed)`: legacy MT19937 seeding (or wraps a bit generator). */
  def RandomState(seed: Any = null): R.RandomState = new R.RandomState(seed)
  /** `np.random.SeedSequence(entropy, spawn_key, pool_size)`. */
  def SeedSequence(entropy: Any = null, spawn_key: Seq[Long] = Seq.empty, pool_size: Int = 4): R.SeedSequence =
    new R.SeedSequence(entropy, spawn_key, pool_size)
  /** `np.random.PCG64(seed)`. */
  def PCG64(seed: Any = null): R.PCG64 = new R.PCG64(seed)
  /** `np.random.PCG64DXSM(seed)`. */
  def PCG64DXSM(seed: Any = null): R.PCG64DXSM = new R.PCG64DXSM(seed)
  /** `np.random.MT19937(seed)` (seeded through a `SeedSequence`). */
  def MT19937(seed: Any = null): R.MT19937 = new R.MT19937(seed)
  /** `np.random.Philox(seed, counter, key)`. */
  def Philox(seed: Any = null, counter: BigInt | Null = null, key: BigInt | Null = null): R.Philox =
    new R.Philox(seed, counter, key)
  /** `np.random.SFC64(seed)`. */
  def SFC64(seed: Any = null): R.SFC64 = new R.SFC64(seed)

  /** `np.random.default_rng(seed)`: a `Generator(PCG64(seed))`. `seed` may be `null` (OS entropy),
    * a non-negative integer, a sequence of integers, a `SeedSequence`, a `BitGenerator` (wrapped),
    * a `Generator` (returned as is) or a `RandomState` (its bit generator is wrapped).
    */
  def default_rng(seed: Any = null): R.Generator = seed match
    case g: R.Generator => g
    case b: R.BitGenerator => new R.Generator(b)
    case rs: R.RandomState => new R.Generator(rs.bit_generator)
    case s => new R.Generator(new R.PCG64(s))

  private val _rand: R.RandomState = new R.RandomState(null: Any)

  /** `np.random.seed(seed)`: re-seeds the global `RandomState`. */
  def seed(seed: Any = null): Unit =
    _rand.bit_generator match
      case _: R.MT19937 => _rand.seed(seed)
      case b =>
        val fresh = b match
          case _: R.PCG64 => new R.PCG64(seed)
          case _: R.PCG64DXSM => new R.PCG64DXSM(seed)
          case _: R.SFC64 => new R.SFC64(seed)
          case _: R.Philox => new R.Philox(seed)
          case other => throw new UnsupportedOperationException(s"cannot reseed ${other.name}")
        b.state = fresh.state

  /** `np.random.get_bit_generator()`. */
  def get_bit_generator(): R.BitGenerator = _rand.bit_generator
  /** `np.random.set_bit_generator(bitgen)`: swaps the global `RandomState`'s bit generator. */
  def set_bit_generator(bitgen: R.BitGenerator): Unit = _rand.initializeBitGenerator(bitgen)

  export _rand.{seed as _, bit_generator as _, *}
