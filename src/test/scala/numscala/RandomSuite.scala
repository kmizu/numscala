package numscala

import numscala.random.*

/** Bit-for-bit comparisons against NumPy 2.4 (see [[RandomRefData]]). */
class RandomSuite extends munit.FunSuite:
  private def R(s: Long = 42L) = np.random.default_rng(s)

  private def flat(x: Any): Seq[Any] = x match
    case a: NDArray[?] => a.toArray.toSeq.flatMap(flat)
    case s: Seq[?] => s.flatMap(flat)
    case a: Array[?] => a.toSeq.flatMap(flat)
    case i: Int => Seq(i.toLong)
    case s: Short => Seq(s.toLong)
    case b: Byte => Seq(b.toLong)
    case f: Float => Seq(f.toDouble)
    case other => Seq(other)

  private def legacy(seed: Long)(f: => Any): Any =
    RandomTestLock.synchronized {
      np.random.seed(seed)
      f
    }

  /** Cases whose computation involves libm transcendental functions: compared to 1e-13 relative
    * (the JVM's exp/log/pow may differ from glibc in the last bit); all others must match exactly.
    */
  private val cases: Seq[(String, () => Any)] = Seq(
    "pcg.random" -> (() => R().random(5)),
    "pcg.raw" -> (() => np.random.PCG64(42).random_raw(3)),
    "pcg.integers" -> (() => R().integers(0, 10, 10)),
    "pcg.integers64" -> (() => R().integers(-(1L << 40), 1L << 40, 5)),
    "pcg.integers_end" -> (() => R().integers(1, 6, size = 8, endpoint = true)),
    "pcg.integers32" -> (() => R().integers(0, 1000, 6, DType.Int32, false)),
    "pcg.integers16" -> (() => R().integers(-5, 300, 6, DType.Int16, false)),
    "pcg.integers8" -> (() => R().integers(-5, 100, 9, DType.Int8, false)),
    "pcg.integersbool" -> (() => R().integers(0, 2, 40, DType.Bool, false)),
    "pcg.normal" -> (() => R().standard_normal(8)),
    "pcg.normal_big" -> (() => R().standard_normal(2000).toSeq.takeRight(5)),
    "pcg.exp" -> (() => R().standard_exponential(8)),
    "pcg.exp_big" -> (() => R().standard_exponential(2000).toSeq.takeRight(5)),
    "pcg.gamma05" -> (() => R().standard_gamma(0.5, 6)),
    "pcg.gamma3" -> (() => R().standard_gamma(3.0, 6)),
    "pcg.beta" -> (() => R().beta(2.0, 3.0, 5)),
    "pcg.beta_small" -> (() => R().beta(0.5, 0.7, 5)),
    "pcg.binom_inv" -> (() => R().binomial(10, 0.3, 8)),
    "pcg.binom_btpe" -> (() => R().binomial(1000, 0.4, 8)),
    "pcg.binom_hi" -> (() => R().binomial(100, 0.9, 8)),
    "pcg.poisson_small" -> (() => R().poisson(3.0, 8)),
    "pcg.poisson_big" -> (() => R().poisson(50.0, 8)),
    "pcg.geometric" -> (() => R().geometric(0.5, 8)),
    "pcg.geometric_small" -> (() => R().geometric(0.05, 8)),
    "pcg.hyper_small" -> (() => R().hypergeometric(5, 7, 4, 8)),
    "pcg.hyper_big" -> (() => R().hypergeometric(100, 200, 50, 8)),
    "pcg.zipf" -> (() => R().zipf(2.0, 8)),
    "pcg.logseries" -> (() => R().logseries(0.6, 8)),
    "pcg.negbin" -> (() => R().negative_binomial(5, 0.5, 8)),
    "pcg.choice" -> (() => R().choice(10, 5)),
    "pcg.choice_norep" -> (() => R().choice(10, 5, replace = false)),
    "pcg.choice_p" -> (() => R().choice(5, 6, p = Seq(0.1, 0.2, 0.3, 0.2, 0.2))),
    "pcg.choice_norep_p" -> (() => R().choice(5, 3, replace = false, p = Seq(0.1, 0.2, 0.3, 0.2, 0.2))),
    "pcg.choice_big_norep" -> (() => R().choice(20000, 5, replace = false)),
    "pcg.choice_big_norep2" -> (() => R().choice(20000, 1000, replace = false).toSeq.take(5)),
    "pcg.permutation" -> (() => R().permutation(10)),
    "pcg.uniform" -> (() => R().uniform(-1, 3, 4)),
    "pcg.laplace" -> (() => R().laplace(1.0, 2.0, 4)),
    "pcg.gumbel" -> (() => R().gumbel(1.0, 2.0, 4)),
    "pcg.logistic" -> (() => R().logistic(1.0, 2.0, 4)),
    "pcg.lognormal" -> (() => R().lognormal(0.5, 0.3, 4)),
    "pcg.rayleigh" -> (() => R().rayleigh(2.0, 4)),
    "pcg.wald" -> (() => R().wald(1.5, 2.0, 4)),
    "pcg.vonmises" -> (() => R().vonmises(0.5, 2.0, 4)),
    "pcg.triangular" -> (() => R().triangular(0, 1, 3, 4)),
    "pcg.pareto" -> (() => R().pareto(3.0, 4)),
    "pcg.weibull" -> (() => R().weibull(1.5, 4)),
    "pcg.power" -> (() => R().power(2.5, 4)),
    "pcg.chisq" -> (() => R().chisquare(3.0, 4)),
    "pcg.ncchisq" -> (() => R().noncentral_chisquare(3.0, 2.0, 4)),
    "pcg.ncchisq_small" -> (() => R().noncentral_chisquare(0.5, 2.0, 4)),
    "pcg.f" -> (() => R().f(3.0, 5.0, 4)),
    "pcg.ncf" -> (() => R().noncentral_f(3.0, 5.0, 1.5, 4)),
    "pcg.t" -> (() => R().standard_t(4.0, 4)),
    "pcg.cauchy" -> (() => R().standard_cauchy(4)),
    "pcg.dirichlet" -> (() => R().dirichlet(Seq(1.0, 2.0, 3.0), 2)),
    "pcg.dirichlet_small" -> (() => R().dirichlet(Seq(0.05, 0.02, 0.08), 2)),
    "pcg.multinomial" -> (() => R().multinomial(20, Seq(0.2, 0.3, 0.5), 3)),
    "pcg.mvhg_marg" -> (() => R().multivariate_hypergeometric(Seq(5, 10, 15), 8, 3)),
    "pcg.mvhg_count" -> (() => R().multivariate_hypergeometric(Seq(5, 10, 15), 8, 3, method = "count")),
    "pcg.normal_bcast" -> (() => R().normal(Seq(0.0, 10.0, 100.0), Seq(Seq(1.0), Seq(2.0)))),
    "pcg.float32" -> (() => R().random(4, DType.Float32)),
    "pcg.normal32" -> (() => R().standard_normal(4, DType.Float32)),
    "pcg.exp32" -> (() => R().standard_exponential(4, DType.Float32, "zig")),
    "pcg.gamma32" -> (() => R().standard_gamma(2.5, 4, DType.Float32)),
    "pcg.exp_inv" -> (() => R().standard_exponential(4, DType.Float64, "inv")),
    "pcg.shuffle2d" -> (() => { val a = np.arange(12).reshape(4, 3); R().shuffle(a); a }),
    "pcg.permuted" -> (() => R().permuted(np.arange(12).reshape(3, 4), axis = 1)),
    "pcg.permuted_none" -> (() => R().permuted(np.arange(12).reshape(3, 4))),
    "pcg.bytes" -> (() => R().bytes(7).map(b => (b & 0xff).toLong).toSeq),
    "pcg.mvn_chol" -> (() =>
      R().multivariate_normal(Seq(1.0, 2.0), Seq(Seq(2.0, 0.5), Seq(0.5, 1.0)), 3, method = "cholesky")),
    "pcg.mvn_diag" -> (() => R().multivariate_normal(Seq(1.0, 2.0), Seq(Seq(1.0, 0.0), Seq(0.0, 4.0)), 3)),
    "pcg.spawn" -> (() => R().spawn(2)(1).random(3)),
    "pcg.seq" -> (() => np.random.default_rng(Seq(1, 2, 3)).random(3)),
    "pcg.big_seed" -> (() => np.random.default_rng((BigInt(1) << 70) + 12345).random(3)),
    "pcg.zero_seed" -> (() => np.random.default_rng(0).random(3)),
    "pcg.jumped" -> (() => np.random.Generator(np.random.PCG64(42).jumped()).random(3)),
    "pcg.advance" -> (() => np.random.Generator(np.random.PCG64(42).advance(1000)).random(3)),
    "dxsm.random" -> (() => np.random.Generator(np.random.PCG64DXSM(42)).random(4)),
    "dxsm.jumped" -> (() => np.random.Generator(np.random.PCG64DXSM(42).jumped(2)).random(3)),
    "mt.gen_random" -> (() => np.random.Generator(np.random.MT19937(42)).random(4)),
    "mt.gen_normal" -> (() => np.random.Generator(np.random.MT19937(42)).standard_normal(4)),
    "mt.gen_ints" -> (() => np.random.Generator(np.random.MT19937(42)).integers(0, 100, 6)),
    "mt.jumped" -> (() => np.random.Generator(np.random.MT19937(42).jumped()).random(3)),
    "philox.random" -> (() => np.random.Generator(np.random.Philox(42)).random(6)),
    "philox.ints" -> (() => np.random.Generator(np.random.Philox(42)).integers(0, 100, 6)),
    "philox.jumped" -> (() => np.random.Generator(np.random.Philox(42).jumped()).random(3)),
    "philox.counter_key" -> (() =>
      np.random.Generator(np.random.Philox(counter = BigInt(5), key = BigInt(123456789))).random(3)),
    "sfc.random" -> (() => np.random.Generator(np.random.SFC64(42)).random(4)),
    "ss.state" -> (() => np.random.SeedSequence(12345).generate_state(6)),
    "ss.state64" -> (() => np.random.SeedSequence(12345).generate_state(3, "uint64").map(_.longValue)),
    "ss.spawn" -> (() => np.random.SeedSequence(12345).spawn(3)(2).generate_state(4)),
    "ss.big" -> (() => np.random.SeedSequence(Seq(1L << 40, 7L, 1L << 33)).generate_state(4)),
    // ---- legacy (global RandomState) ----
    "leg.rand" -> (() => legacy(0)(np.random.rand(5))),
    "leg.randn" -> (() => legacy(0)(np.random.randn(5))),
    "leg.randint" -> (() => legacy(0)(np.random.randint(0, 10, 10))),
    "leg.randint_big" -> (() => legacy(0)(np.random.randint(-(1L << 40), 1L << 40, 4))),
    "leg.randint32" -> (() => legacy(0)(np.random.randint(0, 1000, 5, DType.Int32))),
    "leg.randint8" -> (() => legacy(0)(np.random.randint(-3, 50, 7, DType.Int8))),
    "leg.normal" -> (() => legacy(1)(np.random.normal(5.0, 2.0, 5))),
    "leg.uniform" -> (() => legacy(1)(np.random.uniform(2.0, 3.0, 3))),
    "leg.exp" -> (() => legacy(1)(np.random.exponential(2.0, 4))),
    "leg.gamma" -> (() => legacy(1)(np.random.gamma(2.5, 1.5, 4))),
    "leg.gamma_small" -> (() => legacy(1)(np.random.gamma(0.3, 1.0, 4))),
    "leg.beta" -> (() => legacy(1)(np.random.beta(2.0, 5.0, 4))),
    "leg.beta_small" -> (() => legacy(1)(np.random.beta(0.3, 0.6, 4))),
    "leg.binom" -> (() => legacy(1)(np.random.binomial(10, 0.4, 6))),
    "leg.binom_btpe" -> (() => legacy(1)(np.random.binomial(500, 0.3, 6))),
    "leg.poisson" -> (() => legacy(1)(np.random.poisson(4.0, 6))),
    "leg.poisson_big" -> (() => legacy(1)(np.random.poisson(40.0, 6))),
    "leg.geometric" -> (() => legacy(1)(np.random.geometric(0.1, 6))),
    "leg.hyper" -> (() => legacy(1)(np.random.hypergeometric(10, 15, 6, 6))),
    "leg.hyper_big" -> (() => legacy(1)(np.random.hypergeometric(100, 150, 60, 6))),
    "leg.zipf" -> (() => legacy(1)(np.random.zipf(2.5, 6))),
    "leg.logseries" -> (() => legacy(1)(np.random.logseries(0.7, 6))),
    "leg.negbin" -> (() => legacy(1)(np.random.negative_binomial(3, 0.4, 6))),
    "leg.chisq" -> (() => legacy(1)(np.random.chisquare(4.0, 4))),
    "leg.ncchisq" -> (() => legacy(1)(np.random.noncentral_chisquare(4.0, 1.0, 4))),
    "leg.f" -> (() => legacy(1)(np.random.f(4.0, 6.0, 4))),
    "leg.ncf" -> (() => legacy(1)(np.random.noncentral_f(4.0, 6.0, 1.0, 4))),
    "leg.t" -> (() => legacy(1)(np.random.standard_t(5.0, 4))),
    "leg.cauchy" -> (() => legacy(1)(np.random.standard_cauchy(4))),
    "leg.pareto" -> (() => legacy(1)(np.random.pareto(2.0, 4))),
    "leg.weibull" -> (() => legacy(1)(np.random.weibull(2.0, 4))),
    "leg.power" -> (() => legacy(1)(np.random.power(2.0, 4))),
    "leg.laplace" -> (() => legacy(1)(np.random.laplace(0.0, 1.0, 4))),
    "leg.gumbel" -> (() => legacy(1)(np.random.gumbel(0.0, 1.0, 4))),
    "leg.logistic" -> (() => legacy(1)(np.random.logistic(0.0, 1.0, 4))),
    "leg.lognormal" -> (() => legacy(1)(np.random.lognormal(0.0, 1.0, 4))),
    "leg.rayleigh" -> (() => legacy(1)(np.random.rayleigh(1.0, 4))),
    "leg.wald" -> (() => legacy(1)(np.random.wald(1.0, 1.0, 4))),
    "leg.vonmises" -> (() => legacy(1)(np.random.vonmises(0.0, 4.0, 4))),
    "leg.triangular" -> (() => legacy(1)(np.random.triangular(0.0, 0.5, 1.0, 4))),
    "leg.choice" -> (() => legacy(1)(np.random.choice(10, 5))),
    "leg.choice_norep" -> (() => legacy(1)(np.random.choice(10, 5, replace = false))),
    "leg.choice_p" -> (() => legacy(1)(np.random.choice(4, 5, p = Seq(0.1, 0.2, 0.3, 0.4)))),
    "leg.permutation" -> (() => legacy(1)(np.random.permutation(10))),
    "leg.shuffle" -> (() => legacy(1) { val a = np.arange(10); np.random.shuffle(a); a }),
    "leg.multinomial" -> (() => legacy(1)(np.random.multinomial(20, Seq(0.25, 0.25, 0.5), 2))),
    "leg.dirichlet" -> (() => legacy(1)(np.random.dirichlet(Seq(1.0, 2.0, 3.0), 2))),
    "leg.std_exp" -> (() => legacy(1)(np.random.standard_exponential(4))),
    "leg.std_gamma" -> (() => legacy(1)(np.random.standard_gamma(2.0, 4))),
    "leg.random_integers" -> (() => legacy(1)(np.random.random_integers(1, 6, 6))),
    "leg.tomaxint" -> (() => np.random.RandomState(1).tomaxint(3)),
    "leg.arr_seed" -> (() => np.random.RandomState(Seq(1, 2, 3)).rand(3)),
    "leg.mvn" -> (() =>
      legacy(1)(np.random.multivariate_normal(Seq(0.0, 1.0), Seq(Seq(1.0, 0.0), Seq(0.0, 9.0)), 2))),
    "leg.bytes" -> (() => legacy(1)(np.random.bytes(6).map(b => (b & 0xff).toLong).toSeq))
  )

  private def close(a: Double, b: Double): Boolean =
    a == b || (a.isNaN && b.isNaN) || math.abs(a - b) <= 1e-13 * math.max(math.abs(a), math.abs(b))

  for (name, f) <- cases do
    test(s"numpy reference: $name") {
      val expected = RandomRefData.data(name)
      val got = flat(f())
      assertEquals(got.length, expected.length, s"length for $name")
      val exact = got.zip(expected).forall { case (g, e) => g == e }
      if !exact then
        val near = got.zip(expected).forall {
          case (g: Double, e: Double) => close(g, e)
          case (g, e) => g == e
        }
        assert(near, s"$name\n  got:      $got\n  expected: $expected")
    }

  test("all reference cases are exercised") {
    assertEquals(cases.map(_._1).toSet, RandomRefData.data.keySet)
  }
