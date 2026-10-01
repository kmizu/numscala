package numscala

import numscala.random.*

/** API-level behaviour of `np.random`: broadcasting, dtypes, scalar overloads, state handling,
  * error cases and statistical sanity.
  */
class RandomApiSuite extends munit.FunSuite:
  private def R(s: Long = 42L) = np.random.default_rng(s)

  private def flat(x: Any): Seq[Any] = x match
    case a: NDArray[?] => a.toArray.toSeq.flatMap(flat)
    case s: Seq[?] => s.flatMap(flat)
    case a: Array[?] => a.toSeq.flatMap(flat)
    case t: Tuple => t.productIterator.toSeq.flatMap(flat)
    case i: Int => Seq(i.toLong)
    case s: Short => Seq(s.toLong)
    case b: Byte => Seq(b.toLong)
    case f: Float => Seq(f.toDouble)
    case other => Seq(other)

  private def legacy(seed: Any)(f: => Any): Any =
    RandomTestLock.synchronized {
      np.random.seed(seed)
      f
    }

  private val cases: Seq[(String, () => Any)] = Seq(
    "api.int_scalar" -> (() => Seq(R().integers(5))),
    "api.int_bcast" -> (() => R().integers(Seq(1, 2, 3), Seq(10, 20, 30))),
    "api.int_mixed" -> (() => R().integers(0, Seq(3L, 300L, 70000L, 1L << 40, 1L << 62))),
    "api.int_bcast_size" -> (() => R().integers(Seq(0, 100), 200, size = Seq(3, 2))),
    "api.int_full" -> (() => R().integers(Long.MinValue, Long.MaxValue, size = 4, endpoint = true)),
    "api.int_bool_bcast" -> (() => R().integers(0, Seq(1, 2, 2, 2), null, DType.Bool, false)),
    "api.int8_bcast" -> (() => R().integers(Seq(-128, 0), Seq(127, 10), null, DType.Int8, false)),
    "api.choice_axis1" -> (() => R().choice(np.arange(12.0).reshape(3, 4), 2, axis = 1)),
    "api.choice_2d_size" -> (() => R().choice(np.arange(5) * 10, Seq(2, 3))),
    "api.choice_noshuffle" -> (() => R().choice(30, 6, replace = false, shuffle = false)),
    "api.choice_bigshuffle" -> (() => R().choice(20000, 600, replace = false, shuffle = false).toSeq.take(6)),
    "api.binom_bcast" -> (() => R().binomial(Seq(10, 20), Seq(0.5))),
    "api.binom_size" -> (() => R().binomial(Seq(5, 50), Seq(0.2, 0.7), size = Seq(2, 2))),
    "api.poisson_bcast" -> (() => R().poisson(Seq(1.0, 100.0), size = Seq(3, 2))),
    "api.hyper_bcast" -> (() => R().hypergeometric(Seq(5, 10), Seq(5, 10), Seq(3, 4))),
    "api.multinomial_vec" -> (() => R().multinomial(Seq(10, 20), Seq(Seq(0.5, 0.5), Seq(0.2, 0.8)))),
    "api.multinomial_size" -> (() => R().multinomial(10, Seq(0.3, 0.7), size = Seq(2, 2))),
    "api.uniform_arr" -> (() => R().uniform(Seq(0.0, 10.0), Seq(1.0, 20.0))),
    "api.normal_scalar" -> (() => Seq(R().normal())),
    "api.normal_scalar2" -> (() => Seq(R().normal(3.0, 2.0))),
    "api.gamma_bcast" -> (() => R().gamma(Seq(0.5, 2.0, 5.0), 2.0)),
    "api.zipf_big" -> (() => R().zipf(1.5, 10)),
    "api.vonmises_big" -> (() => R().vonmises(1.0, 2e6, 3)),
    "api.vonmises_small" -> (() => R().vonmises(1.0, 1e-6, 3)),
    "api.logseries_hi" -> (() => R().logseries(0.99, 10)),
    "api.geometric_one" -> (() => R().geometric(1.0, 3)),
    "api.binom_zero" -> (() => R().binomial(0, 0.5, 3)),
    "api.binom_p1" -> (() => R().binomial(7, 1.0, 3)),
    "api.mvhg_big" -> (() => R().multivariate_hypergeometric(Seq(50, 60, 70), 100, 2)),
    "api.mvhg_count_big" -> (() => R().multivariate_hypergeometric(Seq(5, 6, 7), 12, 2, method = "count")),
    "api.triangular_arr" -> (() => R().triangular(Seq(0.0, 1.0), Seq(0.5, 1.5), Seq(1.0, 3.0))),
    "api.exponential_arr" -> (() => R().exponential(Seq(1.0, 2.0, 3.0))),
    "api.negbin_arr" -> (() => R().negative_binomial(Seq(1, 10), Seq(0.3, 0.6))),
    "api.dirichlet_mixed" -> (() => R().dirichlet(Seq(0.05, 0.0, 0.02), 2)),
    "api.random_shape" -> (() => R().random(Seq(2, 3))),
    "api.permutation_2d" -> (() => R().permutation(np.arange(9).reshape(3, 3))),
    "api.permutation_axis" -> (() => R().permutation(np.arange(9).reshape(3, 3), axis = 1)),
    "api.mvn_eigh" -> (() =>
      R().multivariate_normal(Seq(0.0, 0.0), Seq(Seq(4.0, 0.0), Seq(0.0, 1.0)), 2, method = "eigh")),
    "api.spawn0" -> (() => R().spawn(1).head.integers(0, 1000, 4)),
    "api.ss_spawn_rng" -> (() => np.random.default_rng(np.random.SeedSequence(7).spawn(2).head).random(2)),
    "api.philox_adv" -> (() => np.random.Generator(np.random.Philox(1).advance((BigInt(1) << 70) + 5)).random(2)),
    "api.sfc_ints" -> (() => np.random.Generator(np.random.SFC64(3)).integers(0, 7, 5)),
    "api.mt_jump2" -> (() => np.random.Generator(np.random.MT19937(3).jumped(2)).integers(0, 1000, 3)),
    "api.mt_after_gen" -> (() => { val b = np.random.MT19937(5); b.random_raw(700); np.random.Generator(b).random(2) }),
    "api.random_raw_mt" -> (() => np.random.MT19937(5).random_raw(3)),
    "api.uint32_buffer" -> (() => { val g = R(); g.integers(0, 10, 3, DType.Int32, false); g.random(1) }),
    "api.int16_buffer" -> (() => {
      val g = R(); g.integers(0, 10, 3, DType.Int16, false); g.integers(0, 10, 2, DType.Int16, false)
    }),
    "leg.rand_scalar" -> (() => legacy(3)(Seq(np.random.rand()))),
    "leg.randn_scalar" -> (() => legacy(3)(Seq(np.random.randn(), np.random.randn(), np.random.randn()))),
    "leg.randint_bool" -> (() => legacy(3)(np.random.randint(0, 2, 10, DType.Bool))),
    "leg.randint_bcast" -> (() => legacy(3)(np.random.randint(Seq(0, 10), Seq(5, 1000)))),
    "leg.randint16" -> (() => legacy(3)(np.random.randint(-100, 100, 5, DType.Int16))),
    "leg.state_gauss" -> (() => legacy(3) { np.random.randn(); val s = np.random.get_state(); Seq(s._4.toDouble, s._5) }),
    "leg.state_pos" -> (() => legacy(3) { np.random.rand(10); Seq(np.random.get_state()._3) }),
    "leg.state_key" -> (() => legacy(3)(np.random.get_state()._2.toSeq.take(4))),
    "leg.seed_arr" -> (() => legacy(Seq(5, 6, 7))(np.random.rand(3))),
    "leg.seed_big" -> (() => legacy(4294967295L)(np.random.rand(2))),
    "leg.hyper_small" -> (() => legacy(2)(np.random.hypergeometric(3, 4, 5, 6))),
    "leg.binom_hi" -> (() => legacy(2)(np.random.binomial(20, 0.8, 6))),
    "leg.vonmises_small" -> (() => legacy(2)(np.random.vonmises(0.0, 1e-9, 3))),
    "leg.choice_arr" -> (() => legacy(2)(np.random.choice(np.array(1.5, 2.5, 3.5), 4))),
    "leg.choice_norep_p" -> (() =>
      legacy(2)(np.random.choice(5, 3, replace = false, p = Seq(0.1, 0.2, 0.3, 0.2, 0.2)))),
    "leg.normal_bcast" -> (() => legacy(2)(np.random.normal(Seq(0.0, 1.0), Seq(Seq(1.0), Seq(2.0))))),
    "leg.uniform_bcast" -> (() => legacy(2)(np.random.uniform(Seq(0.0, 10.0), 20.0, size = Seq(2, 2)))),
    "leg.permutation_2d" -> (() => legacy(2)(np.random.permutation(np.arange(6).reshape(3, 2)))),
    "leg.zipf_hi" -> (() => legacy(2)(np.random.zipf(1.2, 5))),
    "leg.logser_hi" -> (() => legacy(2)(np.random.logseries(0.95, 6))),
    "leg.ncchisq_small" -> (() => legacy(2)(np.random.noncentral_chisquare(0.8, 3.0, 4))),
    "leg.rs_pcg" -> (() => np.random.RandomState(np.random.PCG64(9)).standard_normal(3))
  )

  private def close(a: Double, b: Double): Boolean =
    a == b || (a.isNaN && b.isNaN) || math.abs(a - b) <= 1e-13 * math.max(math.abs(a), math.abs(b))

  for (name, f) <- cases do
    test(s"numpy reference: $name") {
      val expected = RandomRefData.api(name)
      val got = flat(f())
      assertEquals(got.length, expected.length, s"length for $name")
      val ok = got.zip(expected).forall {
        case (g: Double, e: Double) => close(g, e)
        case (g, e) => g == e
      }
      assert(ok, s"$name\n  got:      $got\n  expected: $expected")
    }

  test("all api reference cases are exercised") {
    assertEquals(cases.map(_._1).toSet, RandomRefData.api.keySet)
  }

  // ------------------------------------------------------------------ shapes & types
  test("scalar vs array overloads and shapes") {
    val g = R()
    val d: Double = g.random()
    assert(d >= 0 && d < 1)
    val a: NDArray[Double] = g.random(2, 3)
    assertEquals(a.shape, Seq(2, 3))
    assertEquals(g.standard_normal(Seq(4, 1)).shape, Seq(4, 1))
    val n: Double = g.normal(1.0, 2.0)
    assert(!n.isNaN)
    assertEquals(g.normal(1.0, 2.0, 5).shape, Seq(5))
    assertEquals(g.normal(size = Seq(2, 2)).shape, Seq(2, 2))
    assertEquals(g.normal(np.zeros(3), 1.0).shape, Seq(3))
    val i: Long = g.integers(10)
    assert(i >= 0 && i < 10)
    val p: Long = g.poisson(4.0)
    assert(p >= 0)
    assertEquals(g.poisson(size = 7).shape, Seq(7))
    val c: Long = g.choice(5)
    assert(c >= 0 && c < 5)
    val ce: Double = g.choice(np.array(1.0, 2.0, 3.0))
    assert(Set(1.0, 2.0, 3.0).contains(ce))
    val ci: NDArray[Long] = g.choice(7, 3)
    assertEquals(ci.shape, Seq(3))
    val ca: NDArray[Double] = g.choice(np.array(1.0, 2.0), size = Seq(2, 2))
    assertEquals(ca.shape, Seq(2, 2))
    assertEquals(g.integers(0, 10, size = 0).shape, Seq(0))
    assertEquals(g.integers(0, 5, 3, DType.Int32, false).dtype.name, "int32")
    assertEquals(g.random(3, DType.Float32).dtype.name, "float32")
    assertEquals(g.multinomial(5, Seq(0.5, 0.5)).shape, Seq(2))
    assertEquals(g.dirichlet(Seq(1.0, 1.0), Seq(4)).shape, Seq(4, 2))
    assertEquals(g.multivariate_normal(Seq(0.0, 0.0), Seq(Seq(1.0, 0.0), Seq(0.0, 1.0)), Seq(3, 2)).shape, Seq(3, 2, 2))
    assertEquals(g.multivariate_normal(Seq(0.0, 0.0), Seq(Seq(1.0, 0.0), Seq(0.0, 1.0))).shape, Seq(2))
    assertEquals(g.uniform(0.0, 1.0, 2, 2).shape, Seq(2, 2))
    val zero = g.normal(0.0, 1.0, size = Seq.empty[Int])
    assertEquals(zero.ndim, 0)
  }

  test("module-level legacy functions") { RandomTestLock.synchronized {
    np.random.seed(0)
    assertEqualsDouble(np.random.random(), 0.5488135039273248, 0.0)
    assertEqualsDouble(np.random.random_sample(), 0.7151893663724195, 0.0)
    assertEqualsDouble(np.random.ranf(), 0.6027633760716439, 0.0)
    np.random.seed(0)
    assertEquals(np.random.sample(3).toList, List(0.5488135039273248, 0.7151893663724195, 0.6027633760716439))
    np.random.seed(0)
    assertEqualsDouble(np.random.randn(), 1.764052345967664, 0.0)
    assertEqualsDouble(np.random.standard_normal(), 0.4001572083672233, 0.0)
    assertEqualsDouble(np.random.normal(), 0.9787379841057392, 0.0)
    np.random.seed(0)
    assertEquals(np.random.randn(3).toList, List(1.764052345967664, 0.4001572083672233, 0.9787379841057392))
    val r: Long = np.random.randint(10)
    assert(r >= 0 && r < 10)
    assert(np.random.random_integers(3) >= 1)
    assertEquals(np.random.get_bit_generator().name, "MT19937")
  } }

  test("legacy state round trip") {
    val rs = np.random.RandomState(123)
    rs.randn()
    val st = rs.get_state()
    assertEquals(st._1, "MT19937")
    assertEquals(st._4, 1)
    val a = rs.randn(5).toList
    rs.set_state(st)
    assertEquals(rs.randn(5).toList, a)
    val m = rs.get_state(legacy = false).asInstanceOf[Map[String, Any]]
    assertEquals(m("bit_generator"), "MT19937")
    val b = rs.random_sample(4).toList
    rs.set_state(m)
    assertEquals(rs.random_sample(4).toList, b)
  }

  test("bit generator state round trips") {
    val gens: Seq[BitGenerator] = Seq(np.random.PCG64(1), np.random.PCG64DXSM(2), np.random.MT19937(3),
      np.random.Philox(4), np.random.SFC64(5))
    for bgen <- gens do
      bgen.nextUInt32() // leave a buffered half word
      val st = bgen.state
      val g = np.random.Generator(bgen)
      val x = g.random(5).toList
      val y = g.integers(0, 100, 5, DType.Int32, false).toList
      bgen.state = st
      assertEquals(g.random(5).toList, x, bgen.name)
      assertEquals(g.integers(0, 100, 5, DType.Int32, false).toList, y, bgen.name)
    intercept[IllegalArgumentException](np.random.PCG64(1).state = np.random.SFC64(1).state)
  }

  test("set_bit_generator swaps the global generator") { RandomTestLock.synchronized {
    val old = np.random.get_bit_generator()
    try
      np.random.set_bit_generator(np.random.PCG64(9))
      assertEquals(np.random.standard_normal(3).toList, List(-0.39576329981686187, 0.6873493216342598, 1.3570237555923859))
      np.random.seed(9)
      assertEquals(np.random.get_bit_generator().name, "PCG64")
    finally np.random.set_bit_generator(old)
  } }

  test("seed sequence details") {
    val ss = np.random.SeedSequence(12345)
    assertEquals(ss.entropy, 12345)
    assertEquals(ss.pool.length, 4)
    val kids = ss.spawn(2)
    assertEquals(kids.map(_.spawn_key), Seq(Seq(0L), Seq(1L)))
    assertEquals(ss.n_children_spawned, 2)
    assertEquals(ss.spawn(1).head.spawn_key, Seq(2L))
    assert(ss.toString.contains("entropy=12345"))
    assertEquals(np.random.SeedSequence("0x3039").generate_state(2).toList, ss.generate_state(2).toList)
    assertEquals(np.random.SeedSequence("12345").generate_state(2).toList, ss.generate_state(2).toList)
    intercept[IllegalArgumentException](np.random.SeedSequence(-1))
    intercept[IllegalArgumentException](np.random.SeedSequence(1.5))
    intercept[IllegalArgumentException](np.random.SeedSequence(1, pool_size = 2))
    val fresh = np.random.default_rng()
    assert(fresh.random() < 1.0)
    assert(fresh.bit_generator.seed_seq != null)
  }

  test("default_rng accepts generators and bit generators") {
    val g = R()
    assert(np.random.default_rng(g) eq g)
    val b = np.random.PCG64(42)
    assertEquals(np.random.default_rng(b).bit_generator, b)
    val rs = np.random.RandomState(1)
    assertEquals(np.random.default_rng(rs).bit_generator, rs.bit_generator)
    assertEquals(g.toString, "Generator(PCG64)")
  }

  // ------------------------------------------------------------------ errors
  test("parameter validation errors") {
    val g = R()
    def bad(msg: String)(f: => Any): Unit =
      val e = intercept[IllegalArgumentException](f)
      assert(e.getMessage.contains(msg), s"'${e.getMessage}' should contain '$msg'")
    bad("scale < 0")(g.normal(0.0, -1.0))
    bad("scale < 0")(g.normal(0.0, Seq(1.0, -1.0)))
    bad("a <= 0")(g.beta(0.0, 1.0))
    bad("p < 0, p > 1 or p is NaN")(g.binomial(10, 1.5))
    bad("n < 0")(g.binomial(-1, 0.5))
    bad("lam < 0")(g.poisson(-1.0))
    bad("lam value too large")(g.poisson(1e20))
    bad("a <= 1")(g.zipf(1.0))
    bad("p <= 0, p > 1")(g.geometric(0.0))
    bad("p < 0, p >= 1")(g.logseries(1.0))
    bad("ngood + nbad < nsample")(g.hypergeometric(2, 2, 5))
    bad("both ngood and nbad must be less than")(g.hypergeometric(2000000000L, 2, 5))
    bad("left > mode")(g.triangular(2.0, 1.0, 3.0))
    bad("mode > right")(g.triangular(0.0, 4.0, 3.0))
    bad("left == right")(g.triangular(1.0, 1.0, 1.0))
    bad("high <= 0")(g.integers(0))
    bad("low >= high")(g.integers(5, 5))
    bad("low > high")(g.integers(5, 4, size = 1, endpoint = true))
    bad("high is out of bounds for int8")(g.integers(0, 200, 1, DType.Int8, false))
    bad("low is out of bounds for int32")(g.integers(-(1L << 40), 0, 1, DType.Int32, false))
    bad("Cannot take a larger sample")(g.choice(3, 5, replace = false))
    bad("Probabilities do not sum to 1")(g.choice(3, 2, p = Seq(0.5, 0.6, 0.1)))
    bad("Probabilities are not non-negative")(g.choice(2, 2, p = Seq(1.5, -0.5)))
    bad("a and p must have same size")(g.choice(3, 2, p = Seq(0.5, 0.5)))
    bad("a must be a positive integer")(g.choice(0, 2))
    bad("Fewer non-zero entries in p than size")(g.choice(3, 3, replace = false, p = Seq(0.5, 0.5, 0.0)))
    bad("sum(pvals[:-1]) > 1.0")(g.multinomial(5, Seq(0.8, 0.8, 0.1)))
    bad("nsample > sum(colors)")(g.multivariate_hypergeometric(Seq(1, 2), 5))
    bad("method must be")(g.multivariate_hypergeometric(Seq(1, 2), 1, method = "x"))
    bad("alpha < 0")(g.dirichlet(Seq(1.0, -1.0)))
    bad("Output size")(g.normal(Seq(0.0, 1.0, 2.0), 1.0, size = Seq(2)))
    bad("mean and cov must have same length")(g.multivariate_normal(Seq(0.0), Seq(Seq(1.0, 0.0), Seq(0.0, 1.0))))
    bad("covariance is not symmetric positive-semidefinite")(
      g.multivariate_normal(Seq(0.0, 0.0), Seq(Seq(1.0, 2.0), Seq(2.0, 1.0)), check_valid = "raise"))
    bad("n too large or p too small")(g.negative_binomial(1e18, 1e-5))
    intercept[ArithmeticException](g.uniform(0.0, Double.PositiveInfinity))
    intercept[LinAlgError](
      g.multivariate_normal(Seq(0.0, 0.0), Seq(Seq(1.0, 2.0), Seq(2.0, 1.0)), method = "cholesky"))
    bad("Seed must be between 0 and 2**32 - 1")(np.random.RandomState(1L << 33))
    bad("Seed must be non-empty")(np.random.RandomState(Seq.empty[Int]))
    val rs = np.random.RandomState(0)
    bad("n < 0")(rs.binomial(-1, 0.5))
    bad("nsample < 1")(rs.hypergeometric(3, 3, 0))
    bad("a must be 1-dimensional")(rs.choice(np.zeros(2, 2), 1))
    bad("alpha <= 0")(rs.dirichlet(Seq(1.0, 0.0)))
    bad("pvals must be a 1-d sequence")(rs.multinomial(3, 0.5))
  }

  // ------------------------------------------------------------------ statistics
  private def meanVar(a: NDArray[?]): (Double, Double) =
    val xs = a.toArray.map {
      case d: Double => d
      case l: Long => l.toDouble
      case other => other.toString.toDouble
    }
    val m = xs.sum / xs.length
    (m, xs.map(x => (x - m) * (x - m)).sum / xs.length)

  private def checkMoments(name: String, a: NDArray[?], mean: Double, variance: Double, tol: Double = 0.05): Unit =
    val (m, v) = meanVar(a)
    assert(math.abs(m - mean) <= tol * math.max(1.0, math.abs(mean)), s"$name mean $m vs $mean")
    assert(math.abs(v - variance) <= 3 * tol * math.max(1.0, variance), s"$name var $v vs $variance")

  test("statistical sanity of the Generator distributions") {
    val g = np.random.default_rng(2024)
    val n = 200000
    checkMoments("random", g.random(n), 0.5, 1.0 / 12)
    checkMoments("normal", g.normal(2.0, 3.0, n), 2.0, 9.0)
    checkMoments("exponential", g.exponential(2.0, n), 2.0, 4.0)
    checkMoments("gamma", g.gamma(3.0, 2.0, n), 6.0, 12.0)
    checkMoments("gamma<1", g.gamma(0.4, 1.0, n), 0.4, 0.4)
    checkMoments("beta", g.beta(2.0, 3.0, n), 0.4, 0.04)
    checkMoments("chisquare", g.chisquare(5.0, n), 5.0, 10.0)
    checkMoments("binomial", g.binomial(100, 0.3, n), 30.0, 21.0)
    checkMoments("poisson", g.poisson(7.5, n), 7.5, 7.5)
    checkMoments("poisson big", g.poisson(150.0, n), 150.0, 150.0)
    checkMoments("geometric", g.geometric(0.2, n), 5.0, 20.0)
    checkMoments("negative_binomial", g.negative_binomial(5, 0.5, n), 5.0, 10.0)
    checkMoments("hypergeometric", g.hypergeometric(30, 70, 20, n), 6.0, 20 * 0.3 * 0.7 * 80.0 / 99)
    checkMoments("laplace", g.laplace(1.0, 2.0, n), 1.0, 8.0)
    checkMoments("logistic", g.logistic(0.0, 1.0, n), 0.0, math.Pi * math.Pi / 3)
    checkMoments("gumbel", g.gumbel(0.0, 1.0, n), 0.5772156649, math.Pi * math.Pi / 6)
    checkMoments("rayleigh", g.rayleigh(1.0, n), math.sqrt(math.Pi / 2), (4 - math.Pi) / 2)
    checkMoments("weibull", g.weibull(1.0, n), 1.0, 1.0)
    checkMoments("pareto", g.pareto(5.0, n), 0.25, 5.0 / (16 * 3))
    checkMoments("power", g.power(3.0, n), 0.75, 3.0 / (16 * 5))
    checkMoments("triangular", g.triangular(0.0, 1.0, 3.0, n), 4.0 / 3, (9 + 1 - 3) / 18.0)
    checkMoments("uniform", g.uniform(-2.0, 2.0, n), 0.0, 16.0 / 12)
    checkMoments("wald", g.wald(2.0, 4.0, n), 2.0, 2.0)
    checkMoments("lognormal", g.lognormal(0.0, 0.5, n), math.exp(0.125), (math.exp(0.25) - 1) * math.exp(0.25))
    checkMoments("standard_t", g.standard_t(10.0, n), 0.0, 1.25)
    checkMoments("f", g.f(5.0, 20.0, n), 20.0 / 18, 2 * 400.0 * 23 / (5 * 18 * 18 * 16))
    checkMoments("noncentral_chisquare", g.noncentral_chisquare(3.0, 2.0, n), 5.0, 14.0)
    checkMoments("noncentral_f", g.noncentral_f(5.0, 20.0, 1.0, n), 20.0 * 6 / (5 * 18), 1.0, tol = 0.1)
    checkMoments("vonmises", g.vonmises(0.5, 4.0, n), 0.5, 0.2934, tol = 0.05)
    checkMoments("logseries", g.logseries(0.5, n), 0.5 / (0.5 * math.log(2)), 0.5 / (0.25 * math.log(2)) - math.pow(1 / math.log(2), 2))
    checkMoments("integers", g.integers(0, 10, n), 4.5, 8.25)
    checkMoments("zipf", g.zipf(4.0, n), 1.0823232 / 1.0, 0.25, tol = 0.1)
    checkMoments("standard_normal f32", g.standard_normal(n, DType.Float32).astypeOf(DType.Float64), 0.0, 1.0)
    checkMoments("standard_exponential f32", g.standard_exponential(n, DType.Float32, "zig").astypeOf(DType.Float64), 1.0, 1.0)
    checkMoments("standard_gamma f32", g.standard_gamma(0.5, n, DType.Float32).astypeOf(DType.Float64), 0.5, 0.5)
    val mv = g.multivariate_normal(Seq(1.0, -1.0), Seq(Seq(2.0, 0.6), Seq(0.6, 1.0)), 50000)
    val xs = mv(::, 0).toArray; val ys = mv(::, 1).toArray
    val mx = xs.sum / xs.length; val my = ys.sum / ys.length
    val cov = xs.indices.map(i => (xs(i) - mx) * (ys(i) - my)).sum / xs.length
    assertEqualsDouble(cov, 0.6, 0.05)
    val d = g.dirichlet(Seq(1.0, 2.0, 3.0), 20000)
    assertEqualsDouble(d(::, 2).toArray.sum / 20000, 0.5, 0.01)
    val mn = g.multinomial(100, Seq(0.2, 0.3, 0.5), 10000)
    assertEqualsDouble(mn(::, 1).toArray.map(_.toDouble).sum / 10000, 30.0, 0.5)
    assert(mn.sum(axis = 1).toArray.forall(_ == 100L))
  }

  test("legacy distributions sanity") {
    val rs = np.random.RandomState(7)
    val n = 100000
    checkMoments("legacy normal", rs.normal(1.0, 2.0, n), 1.0, 4.0)
    checkMoments("legacy gamma", rs.gamma(2.0, 3.0, n), 6.0, 18.0)
    checkMoments("legacy beta", rs.beta(0.5, 0.5, n), 0.5, 0.125)
    checkMoments("legacy binomial", rs.binomial(40, 0.6, n), 24.0, 9.6)
    checkMoments("legacy hypergeometric", rs.hypergeometric(50, 50, 30, n), 15.0, 30 * 0.25 * 70 / 99.0)
    checkMoments("legacy geometric", rs.geometric(0.1, n), 10.0, 90.0, tol = 0.1)
    checkMoments("legacy wald", rs.wald(1.0, 3.0, n), 1.0, 1.0 / 3)
    checkMoments("legacy exponential", rs.standard_exponential(n), 1.0, 1.0)
  }

  // ------------------------------------------------------------------ permutations
  test("shuffle, permutation and permuted are permutations") {
    val g = R(5)
    val a = np.arange(20.0)
    g.shuffle(a)
    assertEquals(a.toArray.sorted.toList, (0 until 20).map(_.toDouble).toList)
    val m = np.arange(12).reshape(3, 4)
    g.shuffle(m, axis = 1)
    assertEquals(m.toArray.toList.grouped(4).map(_.sorted).toList.flatten.sorted, (0 until 12).toList)
    assertEquals(m(::, 0).toArray.map(_ % 4).distinct.length, 1)
    val p = g.permutation(np.arange(10L))
    assertEquals(p.toArray.sorted.toList, (0L until 10L).toList)
    val pm = g.permuted(np.arange(12).reshape(3, 4), axis = 0)
    for j <- 0 until 4 do assertEquals(pm(::, j).toArray.sorted.toList, List(j, j + 4, j + 8))
    val buf = scala.collection.mutable.ArrayBuffer(1, 2, 3, 4, 5)
    g.shuffle(buf)
    assertEquals(buf.sorted.toList, List(1, 2, 3, 4, 5))
    val strided = np.arange(10.0)("::2")
    g.shuffle(strided)
    assertEquals(strided.toArray.sorted.toList, List(0.0, 2.0, 4.0, 6.0, 8.0))
    val e = np.zeros(0)
    g.shuffle(e)
    assertEquals(g.permutation(0).size, 0)
    val noRep = g.choice(1000, 1000, replace = false)
    assertEquals(noRep.toArray.distinct.length, 1000)
  }

  test("bytes and raw draws") {
    val g = R()
    assertEquals(g.bytes(0).length, 0)
    assertEquals(g.bytes(9).length, 9)
    val raw = np.random.PCG64(42).random_raw()
    assertEquals(raw, -4169774921698171256L)
    assertEquals(np.random.PCG64(42).random_raw(2, 2).shape, Seq(2, 2))
  }

  test("RandomState extras") {
    val rs = np.random.RandomState(0)
    assertEquals(rs.rand(2, 3).shape, Seq(2, 3))
    assertEquals(rs.randn(2, 2).shape, Seq(2, 2))
    assert(rs.tomaxint() >= 0)
    val r: Long = rs.random_integers(1, 6)
    assert(r >= 1 && r <= 6)
    assertEquals(rs.random_integers(1, 6, 10).toArray.forall(x => x >= 1 && x <= 6), true)
    rs.seed(0)
    assertEqualsDouble(rs.rand(), 0.5488135039273248, 0.0)
    assertEquals(rs.toString, "RandomState(MT19937)")
    assertEquals(rs.randint(0, 5, 3, DType.Int16).dtype.name, "int16")
    val c: Long = rs.choice(3)
    assert(c < 3)
    assertEquals(rs.choice(np.array(1.0, 2.0), 3).shape, Seq(3))
    intercept[UnsupportedOperationException](np.random.RandomState(np.random.PCG64(1)).seed(3))
  }

/** Serialises tests that use the global `np.random` state (suites run in parallel). */
object RandomTestLock
