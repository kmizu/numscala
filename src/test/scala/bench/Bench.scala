package bench

import numscala.*

/** Micro benchmarks: `sbt "Test/runMain bench.Bench"`. */
object Bench:
  def time[A](label: String, reps: Int = 5)(f: => A): Unit =
    var w = 0
    while w < 10 do { f; w += 1 }
    val t0 = System.nanoTime()
    var i = 0
    while i < reps do { f; i += 1 }
    println(f"$label%-30s ${(System.nanoTime() - t0) / 1e6 / reps}%8.2f ms")

  def main(args: Array[String]): Unit =
    val n = 1_000_000
    val a = np.arange(n.toDouble)
    val b = np.ones(n)
    time("add scalar warm")(a + 1.0)
    time("add 1e6", 20)(a + b)
    time("add self 1e6", 20)(a + a)
    time("add scalar 1e6")(a + 1.0)
    time("map sqrt 1e6")(a.map(math.sqrt))
    time("sum 1e6")(a.sum())
    time("mean axis0 1000x1000")(a.reshape(1000, 1000).mean(0))
    time("transpose copy 1000x1000")(a.reshape(1000, 1000).T.copy())
    time("compare 1e6")(a > 5.0)
    time("mask index 1e6")(a(a > 500000.0))
    val m = np.arange(250000.0).reshape(500, 500)
    time("matmul 500x500", 2)(m @@ m)
    val ai = np.arange(n)
    time("int add 1e6")(ai + ai)
    val shuffled = a.reshape(1000, 1000).T.copy().ravel()
    time("sort 1e6", 2)(shuffled.sorted)
    time("argsort 1e6", 2)(shuffled.argsort())
    time("astype int 1e6")(a.astype[Int])
