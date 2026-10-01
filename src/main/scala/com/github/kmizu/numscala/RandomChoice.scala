package com.github.kmizu.numscala.random

/** Sampling-without-replacement helpers of `choice`. */
private[numscala] object RandomChoice:
  /** NumPy's `_shuffle_int`: Fisher–Yates on `data[first:n]` using Lemire bounded draws. */
  def shuffleInt(bg: BitGenerator, n: Long, first: Long, data: Array[Long]): Unit =
    var i = n - 1
    while i >= first do
      val j = Bounded.uint64(bg, 0L, i, 0L, false).toInt
      val t = data(j); data(j) = data(i.toInt); data(i.toInt) = t
      i -= 1

  /** `Generator.choice(pop, size, replace=False)` without `p` (Floyd's algorithm / tail shuffle). */
  def noReplace(bg: BitGenerator, popSize: Long, size: Long, shuffle: Boolean): Array[Long] =
    val cutoff = if shuffle then 50 else 20
    if popSize > 10000 && size > popSize / cutoff then
      val idx = Array.tabulate(popSize.toInt)(_.toLong)
      shuffleInt(bg, popSize, math.max(popSize - size, 1), idx)
      idx.drop((popSize - size).toInt)
    else
      val idx = new Array[Long](size.toInt)
      val setSize0 = (1.2 * size).toLong
      val mask = Bounded.genMask(setSize0)
      val setSize = 1 + mask
      val hashSet = Array.fill(setSize.toInt)(-1L)
      var j = popSize - size
      while j < popSize do
        val v = Bounded.uint64(bg, 0L, j, 0L, false)
        var loc = (v & mask).toInt
        while hashSet(loc) != -1L && hashSet(loc) != v do loc = ((loc + 1) & mask).toInt
        if hashSet(loc) == -1L then
          hashSet(loc) = v
          idx((j - popSize + size).toInt) = v
        else
          loc = (j & mask).toInt
          while hashSet(loc) != -1L do loc = ((loc + 1) & mask).toInt
          hashSet(loc) = j
          idx((j - popSize + size).toInt) = j
        j += 1
      if shuffle then shuffleInt(bg, size, 1, idx)
      idx

  /** `choice(..., replace=False, p=p)`: repeated weighted draws keeping first occurrences. */
  def noReplaceP(r: RandomCore, p: Array[Double], size: Int): Array[Long] =
    val found = new Array[Long](size)
    var nUniq = 0
    while nUniq < size do
      val x = Array.fill(size - nUniq)(r.bg.nextDouble())
      if nUniq > 0 then
        var k = 0
        while k < nUniq do { p(found(k).toInt) = 0.0; k += 1 }
      val cdf = RandomArrays.cumsum(p)
      val last = cdf(cdf.length - 1)
      for i <- cdf.indices do cdf(i) /= last
      val seen = scala.collection.mutable.LinkedHashSet.empty[Long]
      x.foreach(u => seen += RandomArrays.searchRight(cdf, u).toLong)
      for v <- seen do
        found(nUniq) = v
        nUniq += 1
    found
