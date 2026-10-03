package com.github.kmizu.numscala.cpu.vector25

/** Checks that the forked test JVM can load `jdk.incubator.vector`. */
class SmokeSuite extends munit.FunSuite:
  test("incubator vector module is available in the test JVM") {
    assert(ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent)
    assert(VectorInfo.lanes >= 1)
  }
