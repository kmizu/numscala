package com.github.kmizu.numscala.cpu.vector25

import jdk.incubator.vector.FloatVector

/** Facts about the Vector API species used by the `vector25` backend. */
object VectorInfo:
  /** The preferred Float32 species of this JVM/CPU (`FloatVector.SPECIES_PREFERRED`). */
  val species = FloatVector.SPECIES_PREFERRED
  /** Float32 lanes per vector of the preferred species. */
  def lanes: Int = species.length()
