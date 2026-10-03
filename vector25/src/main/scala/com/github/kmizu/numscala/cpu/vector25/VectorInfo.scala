package com.github.kmizu.numscala.cpu.vector25

/** Facts about the Vector API species used by the `vector25` backend. */
object VectorInfo:
  /** The preferred Float32 species of this JVM/CPU (`FloatVector.SPECIES_PREFERRED`). */
  def species = Species.S
  /** Float32 lanes per vector of the preferred species. */
  def lanes: Int = species.length()
