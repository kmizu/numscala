package com.github.kmizu.numscala.cpu.vector25;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * The Float32 species used by the vector25 kernels, held in Java {@code static final} fields so that
 * HotSpot treats it as a constant and intrinsifies the Vector API calls (a Scala object {@code val} is an
 * instance field and would leave every vector boxed).
 */
public final class Species {
    private Species() {}

    /** {@code FloatVector.SPECIES_PREFERRED}. */
    public static final VectorSpecies<Float> S = FloatVector.SPECIES_PREFERRED;

    /** Lanes of {@link #S}. */
    public static final int L = S.length();
}
