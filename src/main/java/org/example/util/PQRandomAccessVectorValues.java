package org.example.util;

import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.example.PQuantization;

import java.util.List;

/**
 * A PQ-backed implementation of the {@link RandomAccessVectorValues} interface.
 */
public class PQRandomAccessVectorValues implements RandomAccessVectorValues<float[]> {
    private final List<byte[]> encoded;
    private final PQuantization pq;

    public PQRandomAccessVectorValues(List<byte[]> vectors, PQuantization pq) {
        this.encoded = vectors;
        this.pq = pq;
    }

    @Override
    public int size() {
        return encoded.size();
    }

    @Override
    public int dimension() {
        return pq.getDimensions();
    }

    @Override
    public float[] vectorValue(int targetOrd) {
        return pq.decode(encoded.get(targetOrd));
    }

    @Override
    public PQRandomAccessVectorValues copy() {
        // the copy method is called as a workaround for Lucene's implementations not being re-entrant.
        // if you are already re-entrant, you really don't need a new copy, but HGBuilder.build
        // explicitly checks for object identity so we'll do a shallow copy to make it happy.
        return new PQRandomAccessVectorValues(encoded, pq);
    }
}
