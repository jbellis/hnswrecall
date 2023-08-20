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
    private final float[] vector;

    public PQRandomAccessVectorValues(List<byte[]> vectors, PQuantization pq) {
        this.encoded = vectors;
        this.pq = pq;
        this.vector = new float[pq.getDimensions()];
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
        pq.decode(encoded.get(targetOrd), vector);
        return vector;
    }

    @Override
    public PQRandomAccessVectorValues copy() {
        return new PQRandomAccessVectorValues(encoded, pq);
    }
}
