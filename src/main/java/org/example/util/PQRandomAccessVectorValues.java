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
        throw new UnsupportedOperationException();
    }

    public float decodedScore(int targetOrd, float[] query) {
        return (1 + pq.decodedDotProduct(encoded.get(targetOrd), query)) / 2;
    }

    @Override
    public PQRandomAccessVectorValues copy() {
        return new PQRandomAccessVectorValues(encoded, pq);
    }
}
