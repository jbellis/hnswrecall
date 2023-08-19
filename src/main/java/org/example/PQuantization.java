package org.example;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;

import java.util.List;
import java.util.stream.IntStream;

import static org.example.PQUtil.createCodebooks;
import static org.example.PQUtil.subFrom;

public class PQuantization {
    private final List<List<CentroidCluster<DoublePoint>>> codebooks;
    private final int M;
    private final double[] centroid;

    /**
     * Constructor for PQQuantization. Initializes the codebooks by clustering
     * the input data using Product Quantization.
     *
     * M = number of subspaces
     * K = number of clusters per subspace
     */
    public PQuantization(List<float[]> vectors, int M, int K) {
        this.M = M;
        int dimensions = vectors.get(0).length;
        assert dimensions % M == 0 : "The number of dimensions must be divisible by " + M;
        centroid = PQUtil.centroidOf(vectors);
        // subtract the centroid from each vector
        var centeredVectors = vectors.stream().parallel().map(v -> subFrom(v, centroid)).toList();
        codebooks = createCodebooks(centeredVectors, M, K);
    }

    public List<byte[]> quantizeAll(List<float[]> vectors) {
        return vectors.stream().parallel().map(this::quantize).toList();
    }

    /**
     * Quantizes the input vector using the generated codebooks.
     *
     * @return The quantized value represented as an integer.
     */
    public byte[] quantize(float[] vector) {
        float[] centered = subFrom(vector, centroid);
        int subvectorSize = centered.length / M;
        List<Integer> indices = IntStream.range(0, M)
                .mapToObj(m -> {
                    // find the closest centroid in the corresponding codebook to each subvector
                    return PQUtil.closetCentroidIndex(PQUtil.getSubVector(centered, m, subvectorSize), codebooks.get(m));
                })
                .toList();

        return PQUtil.toBytes(indices, M);
    }
}
