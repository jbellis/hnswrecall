package org.example;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;

import java.util.List;
import java.util.stream.IntStream;

public class PQuantization {
    private final List<List<CentroidCluster<DoublePoint>>> codebooks;
    private final int M;

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
        codebooks = PQUtil.createCodebooks(vectors, M, K);
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
        int subvectorSize = vector.length / M;
        List<Integer> indices = IntStream.range(0, M)
                .mapToObj(m -> {
                    // find the closest centroid in the corresponding codebook to each subvector
                    return PQUtil.closetCentroidIndex(PQUtil.getSubVector(vector, m, subvectorSize), codebooks.get(m));
                })
                .toList();

        return PQUtil.toBytes(indices, M);
    }
}
