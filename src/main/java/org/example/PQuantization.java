package org.example;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;

import java.util.AbstractMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class PQuantization {
    private final List<List<CentroidCluster<DoublePoint>>> codebooks;
    // number of subspaces
    private final int M = 4;

    /**
     * Constructor for PQQuantization. Initializes the codebooks by clustering
     * the input data using Product Quantization.
     */
    public PQuantization(List<float[]> vectors) {
        int dimensions = vectors.get(0).length;
        assert dimensions % M == 0 : "The number of dimensions must be divisible by " + M;
        // Number of centroids per subspace
        int k = 256;
        codebooks = createCodebooks(vectors, M, k);
    }

    static List<List<CentroidCluster<DoublePoint>>> createCodebooks(List<float[]> vectors, int M, int K) {
        int subvectorSize = vectors.get(0).length / M;
        return IntStream.range(0, M).parallel()
                .mapToObj(m -> {
                    List<DoublePoint> subvectors = vectors.stream().parallel()
                            .map(vector -> new DoublePoint(getSubVector(vector, m, subvectorSize)))
                            .collect(Collectors.toList());
                    KMeansPlusPlusClusterer<DoublePoint> clusterer = new KMeansPlusPlusClusterer<>(K, 100, PQuantization::distanceBetween);
                    return clusterer.cluster(subvectors);
                })
                .toList();
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
                    double[] subvector = getSubVector(vector, m, subvectorSize);
                    List<CentroidCluster<DoublePoint>> codebook = codebooks.get(m);
                    return IntStream.range(0, codebook.size())
                            .mapToObj(i -> new AbstractMap.SimpleEntry<>(i, distanceBetween(subvector, codebook.get(i).getCenter().getPoint())))
                            .min(Map.Entry.comparingByValue())
                            .map(Map.Entry::getKey)
                            .get();
                })
                .toList();

        // convert indexes to bytes, in a manner such that naive euclidean distance will still capture
        // the correct distance between indexes
        byte[] q = new byte[M];
        for (int m = 0; m < M; m++) {
            int centroidIndex = indices.get(m);
            byte byteValue;
            if (centroidIndex < 128) {
                byteValue = (byte) (centroidIndex - 128);
            } else {
                byteValue = (byte) (centroidIndex - 128);
            }
            q[m] = byteValue;
        }
        return q;
    }

    /**
     * Extracts the m-th subvector from a single vector.
     *
     * @return The m-th subvector.
     */
    static double[] getSubVector(float[] vector, int m, int subvectorSize) {
        double[] subvector = new double[subvectorSize];
        for (int i = 0; i < subvectorSize; i++) {
            subvector[i] = vector[m * subvectorSize + i];
        }
        return subvector;
    }

    static double distanceBetween(double[] vector1, double[] vector2) {
        var sum = 0.0;
        for (var i = 0; i < vector1.length; i += DoubleVector.SPECIES_PREFERRED.length()) {
            var a = DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, vector1, i);
            var b = DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, vector2, i);
            var diff = a.sub(b);
            var square = diff.mul(diff);
            sum += square.reduceLanes(VectorOperators.ADD);
        }
        // tail
        for (var i = vector1.length - vector1.length % DoubleVector.SPECIES_PREFERRED.length(); i < vector1.length; i++) {
            var diff = vector1[i] - vector2[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
}
