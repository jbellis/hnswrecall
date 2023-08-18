package org.example;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;


public class PQuantization {
    private final int M = 4;  // Number of subvectors
    private final int K = 256; // Number of centroids per subspace
    private final int subvectorSize; // Size of each subvector
    private List<List<CentroidCluster<DoublePoint>>> codebooks;

    /**
     * Constructor for PQQuantization. Initializes the codebooks by clustering
     * the input data using Product Quantization.
     */
    public PQuantization(double[][] data) {
        subvectorSize = data[0].length / M;
        codebooks = IntStream.range(0, M).parallel()
                .mapToObj(m -> {
                    List<DoublePoint> subvectors = getSubVectors(data, m);
                    KMeansPlusPlusClusterer<DoublePoint> clusterer = new KMeansPlusPlusClusterer<>(K);
                    return clusterer.cluster(subvectors);
                })
                .collect(Collectors.toList());
    }

    /**
     * Quantizes the input vector using the generated codebooks.
     *
     * @return The quantized value represented as an integer.
     */
    public int quantize(double[] vector) {
        List<Integer> indices = IntStream.range(0, M).parallel()
                .mapToObj(m -> {
                    double[] subvector = getSubVector(vector, m);
                    return findClosestCentroidIndex(subvector, codebooks.get(m));
                })
                .collect(Collectors.toList());

        int value = 0;
        for (int index : indices) {
            value = (value << 8) | index;  // Encode the index in the integer
        }
        return value;
    }

    /**
     * Extracts the m-th subvector from each vector in the dataset.
     *
     * @return A list of m-th subvectors.
     */
    private List<DoublePoint> getSubVectors(double[][] data, int m) {
        return Stream.of(data).parallel()
                .map(vector -> new DoublePoint(getSubVector(vector, m)))
                .collect(Collectors.toList());
    }

    /**
     * Extracts the m-th subvector from a single vector.
     *
     * @return The m-th subvector.
     */
    private double[] getSubVector(double[] vector, int m) {
        double[] subvector = new double[subvectorSize];
        System.arraycopy(vector, m * subvectorSize, subvector, 0, subvectorSize);
        return subvector;
    }

    /**
     * Finds the index of the closest centroid in the codebook for the given subvector.
     *
     * @return The index of the closest centroid.
     */
    private int findClosestCentroidIndex(double[] subvector, List<CentroidCluster<DoublePoint>> codebook) {
        DoublePoint point = new DoublePoint(subvector);
        return IntStream.range(0, codebook.size()).parallel()
                .boxed()
                .min((i1, i2) -> Double.compare(
                        point.distanceFrom(codebook.get(i1).getCenter()),
                        point.distanceFrom(codebook.get(i2).getCenter())))
                .orElse(-1);
    }
}
