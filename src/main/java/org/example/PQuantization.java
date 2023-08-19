package org.example;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;

import java.util.AbstractMap;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class PQuantization {
    private final List<List<CentroidCluster<DoublePoint>>> codebooks;
    private final int M;
    private final double[] centroid;
    private final int[] subvectorSizes; // added member variable

    /**
     * Constructor for PQQuantization. Initializes the codebooks by clustering
     * the input data using Product Quantization.
     *
     * M = number of subspaces
     * K = number of clusters per subspace
     */
    public PQuantization(List<float[]> vectors, int M, int K) {
        this.M = M;
        centroid = centroidOf(vectors);
        subvectorSizes = getSubvectorSizes(vectors.get(0).length, M);
        // subtract the centroid from each vector
        var centeredVectors = vectors.stream().parallel().map(v -> subFrom(v, centroid)).toList();
        // TODO compute optimal rotation as well
        codebooks = createCodebooks(centeredVectors, M, K, subvectorSizes);
    }

    public List<byte[]> encodeAll(List<float[]> vectors) {
        return vectors.stream().parallel().map(this::encode).toList();
    }

    /**
     * Quantizes the input vector using the generated codebooks.
     *
     * @return The quantized value represented as an integer.
     */
    public byte[] encode(float[] vector) {
        float[] centered = subFrom(vector, centroid);

        List<Integer> indices = IntStream.range(0, M)
                .mapToObj(m -> {
                    // find the closest centroid in the corresponding codebook to each subvector
                    return closetCentroidIndex(getSubVector(centered, m, subvectorSizes), codebooks.get(m));
                })
                .toList();

        return toBytes(indices, M);
    }

    /**
     * Decodes the quantized representation (byte array) to its approximate original vector.
     *
     * @return The approximate original vector.
     */
    public float[] decode(byte[] encoded) {
        float[] reconstructed = new float[centroid.length]; // The reconstructed vector should have the same length as the original vector.

        int offset = 0; // starting position in the reconstructed array for the current subvector
        for (int m = 0; m < M; m++) {
            byte byteValue = encoded[m];
            int centroidIndex = byteValue + 128; // reverse the operation done in toBytes()
            double[] centroidSubvector = codebooks.get(m).get(centroidIndex).getCenter().getPoint();

            for (int i = 0; i < subvectorSizes[m]; i++) {
                reconstructed[offset + i] = (float) centroidSubvector[i];
            }
            offset += subvectorSizes[m]; // move to the next subvector's starting position
        }

        // Add back the global centroid to get the approximate original vector.
        for (int i = 0; i < reconstructed.length; i++) {
            reconstructed[i] += (float) centroid[i];
        }

        return reconstructed;
    }

    public int getDimensions() {
        return centroid.length;
    }

    static void printCodebooks(List<List<CentroidCluster<DoublePoint>>> result) {
        List<List<String>> strings = result.stream().map(L -> L.stream().map(C -> arraySummary(C.getCenter().getPoint())).toList()).toList();
        System.out.printf("Codebooks: [%s]%n", String.join("\n ", strings.stream().map(L -> "[" + String.join(", ", L) + "]").toList()));
    }

    private static String arraySummary(double[] a) {
        String[] b = Arrays.stream(a, 0, Math.min(4, a.length)).mapToObj(String::valueOf).toArray(String[]::new);
        if (a.length > 4) {
            b[3] = "... (%s)".formatted(a.length);
        }
        return "[" + String.join(", ", b) + "]";
    }

    static List<List<CentroidCluster<DoublePoint>>> createCodebooks(List<float[]> vectors, int M, int K, int[] subvectorSizes) {
        return IntStream.range(0, M).parallel()
                .mapToObj(m -> {
                    List<DoublePoint> subvectors = vectors.stream().parallel()
                            .map(vector -> new DoublePoint(getSubVector(vector, m, subvectorSizes)))
                            .collect(Collectors.toList());
                    KMeansPlusPlusClusterer<DoublePoint> clusterer = new KMeansPlusPlusClusterer<>(K, 15, PQuantization::distanceBetween);
                    List<CentroidCluster<DoublePoint>> L = clusterer.cluster(subvectors);
                    // sort the clusters lexicographically by their centroid double[] values
                    L.sort((c1, c2) -> {
                        double[] p1 = c1.getCenter().getPoint();
                        double[] p2 = c2.getCenter().getPoint();
                        for (int i = 0; i < p1.length; i++) {
                            if (p1[i] != p2[i]) {
                                return Double.compare(p1[i], p2[i]);
                            }
                        }
                        return 0;
                    });
                    return L;
                })
                .toList();
    }

    static float[] subFrom(float[] v, double[] centroid) {
        // TODO use vectorized operations
        float[] centered = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            centered[i] = v[i] - (float) centroid[i];
        }
        return centered;
    }

    static double[] centroidOf(List<float[]> vectors) {
        return IntStream.range(0, vectors.get(0).length).mapToDouble(i -> {
            return vectors.stream().parallel().mapToDouble(v -> v[i]).sum() / vectors.size();
        }).toArray();
    }

    static int closetCentroidIndex(double[] subvector, List<CentroidCluster<DoublePoint>> codebook) {
        return IntStream.range(0, codebook.size())
                .mapToObj(i -> new AbstractMap.SimpleEntry<>(i, distanceBetween(subvector, codebook.get(i).getCenter().getPoint())))
                .min(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .get();
    }

    /**
     * convert indexes to bytes, in a manner such that naive euclidean distance will still capture
     * the correct distance between indexes
     */
    static byte[] toBytes(List<Integer> indices, int M) {
        byte[] q = new byte[M];
        for (int m = 0; m < M; m++) {
            int centroidIndex = indices.get(m);
            byte byteValue = (byte) (centroidIndex - 128);
            q[m] = byteValue;
        }
        return q;
    }

    /**
     * Extracts the m-th subvector from a single vector.
     *
     * @return The m-th subvector.
     */
    static double[] getSubVector(float[] vector, int m, int[] subvectorSizes) {
        double[] subvector = new double[subvectorSizes[m]];
        // calculate the offset for the m-th subvector
        int offset = Arrays.stream(subvectorSizes, 0, m).sum();
        // copy
        for (int i = 0; i < subvectorSizes[m]; i++) {
            subvector[i] = vector[offset + i];
        }
        return subvector;
    }

    static double distanceBetween(double[] vector1, double[] vector2) {
        var sum = 0.0;
        int vectorizedLength = (vector1.length / DoubleVector.SPECIES_PREFERRED.length()) * DoubleVector.SPECIES_PREFERRED.length();

        // Process the vectorized part
        for (var i = 0; i < vectorizedLength; i += DoubleVector.SPECIES_PREFERRED.length()) {
            var a = DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, vector1, i);
            var b = DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, vector2, i);
            var diff = a.sub(b);
            var square = diff.mul(diff);
            sum += square.reduceLanes(VectorOperators.ADD);
        }

        // Process the tail
        for (var i = vectorizedLength; i < vector1.length; i++) {
            var diff = vector1[i] - vector2[i];
            sum += diff * diff;
        }

        return Math.sqrt(sum);
    }

    static int[] getSubvectorSizes(int dimensions, int M) {
        int[] sizes = new int[M];
        int baseSize = dimensions / M;
        int remainder = dimensions % M;
        // distribute the remainder among the subvectors
        for (int i = 0; i < M; i++) {
            sizes[i] = baseSize + (i < remainder ? 1 : 0);
        }
        return sizes;
    }
}
