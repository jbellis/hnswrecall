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

public class PQUtil {
    static void printCodebooks(List<List<CentroidCluster<DoublePoint>>> result) {
        List<List<String>> strings = result.stream().map(L -> L.stream().map(C -> arraySummary(C.getCenter().getPoint())).toList()).toList();
        System.out.printf("Codebooks: [%s]%n", String.join("\n ", strings.stream().map(L -> "[" + String.join(", ", L) + "]").toList()));
    }

    private static String arraySummary(double[] a) {
        String[] b = Arrays.stream(a, 0, 4).mapToObj(String::valueOf).toArray(String[]::new);
        b[3] = "... (%s)".formatted(a.length);
        return "[" + String.join(", ", b) + "]";
    }

    static List<List<CentroidCluster<DoublePoint>>> createCodebooks(List<float[]> vectors, int M, int K) {
        // split vectors into M subvectors
        int subvectorSize = vectors.get(0).length / M;
        return IntStream.range(0, M).parallel()
                .mapToObj(m -> {
                    List<DoublePoint> subvectors = vectors.stream().parallel()
                            .map(vector -> new DoublePoint(getSubVector(vector, m, subvectorSize)))
                            .collect(Collectors.toList());
                    KMeansPlusPlusClusterer<DoublePoint> clusterer = new KMeansPlusPlusClusterer<>(K, 15, PQUtil::distanceBetween);
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

    static byte[] toBytes(List<Integer> indices, int M) {
        // convert indexes to bytes, in a manner such that naive euclidean distance will still capture
        // the correct distance between indexes
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
    static double[] getSubVector(float[] vector, int m, int subvectorSize) {
        double[] subvector = new double[subvectorSize];
        for (int i = 0; i < subvectorSize; i++) {
            subvector[i] = vector[m * subvectorSize + i];
        }
        return subvector;
    }

    static double distanceBetween(double[] vector1, double[] vector2) {
        var sum = 0.0;
        if (vector1.length >= DoubleVector.SPECIES_PREFERRED.length()) {
            for (var i = 0; i < vector1.length; i += DoubleVector.SPECIES_PREFERRED.length()) {
                var a = DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, vector1, i);
                var b = DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, vector2, i);
                var diff = a.sub(b);
                var square = diff.mul(diff);
                sum += square.reduceLanes(VectorOperators.ADD);
            }
        }
        // tail
        for (var i = vector1.length - vector1.length % DoubleVector.SPECIES_PREFERRED.length(); i < vector1.length; i++) {
            var diff = vector1[i] - vector2[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
}
