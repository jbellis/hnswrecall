package org.example;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.lucene.util.VectorUtil;
import org.example.util.KMeansPlusPlusFloatClusterer;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.example.util.SimdOps.*;

public class PQuantization {
    private final Encoder encoder;
    private final int M;
    private final int originalDimension;
    private final float[] globalCentroid;
    private final int[] subvectorSizes;

    /**
     * Constructor for PQQuantization. Initializes the codebooks by clustering
     * the input data using Product Quantization.
     *
     * M = number of subspaces
     * K = number of clusters per subspace
     */
    public PQuantization(List<float[]> vectors, int M, int K, boolean globallyCenter) {
        this.M = M;
        originalDimension = vectors.get(0).length;
        subvectorSizes = getSubvectorSizes(originalDimension, M);
        if (globallyCenter) {
            globalCentroid = KMeansPlusPlusFloatClusterer.centroidOf(vectors);
            // subtract the centroid from each vector
            vectors = vectors.stream().parallel().map(v -> simdSub(v, globalCentroid)).toList();
            // TODO compute optimal rotation as well
        } else {
            globalCentroid = null;
        }
        encoder = createEncoder(vectors, M, K, subvectorSizes);
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
        if (globalCentroid != null) {
            vector = simdSub(vector, globalCentroid);
        }
        vector = simdMultiplyMatrix(vector, encoder.rotationMatrix);

        float[] finalVector = vector;
        List<Integer> indices = IntStream.range(0, M)
                .mapToObj(m -> {
                    // find the closest centroid in the corresponding codebook to each subvector
                    return closetCentroidIndex(getSubVector(finalVector, m, subvectorSizes), encoder.codebooks.get(m));
                })
                .toList();

        return toBytes(indices, M);
    }

    /**
     * Decodes the quantized representation (byte array) to its approximate original vector.
     */
    public void decode(byte[] encoded, float[] target) {
        int offset = 0; // starting position in the target array for the current subvector
        for (int m = 0; m < M; m++) {
            int centroidIndex = Byte.toUnsignedInt(encoded[m]);
            float[] centroidSubvector = encoder.codebooks.get(m).get(centroidIndex);
            System.arraycopy(centroidSubvector, 0, target, offset, subvectorSizes[m]);
            offset += subvectorSizes[m];
        }

        var t = simdMultiplyMatrix(target, encoder.rotationTransposed);
        System.arraycopy(t, 0, target, 0, target.length);
        if (globalCentroid != null) {
            // Add back the global centroid to get the approximate original vector.
            simdAddInPlace(target, globalCentroid);
        }
    }

    public int getDimensions() {
        return originalDimension;
    }

    static void printCodebooks(List<List<float[]>> codebooks) {
        List<List<String>> strings = codebooks.stream()
                .map(L -> L.stream()
                        .map(PQuantization::arraySummary)
                        .collect(Collectors.toList()))
                .toList();
        System.out.printf("Codebooks: [%s]%n", String.join("\n ", strings.stream()
                .map(L -> "[" + String.join(", ", L) + "]")
                .toList()));
    }

    private static String arraySummary(float[] a) {
        List<String> b = new ArrayList<>();
        for (int i = 0; i < Math.min(4, a.length); i++) {
            b.add(String.valueOf(a[i]));
        }
        if (a.length > 4) {
            b.set(3, "... (" + a.length + ")");
        }
        return "[" + String.join(", ", b) + "]";
    }

    private record Encoder(List<List<float[]>> codebooks, float[][] rotationMatrix, float[][] rotationTransposed) { }

    static Encoder createEncoder(List<float[]> vectors, int M, int K, int[] subvectorSizes) {
        var clusterers = IntStream.range(0, M).parallel()
                .mapToObj(m -> {
                    List<float[]> subvectors = getSubvectors(vectors, subvectorSizes, m);
                    return new KMeansPlusPlusFloatClusterer(subvectors, K, VectorUtil::squareDistance);
                }).toList();

        List<List<float[]>> centroids = clusterers.stream().map(KMeansPlusPlusFloatClusterer::getCentroids).toList();
        float[][] rotationMatrix = null;
        for (int i = 0; i < 15; i++) {
            rotationMatrix = computeRotationMatrix(vectors, centroids);
            var RM = rotationMatrix;
            var rotatedVectors = vectors.stream().map(v -> simdMultiplyMatrix(v, RM)).toList();
            centroids = IntStream.range(0, M).parallel()
                    .mapToObj(m -> {
                        var clusterer = clusterers.get(m);
                        List<float[]> subvectors = getSubvectors(rotatedVectors, subvectorSizes, m);
                        clusterer.clusterOnce(subvectors);
                        return clusterer.getCentroids();
                    }).toList();
        }
        return new Encoder(centroids, rotationMatrix, transpose(rotationMatrix));
    }

    private static List<float[]> getSubvectors(List<float[]> vectors, int[] subvectorSizes, int m) {
        List<float[]> subvectors = vectors.stream().parallel()
                .map(vector -> getSubVector(vector, m, subvectorSizes))
                .toList();
        return subvectors;
    }

    static int closetCentroidIndex(float[] subvector, List<float[]> codebook) {
        return IntStream.range(0, codebook.size())
                .mapToObj(i -> new AbstractMap.SimpleEntry<>(i, VectorUtil.squareDistance(subvector, codebook.get(i))))
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
            q[m] = (byte) (int) indices.get(m);
        }
        return q;
    }

    /**
     * Extracts the m-th subvector from a single vector.
     *
     * @return The m-th subvector.
     */
    static float[] getSubVector(float[] vector, int m, int[] subvectorSizes) {
        float[] subvector = new float[subvectorSizes[m]];
        // calculate the offset for the m-th subvector
        int offset = Arrays.stream(subvectorSizes, 0, m).sum();
        System.arraycopy(vector, offset, subvector, 0, subvectorSizes[m]);
        return subvector;
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

    public static float[][] transpose(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;

        float[][] transposed = new float[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }

    static float[][] computeRotationMatrix(List<float[]> vectors, List<List<float[]>> centroids) {
        int dim = vectors.get(0).length;
        int numCentroids = centroids.stream().mapToInt(List::size).sum();

        // Convert the training data and centroids into matrices.
        RealMatrix X = MatrixUtils.createRealMatrix(vectors.size(), dim);
        IntStream.range(0, vectors.size()).parallel().forEach(i -> {
            var v = vectors.get(i);
            for (int j = 0; j < v.length; j++) {
                X.setEntry(i, j, v[j]);
            }
        });

        RealMatrix C = MatrixUtils.createRealMatrix(numCentroids, dim);
        IntStream.range(0, centroids.size()).parallel().forEach(i -> {
            var codebook = centroids.get(i);
            for (int j = 0; j < codebook.size(); j++) {
                var centroid = codebook.get(j);
                for (int k = 0; k < centroid.length; k++) {
                    C.setEntry(j, k, centroid[k]);
                }
            }
        });

        // Compute the correlation matrix between the original data and the quantized data.
        RealMatrix correlationMatrix = X.transpose().multiply(X); // In place of rotated and quantized data, using just rotated for simplicity.

        // Perform SVD on the correlation matrix.
        SingularValueDecomposition svd = new SingularValueDecomposition(correlationMatrix);
        RealMatrix U = svd.getU();
        RealMatrix VT = svd.getVT();

        // Compute the new rotation matrix from the singular vectors as R^T = U V^T.
        RealMatrix R = U.multiply(VT);

        // Convert the rotation matrix from double to float.
        float[][] rotationMatrix = new float[R.getRowDimension()][R.getColumnDimension()];
        IntStream.range(0, R.getRowDimension()).parallel().forEach(i -> {
            for (int j = 0; j < R.getColumnDimension(); j++) {
                rotationMatrix[i][j] = (float) R.getEntry(i, j);
            }
        });

        return rotationMatrix;
    }
}
