package org.example;

import org.apache.lucene.util.VectorUtil;
import org.example.util.KMeansPlusPlusFloatClusterer;
import org.example.util.PCA;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.example.util.SimdOps.*;

public class PQuantization {
    private final List<List<float[]>> codebooks;
    private final int M;
    private final float[] globalCentroid;
    private final float[][] rotationMatrix;
    private final float[][] transposedRotationMatrix;
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
        globalCentroid = KMeansPlusPlusFloatClusterer.centroidOf(vectors);
        subvectorSizes = getSubvectorSizes(vectors.get(0).length, M);
        // subtract the centroid from each vector
        var centeredVectors = vectors.stream().parallel().map(v -> simdSub(v, globalCentroid)).toList();
        rotationMatrix = PCA.computePCARotation(centeredVectors);
        transposedRotationMatrix = PCA.transpose(rotationMatrix);
        var rotatedVectors = centeredVectors.stream().parallel().map(v -> simdMultiplyMatrix(v, rotationMatrix)).toList();
        codebooks = createCodebooks(rotatedVectors, M, K, subvectorSizes);
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
        float[] centered = simdSub(vector, globalCentroid);
        float[] rotated = simdMultiplyMatrix(centered, rotationMatrix);

        List<Integer> indices = IntStream.range(0, M)
                .mapToObj(m -> {
                    // find the closest centroid in the corresponding codebook to each subvector
                    return closetCentroidIndex(getSubVector(rotated, m, subvectorSizes), codebooks.get(m));
                })
                .toList();

        return toBytes(indices, M);
    }

    /**
     * Decodes the quantized representation (byte array) to its approximate original vector.
     *
     * @return The approximate original vector.
     */
    public float[] decode(byte[] encoded, float[] target) {
        int offset = 0; // starting position in the target array for the current subvector
        for (int m = 0; m < M; m++) {
            int centroidIndex = Byte.toUnsignedInt(encoded[m]);
            float[] centroidSubvector = codebooks.get(m).get(centroidIndex);
            System.arraycopy(centroidSubvector, 0, target, offset, subvectorSizes[m]);
            offset += subvectorSizes[m];
        }

        // Invert the rotation and centering to get the approximate original vector.
        var unrotated = simdMultiplyMatrix(target, transposedRotationMatrix);
        simdAddInPlace(unrotated, globalCentroid);
        return unrotated;
    }

    public int getDimensions() {
        return globalCentroid.length;
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


    static List<List<float[]>> createCodebooks(List<float[]> vectors, int M, int K, int[] subvectorSizes) {
        return IntStream.range(0, M).parallel()
                .mapToObj(m -> {
                    List<float[]> subvectors = vectors.stream().parallel()
                            .map(vector -> getSubVector(vector, m, subvectorSizes))
                            .toList();
                    var clusterer = new KMeansPlusPlusFloatClusterer(K, 15, VectorUtil::squareDistance);
                    return clusterer.cluster(subvectors);
                })
                .toList();
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
}
