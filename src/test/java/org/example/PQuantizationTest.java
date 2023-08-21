package org.example;

import org.apache.lucene.util.VectorUtil;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

public class PQuantizationTest {
    private static final float EPSILON = 1e-6f;

    @Test
    public void testCodebooks() {
        // Using a small, controlled dataset for testing
        List<float[]> testVectors = List.of(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new float[]{1.1f, 2.1f, 3.1f, 4.1f},
                new float[]{2.0f, 3.0f, 4.0f, 5.0f},
                new float[]{2.1f, 3.1f, 4.1f, 5.1f}
        );

        int M = 4;
        int K = 2;  // Since our test dataset is small, let's use a smaller K

        var result = PQuantization.createEncoder(testVectors, M, K, PQuantization.getSubvectorSizes(4, M), true);
        // Print results for manual inspection
        PQuantization.printCodebooks(result.codebooks());

        // quantize the vectors
        var quantized = new PQuantization(testVectors, M, K, false, false).encodeAll(testVectors);
        System.out.printf("Quantized: %s%n", quantized.stream().map(Arrays::toString).toList());
    }

    @Test
    public void testClosetCentroidIndex2D() {
        // 1. Create a sample codebook
        var codebook = List.of(
                new float[]{1.0f, 2.0f},
                new float[]{5.0f, 6.0f},
                new float[]{9.0f, 10.0f});

        // The closest centroid to [6.0, 7.0] is [5.0, 6.0], which is at index 1.
        float[] subvector = {6.0f, 7.0f};
        int closestIndex = PQuantization.closetCentroidIndex(subvector, codebook);
        assertEquals(1, closestIndex);
    }


    @Test
    public void testQuantize1D() {
        // 1. Create a sample codebook
        var codebooks = List.of(
                List.of(new float[]{1.05f}, new float[]{2.05f}),
                List.of(new float[]{2.05f}, new float[]{3.05f}),
                List.of(new float[]{3.05f}, new float[]{4.05f}),
                List.of(new float[]{5.05f}, new float[]{4.05f}));

        int M = 4;
        int[] sizes = PQuantization.getSubvectorSizes(4, M);

        var vector = new float[]{1.0f, 2.0f, 3.0f, 4.0f};
        List<Integer> indices = IntStream.range(0, M)
                .mapToObj(m -> {
                    // find the closest centroid in the corresponding codebook to each subvector
                    return PQuantization.closetCentroidIndex(PQuantization.getSubVector(vector, m, sizes), codebooks.get(m));
                })
                .toList();
        assertEquals(List.of(0, 0, 0, 1), indices);
    }

    @Test
    public void testGetSubVector() {
        float[] vector = new float[]{9.0f, 10.0f, 11.0f, 12.0f};
        assertArrayEquals(new float[]{9.0f}, PQuantization.getSubVector(vector, 0, new int[]{1, 1, 1, 1}), EPSILON);
        assertArrayEquals(new float[]{11.0f, 12.0f}, PQuantization.getSubVector(vector, 1, new int[]{2, 2}), EPSILON);
    }

    @Test
    public void testDistanceBetween() {
        float[] vector1 = new float[]{1.0f, 2.0f};
        float[] vector2 = new float[]{3.0f, 4.0f};
        float[] vector3 = new float[]{5.0f, 1.0f};

        assertEquals(0, VectorUtil.squareDistance(vector1, vector1), EPSILON);
        assertEquals(8, VectorUtil.squareDistance(vector1, vector2), EPSILON);
        assertEquals(17, VectorUtil.squareDistance(vector1, vector3), EPSILON);
    }

    @Test
    public void testToBytesConversion() {
        List<List<Integer>> testCases = Arrays.asList(
                Arrays.asList(0, 0, 0, 0),
                Arrays.asList(255, 255, 255, 255),
                Arrays.asList(127, 127, 127, 127),
                Arrays.asList(128, 128, 128, 128),
                Arrays.asList(0, 127, 128, 255)
        );

        List<byte[]> expectedResults = Arrays.asList(
                new byte[]{-128, -128, -128, -128},
                new byte[]{127, 127, 127, 127},
                new byte[]{-1, -1, -1, -1},
                new byte[]{0, 0, 0, 0},
                new byte[]{-128, -1, 0, 127}
        );

        for (int i = 0; i < testCases.size(); i++) {
            byte[] result = PQuantization.toBytes(testCases.get(i), 4);
            byte[] expected = expectedResults.get(i);
            if (Arrays.equals(result, expected)) {
                System.out.println("Test case " + (i + 1) + " passed!");
            } else {
                System.out.println("Test case " + (i + 1) + " failed. Expected " + Arrays.toString(expected) + " but got " + Arrays.toString(result));
            }
        }
    }

    @Test
    public void testGetSubVectorForUnevenSizes() {
        float[] vector = new float[]{9.0f, 10.0f, 11.0f, 12.0f, 13.0f};
        var sizes = new int[]{3, 2};
        assertArrayEquals(new float[]{9.0f, 10.0f, 11.0f}, PQuantization.getSubVector(vector, 0, sizes), EPSILON);
        assertArrayEquals(new float[]{12.0f, 13.0f}, PQuantization.getSubVector(vector, 1, sizes), EPSILON);
    }

    @Test
    public void testCodebooksForUnevenSizes() {
        // Using a small, controlled dataset for testing with an uneven size
        List<float[]> testVectors = List.of(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                new float[]{1.1f, 2.1f, 3.1f, 4.1f, 5.1f},
                new float[]{2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new float[]{2.1f, 3.1f, 4.1f, 5.1f, 6.1f}
        );

        int M = 3;
        int K = 2;  // Since our test dataset is small, let's use a smaller K

        var result = PQuantization.createEncoder(testVectors, M, K, PQuantization.getSubvectorSizes(5, M), true);
        assertNotNull(result);
        assertEquals(M, result.codebooks().size());
        // Print results for manual inspection
         PQuantization.printCodebooks(result.codebooks());
    }

    @Test
    public void testSameLengthVectors() {
        float[] v1 = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] v2 = {2.0f, 3.0f, 4.0f, 5.0f};
        float expected = 4.0f;
        float result = VectorUtil.squareDistance(v1, v2);
        assertEquals(expected, result, 1e-9);
    }

    @Test
    public void testVectorLengthNotDivisibleBySpeciesPreferred() {
        float[] v1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float[] v2 = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float expected = 5.0f;
        float result = VectorUtil.squareDistance(v1, v2);
        assertEquals(expected, result, 1e-9);
    }

    @Test
    public void testEdgeCaseEmptyVectors() {
        float[] v1 = {};
        float[] v2 = {};
        float expected = 0.0f;
        float result = VectorUtil.squareDistance(v1, v2);
        assertEquals(expected, result, 1e-9);
    }

    @Test
    public void testEdgeCaseSingleElementVectors() {
        float[] v1 = {1.0f};
        float[] v2 = {2.0f};
        float expected = 1.0f;
        float result = VectorUtil.squareDistance(v1, v2);
        assertEquals(expected, result, 1e-9);
    }

    @Test
    public void testEncodeDecodeRoundTrip() {
        // Using a controlled dataset for testing
        List<float[]> testVectors = List.of(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new float[]{1.1f, 2.1f, 3.1f, 4.1f},
                new float[]{2.0f, 3.0f, 4.0f, 5.0f},
                new float[]{2.1f, 3.1f, 4.1f, 5.1f}
        );

        int M = 4;
        int K = 2;  // Since our test dataset is small, let's use a smaller K
        PQuantization pq = new PQuantization(testVectors, M, K, false, false);

        for (float[] originalVector : testVectors) {
            byte[] encoded = pq.encode(originalVector);
            float[] decoded = new float[originalVector.length];
            pq.decode(encoded, decoded);

            for (int i = 0; i < originalVector.length; i++) {
                assertTrue(Math.abs(originalVector[i] - decoded[i]) < 0.1,
                        "Difference detected in component " + i + ": Expected " + originalVector[i] + ", but got " + decoded[i]);
            }
        }
    }

    @Test
    public void testOPQvsPQRandom() {
        for (int i = 0; i < 100; i++) {
            var random = new Random();
            // Generate the synthetic dataset
            int numClusters = random.nextInt(5, 500);
            int pointsPerCluster = random.nextInt(50, 500);
            List<float[]> testData = generateTestData(numClusters, pointsPerCluster);

            int M = 4;  // Number of subvectors
            int K = 2;  // Number of centroids per subvector

            // Encode using standard PQ
            PQuantization pq = new PQuantization(testData, M, K, false, false);
            float pqError = computeReconstructionError(testData, pq);
            System.out.println("PQ Reconstruction Error: " + pqError);

            // Encode using OPQ
            PQuantization opq = new PQuantization(testData, M, K, false, true);
            float opqError = computeReconstructionError(testData, opq);
            System.out.println("OPQ Reconstruction Error: " + opqError);

            assertTrue(opqError < pqError, "OPQ error should be lower than PQ error");
        }
    }

    @Test
    public void testOPQvsPQSmall() {
        List<float[]> testData = generateSmallTestData();

        int M = 4;  // Number of subvectors
        int K = 2;  // Number of centroids per subvector

        // Encode using standard PQ
        PQuantization pq = new PQuantization(testData, M, K, false, false);
        float pqError = computeReconstructionError(testData, pq);
        System.out.println("PQ Reconstruction Error: " + pqError);

        // Encode using OPQ
        PQuantization opq = new PQuantization(testData, M, K, false, true);
        float opqError = computeReconstructionError(testData, opq);
        System.out.println("OPQ Reconstruction Error: " + opqError);

        assertTrue(opqError < pqError, "OPQ error should be lower than PQ error");
    }

    public List<float[]> generateSmallTestData() {
        List<float[]> data = new ArrayList<>();

        // Generate two clusters in 8D:
        // Cluster 1: elongated along the first half-diagonal.
        // Cluster 2: elongated along the second half-diagonal.

        // Cluster 1
        float[] centroid1 = {5, 5, 5, 5, 0, 0, 0, 0};
        for (int j = 0; j < 50; j++) {
            float[] point = new float[8];
            for (int d = 0; d < 4; d++) {
                point[d] = centroid1[d] + (float) Math.random() - 0.5f;  // Perturb the first four dimensions
            }
            data.add(point);
        }

        // Cluster 2
        float[] centroid2 = {0, 0, 0, 0, 5, 5, 5, 5};
        for (int j = 0; j < 50; j++) {
            float[] point = new float[8];
            for (int d = 4; d < 8; d++) {
                point[d] = centroid2[d] + (float) Math.random() - 0.5f;  // Perturb the last four dimensions
            }
            data.add(point);
        }

        return data;
    }

    private float computeReconstructionError(List<float[]> data, PQuantization quantizer) {
        float totalError = 0.0f;
        for (float[] originalVector : data) {
            byte[] encoded = quantizer.encode(originalVector);
            float[] decoded = new float[originalVector.length];
            quantizer.decode(encoded, decoded);
            totalError += VectorUtil.squareDistance(originalVector, decoded);
        }
        return totalError / data.size();
    }

    public List<float[]> generateTestData(int numClusters, int pointsPerCluster) {
        Random random = new Random();
        List<float[]> data = new ArrayList<>();

        // Generate clustered data in 8D
        for (int i = 0; i < numClusters; i++) {
            float[] centroid = new float[8];
            for (int d = 0; d < 8; d++) {
                centroid[d] = random.nextFloat() * 10;
            }

            for (int j = 0; j < pointsPerCluster; j++) {
                float[] point = new float[8];
                for (int d = 0; d < 8; d++) {
                    point[d] = centroid[d] + random.nextFloat() - 0.5f;  // Perturb each dimension
                }
                data.add(point);
            }
        }

        return data;
    }
}
