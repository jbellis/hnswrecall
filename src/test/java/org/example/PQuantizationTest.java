package org.example;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import static org.example.PQuantization.distanceBetween;
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

        List<List<CentroidCluster<DoublePoint>>> result = PQuantization.createCodebooks(testVectors, M, K, PQuantization.getSubvectorSizes(4, M));
        PQuantization.printCodebooks(result);

        // quantize the vectors
        var quantized = new PQuantization(testVectors, M, K).encodeAll(testVectors);
        System.out.printf("Quantized: %s%n", quantized.stream().map(Arrays::toString).toList());
    }

    @Test
    public void testClosetCentroidIndex2D() {
        // 1. Create a sample codebook
        List<CentroidCluster<DoublePoint>> codebook = List.of(
                new CentroidCluster<>(new DoublePoint(new double[]{1.0, 2.0})),
                new CentroidCluster<>(new DoublePoint(new double[]{5.0, 6.0})),
                new CentroidCluster<>(new DoublePoint(new double[]{9.0, 10.0}))
        );

        // The closest centroid to [6.0, 7.0] is [5.0, 6.0], which is at index 1.
        double[] subvector = {6.0, 7.0};
        int closestIndex = PQuantization.closetCentroidIndex(subvector, codebook);
        assertEquals(1, closestIndex);
    }


    @Test
    public void testQuantize1D() {
        // 1. Create a sample codebook
        List<List<CentroidCluster<DoublePoint>>> codebooks = List.of(
                List.of(new CentroidCluster<>(new DoublePoint(new double[]{1.05})),
                        new CentroidCluster<>(new DoublePoint(new double[]{2.05}))),
                List.of(new CentroidCluster<>(new DoublePoint(new double[]{2.05})),
                        new CentroidCluster<>(new DoublePoint(new double[]{3.05}))),
                List.of(new CentroidCluster<>(new DoublePoint(new double[]{3.05})),
                        new CentroidCluster<>(new DoublePoint(new double[]{4.05}))),
                List.of(new CentroidCluster<>(new DoublePoint(new double[]{5.05})),
                        new CentroidCluster<>(new DoublePoint(new double[]{4.05}))));

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
        assertArrayEquals(new double[]{9.0f}, PQuantization.getSubVector(vector, 0, new int[]{1, 1, 1, 1}), EPSILON);
        assertArrayEquals(new double[]{11.0, 12.0}, PQuantization.getSubVector(vector, 1, new int[]{2, 2}), EPSILON);
    }

    @Test
    public void testDistanceBetween() {
        double[] vector1 = new double[]{1.0, 2.0};
        double[] vector2 = new double[]{3.0, 4.0};
        double[] vector3 = new double[]{5.0, 1.0};

        assertEquals(0, distanceBetween(vector1, vector1), EPSILON);
        assertEquals(Math.sqrt(8), distanceBetween(vector1, vector2), EPSILON);
        assertEquals(Math.sqrt(17), distanceBetween(vector1, vector3), EPSILON);
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
        assertArrayEquals(new double[]{9.0, 10.0, 11.0}, PQuantization.getSubVector(vector, 0, sizes), EPSILON);
        assertArrayEquals(new double[]{12.0, 13.0}, PQuantization.getSubVector(vector, 1, sizes), EPSILON);
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

        List<List<CentroidCluster<DoublePoint>>> result = PQuantization.createCodebooks(testVectors, M, K, PQuantization.getSubvectorSizes(5, M));
        assertNotNull(result);
        assertEquals(M, result.size());

        // Print results for manual inspection
        PQuantization.printCodebooks(result);
    }

    @Test
    public void testSameLengthVectors() {
        double[] v1 = {1.0, 2.0, 3.0, 4.0};
        double[] v2 = {2.0, 3.0, 4.0, 5.0};
        double expected = Math.sqrt(4.0);
        double result = distanceBetween(v1, v2);
        assertEquals(expected, result, 1e-9);
    }

    @Test
    public void testVectorLengthNotDivisibleBySpeciesPreferred() {
        double[] v1 = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] v2 = {2.0, 3.0, 4.0, 5.0, 6.0};
        double expected = Math.sqrt(5.0);
        double result = distanceBetween(v1, v2);
        assertEquals(expected, result, 1e-9);
    }

    @Test
    public void testEdgeCaseEmptyVectors() {
        double[] v1 = {};
        double[] v2 = {};
        double expected = 0.0;
        double result = distanceBetween(v1, v2);
        assertEquals(expected, result, 1e-9);
    }

    @Test
    public void testEdgeCaseSingleElementVectors() {
        double[] v1 = {1.0};
        double[] v2 = {2.0};
        double expected = 1.0;
        double result = distanceBetween(v1, v2);
        assertEquals(expected, result, 1e-9);
    }
}
