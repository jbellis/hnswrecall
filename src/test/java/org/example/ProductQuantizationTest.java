package org.example;

import org.apache.lucene.util.VectorUtil;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

public class ProductQuantizationTest {
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

        var result = ProductQuantization.createCodebooks(testVectors, M, ProductQuantization.getSubvectorSizes(4, M));
        // Print results for manual inspection
        ProductQuantization.printCodebooks(result);

        // quantize the vectors
        var quantized = new ProductQuantization(testVectors, M, false).encodeAll(testVectors);
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
        int closestIndex = ProductQuantization.closetCentroidIndex(subvector, codebook);
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
        int[] sizes = ProductQuantization.getSubvectorSizes(4, M);

        var vector = new float[]{1.0f, 2.0f, 3.0f, 4.0f};
        List<Integer> indices = IntStream.range(0, M)
                .mapToObj(m -> {
                    // find the closest centroid in the corresponding codebook to each subvector
                    return ProductQuantization.closetCentroidIndex(ProductQuantization.getSubVector(vector, m, sizes), codebooks.get(m));
                })
                .toList();
        assertEquals(List.of(0, 0, 0, 1), indices);
    }

    @Test
    public void testGetSubVector() {
        float[] vector = new float[]{9.0f, 10.0f, 11.0f, 12.0f};
        assertArrayEquals(new float[]{9.0f}, ProductQuantization.getSubVector(vector, 0, new int[]{1, 1, 1, 1}), EPSILON);
        assertArrayEquals(new float[]{11.0f, 12.0f}, ProductQuantization.getSubVector(vector, 1, new int[]{2, 2}), EPSILON);
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
            byte[] result = ProductQuantization.toBytes(testCases.get(i), 4);
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
        assertArrayEquals(new float[]{9.0f, 10.0f, 11.0f}, ProductQuantization.getSubVector(vector, 0, sizes), EPSILON);
        assertArrayEquals(new float[]{12.0f, 13.0f}, ProductQuantization.getSubVector(vector, 1, sizes), EPSILON);
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

        var result = ProductQuantization.createCodebooks(testVectors, M, ProductQuantization.getSubvectorSizes(5, M));
        assertNotNull(result);
        assertEquals(M, result.size());
        // Print results for manual inspection
         ProductQuantization.printCodebooks(result);
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
        ProductQuantization pq = new ProductQuantization(testVectors, M, false);

        for (float[] originalVector : testVectors) {
            byte[] encoded = pq.encode(originalVector);
            float[] target = new float[originalVector.length];
            float[] decoded = pq.decode(encoded, target);

            for (int i = 0; i < originalVector.length; i++) {
                assertTrue(Math.abs(originalVector[i] - decoded[i]) < 0.1,
                        "Difference detected in component " + i + ": Expected " + originalVector[i] + ", but got " + decoded[i]);
            }
        }
    }
}
