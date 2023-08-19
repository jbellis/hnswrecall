package org.example;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.util.Arrays;
import java.util.List;

public class PQuantizationTest {

    private PQuantization pqQuantization;
    private static final float EPSILON = 1e-6f;

    @BeforeEach
    public void setUp() {
        List<float[]> sampleData = Arrays.asList(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new float[]{5.0f, 6.0f, 7.0f, 8.0f}
        );
        pqQuantization = new PQuantization(sampleData);
    }

    @Test
    public void testQuantizeAll() {
        List<float[]> vectors = Arrays.asList(
                new float[]{9.0f, 10.0f, 11.0f, 12.0f}
        );
        List<byte[]> quantized = pqQuantization.quantizeAll(vectors);
        // Check the size and possibly some properties of the quantized output
        assertEquals(1, quantized.size());
        // More assertions based on expected quantization
    }

    @Test
    public void testQuantize() {
        float[] vector = new float[]{9.0f, 10.0f, 11.0f, 12.0f};
        byte[] quantized = pqQuantization.quantize(vector);
        // More assertions based on expected quantization
    }

    @Test
    public void testGetSubVector() {
        float[] vector = new float[]{9.0f, 10.0f, 11.0f, 12.0f};
        double[] subVector = pqQuantization.getSubVector(vector, 1);
        assertArrayEquals(new double[]{11.0, 12.0}, subVector, EPSILON);
    }

    @Test
    public void testDistanceBetween() {
        double[] vector1 = new double[]{1.0, 2.0};
        double[] vector2 = new double[]{3.0, 4.0};
        double distance = pqQuantization.distanceBetween(vector1, vector2);
        assertEquals(Math.sqrt(8.0), distance, EPSILON);
    }
}
