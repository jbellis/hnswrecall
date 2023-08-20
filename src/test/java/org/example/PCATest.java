package org.example;

import org.example.util.PCA;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.example.util.SimdOps.simdMultiplyMatrix;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class PCATest {

    @Test
    public void testComputePCARotation() {
        // Create some sample data that lies approximately on the line y = x in 2D.
        List<float[]> data = List.of(
                new float[] {0.0f, 0.0f},
                new float[] {1.0f, 1.1f},
                new float[] {-1.0f, -0.9f},
                new float[] {2.0f, 2.1f},
                new float[] {-2.0f, -2.2f}
        );

        float[][] rotation = PCA.computePCARotation(data);
        System.out.println(Arrays.deepToString(rotation));

        // Since the data lies on the line y = x, the first principal component
        // should be close to the direction (1/sqrt(2), 1/sqrt(2))
        float expectedDirection = Math.abs((float) (1.0 / Math.sqrt(2)));

        assertEquals(expectedDirection, Math.abs(rotation[0][0]), 0.01);
        assertEquals(expectedDirection, Math.abs(rotation[1][0]), 0.01);
    }

    @Test
    public void testTransposeAndInverseRotation() {
        // Define a simple 2x2 rotation matrix that represents a 90-degree rotation.
        float[][] rotationMatrix = {
                {0.0f, -1.0f},
                {1.0f, 0.0f}
        };

        float[][] transposed = PCA.transpose(rotationMatrix);

        // A vector to test
        float[] originalVector = {1.0f, 0.0f};
        float[] rotatedVector = simdMultiplyMatrix(originalVector, rotationMatrix);
        float[] unrotatedVector = simdMultiplyMatrix(rotatedVector, transposed);

        // Check if unrotatedVector is close to originalVector
        assertArrayEquals(originalVector, unrotatedVector, 0.01f);
    }
}
