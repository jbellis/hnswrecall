package org.example;

import org.example.util.KMeansPlusPlusFloatClusterer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class KMeansPlusPlusFloatClustererTest {
    private Random random = new Random();

    @Test
    public void testCluster() {
        List<float[]> points = new ArrayList<>();
        points.add(new float[] {1f, 2f});
        points.add(new float[] {3f, 4f});

        KMeansPlusPlusFloatClusterer clusterer = new KMeansPlusPlusFloatClusterer(1, 10,
                (a, b) -> euclideanDistance(a, b));

        List<float[]> centroids = clusterer.cluster(points);

        assertEquals(1, centroids.size());
        assertArrayEquals(new float[] {2f, 3f}, centroids.get(0), 0.01f);
    }

    @Test
    public void testTooManyClusters() {
        List<float[]> points = new ArrayList<>();
        points.add(new float[] {1f, 2f});

        assertThrows(IllegalArgumentException.class, () -> {
            KMeansPlusPlusFloatClusterer clusterer = new KMeansPlusPlusFloatClusterer(2, 10,
                    (a, b) -> euclideanDistance(a, b));
            clusterer.cluster(points);
        });
        KMeansPlusPlusFloatClusterer clusterer = new KMeansPlusPlusFloatClusterer(2, 10,
                (a, b) -> euclideanDistance(a, b));

        clusterer.cluster(points);
    }

    @Test
    public void testRepeatedPoints() {
        float[] point = new float[] {1f, 2f};
        List<float[]> points = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            points.add(point);
        }

        KMeansPlusPlusFloatClusterer clusterer = new KMeansPlusPlusFloatClusterer(10, 10,
                (a, b) -> euclideanDistance(a, b));

        List<float[]> centroids = clusterer.cluster(points);

        assertEquals(10, centroids.size());
        for (float[] centroid : centroids) {
            assertArrayEquals(point, centroid, 0.01f);
        }
    }

    private double euclideanDistance(float[] a, float[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }

}
