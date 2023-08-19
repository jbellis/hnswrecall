package org.example;

import org.example.util.KMeansPlusPlusFloatClusterer;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

public class KMeansPlusPlusFloatClustererTest {
    @Test
    public void testMultiplePointsOneCluster() {
        List<float[]> points = Arrays.asList(
                new float[] {1, 2},
                new float[] {1.1f, 2.2f},
                new float[] {0.9f, 1.8f},
                new float[] {1.2f, 2.1f}
        );

        KMeansPlusPlusFloatClusterer clusterer = new KMeansPlusPlusFloatClusterer(1, 10,
                this::euclideanDistance);

        List<float[]> centroids = clusterer.cluster(points);

        assertEquals(1, centroids.size());
        assertArrayEquals(new float[] {1.05f, 2.025f}, centroids.get(0), 0.01f);
    }

    @Test
    public void testDistinctClusters() {
        List<float[]> points = Arrays.asList(
                new float[] {1, 1},
                new float[] {1.1f, 1.2f},
                new float[] {1.2f, 1.1f},
                new float[] {10, 10},
                new float[] {10.1f, 10.2f},
                new float[] {9.9f, 10.1f}
        );

        KMeansPlusPlusFloatClusterer clusterer = new KMeansPlusPlusFloatClusterer(2, 10,
                this::euclideanDistance);

        List<float[]> centroids = clusterer.cluster(points);

        // Sort centroids based on their first dimension
        centroids.sort(Comparator.comparingDouble(a -> a[0]));

        assertEquals(2, centroids.size());
        assertArrayEquals(new float[] {1.1f, 1.1f}, centroids.get(0), 0.01f);
        assertArrayEquals(new float[] {10f, 10.1f}, centroids.get(1), 0.01f);
    }

    @Test
    public void testZeroPoints() {
        List<float[]> points = new ArrayList<>();

        KMeansPlusPlusFloatClusterer clusterer = new KMeansPlusPlusFloatClusterer(2, 10,
                this::euclideanDistance);

        assertThrows(IllegalArgumentException.class, () -> clusterer.cluster(points));
    }

    @Test
    public void testAllPointsIdentical() {
        float[] point = new float[] {1, 2};
        List<float[]> points = Collections.nCopies(100, point);

        KMeansPlusPlusFloatClusterer clusterer = new KMeansPlusPlusFloatClusterer(10, 10,
                this::euclideanDistance);

        List<float[]> centroids = clusterer.cluster(points);

        assertEquals(10, centroids.size());
        for (float[] centroid : centroids) {
            assertArrayEquals(point, centroid, 0.01f);
        }
    }

    @Test
    public void testCluster() {
        List<float[]> points = new ArrayList<>();
        points.add(new float[] {1f, 2f});
        points.add(new float[] {3f, 4f});

        KMeansPlusPlusFloatClusterer clusterer = new KMeansPlusPlusFloatClusterer(1, 10,
                this::euclideanDistance);

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
                    this::euclideanDistance);
            clusterer.cluster(points);
        });
    }

    @Test
    public void testRepeatedPoints() {
        float[] point = new float[] {1f, 2f};
        List<float[]> points = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            points.add(point);
        }

        KMeansPlusPlusFloatClusterer clusterer = new KMeansPlusPlusFloatClusterer(10, 10,
                this::euclideanDistance);

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
