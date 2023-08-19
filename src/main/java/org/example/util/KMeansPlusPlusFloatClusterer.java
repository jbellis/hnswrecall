package org.example.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;

import static org.example.util.SimdOps.simdDivInPlace;
import static org.example.util.SimdOps.simdSum;

public class KMeansPlusPlusFloatClusterer {

    private final int k;
    private final int maxIterations;
    private final BiFunction<float[], float[], Float> distanceFunction;
    private final Random random;

    public KMeansPlusPlusFloatClusterer(int k, int maxIterations, BiFunction<float[], float[], Float> distanceFunction) {
        if (k <= 0) {
            throw new IllegalArgumentException("Number of clusters must be positive.");
        }
        this.k = k;
        this.maxIterations = maxIterations;
        this.distanceFunction = distanceFunction;
        this.random = new Random();
    }

    public List<float[]> cluster(List<float[]> points) {
        if (k > points.size()) {
            throw new IllegalArgumentException("Number of clusters cannot exceed number of points.");
        }

        List<float[]> centroids = chooseInitialCentroids(points);
        int[] assignments = new int[points.size()];
        assignPointsToClusters(centroids, points, assignments);

        for (int i = 0; i < maxIterations; i++) {
            List<float[]> newCentroids = new ArrayList<>();
            for (int j = 0; j < centroids.size(); j++) {
                List<float[]> clusterPoints = getPointsForCluster(j, points, assignments);
                if (clusterPoints.isEmpty()) {
                    // Handle empty cluster by re-initializing the centroid
                    newCentroids.add(getNextCentroid(centroids, points));
                } else {
                    newCentroids.add(centroidOf(clusterPoints));
                }
            }
            reassignPointsToClusters(newCentroids, points, assignments);
            centroids = newCentroids;
            System.out.println("Iteration " + i + " complete");
        }

        return centroids;
    }

    private List<float[]> chooseInitialCentroids(List<float[]> points) {
        List<float[]> centroids = new ArrayList<>();
        centroids.add(points.get(random.nextInt(points.size())));

        for (int i = 1; i < k; i++) {
            float[] nextCentroid = getNextCentroid(centroids, points);
            centroids.add(nextCentroid);
        }

        return centroids;
    }

    private float[] getNextCentroid(List<float[]> centroids, List<float[]> points) {
        double[] distances = new double[points.size()];
        double total = 0;

        for (int i = 0; i < points.size(); i++) {
            distances[i] = minDistanceToCentroid(points.get(i), centroids);
            total += distances[i];
        }

        double r = random.nextDouble() * total;

        for (int i = 0; i < distances.length; i++) {
            r -= distances[i];
            if (r <= 0) {
                return points.get(i);
            }
        }

        // Throw if we couldn't find a centroid
        throw new IllegalStateException("Failed to select a centroid using the weighted distribution");
    }

    private void assignPointsToClusters(List<float[]> centroids, List<float[]> points, int[] assignments) {
        for (int i = 0; i < points.size(); i++) {
            float[] point = points.get(i);
            assignments[i] = getNearestCluster(point, centroids);
        }
    }

    private void reassignPointsToClusters(List<float[]> newCentroids, List<float[]> points, int[] assignments) {
        for (int i = 0; i < points.size(); i++) {
            float[] point = points.get(i);
            assignments[i] = getNearestCluster(point, newCentroids);
        }
    }

    private int getNearestCluster(float[] point, List<float[]> centroids) {
        double minDistance = Double.MAX_VALUE;
        int nearestCluster = 0;
        for (int i = 0; i < centroids.size(); i++) {
            float[] centroid = centroids.get(i);
            double distance = distanceFunction.apply(point, centroid);
            if (distance < minDistance) {
                minDistance = distance;
                nearestCluster = i;
            }
        }
        return nearestCluster;
    }

    private double minDistanceToCentroid(float[] point, List<float[]> centroids) {
        double minDistance = Double.MAX_VALUE;
        for (float[] centroid : centroids) {
            double distance = distanceFunction.apply(point, centroid);
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
        return minDistance;
    }

    private List<float[]> getPointsForCluster(int centroidIndex, List<float[]> points, int[] assignments) {
        List<float[]> clusterPoints = new ArrayList<>();
        for (int i = 0; i < points.size(); i++) {
            if (assignments[i] == centroidIndex) {
                clusterPoints.add(points.get(i));
            }
        }
        return clusterPoints;
    }

    public static float[] centroidOf(List<float[]> points) {
        if (points.isEmpty()) {
            throw new IllegalArgumentException("Can't compute centroid of empty points list");
        }

        float[] centroid = simdSum(points);
        simdDivInPlace(centroid, points.size());

        return centroid;
    }
}
