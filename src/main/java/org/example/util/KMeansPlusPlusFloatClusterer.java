package org.example.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;

public class KMeansPlusPlusFloatClusterer {

    private final int k;
    private final int maxIterations;
    private final BiFunction<float[], float[], Double> distanceFunction;
    private final Random random;

    public KMeansPlusPlusFloatClusterer(int k, int maxIterations, BiFunction<float[], float[], Double> distanceFunction) {
        this.k = k;
        this.maxIterations = maxIterations;
        this.distanceFunction = distanceFunction;
        this.random = new Random();
    }

    public List<float[]> cluster(List<float[]> points) {
        List<float[]> centroids = chooseInitialCentroids(points);
        int[] assignments = new int[points.size()];
        assignPointsToClusters(centroids, points, assignments);

        for (int i = 0; i < maxIterations; i++) {
            List<float[]> newCentroids = new ArrayList<>();
            for (float[] centroid : centroids) {
                newCentroids.add(computeCentroid(getPointsForCluster(centroid, points, assignments)));
            }
            reassignPointsToClusters(newCentroids, points, assignments);
            centroids = newCentroids;
        }

        return centroids;
    }

    private List<float[]> chooseInitialCentroids(List<float[]> points) {
        List<float[]> centroids = new ArrayList<>();
        centroids.add(points.get(random.nextInt(points.size())));

        while (centroids.size() < k) {
            float[] farthest = getFarthestPoint(centroids, points);
            centroids.add(farthest);
        }

        return centroids;
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

    private float[] getFarthestPoint(List<float[]> centroids, List<float[]> points) {
        double maxDistance = Double.NEGATIVE_INFINITY;
        float[] farthest = null;
        for (float[] point : points) {
            double distance = minDistanceToCentroid(point, centroids);
            if (distance > maxDistance) {
                maxDistance = distance;
                farthest = point;
            }
        }
        return farthest;
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

    private List<float[]> getPointsForCluster(float[] centroid, List<float[]> points, int[] assignments) {
        List<float[]> clusterPoints = new ArrayList<>();
        for (int i = 0; i < points.size(); i++) {
            if (assignments[i] == getCentroidIndex(centroid, assignments)) {
                clusterPoints.add(points.get(i));
            }
        }
        return clusterPoints;
    }

    private int getCentroidIndex(float[] centroid, int[] assignments) {
        for (int i = 0; i < assignments.length; i++) {
            if (assignments[i] == -1) {
                assignments[i] = i;
                return i;
            }
        }
        return -1;
    }

    private float[] computeCentroid(List<float[]> points) {
        if (points.isEmpty()) {
            throw new IllegalArgumentException("Can't compute centroid of empty points list");
        }

        float[] centroid = new float[points.get(0).length];
        for (float[] point : points) {
            for (int i = 0; i < centroid.length; i++) {
                centroid[i] += point[i];
            }
        }
        for (int i = 0; i < centroid.length; i++) {
            centroid[i] /= points.size();
        }
        return centroid;
    }

}
