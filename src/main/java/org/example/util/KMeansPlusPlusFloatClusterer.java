package org.example.util;

import java.util.ArrayList;
import java.util.Arrays;
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
    private final List<List<float[]>> clusterPoints;
    private final float[][] centroidDistances;


    /**
     * Constructs a KMeansPlusPlusFloatClusterer with the specified number of clusters,
     * maximum iterations, and distance function.
     *
     * @param k number of clusters.
     * @param maxIterations maximum number of iterations for the clustering process.
     * @param distanceFunction a function to compute the distance between two points.
     */
    public KMeansPlusPlusFloatClusterer(int k, int maxIterations, BiFunction<float[], float[], Float> distanceFunction) {
        if (k <= 0) {
            throw new IllegalArgumentException("Number of clusters must be positive.");
        }
        this.k = k;
        this.maxIterations = maxIterations;
        this.distanceFunction = distanceFunction;
        this.random = new Random();
        this.clusterPoints = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            this.clusterPoints.add(new ArrayList<>());
        }
        centroidDistances = new float[k][k];
    }

    /**
     * Performs clustering on the provided set of points.
     *
     * @param points a list of points to be clustered.
     * @return a list of cluster centroids.
     */
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
                if (clusterPoints.get(j).isEmpty()) {
                    // Handle empty cluster by re-initializing the centroid
                    newCentroids.add(points.get(random.nextInt(points.size())));
                } else {
                    newCentroids.add(centroidOf(clusterPoints.get(j)));
                }
            }
            assignPointsToClusters(newCentroids, points, assignments);
            centroids = newCentroids;
            // update centroid distances
            for (int m = 0; m < centroids.size(); m++) {
                for (int n = m + 1; n < centroids.size(); n++) {
                    float distance = distanceFunction.apply(centroids.get(m), centroids.get(n));
                    centroidDistances[m][n] = distance;
                    centroidDistances[n][m] = distance; // Distance matrix is symmetric
                }
            }
        }

        return centroids;
    }

    /**
     * Chooses the initial centroids for clustering.
     *
     * The first centroid is chosen randomly from the data points. Subsequent centroids
     * are selected with a probability proportional to the square of their distance
     * to the nearest existing centroid. This ensures that the centroids are spread out
     * across the data and not initialized too closely to each other, leading to better
     * convergence and potentially improved final clusterings.
     * *
     * @param points a list of points from which centroids are chosen.
     * @return a list of initial centroids.
     */
    private List<float[]> chooseInitialCentroids(List<float[]> points) {
        List<float[]> centroids = new ArrayList<>();
        float[] distances = new float[points.size()];
        Arrays.fill(distances, Float.MAX_VALUE);

        // Choose the first centroid randomly
        float[] firstCentroid = points.get(random.nextInt(points.size()));
        centroids.add(firstCentroid);
        for (int i = 0; i < points.size(); i++) {
            float distance1 = distanceFunction.apply(points.get(i), firstCentroid);
            distances[i] = Math.min(distances[i], distance1);
        }

        // For each subsequent centroid
        for (int i = 1; i < k; i++) {
            float totalDistance = 0;
            for (float distance : distances) {
                totalDistance += distance;
            }

            float r = random.nextFloat() * totalDistance;
            int selectedIdx = -1;
            for (int j = 0; j < distances.length; j++) {
                r -= distances[j];
                if (r <= 0) {
                    selectedIdx = j;
                    break;
                }
            }

            if (selectedIdx == -1) {
                throw new IllegalStateException("Failed to select a centroid using the weighted distribution.");
            }

            float[] nextCentroid = points.get(selectedIdx);
            centroids.add(nextCentroid);

            // Update distances, but only if the new centroid provides a closer distance
            for (int j = 0; j < points.size(); j++) {
                float newDistance = distanceFunction.apply(points.get(j), nextCentroid);
                distances[j] = Math.min(distances[j], newDistance);
            }
        }

        return centroids;
    }

    /**
     * Assigns points to the nearest cluster.
     *
     * @param centroids a list of centroids.
     * @param points a list of points to be assigned.
     * @param assignments an array to store the cluster assignments.
     */
    private void assignPointsToClusters(List<float[]> centroids, List<float[]> points, int[] assignments) {
        for (List<float[]> cluster : clusterPoints) {
            cluster.clear();
        }

        for (int i = 0; i < points.size(); i++) {
            float[] point = points.get(i);
            int clusterIndex = getNearestCluster(point, centroids);
            clusterPoints.get(clusterIndex).add(point);
            assignments[i] = clusterIndex;
        }
    }

    private int getNearestCluster(float[] point, List<float[]> centroids) {
        float minDistance = Float.MAX_VALUE;
        int nearestCluster = 0;

        for (int i = 0; i < centroids.size(); i++) {
            if (i != nearestCluster) {
                // Using triangle inequality to potentially skip the computation
                float potentialMinDistance = Math.abs(minDistance - centroidDistances[nearestCluster][i]);
                if (potentialMinDistance >= minDistance) {
                    continue;
                }
            }

            float distance = distanceFunction.apply(point, centroids.get(i));
            if (distance < minDistance) {
                minDistance = distance;
                nearestCluster = i;
            }
        }

        return nearestCluster;
    }

    /**
     * Computes the centroid of a set of points.
     *
     * @param points a list of points.
     * @return the computed centroid.
     */
    public static float[] centroidOf(List<float[]> points) {
        if (points.isEmpty()) {
            throw new IllegalArgumentException("Can't compute centroid of empty points list");
        }

        float[] centroid = simdSum(points);
        simdDivInPlace(centroid, points.size());

        return centroid;
    }
}
