package org.example.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.BiFunction;

import static org.example.util.SimdOps.simdDivInPlace;
import static org.example.util.SimdOps.simdSum;

/**
 * A KMeans++ implementation for float vectors.  Optimizes to use SIMD vector
 * instructions, and to use the triangle inequality to skip distance calculations.
 * Roughly 3x faster than using the apache commons math implementation (with
 * conversions to double[]).
 */
public class KMeansPlusPlusFloatClusterer {
    private final BiFunction<float[], float[], Float> distanceFunction;
    private final Random random;
    private final List<List<float[]>> clusterPoints;
    private final float[][] centroidDistances;
    private List<float[]> points;
    private final int[] assignments;
    private final List<float[]> centroids;

    /**
     * Constructs a KMeansPlusPlusFloatClusterer with the specified number of clusters,
     * maximum iterations, and distance function.
     *
     * @param k number of clusters.
     * @param distanceFunction a function to compute the distance between two points.
     */
    public KMeansPlusPlusFloatClusterer(List<float[]> points, int k, BiFunction<float[], float[], Float> distanceFunction) {
        this(chooseInitialCentroids(points, k, distanceFunction), points, k, distanceFunction);
    }

    public KMeansPlusPlusFloatClusterer(List<float[]> centroids, List<float[]> points, int k, BiFunction<float[], float[], Float> distanceFunction) {
        if (k <= 0) {
            throw new IllegalArgumentException("Number of clusters must be positive.");
        }
        if (k > points.size()) {
            throw new IllegalArgumentException("Number of clusters cannot exceed number of points.");
        }

        this.points = points;
        this.distanceFunction = distanceFunction;
        this.random = new Random();
        this.clusterPoints = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            this.clusterPoints.add(new ArrayList<>());
        }
        centroidDistances = new float[k][k];
        this.centroids = centroids;
        updateCentroidDistances();
        assignments = new int[points.size()];
        assignPointsToClusters();
    }

    public List<float[]> getCentroids() {
        return centroids;
    }

    /**
     * Performs clustering on the provided set of points.
     *
     * @return a list of cluster centroids.
     */
    public List<float[]> cluster(int maxIterations) {
        for (int i = 0; i < maxIterations; i++) {
            int changedCount = clusterOnce(points);
            if (changedCount <= 0.01 * points.size()) {
                break;
            }
        }
        return centroids;
    }

    public int clusterOnce(List<float[]> newPoints) {
        points = newPoints;

        for (int j = 0; j < centroids.size(); j++) {
            if (clusterPoints.get(j).isEmpty()) {
                // Handle empty cluster by re-initializing the centroid
                centroids.set(j, points.get(random.nextInt(points.size())));
            } else {
                centroids.set(j, centroidOf(clusterPoints.get(j)));
            }
        }
        int changedCount = assignPointsToClusters();
        updateCentroidDistances();

        return changedCount;
    }

    private void updateCentroidDistances() {
        for (int m = 0; m < centroids.size(); m++) {
            for (int n = m + 1; n < centroids.size(); n++) {
                float distance = distanceFunction.apply(centroids.get(m), centroids.get(n));
                centroidDistances[m][n] = distance;
                centroidDistances[n][m] = distance; // Distance matrix is symmetric
            }
        }
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
    private static List<float[]> chooseInitialCentroids(List<float[]> points, int k, BiFunction<float[], float[], Float> distanceFunction) {
        var random = ThreadLocalRandom.current();
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
                if (r < 1e-6) {
                    selectedIdx = j;
                    break;
                }
            }

            if (selectedIdx == -1) {
                selectedIdx = random.nextInt(points.size());
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
     * Assigns points to the nearest cluster.  The results are stored as ordinals in `assignments`
     */
    private int assignPointsToClusters() {
        int changedCount = 0;

        for (List<float[]> cluster : clusterPoints) {
            cluster.clear();
        }

        for (int i = 0; i < points.size(); i++) {
            float[] point = points.get(i);
            int clusterIndex = getNearestCluster(point, centroids);

            // Check if assignment has changed
            if (assignments[i] != clusterIndex) {
                changedCount++;
            }

            clusterPoints.get(clusterIndex).add(point);
            assignments[i] = clusterIndex;
        }

        return changedCount;
    }

    /**
     * Return the index of the closest centroid to the given point
     */
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

    public static float[] centroidOf(List<float[]> points) {
        if (points.isEmpty()) {
            throw new IllegalArgumentException("Can't compute centroid of empty points list");
        }

        float[] centroid = simdSum(points);
        simdDivInPlace(centroid, points.size());

        return centroid;
    }
}
