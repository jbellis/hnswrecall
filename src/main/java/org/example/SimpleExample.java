package org.example;

import org.apache.lucene.index.*;
import org.apache.lucene.util.hnsw.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/**
 * A simple example of how to use the HNSW graph implementation.
 *
 * Creates a random universe of vectors, and performs search of random vectors against it.
 *
 * No correctness checks are performed, this is just an example of the API.
 */
public class SimpleExample {
    private static final VectorSimilarityFunction similarityFunction = VectorSimilarityFunction.COSINE;
    private static final Random random = new Random();

    public static void main(String[] args) throws IOException {
        // Create a random vector universe
        var vectorDimensions = 1500;
        var universeSize = 10_000;
        var universe = new ArrayList<float[]>(universeSize);
        for (var i = 0; i < universeSize; i++) {
            universe.add(randomVector(vectorDimensions));
        }

        // construct a HNSW graph of the universe
        var ravv = new ListRandomAccessVectorValues(universe, vectorDimensions);
        var builder = HnswGraphBuilder.create(ravv, VectorEncoding.FLOAT32, similarityFunction, 16, 100, random.nextInt());
        var hnsw = builder.build(ravv.copy());

        // search for the nearest neighbors of a random vector
        var queryVector = randomVector(vectorDimensions);
        var nn = HnswGraphSearcher.search(queryVector, 10, ravv.copy(), VectorEncoding.FLOAT32, similarityFunction, hnsw, null, Integer.MAX_VALUE);
        System.out.println("Nearest neighbors of " + Arrays.toString(queryVector) + ":");
        for (var i : nn.nodes()) {
            var neighbor = universe.get(i);
            var similarity = similarityFunction.compare(queryVector, neighbor);
            System.out.println("  " + Arrays.toString(neighbor) + " (similarity: " + similarity + ")");
        }
    }

    private static float[] randomVector(int vectorDimension) {
        var queryVector = new float[vectorDimension];
        for (var i = 0; i < vectorDimension; i++) {
            queryVector[i] = random.nextFloat();
        }
        return queryVector;
    }
}
