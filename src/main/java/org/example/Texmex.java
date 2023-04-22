package org.example;

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.example.util.ListRandomAccessVectorValues;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.IntStream;

/**
 * Tests HNSW against vectors from the Texmex dataset: http://corpus-texmex.irisa.fr/
 */
public class Texmex {
    public static ArrayList<float[]> readFvecs(String filePath) throws IOException {
        var vectors = new ArrayList<float[]>();
        try (var dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {
            while (dis.available() > 0) {
                var dimension = Integer.reverseBytes(dis.readInt());
                assert dimension > 0 : dimension;
                var buffer = new byte[dimension * Float.BYTES];
                dis.readFully(buffer);
                var byteBuffer = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN);

                var vector = new float[dimension];
                for (var i = 0; i < dimension; i++) {
                    vector[i] = byteBuffer.getFloat();
                }
                vectors.add(vector);
            }
        }
        return vectors;
    }

    private static ArrayList<HashSet<Integer>> readIvecs(String filename) {
        var groundTruthTopK = new ArrayList<HashSet<Integer>>();

        try (var dis = new DataInputStream(new FileInputStream(filename))) {
            while (dis.available() > 0) {
                var numNeighbors = Integer.reverseBytes(dis.readInt());
                var neighbors = new HashSet<Integer>(numNeighbors);

                for (var i = 0; i < numNeighbors; i++) {
                    var neighbor = Integer.reverseBytes(dis.readInt());
                    neighbors.add(neighbor);
                }

                groundTruthTopK.add(neighbors);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return groundTruthTopK;
    }

    public static double testRecall(String siftName) throws IOException {
        var baseVectors = readFvecs(String.format("%s/%s_base.fvecs", siftName, siftName));
        var queryVectors = readFvecs(String.format("%s/%s_query.fvecs", siftName, siftName));
        var groundTruth = readIvecs(String.format("%s/%s_groundtruth.ivecs", siftName, siftName));

        var ravv = new ListRandomAccessVectorValues(baseVectors, baseVectors.get(0).length);
        var builder = HnswGraphBuilder.create(ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, 16, 100, ThreadLocalRandom.current().nextInt());
        var hnsw = builder.build(ravv.copy());

        var topKfound = new AtomicInteger(0);
        var topK = 100;
        IntStream.range(0, queryVectors.size()).forEach(i -> {
            var queryVector = queryVectors.get(i);
            NeighborQueue nn;
            try {
                // search is not threadsafe, not sure why
                nn = HnswGraphSearcher.search(queryVector, 100, ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.COSINE, hnsw, null, Integer.MAX_VALUE);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            var gt = groundTruth.get(i);
            var n = IntStream.range(0, topK).filter(j -> gt.contains(nn.nodes()[j])).count();
            topKfound.addAndGet((int) n);
        });
        return (double) topKfound.get() / (queryVectors.size() * topK);
    }

    public static void main(String[] args) throws IOException {
        // Average recall and standard deviation over 100 runs
        int numRuns = 100;

        DoubleAdder totalRecall = new DoubleAdder();
        DoubleAdder totalRecallSquared = new DoubleAdder();

        IntStream.range(0, numRuns).parallel()
                .mapToDouble(i -> {
                    try {
                        return testRecall("siftsmall");
                    } catch (IOException e) {
                        e.printStackTrace();
                        return 0;
                    }
                })
                .forEach(recall -> {
                    totalRecall.add(recall);
                    totalRecallSquared.add(recall * recall);
                });

        double averageRecall = totalRecall.doubleValue() / numRuns;
        double variance = (totalRecallSquared.doubleValue() / numRuns) - (averageRecall * averageRecall);
        double stdev = Math.sqrt(variance);

        System.out.println("Average Recall: " + averageRecall);
        System.out.println("Standard Deviation: " + stdev);
    }
}
