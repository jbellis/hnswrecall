package org.example;

import io.jhdf.HdfFile;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.NamedThreadFactory;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.*;
import org.example.util.ListRandomAccessVectorValues;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Supplier;
import java.util.stream.IntStream;

public class Bench {
    public static void testRecall(List<float[]> baseVectors, List<float[]> queryVectors, List<Set<Integer>> groundTruth) throws IOException, ExecutionException, InterruptedException {
        var ravv = new ListRandomAccessVectorValues(baseVectors, baseVectors.get(0).length);
        var topK = groundTruth.get(0).size();
        int M = 16;
        int beamWidth = 100;

        var start = System.nanoTime();
        ConcurrentOnHeapHnswGraph hnsw = buildGraph(ravv, M, beamWidth);
        long buildNanos = System.nanoTime() - start;

        int queryRuns = 10;
        start = System.nanoTime();
        var pqr = performQueries(queryVectors, groundTruth, ravv, hnsw::getView, topK, queryRuns);
        long queryNanos = System.nanoTime() - start;
        var recall = ((double) pqr.topKFound) / (queryRuns * queryVectors.size() * topK);
        System.out.format("HNSW: top %d recall %.4f, build %.2fs, queryx%d %.2fs. %s nodes visited%n",
                topK, recall, buildNanos / 1_000_000_000.0, queryRuns, queryNanos / 1_000_000_000.0, pqr.nodesVisited);
    }

    private static ConcurrentOnHeapHnswGraph buildGraph(ListRandomAccessVectorValues ravv, int M, int beamWidth) throws IOException, InterruptedException, ExecutionException {
        var builder = ConcurrentHnswGraphBuilder.create(ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.DOT_PRODUCT, M, beamWidth);
        int buildThreads = Runtime.getRuntime().availableProcessors();
        var es = Executors.newFixedThreadPool(
                buildThreads, new NamedThreadFactory("Concurrent HNSW builder"));
        var hnsw = builder.buildAsync(ravv.copy(), es, buildThreads).get();
        es.shutdown();
        return hnsw;
    }

    private static float normOf(float[] baseVector) {
        float norm = 0;
        for (float v : baseVector) {
            norm += v * v;
        }
        return (float) Math.sqrt(norm);
    }

    private record ResultSummary(int topKFound, int nodesVisited) { }

    private static ResultSummary performQueries(List<float[]> queryVectors, List<Set<Integer>> groundTruth, ListRandomAccessVectorValues ravv, Supplier<HnswGraph> graphSupplier, int topK, int queryRuns) {
        LongAdder topKfound = new LongAdder();
        LongAdder nodesVisited = new LongAdder();
        for (int k = 0; k < queryRuns; k++) {
            IntStream.range(0, queryVectors.size()).parallel().forEach(i -> {
                var queryVector = queryVectors.get(i);
                NeighborQueue nn;
                try {
                    // the code supports for querying for more than topK, and then using
                    // just the topK results to get more accuracy
                    // (see FAQ entry about "LIMIT 2 query will return a better result in the first row than a LIMIT 1 query"
                    // in https://docs.google.com/document/d/1P_elPGmuiwuwzzIkYMie6z4gmKH3XX8qGaNso4NyY-Q)
                    int overquery = topK; // 4 * topK
                    nn = HnswGraphSearcher.search(queryVector, overquery, ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.DOT_PRODUCT, graphSupplier.get(), null, Integer.MAX_VALUE);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
                var gt = groundTruth.get(i);
                var a = new int[nn.size()];
                for (int j = a.length - 1; j >= 0; j--) {
                    a[j] = nn.pop();
                }
                var n = IntStream.range(0, Math.min(a.length, topK)).filter(j -> gt.contains(a[j])).count();
                topKfound.add(n);
                nodesVisited.add(nn.visitedCount());
            });
        }
        return new ResultSummary((int) topKfound.sum(), (int) nodesVisited.sum());
    }

    private static void computeRecallFor(String pathStr) throws IOException, ExecutionException, InterruptedException {
        float[][] baseVectors;
        float[][] queryVectors;
        int[][] groundTruth;
        try (HdfFile hdf = new HdfFile(Paths.get(pathStr))) {
            baseVectors = (float[][]) hdf.getDatasetByPath("train").getData();
            queryVectors = (float[][]) hdf.getDatasetByPath("test").getData();
            groundTruth = (int[][]) hdf.getDatasetByPath("neighbors").getData();
        }

        // verify that vectors are normalized and sane
        List<float[]> scrubbedBaseVectors = new ArrayList<>(baseVectors.length);
        List<float[]> scrubbedQueryVectors = new ArrayList<>(queryVectors.length);
        List<Set<Integer>> gtSet = new ArrayList<>(groundTruth.length);
        // remove zero vectors, noting that this will change the indexes of the ground truth answers
        // [nytimes dataset includes zero vectors, wtf were they thinking]
        Map<Integer, Integer> rawToScrubbed = new HashMap<>();
        {
            int j = 0;
            for (int i = 0; i < baseVectors.length; i++) {
                float[] v = baseVectors[i];
                if (Math.abs(normOf(v)) > 1e-5) {
                    scrubbedBaseVectors.add(v);
                    rawToScrubbed.put(i, j++);
                }
            }
        }
        for (int i = 0; i < queryVectors.length; i++) {
            float[] v = queryVectors[i];
            if (Math.abs(normOf(v)) > 1e-5) {
                scrubbedQueryVectors.add(v);
                var gt = new HashSet<Integer>();
                for (int j = 0; j < groundTruth[i].length; j++) {
                    gt.add(rawToScrubbed.get(groundTruth[i][j]));
                }
                gtSet.add(gt);
            }
        }
        // now that the zero vectors are removed, we can normalize
        if (Math.abs(normOf(baseVectors[0]) - 1.0) > 1e-5) {
            normalizeAll(scrubbedBaseVectors);
            normalizeAll(scrubbedQueryVectors);
        }
        assert scrubbedQueryVectors.size() == gtSet.size();
        // clear the reference so it can be GC'd
        baseVectors = null;
        queryVectors = null;
        groundTruth = null;

        System.out.format("%s: %d base and %d query vectors loaded, dimensions %d%n",
                pathStr, scrubbedBaseVectors.size(), scrubbedQueryVectors.size(), scrubbedBaseVectors.get(0).length);

        testRecall(scrubbedBaseVectors, scrubbedQueryVectors, gtSet);
    }

    private static void normalizeAll(Iterable<float[]> vectors) {
        for (float[] v : vectors) {
            VectorUtil.l2normalize(v);
        }
    }

    public static void main(String[] args) throws IOException, ExecutionException, InterruptedException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        computeRecallFor("hdf5/nytimes-256-angular.hdf5");
    }
}
