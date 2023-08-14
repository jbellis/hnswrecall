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

/**
 * Tests HNSW against vectors from the Texmex dataset
 */
public class Texmex {
    public static void testRecall(int M, int beamWidth, VectorSimilarityFunction similarityFunction, List<float[]> baseVectors, List<float[]> queryVectors, List<Set<Integer>> groundTruth)
            throws IOException, ExecutionException, InterruptedException
    {
        var ravv = new ListRandomAccessVectorValues(baseVectors, baseVectors.get(0).length);
        var topK = groundTruth.get(0).size();

        // build the graphs on multiple threads
        var start = System.nanoTime();
        var builder = ConcurrentHnswGraphBuilder.create(ravv, VectorEncoding.FLOAT32, similarityFunction, M, beamWidth);
        int buildThreads = 24;
        var es = Executors.newFixedThreadPool(buildThreads, new NamedThreadFactory("Concurrent HNSW builder"));
        var hnsw = builder.buildAsync(ravv.copy(), es, buildThreads).get();
        var vBuilder = new VamanaGraphBuilder<>(hnsw, ravv, VectorEncoding.FLOAT32, similarityFunction, 2 * beamWidth);
        long buildNanos = System.nanoTime() - start;
        es.shutdown();

        // query hnsw baseline
        int queryRuns = 10;
        start = System.nanoTime();
        var pqr = performQueries(queryVectors, groundTruth, ravv, hnsw::getView, topK, queryRuns);
        long queryNanos = System.nanoTime() - start;
        var recall = ((double) pqr.topKFound) / (queryRuns * queryVectors.size() * topK);
        System.out.format("HNSW M=%d ef=%d: top %d recall %.4f, build %.2fs, query %.2fs. %s nodes visited%n",
                M, beamWidth, topK, recall, buildNanos / 1_000_000_000.0, queryNanos / 1_000_000_000.0, pqr.nodesVisited);

        // query vamana
        var vStart = System.nanoTime();
        ResultSummary vqr;
        long vBuildNanos;
        float alpha = 1.5f;
        // x2 b/c OnHeapHnswGraph doubles connections on L0
        var vamana = es.submit(() -> vBuilder.buildVamana(2 * M, alpha)).get();
        vBuildNanos = System.nanoTime() - vStart;
        vStart = System.nanoTime();
        vqr = vamanaQueries(vamana, queryVectors, groundTruth, ravv, topK, queryRuns);
        var vQueryNanos = System.nanoTime() - vStart;
        var vRecall = ((double) vqr.topKFound) / (queryRuns * queryVectors.size() * topK);
        System.out.format("Vamana M=%d ef=%d alpha=%.2f: top %d recall %.4f, build %.2fs, query %.2fs. %s nodes visited%n",
                M, beamWidth, alpha, topK, vRecall, vBuildNanos / 1_000_000_000.0, vQueryNanos / 1_000_000_000.0, vqr.nodesVisited);
    }

    private static float normOf(float[] baseVector) {
        float norm = 0;
        for (float v : baseVector) {
            norm += v * v;
        }
        return (float) Math.sqrt(norm);
    }

    private record ResultSummary(int topKFound, int nodesVisited) { }

    private static ResultSummary vamanaQueries(ConcurrentVamanaGraph vamana, List<float[]> queryVectors, List<Set<Integer>> groundTruth, ListRandomAccessVectorValues ravv, int topK, int queryRuns) {
        LongAdder topKfound = new LongAdder();
        LongAdder nodesVisited = new LongAdder();
        var greedySearcher = ThreadLocal.withInitial(() -> new VamanaSearcher<>(vamana, ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.DOT_PRODUCT));
        for (int k = 0; k < queryRuns; k++) {
            IntStream.range(0, queryVectors.size()).parallel().forEach(i -> {
                var queryVector = queryVectors.get(i);
                VamanaSearcher.QueryResult qr;
                try {
                    qr = greedySearcher.get().search(queryVector, 2 * topK);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
                var gt = groundTruth.get(i);
                int[] resultNodes = qr.results.node();
                var n = IntStream.range(0, Math.min(resultNodes.length, topK)).filter(j -> gt.contains(resultNodes[j])).count();
                topKfound.add(n);
                nodesVisited.add(qr.visitedCount);
            });
        }
        return new ResultSummary((int) topKfound.sum(), (int) nodesVisited.sum());
    }

    private static ResultSummary performQueries(List<float[]> queryVectors, List<Set<Integer>> groundTruth, ListRandomAccessVectorValues ravv, Supplier<HnswGraph> graphSupplier, int topK, int queryRuns) {
        LongAdder topKfound = new LongAdder();
        LongAdder nodesVisited = new LongAdder();
        for (int k = 0; k < queryRuns; k++) {
            IntStream.range(0, queryVectors.size()).parallel().forEach(i -> {
                var queryVector = queryVectors.get(i);
                NeighborQueue nn;
                try {
                    nn = HnswGraphSearcher.search(queryVector, topK, ravv, VectorEncoding.FLOAT32, VectorSimilarityFunction.DOT_PRODUCT, graphSupplier.get(), null, Integer.MAX_VALUE);
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

    private static void computeRecallFor(int M, int beamWidth, String pathStr) throws IOException, ExecutionException, InterruptedException {
        // infer the similarity
        VectorSimilarityFunction similarityFunction;
        if (pathStr.contains("angular")) {
            similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
        } else if (pathStr.contains("euclidean")) {
            similarityFunction = VectorSimilarityFunction.EUCLIDEAN;
        } else {
            throw new IllegalArgumentException("Unknown similarity function -- expected angular or euclidean for " + pathStr);
        }

        // read the data
        float[][] baseVectors;
        float[][] queryVectors;
        int[][] groundTruth;
        try (HdfFile hdf = new HdfFile(Paths.get(pathStr))) {
            baseVectors = (float[][]) hdf.getDatasetByPath("train").getData();
            queryVectors = (float[][]) hdf.getDatasetByPath("test").getData();
            groundTruth = (int[][]) hdf.getDatasetByPath("neighbors").getData();
        }

        List<float[]> scrubbedBaseVectors;
        List<float[]> scrubbedQueryVectors;
        List<Set<Integer>> gtSet;
        if (similarityFunction == VectorSimilarityFunction.DOT_PRODUCT) {
            // verify that vectors are normalized and sane
            scrubbedBaseVectors = new ArrayList<>(baseVectors.length);
            scrubbedQueryVectors = new ArrayList<>(queryVectors.length);
            gtSet = new ArrayList<>(groundTruth.length);
            // remove zero vectors, noting that this will change the indexes of the ground truth answers
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
        } else {
            scrubbedBaseVectors = Arrays.asList(baseVectors);
            scrubbedQueryVectors = Arrays.asList(queryVectors);
            gtSet = new ArrayList<>(groundTruth.length);
            for (int[] gt : groundTruth) {
                var gtSetForQuery = new HashSet<Integer>();
                for (int i : gt) {
                    gtSetForQuery.add(i);
                }
                gtSet.add(gtSetForQuery);
            }
        }

        System.out.format("%s: %d base and %d query vectors loaded, dimensions %d%n",
                pathStr, scrubbedBaseVectors.size(), scrubbedQueryVectors.size(), scrubbedBaseVectors.get(0).length);

        testRecall(M, beamWidth, similarityFunction, scrubbedBaseVectors, scrubbedQueryVectors, gtSet);
    }

    private static void normalizeAll(Iterable<float[]> vectors) {
        for (float[] v : vectors) {
            VectorUtil.l2normalize(v);
        }
    }

    public static void main(String[] args) throws IOException, ExecutionException, InterruptedException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        for (int M : List.of(8, 12, 16, 24, 32, 40)) {
            for (int beamWidth: List.of(80, 100, 120, 160, 200, 400, 600, 800)) {
                // angular
                computeRecallFor(M, beamWidth, "hdf5/nytimes-256-angular.hdf5");
                computeRecallFor(M, beamWidth, "hdf5/glove-100-angular.hdf5");
                computeRecallFor(M, beamWidth, "hdf5/glove-200-angular.hdf5");

                // euclidean
                computeRecallFor(M, beamWidth, "hdf5/sift-128-euclidean.hdf5");
                computeRecallFor(M, beamWidth, "hdf5/fashion-mnist-784-euclidean.hdf5");

                // need large file support
//              computeRecallFor(M, beamWidth, "hdf5/deep-image-96-angular.hdf5");
//              computeRecallFor(M, beamWidth, "hdf5/gist-960-euclidean.hdf5");
            }
        }

    }
}
