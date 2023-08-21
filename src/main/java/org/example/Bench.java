package org.example;

import io.jhdf.HdfFile;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.NamedThreadFactory;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.*;
import org.example.util.ListRandomAccessVectorValues;
import org.example.util.PQRandomAccessVectorValues;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Tests HNSW against vectors from the Texmex dataset
 */
public class Bench {
    private static void testRecall(int M, int efConstruction, List<Integer> efSearchOptions, DataSet ds, PQRandomAccessVectorValues pqvv)
            throws ExecutionException, InterruptedException
    {
        var floatVectors = new ListRandomAccessVectorValues<>(ds.baseVectors, ds.baseVectors.get(0).length);
        var topK = ds.groundTruth.get(0).size();

        // build the graphs on multiple threads
        var start = System.nanoTime();
        var builder = new ConcurrentHnswGraphBuilder<>(floatVectors, VectorEncoding.FLOAT32, ds.similarityFunction, M, efConstruction, 1.5f, 1.4f);
        int buildThreads = Runtime.getRuntime().availableProcessors();
        var es = Executors.newFixedThreadPool(buildThreads, new NamedThreadFactory("Concurrent HNSW builder"));
        var hnsw = builder.buildAsync(floatVectors.copy(), es, buildThreads).get();
        es.shutdown();
        System.out.format("HNSW M=%d ef=%d build in %.2fs,%n",
                M, efConstruction, (System.nanoTime() - start) / 1_000_000_000.0);

        int queryRuns = 2;
        for (int overquery : efSearchOptions) {
            start = System.nanoTime();
            var pqr = performQueries(ds, pqvv, hnsw::getView, topK, topK * overquery, queryRuns);
            var recall = ((double) pqr.topKFound) / (queryRuns * ds.queryVectors.size() * topK);
            System.out.format("HNSW top %d/%d recall %.4f, query %.2fs. %s nodes visited%n",
                    topK, overquery, recall, (System.nanoTime() - start) / 1_000_000_000.0, pqr.nodesVisited);
        }
    }

    private static float normOf(float[] baseVector) {
        float norm = 0;
        for (float v : baseVector) {
            norm += v * v;
        }
        return (float) Math.sqrt(norm);
    }

    private record ResultSummary(int topKFound, int nodesVisited) { }

    private static long topKCorrect(int topK, int[] resultNodes, Set<Integer> gt) {
        int count = Math.min(resultNodes.length, topK);
        // stream the first count results into a Set
        var resultSet = Arrays.stream(resultNodes, 0, count)
                .boxed()
                .collect(Collectors.toSet());
        assert resultSet.size() == count : String.format("%s duplicate results out of %s", count - resultSet.size(), count);
        return resultSet.stream().filter(gt::contains).count();
    }

    private static long topKCorrect(int topK, NeighborQueue nn, Set<Integer> gt) {
        var a = new int[nn.size()];
        for (int j = a.length - 1; j >= 0; j--) {
            a[j] = nn.pop();
        }
        return topKCorrect(topK, a, gt);
    }

    private static ResultSummary performQueries(DataSet ds,
                                                PQRandomAccessVectorValues pqvv,
                                                Supplier<HnswGraph> graphSupplier,
                                                int topK,
                                                int efSearch,
                                                int queryRuns)
    {
        assert efSearch >= topK;
        LongAdder topKfound = new LongAdder();
        LongAdder nodesVisited = new LongAdder();
        for (int k = 0; k < queryRuns; k++) {
            IntStream.range(0, ds.queryVectors.size()).parallel().forEach(i -> {
                var queryVector = ds.queryVectors.get(i);
                NeighborQueue nn;
                try {
                    NeighborSimilarity.ScoreFunction sf = (other) -> pqvv.decodedScore(other, queryVector);
                    nn = new HnswSearcher.Builder<>(graphSupplier.get(), pqvv, sf)
                            .build()
                            .search(efSearch, null, Integer.MAX_VALUE);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }

                int[] results;
                if (efSearch > topK) {
                    // re-rank the quantized results by the original similarity function
                    // Decorate
                    int[] raw = new int[nn.size()];
                    for (int j = raw.length - 1; j >= 0; j--) {
                        raw[j] = nn.pop();
                    }
                    // Pair each item in `raw` with its computed similarity
                    Map.Entry<Integer, Double>[] decorated = new AbstractMap.SimpleEntry[raw.length];
                    for (int j = 0; j < raw.length; j++) {
                        double similarity = ds.similarityFunction.compare(queryVector, ds.baseVectors.get(raw[j]));
                        decorated[j] = new AbstractMap.SimpleEntry<>(raw[j], similarity);
                    }
                    // Sort based on the computed similarity
                    Arrays.sort(decorated, (p1, p2) -> Double.compare(p2.getValue(), p1.getValue())); // Note the order for reversed sort
                    // Undecorate
                    results = Arrays.stream(decorated).mapToInt(Map.Entry::getKey).toArray();
                } else {
                    results = new int[nn.size()];
                    for (int j = results.length - 1; j >= 0; j--) {
                        results[j] = nn.pop();
                    }
                }

                var gt = ds.groundTruth.get(i);
                var n = topKCorrect(topK, results, gt);
                topKfound.add(n);
                nodesVisited.add(nn.visitedCount());
            });
        }
        return new ResultSummary((int) topKfound.sum(), (int) nodesVisited.sum());
    }

    record DataSet(VectorSimilarityFunction similarityFunction, List<float[]> baseVectors, List<float[]> queryVectors, List<Set<Integer>> groundTruth) { }

    private static DataSet load(String pathStr) {
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

        System.out.format("%n%s: %d base and %d query vectors loaded, dimensions %d%n",
                pathStr, scrubbedBaseVectors.size(), scrubbedQueryVectors.size(), scrubbedBaseVectors.get(0).length);

        return new DataSet(similarityFunction, scrubbedBaseVectors, scrubbedQueryVectors, gtSet);
    }

    private static void normalizeAll(Iterable<float[]> vectors) {
        for (float[] v : vectors) {
            VectorUtil.l2normalize(v);
        }
    }

    public static void main(String[] args) throws IOException, ExecutionException, InterruptedException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        var files = List.of(
                "hdf5/nytimes-256-angular.hdf5",
                "hdf5/glove-100-angular.hdf5",
                "hdf5/sift-128-euclidean.hdf5",
                "hdf5/glove-200-angular.hdf5");
        var mGrid = List.of(8, 16, 24, 32, 48, 64);
        var efConstructionGrid = List.of(60, 160, 200, 400, 600, 800);
        var efSearchFactor = List.of(1, 2);
        // large files not yet supported
//                "hdf5/deep-image-96-angular.hdf5",
//                "hdf5/gist-960-euclidean.hdf5");
        for (var f : files) {
            gridSearch(f, mGrid, efConstructionGrid, efSearchFactor);
        }

        // tiny dataset, don't waste time building a huge index
        files = List.of("hdf5/fashion-mnist-784-euclidean.hdf5");
        mGrid = List.of(8, 12, 16, 24);
        efConstructionGrid = List.of(40, 60, 80, 100, 120, 160);
        for (var f : files) {
            gridSearch(f, mGrid, efConstructionGrid, efSearchFactor);
        }
    }

    private static void gridSearch(String f, List<Integer> mGrid, List<Integer> efConstructionGrid, List<Integer> efSearchFactor) throws ExecutionException, InterruptedException {
        var ds = load(f);

        var start = System.nanoTime();
        var pqDims = ds.baseVectors.get(0).length / 2;
        ProductQuantization pq = new ProductQuantization(ds.baseVectors, pqDims, ds.similarityFunction == VectorSimilarityFunction.EUCLIDEAN);
        System.out.format("PQ@%s build %.2fs,%n", pqDims, (System.nanoTime() - start) / 1_000_000_000.0);

        start = System.nanoTime();
        var quantizedVectors = pq.encodeAll(ds.baseVectors);
        System.out.format("PQ encode %.2fs,%n", (System.nanoTime() - start) / 1_000_000_000.0);

        for (int M : mGrid) {
            for (int beamWidth : efConstructionGrid) {
                testRecall(M, beamWidth, efSearchFactor, ds, new PQRandomAccessVectorValues(quantizedVectors, pq));
            }
        }
    }
}
