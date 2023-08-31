package org.example;

import io.jhdf.HdfFile;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.example.util.ListRandomAccessVectorValues;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Tests HNSW against vectors from the Texmex dataset
 */
public class Bench {
    private static int threadCount = Runtime.getRuntime().availableProcessors();

    private static void testRecall(int M, int efConstruction, List<Integer> efSearchOptions, DataSet ds)
            throws ExecutionException, InterruptedException, IOException {
        var ravv = new ListRandomAccessVectorValues(ds.baseVectors, ds.baseVectors.get(0).length);
        var topK = ds.groundTruth.get(0).size();

        // build the graphs on multiple threads
        var start = System.nanoTime();
        var builder = HnswGraphBuilder.create(ravv, VectorEncoding.FLOAT32, ds.similarityFunction, M, efConstruction, 42);
        var hnsw = builder.build(ravv.copy());
        long buildNanos = System.nanoTime() - start;

        int queryRuns = 1;
        for (int overquery : efSearchOptions) {
            start = System.nanoTime();
            var pqr = performQueries(ds, ravv, () -> hnsw, topK, topK * overquery, queryRuns);
            var recall = ((double) pqr.topKFound) / (queryRuns * ds.queryVectors.size() * topK);
            System.out.format("HNSW   M=%d ef=%d: top %d/%d recall %.4f, build %.2fs, query %.2fs. %s nodes visited%n",
                    M, efConstruction, topK, overquery, recall, buildNanos / 1_000_000_000.0, (System.nanoTime() - start) / 1_000_000_000.0, pqr.nodesVisited);
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

    private static long topKCorrect(int topK, TopDocs nn, Set<Integer> gt) {
        var a = new int[nn.scoreDocs.length];
        for (int j = a.length - 1; j >= 0; j--) {
            a[j] = nn.scoreDocs[j].doc;
        }
        return topKCorrect(topK, a, gt);
    }

    private static ResultSummary performQueries(DataSet ds, ListRandomAccessVectorValues ravv, Supplier<HnswGraph> graphSupplier, int topK, int efSearch, int queryRuns) {
        assert efSearch >= topK;
        LongAdder topKfound = new LongAdder();
        LongAdder nodesVisited = new LongAdder();
        for (int k = 0; k < queryRuns; k++) {
            IntStream.range(0, ds.queryVectors.size()).forEach(i -> {
                var queryVector = ds.queryVectors.get(i);
                KnnCollector nn;
                try {
                    nn = HnswGraphSearcher.search(queryVector, efSearch, ravv, VectorEncoding.FLOAT32, ds.similarityFunction, graphSupplier.get(), null, Integer.MAX_VALUE);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
                var gt = ds.groundTruth.get(i);
                var n = topKCorrect(topK, nn.topDocs(), gt);
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
        System.out.println("Using " + 0 + " threads");

        var files = List.of(
        "hdf5/nytimes-256-angular.hdf5",
        "hdf5/glove-100-angular.hdf5",
        "hdf5/glove-200-angular.hdf5",
        "hdf5/sift-128-euclidean.hdf5");
        var mGrid = List.of(16);
        var efConstructionGrid = List.of(120);
        var efSearchFactor = List.of(1);
        // large files not yet supported
//                "hdf5/deep-image-96-angular.hdf5",
//                "hdf5/gist-960-euclidean.hdf5");
        for (var f : files) {
            gridSearch(f, mGrid, efConstructionGrid, efSearchFactor);
        }
    }

    private static void gridSearch(String f, List<Integer> mGrid, List<Integer> efConstructionGrid, List<Integer> efSearchFactor) throws ExecutionException, InterruptedException, IOException {
        var ds = load(f);
        for (int M : mGrid) {
            for (int beamWidth : efConstructionGrid) {
                testRecall(M, beamWidth, efSearchFactor, ds);
            }
        }
    }
}
