package org.example;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.lucene95.Lucene95Codec;
import org.apache.lucene.codecs.lucene95.Lucene95HnswVectorsFormat;
import org.apache.lucene.codecs.lucene95.Lucene95HnswVectorsWriter;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.NamedThreadFactory;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.hnsw.ConcurrentHnswGraphBuilder;
import org.apache.lucene.util.hnsw.ConcurrentOnHeapHnswGraph;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.example.util.Deep1BLoader;
import org.example.util.ListRandomAccessVectorValues;

import static org.apache.lucene.util.StringHelper.ID_LENGTH;

/**
 * Tests HNSW against vectors from the Texmex dataset
 */
public class Bench {
    private static void testRecall(int M, int efConstruction, DataSet ds) throws IOException {
        var topK = 100;
        var queryRuns = 1;
        try (var reader = openReader(new File("luceneindex/deep1B-10M"), ds)) {
            var pqr = queryReader(ds, reader, topK, topK, queryRuns);
            var recall = ((double) pqr.topKFound) / (queryRuns * ds.queryVectors.size() * topK);
        }
    }

    private static ResultSummary queryReader(DataSet ds, KnnVectorsReader reader, int topK, int efSearch, int queryRuns) {
        assert efSearch >= topK;
        LongAdder topKfound = new LongAdder();
        LongAdder nodesVisited = new LongAdder();
        var startQuery = System.nanoTime();
        AtomicInteger completed = new AtomicInteger();
        for (int k = 0; k < queryRuns; k++) {
            IntStream.range(0, ds.queryVectors.size()).parallel().forEach(i -> {
                var queryVector = ds.queryVectors.get(i);
                TopDocs docs;
                try {
                    docs = reader.search("MockField", queryVector, efSearch, null, Integer.MAX_VALUE);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
                var gt = ds.groundTruth.get(i);
                var n = topKCorrect(topK, docs, gt);
                topKfound.add(n);
                var elapsed = System.nanoTime() - startQuery;
                var rate = completed.incrementAndGet() / (elapsed / 1_000_000_000.0);
                System.out.println(String.format("Completed %s queries at %.2f qps", completed.get(), rate));
            });
        }
        return new ResultSummary((int) topKfound.sum(), (int) nodesVisited.sum());
    }

    private static KnnVectorsReader openReader(File vectorPath, DataSet ds) throws IOException {
        Directory directory = FSDirectory.open(vectorPath.toPath().getParent());

        FieldInfo fieldInfo = createFieldInfoForVector(ds);
        FieldInfos fieldInfos = new FieldInfos(Collections.singletonList(fieldInfo).toArray(new FieldInfo[0]));
        String segmentName = vectorPath.getName();
        var segmentId = new byte[ID_LENGTH];
        SegmentInfo segmentInfo = new SegmentInfo(directory, Version.LATEST, Version.LATEST, segmentName, 100_000_000, false, Lucene95Codec.getDefault(), Collections.emptyMap(), segmentId, Collections.emptyMap(), null);

        SegmentReadState state = new SegmentReadState(directory, segmentInfo, fieldInfos, IOContext.DEFAULT);
        return new Lucene95HnswVectorsFormat(16, 100).fieldsReader(state);
    }

    private static FieldInfo createFieldInfoForVector(DataSet ds)
    {
        String name = "MockField";
        int number = 0;
        boolean storeTermVector = false;
        boolean omitNorms = false;
        boolean storePayloads = false;
        IndexOptions indexOptions = IndexOptions.NONE;
        DocValuesType docValues = DocValuesType.NONE;
        long dvGen = -1;
        Map<String, String> attributes = Map.of();
        int pointDimensionCount = 0;
        int pointIndexDimensionCount = 0;
        int pointNumBytes = 0;
        VectorEncoding vectorEncoding = VectorEncoding.FLOAT32;
        VectorSimilarityFunction vectorSimilarityFunction = ds.similarityFunction;
        boolean softDeletesField = false;

        return new FieldInfo(name, number, storeTermVector, omitNorms, storePayloads, indexOptions, docValues,
                             dvGen, attributes, pointDimensionCount, pointIndexDimensionCount, pointNumBytes,
                             96, vectorEncoding, vectorSimilarityFunction, softDeletesField);
    }

    private record ResultSummary(int topKFound, int nodesVisited) { }

    private static long topKCorrect(int topK, TopDocs docs, Set<Integer> gt) {
        int count = Math.min(docs.scoreDocs.length, topK);
        // stream the first count results into a Set
        var resultSet = Arrays.stream(docs.scoreDocs, 0, count)
                              .map(sd -> sd.doc)
                              .collect(Collectors.toSet());
        assert resultSet.size() == count : String.format("%s duplicate results out of %s", count - resultSet.size(), count);
        return resultSet.stream().filter(gt::contains).count();
    }

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

    record DataSet(VectorSimilarityFunction similarityFunction, List<float[]> baseVectors, List<float[]> queryVectors, List<? extends Set<Integer>> groundTruth) { }

    private static DataSet load() throws IOException {
        var queryVectors = Deep1BLoader.readFBin("bigann-data/deep1b/query.public.10K.fbin", 10_000);
        var gt = Deep1BLoader.readGT("bigann-data/deep1b/deep-100M");
        return new DataSet(VectorSimilarityFunction.EUCLIDEAN, null, queryVectors, gt);
    }

    public static void main(String[] args) throws IOException, ExecutionException, InterruptedException {
        System.out.println("Heap space available is " + Runtime.getRuntime().maxMemory());

        var mGrid = List.of(16);
        var efConstructionGrid = List.of(100);
        var efSearchFactor = List.of(1);
        gridSearch(mGrid, efConstructionGrid, efSearchFactor);
    }

    private static void gridSearch(List<Integer> mGrid, List<Integer> efConstructionGrid, List<Integer> efSearchFactor) throws ExecutionException, InterruptedException, IOException {
        var ds = load();
        for (int M : mGrid) {
            for (int beamWidth : efConstructionGrid) {
                testRecall(M, beamWidth, ds);
            }
        }
    }
}
