package org.example.util;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.stream.IntStream;

public class Deep1BLoader {
    public static List<float[]> readFBin(String filePath, int count) throws IOException {
        var vectors = new float[count][];

        int n;
        int dimension;
        try (var raf = new RandomAccessFile(filePath, "r")) {
            n = Integer.reverseBytes(raf.readInt());
            dimension = Integer.reverseBytes(raf.readInt());
        }
        System.out.printf("%s contains %d vectors; reading %d%n", filePath, n, count);

        var threadCount = 16;
        int perThread = count / threadCount;
        int remainder = count % threadCount;

        IntStream.range(0, threadCount).parallel().forEach(threadNum -> {
            try (var raf = new RandomAccessFile(filePath, "r")) {
                long startPosition = 8L + ((long) threadNum * perThread + Math.min(threadNum, remainder)) * dimension * Float.BYTES;
                raf.seek(startPosition);

                int itemsToProcess = perThread + (threadNum < remainder ? 1 : 0);
                for (int j = 0; j < itemsToProcess; j++) {
                    var buffer = new byte[dimension * Float.BYTES];
                    raf.readFully(buffer);
                    var byteBuffer = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN);

                    var vector = new float[dimension];
                    var floatBuffer = byteBuffer.asFloatBuffer();
                    floatBuffer.get(vector);
                    vectors[threadNum * perThread + (Math.min(threadNum, remainder)) + j] = vector;
                }
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });

        System.out.printf("Completed reading %s%n".formatted(filePath));
        return List.of(vectors);
    }

    public static ArrayList<HashSet<Integer>> readGT(String filePath) {
        var groundTruthTopK = new ArrayList<HashSet<Integer>>();

        try (var dis = new DataInputStream(new FileInputStream(filePath))) {
            var n = Integer.reverseBytes(dis.readInt());
            var topK = Integer.reverseBytes(dis.readInt());
            for (int i = 0; i < n; i++) {
                var neighbors = new HashSet<Integer>(topK);
                for (var j = 0; j < topK; j++) {
                    var neighbor = Integer.reverseBytes(dis.readInt());
                    neighbors.add(neighbor);
                }
                groundTruthTopK.add(neighbors);
            }
            // GT file also contains scores, we don't need those
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }

        System.out.printf("Completed reading %s%n".formatted(filePath));
        return groundTruthTopK;
    }
}
