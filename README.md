# HNSW benchmarking with recall

This repository provides a `Bench` class that benchmarks build time, throughput, and accuracy (recall) for the Lucene HNSW implementation. See the [Lucene JIRA issue](https://issues.apache.org/jira/browse/LUCENE-9004) and [the Elastic blog post](https://www.elastic.co/blog/introducing-approximate-nearest-neighbor-search-in-elasticsearch-8-0) for more background, 
and [the original paper](https://arxiv.org/pdf/1603.09320.pdf) for how it works.

## Prerequisites

You should use Java 20+ to take advantage of Lucene's SIMD optimizations, otherwise you will be half as fast as you should be.

The gradle build is configured to use a local Lucene build, and it requires Jonathan Ellis's Concurrent HNSW implementation.  For now you can get that from his branch here: https://github.com/jbellis/lucene/tree/concurrent4

Then install it with
```bash
$ export JAVA_HOME=/path/to/java-20
$ ./gradlew mavenToLocal
```

## Usage

Download datasets from https://github.com/erikbern/ann-benchmarks/#data-sets.

Currently it's hardoded to look for the nytimes dataset in a subdirectory `hdf5/`.

```bash
$ ./gradlew runBench
```

It takes about 90s to run on my 16 core / 24 thread machine, it's close to perfectly linear wrt number of threads.
