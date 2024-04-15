TODO full README

tldr run the Bench class against hdf5 datasets that you can find at https://github.com/erikbern/ann-benchmarks

Requires JBE's concurrent Lucene code which you can get from https://github.com/jbellis/lucene/tree/concurrent5, then run `gradlew mavenToLocal`

If you're patient, it's a trivial change to mainline Lucene instead, but building the index with a large dataset will take ~days.
