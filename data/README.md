Description of data input
=========================

Data are organized by one cell type per folder. I included data from the 13 cell lines used for the ENCODE-DREAM competition. Each folder contains the following files:

* bigWig file(s). The bigWig format is a standard data format for displaying dense, continuous data that can be visualized in in a genome browser. bigWig files are compressed and indexed for portability and speed. The most important type of bigWig file used in this project is the 1x coverage normalized DNase 5' cut signal. With GM12878 as an example, the following details the requirements and command lines to convert BAM files into a bigWig file:

Required
--------
* [samtools] (https://github.com/samtools/samtools). Useful for manipulating BAM files. If you do genomics, chances are you already have this. Latest version should be sufficient.
* [deepTools] (https://github.com/fidelram/deepTools). Tools for normalizing read coverage in BAM files. Despite its name, it has nothing to do with deep learning.

```
$ ls
DNASE.GM12878.biorep1.techrep1.bam  DNASE.GM12878.biorep2.techrep1.bam
$ samtools merge GM12878.bam DNASE.GM12878.biorep*.techrep1.bam
$ samtools index GM12878.bam
$ bamCoverage --bam ${i}.bam -o ${i}.1x.bw --outFileFormat bigwig --normalizeTo1x 2478297382 --ignoreForNormalization chrX chrM --Offset 1 --binSize 1 --numberOfProcessors 12 --blackListFileName resources/wgEncodeDacMapabilityConsensusExcludable.bed.gz --skipNonCoveredRegions
$ # You can find the wgEncodeDacMapabilityConsensusExcludable.bed.gz file in the resources folder.
$ # Unfortunately, I neglected the Duke blacklist file at the time I generated these bigWig files.
```

* Gzipped BED file(s). Genomic interval files containing TF binding peak locations (both conserved and relaxed peaks).

* bigwig.txt. Two column tab delimited file describing bigWig files to be used for training. First column is the file names and second column is the identifier names each bigWig. When leveraging data from multiple cell lines, the identifier names for each cell line must match and be in the same order.

* chip.txt. Two column tab delimited file describing peak interval files to be used for training. First column is the file names of the gzipped peak BED files and the second column is the name of the transcription factor corresponding to that BED file. Optionally, a third column of the file names of relaxed peaks can be included. If relaxed peak files are included, negative training bins will be drawn from outside of the relaxed peaks; otherwise, negative training bins will be drawn from outside of the conservative peaks.

* meta.txt. Two column tab delimited file describing cell-type specific metadata to be used for training. First column is the metadata value and second column is the identifier names each feature. When leveraging data from multiple cell lines, the identifier names for each cell line must match and be in the same order. In these examples, I used 8 gene expression PCA features. See [here](https://github.com/davidaknowles/tf_net/blob/master/gene_expression_pca.R) for a description of how those features were generated.

bigWig files
============

Due to file size limits, bigWig files are not included in this repository. You can download them from the following links. Make sure you place the downloaded bigWig files into their respective folders. It should be straightforward to tell from the file names which files go where. The first track is the 35 bp mapability uniqueness track and belongs in every folder. I highly recommend creating soft links using the ln command so that you are not wasting disk space with multiple copies of the track.

[Duke 35 bp mapability uniqueness] (http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeMapability/wgEncodeDukeMapabilityUniqueness35bp.bigWig)

[A549 1x DGF] (https://www.synapse.org/#!Synapse:syn14748607)

[GM12878 1x DGF] (https://www.synapse.org/#!Synapse:syn8073652)

[H1-hESC 1x DGF] (https://www.synapse.org/#!Synapse:syn8073583)

[HCT116 1x DGF] (https://www.synapse.org/#!Synapse:syn8074109)

[HeLa-S3 1x DGF] (https://www.synapse.org/#!Synapse:syn8073618)

[HepG2 1x DGF] (https://www.synapse.org/#!Synapse:syn8073517)

[IMR90 1x DGF] (https://www.synapse.org/#!Synapse:syn14749187)

[K562 1x DGF] (https://www.synapse.org/#!Synapse:syn8073483)

[MCF-7 1x DGF] (https://www.synapse.org/#!Synapse:syn8073537)

[PC-3 1x DGF] (https://www.synapse.org/#!Synapse:syn8074140)

[Panc1 1x DGF] (https://www.synapse.org/#!Synapse:syn8074156)

[induced pluripotent stem cell 1x DGF] (https://www.synapse.org/#!Synapse:syn8074090)

[liver 1x DGF] (https://www.synapse.org/#!Synapse:syn8074181)

