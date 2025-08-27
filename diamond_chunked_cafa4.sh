#!/bin/bash
echo $1
echo $2
echo $3
echo $4

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

diamond blastp \
    -q "$1/chunk_1_cafa4.fasta" \
    -d $2 \
    -o "$3/cafa4_k150_out_chunk_1.tsv" \
    --ultra-sensitive -k 150 -f 6 --header -p $4 -b 5 -c 1 \
    --no-unlink \
    && diamond blastp \
    -q "$1/chunk_2_cafa4.fasta" \
    -d $2 \
    -o "$3/cafa4_k150_out_chunk_2.tsv" \
    --ultra-sensitive -k 150 -f 6 --header -p $4 -b 5 -c 1 \
    && diamond blastp \
    -q "$1/chunk_3_cafa4.fasta" \
    -d $2 \
    -o "$3/cafa4_k150_out_chunk_3.tsv" \
    --ultra-sensitive -k 150 -f 6 --header -p $4 -b 5 -c 1 \
    && diamond blastp \
    -q "$1/chunk_4_cafa4.fasta" \
    -d $2 \
    -o "$3/cafa4_k150_out_chunk_4.tsv" \
    --ultra-sensitive -k 150 -f 6 --header -p $4 -b 5 -c 1 \
    && diamond blastp \
    -q "$1/chunk_5_cafa4.fasta" \
    -d $2 \
    -o "$3/cafa4_k150_out_chunk_5.tsv" \
    --ultra-sensitive -k 150 -f 6 --header -p $4 -b 5 -c 1 \
    && diamond blastp \
    -q "$1/chunk_6_cafa4.fasta" \
    -d $2 \
    -o "$3/cafa4_k150_out_chunk_6.tsv" \
    --ultra-sensitive -k 150 -f 6 --header -p $4 -b 5 -c 1 \
    && diamond blastp \
    -q "$1/chunk_7_cafa4.fasta" \
    -d $2 \
    -o "$3/cafa4_k150_out_chunk_7.tsv" \
    --ultra-sensitive -k 150 -f 6 --header -p $4 -b 5 -c 1 \
    && diamond blastp \
    -q "$1/chunk_8_cafa4.fasta" \
    -d $2 \
    -o "$3/cafa4_k150_out_chunk_8.tsv" \
    --ultra-sensitive -k 150 -f 6 --header -p $4 -b 5 -c 1 \
    && diamond blastp \
    -q "$1/chunk_9_cafa4.fasta" \
    -d $2 \
    -o "$3/cafa4_k150_out_chunk_9.tsv" \
    --ultra-sensitive -k 150 -f 6 --header -p $4 -b 5 -c 1 \
    && diamond blastp \
    -q "$1/chunk_10_cafa4.fasta" \
    -d $2 \
    -o "$3/cafa4_k150_out_chunk_10.tsv" \
    --ultra-sensitive -k 150 -f 6 --header -p $4 -b 5 -c 1 \
