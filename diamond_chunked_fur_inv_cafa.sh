#!/bin/bash

diamond blastp \
    -q $1 \
    -d $2 \
    -o $3 \
    --ultra-sensitive -k $4 -f 6 --header -p $5 -b 5 -c 1