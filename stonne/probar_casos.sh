#!/bin/bash

filename=results-outputs_M$1-N$1-K$1-ms$2
csv=results.csv


for ((sparsity = 0; sparsity < 100; sparsity += 10)); do
    for ((T_K = 1; T_K <= $1; T_K *= 2)); do
        for ((T_N = 1; T_N <= $1; T_N *= 2)); do
            echo "./stonne -SparseDense -M=$1 -N=$1 -K=$1 -T_N=$T_N -T_K=$T_K -MK_sparsity=$sparsity -num_ms=$2 -dn_bw=$2 -rn_bw=$2 -accumulation_buffer=1"
            
            ./stonne -SparseDense -M=$1 -N=$1 -K=$1 -T_N=$T_N -T_K=$T_K -MK_sparsity=$sparsity -num_ms=$2 -dn_bw=$2 -rn_bw=$2 -accumulation_buffer=1 > output 2> err
            cycles=$(cat output | grep "Number of cycles running:" | cut -f5 -d" ")
            assert=$(cat err | grep "Assertion")
            echo "sparsity=$sparsity T_K=$T_K T_N=$T_N cycles=$cycles" >> $filename
            if [ ! -z "$assert" ]
            then
                echo "  -> $assert" >> $filename
            fi
            
            if [ -z "$assert" ]
            then
                echo "$1, $2, $sparsity, $T_K, $T_N, $cycles" >> $csv
            fi
        done
    done
done


rm output_stats*
