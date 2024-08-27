#!/bin/bash

usage()
{
    echo "usage: compile_all [[-s solver] [-d directory]] | [-h]]"
}

while [ "$1" != "" ]; do
    case $1 in
        -d | --dir )    shift
                        dir="$1"
                        ;;
        -s | --solver ) shift
                        solver="$1"
                        ;;
        -h | --help )   usage
                        exit
                        ;;
        * )             usage
                        exit 1
    esac
    shift
done


kmers=(k501 k301 k201 k101 sim_k71 sim_k61)
for kmer in $kmers; do
    ./compile_full_benchmark "-f data/ddDapMeze1.MT.$kmer.utg.final.gfa -d $dir -s $solver" >> "out/$dir/$solver.compiled.txt"
done 