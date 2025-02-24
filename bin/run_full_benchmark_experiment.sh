#!/bin/bash
memory=4000
root_dir=/lustre/scratch127/qpg/jc59/full_benchmark

dt=$(date '+%d%m%Y.%H%M')
kmer=301

while [ "$1" != "" ]; do
    case $1 in
        -k | --kmer )           shift
                                kmer="$1"
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

out_suffix=$kmer.$dt
bsub -J "run_benchmark" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q normal \
     -M "$memory" -o "$root_dir/$out_suffix/log.txt" -e "$root_dir/error.$out_suffix/err.txt" -G "qpg" \
     "/nfs/users/nfs_j/jc59/quantumwork/pangenome/bin/full_benchmark_experiment.sh $kmer $dt"