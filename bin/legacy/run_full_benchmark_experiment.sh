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
        -p | --problem )        shift
                                problem="$1"
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

problem_types=("tangle", "oriented")
if [[ ! " ${problem_types[*]} " =~ ${problem} ]]; then
    echo "Solver must be one of ${problem_types[*]}"
    exit 1
fi

outdir=$root_dir/$problem/$kmer.$dt
mkdir -p $outdir

bsub -J "run_benchmark" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q normal \
     -M "$memory" -o "$outdir/log.txt" -e "$outdir/err.txt" -G "qpg" \
     "/nfs/users/nfs_j/jc59/quantumwork/pangenome/bin/full_benchmark_$problem.sh $kmer $dt"