#!/bin/bash

usage()
{
    echo "usage: get_scores_and_times [[-s solver] [-k kmer]] | [-h]]"
}

while [ "$1" != "" ]; do
    case $1 in
        -k | --kmer )   shift
                        kmer="$1"
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

search_pattern="Energy of path"
run_time_pattern="Run time"
quantum_time_pattern="qpu_access_time"

file="out/$solver.compiled.$kmer.txt"
# Read the file line by line
while IFS= read -r line; do
    # Check if the line matches the search pattern
    if echo "$line" | grep -q "$search_pattern"; then
        echo "$line"
        count=1
        # Continue reading lines until one matches the stop pattern
        while IFS= read -r next_line; do
            if echo "$next_line" | grep -q "$run_time_pattern \| $quantum_time_pattern"; then
                echo "$next_line"
                break
            fi
            echo "$next_line"
            ((count++))
            if [ "$count" -ge "$max_lines" ]; then
                break
            fi
        done
    fi
done < "$file"
