#!/bin/bash

usage()
{
    echo "usage: compile_all [[-s solver] [-d directory] [-n name]] | [-h]]"
}

name=""
while [ "$1" != "" ]; do
    case $1 in
        -d | --dir )    shift
                        dir="$1"
                        ;;
        -s | --solver ) shift
                        solver="$1"
                        ;;
        -n | --name )   shift
                        name=".$1"
                        ;;
        -h | --help )   usage
                        exit
                        ;;
        * )             usage
                        exit 1
    esac
    shift
done

rm -f "out/$dir/$solver$name.compiled.txt"
filenames=("both2.syncasm1001.utg.final.gfa" "both.syncasm.utg.final.gfa")
for filename in "${filenames[@]}"; do
    echo $filename >> "out/$dir/$solver$name.compiled.txt"
    source ./bin/compile_full_benchmark.sh "-f" "$filename" "-d" "$dir" "-s" "$solver" "-n" "$name" >> "out/$dir/$solver$name.compiled.txt"
done 