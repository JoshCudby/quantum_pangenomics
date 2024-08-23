#!/bin/bash

usage()
{
    echo "usage: compile_full_benchmark [[-s solver] [-f filename] [-d directory]] | [-h]]"
}

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )   shift
                        filename="$1"
                        ;;
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

if [[ $filename =~ ^data/(.*)$ ]]; then
    filename="${BASH_REMATCH[1]}" 
fi

out_dir="./out/$dir"
file_pattern="$solver.full.$filename*"
search_pattern="Compilation Data"

files=$(find "$out_dir" -type f -name "$file_pattern" )

# Iterate through each file and search for the pattern
for file in $files; do
    # Read the file line by line
    while IFS= read -r line; do
        # Check if the line matches the search pattern
        if echo "$line" | grep -q "$search_pattern"; then
            IFS= read -r next_line;
            echo "$next_line"
        fi
    done < "$file"
done