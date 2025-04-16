#!/bin/bash

# Input files
file1="$1"
file2="$2"
output_file="$3"

# Clear the output file
> "$output_file"

# Read each row from file1
while IFS= read -r line1; do
    # Extract first column
    col1=$(echo "$line1" | awk '{print $1}')
    if [[ $col1 == S ]]; then
        # Match first 2 columns
        key=$(echo "$line1" | awk '{print $1, $2 " "}')
        match_columns=2
    elif [[ $col1 == L ]]; then
        # Match first 5 columns
        key=$(echo "$line1" | awk '{print $1, $2, $3, $4, $5 " "}')
        match_columns=5
    else
        continue  # Skip if the first column doesn't start with S or L
    fi
    # Scan file2 for matching rows
    while IFS= read -r line2; do
        compare_key=$(echo "$line2" | awk -v cols=$match_columns '{for (i=1; i<=cols; i++) printf "%s ", $i; print ""}')
        if [[ "$compare_key" == "$key" ]]; then
            echo "$line2" >> "$output_file"
            echo match $key
            break
        fi
    done < "$file2"

done < "$file1"

echo "Matching rows written to $output_file"
