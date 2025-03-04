#!/bin/bash

input_file="$1"  # Change this to your file name
temp_file="merged_data.tmp"

column_names=("-" "len-t" "len-q" "covered" "used" "ncontig" "nbreaks" "nindel" "ndiff" "identity")  # Adjust as needed

# Remove the temp file if it exists
rm -f "$temp_file"

# Extract numeric data from all tables and merge them
awk '!/^=/ && !/^-/ && !/^#/ && !/^Per/ && NF {print}' "$input_file" >> "$temp_file"

# Calculate column means
awk -v col_names="${column_names[*]}" '
BEGIN {
    split(col_names, names, " ");  # Split column names into an array
}
{
    for (i = 1; i <= NF; i++) {
        sum[i] += $i;
        count[i]++;
    }
}
END {
    for (i = 1; i in sum; i++) {
        col_name = (i <= length(names)) ? names[i] : "Column" i;
        printf "%s mean: %.4f\n", col_name, sum[i] / count[i];
    }
}' "$temp_file"

# Clean up temporary file
rm -f "$temp_file"
