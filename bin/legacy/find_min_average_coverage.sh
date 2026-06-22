#!/bin/bash

file="$1"  # Change this to your actual file
min_avg=999999        # Large initial value
min_table=""          # Store the table with min average
table=""              # Temporary variable to store table data
processing=0          # Flag to track when processing a table

while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^=== ]]; then  # Line starts with "===", marks new table
        if [[ $processing -eq 1 ]]; then
            # Process the captured table
            avg=$(echo "$table" | awk 'NR>1 {sum4+=$4; sum5+=$5; count++} END {if (count > 0) print (sum4/count + sum5/count)/2}')
            if (( $(echo "$avg < $min_avg" | bc -l) )); then
                min_avg=$avg
                min_table="$table"
            fi
        fi
        table=""       # Reset table storage
        processing=1   # Start processing new table
    else
        table+="$line"$'\n'
    fi
done < "$file"

# Process the last table if the file doesn't end with a separator
if [[ $processing -eq 1 ]]; then
    avg=$(echo "$table" | awk 'NR>1 {sum4+=$4; sum5+=$5; count++} END {if (count > 0) print (sum4/count + sum5/count)/2}')
    if (( $(echo "$avg < $min_avg" | bc -l) )); then
        min_avg=$avg
        min_table="$table"
    fi
fi

# Print the table with the minimum average
echo "Table with minimum average in columns 4 and 5:"
echo "$min_table"