#!/bin/bash

input_file="$1" 
batch_size=5  # Number of tables per batch
table_size=5  # Number of rows per table
temp_file="merged_data.tmp"

# Remove the temp file if it exists
rm -f "$temp_file"

# Extract numeric data from all tables and merge them
awk '!/^=/ && !/^-/ && !/^#/ && !/^Per/ && NF {print}' "$input_file" >> "$temp_file"

awk -v batch_size=$batch_size -v table_size=$table_size '
{
    if (NF < 4) next;  # Skip lines with insufficient columns

    row_idx = (NR - 1) % table_size  # Row index within a batch (0-4)
    batch_idx = int((NR - 1) / (batch_size * table_size))  # Batch index
    table_idx = int(((NR % (batch_size * table_size)) / table_size))  # Table index within batch (0-4)
    
    avg = ($4 + $5) / 2  # Average of 3rd and 4th columns

    # Check if this row is the best so far for this row index in the batch
    if (!(batch_idx, row_idx) in max_avg || avg > max_avg[batch_idx, row_idx]) {
        max_avg[batch_idx, row_idx] = avg
        best_row[batch_idx, row_idx] = $0
    }
}
END {
    for (batch = 0; batch * batch_size * table_size < NR; batch++) {
        print "==============="
        print batch
        for (row = 0; row < table_size; row++) {
            if ((batch, row) in best_row) {
                print best_row[batch, row]
            }
        }
        print "----------------"
    }
}' "$temp_file"


# Clean up temporary file
rm -f "$temp_file"