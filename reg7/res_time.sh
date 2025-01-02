#!/bin/bash

# Loop over all CSV files in the current directory
for file in res/times/*.csv; do
  # Check if there are any CSV files
  if [[ -e "$file" ]]; then
    echo "Sorting $file by the third field..."
    # Sort each file by the third field (numeric sort) and save it in place
    sort -t',' -k3,3n "$file" -o "$file"
  else
    echo "No CSV files found."
    exit 1
  fi
done

echo "All CSV files have been sorted by the third field."

python3.12 execution_time_table.py