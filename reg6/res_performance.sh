#!/bin/bash

# Declare an associative array to store counts
declare -A counts

# Loop through all CSV files in the current directory
for file in res/*.csv; do
    # Read each line of the file
    while IFS=',' read -r rank treatment rest; do
        # Trim whitespace from rank and treatment
        rank=$(echo "$rank" | xargs)
        treatment=$(echo "$treatment,$rest" | xargs | cut -d, -f1-2)

        # Check if rank is 0, 1, or 2
        if [[ "$rank" =~ ^[012]$ ]]; then
            # Increment the count for the treatment and rank
            key="$treatment,$rank"
            counts["$key"]=$((counts["$key"] + 1))
        fi
    done < "$file"
done

# Create a new associative array to store consolidated results
declare -A consolidated

# Consolidate counts into the required format
for key in "${!counts[@]}"; do
    treatment="${key%,*}" # Extract treatment
    rank="${key##*,}"    # Extract rank
    count="${counts[$key]}"

    # Initialize consolidated entry if not exists
    if [[ -z "${consolidated[$treatment]}" ]]; then
        consolidated["$treatment"]="rank0,0,rank1,0,rank2,0"
    fi

    # Update counts for the respective rank
    case $rank in
        0) consolidated["$treatment"]=$(echo "${consolidated[$treatment]}" | sed "s/rank0,0/rank0,$count/") ;;
        1) consolidated["$treatment"]=$(echo "${consolidated[$treatment]}" | sed "s/rank1,0/rank1,$count/") ;;
        2) consolidated["$treatment"]=$(echo "${consolidated[$treatment]}" | sed "s/rank2,0/rank2,$count/") ;;
    esac
done

# Write results to a temporary file for sorting
temp_file=$(mktemp)
echo "Sampling, Regressor,rank0,count0,rank1,count1,rank2,count2" > "$temp_file"

for treatment in "${!consolidated[@]}"; do
    count0=$(echo "${consolidated[$treatment]}" | grep -o 'rank0,[0-9]*' | cut -d, -f2)
    echo "$treatment,${consolidated[$treatment]}" >> "$temp_file"
done

# Sort by `count0` (third column, numerically, descending)
sorted_file="sorted_results.csv"
head -n 1 "$temp_file" > "$sorted_file" # Add the header
tail -n +2 "$temp_file" | sort -t',' -k4,4nr >> "$sorted_file"

# Clean up the temporary file
rm "$temp_file"

# Display the sorted results
cat "$sorted_file"
