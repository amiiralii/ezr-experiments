#!/bin/bash

# Declare an associative array to store counts
declare -A counts

# Loop through all CSV files in the current directory
for file in *.csv; do
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

# Print header
echo "Treatment,rank0,count0,rank1,count1,rank2,count2" > final_results.csv

# Print consolidated results
for treatment in "${!consolidated[@]}"; do
    echo "$treatment,${consolidated[$treatment]}" >> final_results.csv
done

# Display the results
cat final_results.csv
