import csv
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(parent_dir)

# Import the parent script
import stats

# Specify the path to your CSV file
file_path = sys.argv[1]
# Open the file and read it
with open(file_path, mode='r') as file:
    csv_reader = csv.reader(file)
    
    times = []
    # Iterate through the rows
    for row in csv_reader:
        a = stats.SOME(txt=f"{row[0]}")
        a.add(float(row[1]))
        times.append(a)

stats.report( times , epsilon=5)
        