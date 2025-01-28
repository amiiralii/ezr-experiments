from ezr import the, DATA, csv, xval, activeLearning, rows, dist
import sys 
import matplotlib.pyplot as plt
import numpy as np

def measure_dependancy(dataset):
    d = DATA().adds(csv(dataset))
    kmeans_clusters = d.kmeansplusplus(rows = d.rows, neighbors=5)
    xdist , ydist = [], []
    for cluster in kmeans_clusters:
        centroid = cluster[0]
        xdist += [d.dist(centroid, r) for r in cluster]
        ydist += [d.disty(centroid, r) for r in cluster]
        
    return [round(x/y,2) for x,y in zip(xdist,ydist) if 0 not in [x,y]]


dataset = sys.argv[1]
measure = measure_dependancy(dataset)
for a in range(len(measure)): 
    if measure[a] >= 4: measure[a] = 4

plt.hist(measure, bins=20 , edgecolor='red')
plt.xlim(0, 3)
plt.xlabel('dist(x) / dist(y)')
plt.ylabel('Frequency')
plt.title(dataset.split("/")[-1])
plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
plt.savefig(f"dependency/{dataset.split("/")[-1][:-4]}.png")