from ezr import the, DATA, csv, xval, activeLearning, rows, dist
import sys 
import matplotlib.pyplot as plt
import numpy as np
import os

def measure_dependancy(dataset):
    d = DATA().adds(csv(dataset))
    kmeans_clusters = d.kmeansplusplus(rows = d.rows, neighbors=5)
    xdist , ydist = [], []
    for cluster in kmeans_clusters:
        centroid = cluster[0]
        xdist += [d.dist(centroid, r) for r in cluster]
        ydist += [d.disty(centroid, r) for r in cluster]
        
    return [round(x/y,4) for x,y in zip(xdist,ydist) if 0 not in [x,y]]

def build_chart(measure):
    nrm, unnorm = 0, 0
    for a in range(len(measure)): 
        if measure[a] >= 4: measure[a] = 4
        if measure[a] >= 4 or measure[a] <= 0.25: unnorm += 1
        else: nrm += 1

    plt.hist(measure, bins=20 , edgecolor='red')
    plt.xlim(0, 3)
    plt.xlabel('dist(x) / dist(y)')
    plt.ylabel('Frequency')
    plt.title(dataset.split("/")[-1])
    plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    plt.savefig(f"dependency/{dataset.split("/")[-1][:-4]}.png")

    return nrm / unnorm

lst = {}
directory = os.getcwd()
for dir in os.listdir(f"{directory}/data/optimize/"):
    for dataset in os.listdir(f"{directory}/data/optimize/{dir}"):
        if dataset[-4:]=='.csv':
            d = f'{directory}/data/optimize/{dir}/{dataset}'
            measure = measure_dependancy(d)
            lst[dataset] = build_chart(measure)

for l,v in lst.items():
    print(f"{l},{round(v,2)}")