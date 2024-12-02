import csv
import sys
import os
import pandas as pd
from ezr import the, DATA, csv, xval, activeLearning, rows, dist

directory = os.getcwd()

first = True
lst = []
lst.append(["Dataset Name","xcols","ycols","rows"])
for dir in os.listdir(f"{directory}/data/optimize/"):
    for dataset in os.listdir(f"{directory}/data/optimize/{dir}"):
        if dataset[-4:]=='.csv':
            d = DATA().adds(csv(f"{directory}/data/optimize/{dir}/{dataset}"))
            lst.append([dataset[:-4], len(d.cols.x), len(d.cols.y), len(d.rows)])
            

df = pd.DataFrame(lst)
df.to_csv("dataset_info.csv",index=False)