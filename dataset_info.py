import csv
import sys
import os
import pandas as pd
from ezr import the, DATA, csv, xval, activeLearning, rows, dist

def cat_ratio(cols):
    return str(sum([ "SYM" in str(type(k)) for k in cols])) + '/' + str(sum([ "NUM" in str(type(k)) for k in cols]))

def wir(cols):
    wir = 2
    for c in cols:
        if "SYM" in str(type(c)):
            mi, ma = min(c.has.values()) , max(c.has.values())
            if wir > (mi / ma): wir = round(mi / ma, 2)
    return wir if wir !=2 else "Num"

def wd(cols):
    mi=-1
    for c in cols:
        if "NUM" in str(type(c)):
            mi = round(c.m2 / (c.sd**2 + 1E-32), 2)
    return mi if mi > 0 else "Sym"


directory = os.getcwd()

first = True
lst = []
###  num/cat = Numeric to Categorical Ratio
###  WIR = Worst Imbalance Ratio
###  WDST = Worst Distribution : m2/sd
lst.append(["Dataset Name","xcols","ycols","rows", "dimension", "num/catX", "num/catY", "WIRX", "WIRY", "WDSTX", "WDSTY"])
for dir in os.listdir(f"{directory}/data/optimize/"):
    for dataset in os.listdir(f"{directory}/data/optimize/{dir}"):
        if dataset[-4:]=='.csv':
            d = DATA().adds(csv(f"{directory}/data/optimize/{dir}/{dataset}"))
            lst.append([dataset[:-4], len(d.cols.x), len(d.cols.y), len(d.rows), round(len(d.rows)/len(d.cols.all),2), cat_ratio(d.cols.x), cat_ratio(d.cols.y), wir(d.cols.x), wir(d.cols.y), wd(d.cols.x), wd(d.cols.y)])
        print(dataset, "done!")
            
print("Derived!")
df = pd.DataFrame(lst)
df.to_csv("dataset_info.csv",index=False)
print("Exported!")