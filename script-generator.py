import sys, os
from ezr import DATA, csv

def printscript(dataset):
  cwd = os.getcwd()
  print(f'python3.12 {cwd}/extend.py {cwd}/{dataset} | tee {cwd}/reg/res/low-res/{dataset.split("/")[-1]} &')
  
print(f"mkdir -p {os.getcwd()}/reg/res")
print(f"mkdir -p {os.getcwd()}/reg/res/low-res")
print(f"rm {os.getcwd()}/reg/res/low-res/*")

for i in os.listdir(f"data/optimize"):
  for j in os.listdir(f"data/optimize/{i}/"):
    if j[-4:] == ".csv":
      if len(DATA().adds(csv("data/optimize/"+i+"/"+j)).rows) < 1000:
        printscript(str("data/optimize/"+i+"/"+j))
  