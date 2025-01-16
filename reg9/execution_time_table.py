import os
import pandas as pd

directory = os.getcwd()

first = True
lst = []
for filename in os.listdir(f"{directory}/res/times/"):
    if filename[-4:]=='.csv':
        tms = {}
        tms['name'] = filename
        with open(f"{directory}/res/times/{filename}", mode='r') as file:
            for line in file:
                l = line.strip().split(',')
                col = str(l[0]+'-'+l[1])
                time = float(l[-1])
                tms[col] = time
        lst.append(tms)

df = pd.DataFrame(lst)
df.to_csv("time_res.csv",index=False)