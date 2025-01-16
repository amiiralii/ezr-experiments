import os
import pandas as pd

def save_stat(stat, line, filename, directory):
    if len(stat)<1:
        return
    dir = directory
    if "d2h" in line:
        dir += "/res-d2h/"
    elif "without abs" in line:
        dir += "/res-noabs/"
    else:
        dir += "/res-abs/"
    dir += filename
    with open(dir, "w") as f:
        for s in range(len(stat)-1,-1,-1):
            f.write(stat[s])
            f.write('\n')
    
    

directory = os.getcwd()
for filename in os.listdir(f"{directory}/res/"):
    if ".csv" in filename:
        with open(f"{directory}/res/{filename}", "r") as file:
            content = file.read().split("\n")
            stat = []
            for i in range(len(content)-1,-1,-1):
                if '+' in content[i]:
                    save_stat(stat, content[i], filename, directory)
                    stat = []
                else:
                    stat.append(content[i])
