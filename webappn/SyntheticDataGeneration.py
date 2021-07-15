import numpy as np
import pandas as pd
import itertools
import random
import csv
import os
base = os.path.dirname(__file__)

def getP(x,y):
    rel_path = "media/Model.csv"
    abs_file_path = os.path.join(base, rel_path)
    dataset = pd.read_csv(abs_file_path,sep=',')
    exactP = dataset.iloc[:].values
    return exactP[x][y]

def synthetic_data(ni,nc,f):
    rel_path = "media/Model.csv"
    abs_file_path = os.path.join(base, rel_path)
    dataset = pd.read_csv(abs_file_path,sep=',')
    exactP = dataset.iloc[:].values
    noItems=(len(exactP))
    noComparison=int(nc)
    frequency=int(f)
    combination=list(itertools.combinations(range(noItems), 2))
    chosenCombination=(random.sample(combination, noComparison))
    l=[]
    for x,y in chosenCombination:
        i=0
        np.random.seed(42)
        p = getP(x,y)
        while(i<frequency):
            h = np.random.binomial(1, p)
            l.append((x,y,h))
            i=i+1

    rel_path = "media/Syntheticdataset.csv"
    abs_file_path = os.path.join(base, rel_path)
    with open(abs_file_path, 'w', newline='') as file:
        fieldnames = ['i', 'j','k']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow
        for x,y,z in l:
            writer.writerow({'i': x, 'j': y, 'k': z})

#noItems=int(input("Number of Items: "))
#noComparison=int(input("Number of Comparisions: "))
#frequency=int(input("Frequency: "))
#synthetic_data(noItems,noComparison,frequency)
