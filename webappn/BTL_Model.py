import numpy as np
import pandas as pd
import itertools
import random
import csv

def getP(x,y):
    dataset = pd.read_csv('media/Model.csv',sep=',')
    exactP = dataset.iloc[:].values
    return exactP[x][y]

def BTLsynthetic_data(ni,nc,f):
    noItems=int(ni)
    noComparison=int(nc)
    frequency=int(f)
    p=[pow(3,i) for i in range(noItems)]
    #p=np.random.uniform(low=0.0, high=1.0, size=noItems)

    #p=[100,200,300,400,500,600,700,800,900,1000]
    #p=[10,20,30,40,50,60,70,80,90,100]
    #p=[1,2,3,4,5,6,7,8,9,10]
    #print(p)
    """
    prob = [ [ 0 for i in range(noItems) ] for j in range(noItems) ]
    for i in range(noItems):
        for j in range(noItems):
            if(i<j):
                prob[i][j] = random.uniform(0.0, 0.25)
                prob[j][i] = round(1-prob[i][j],2)
    """
    prob = [[0 for i in range(noItems)]for j in range(noItems)]
    for i in range(noItems):
        for j in range(noItems):
            if(i<j):
                prob[i][j] = round((p[j]/(p[i]+p[j])),2)
                prob[j][i] = round(1-prob[i][j],2)

    #print(prob)
    fieldnames=[]
    for i in range(noItems):
        fieldnames.append(i)
    with open('media/Model.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow
    with open('media/Model.csv', 'a', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(prob)
    combination=list(itertools.combinations(range(noItems), 2))
    #print(combination)
    chosenCombination=(random.sample(combination, noComparison))
    #print(chosenCombination)
    l=[]
    for x,y in chosenCombination:
        np.random.seed(42)
        p = getP(x,y)
        #print(p)
        i=0
        while(i<frequency):
            h = np.random.binomial(1, p)
            l.append((x,y,h))
            i=i+1
    with open('media/Syntheticdataset.csv', 'w', newline='') as file:
        fieldnames = ['i', 'j','k']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow
        for x,y,z in l:
            writer.writerow({'i': x, 'j': y, 'k': z})

"""
noItems=int(input())
p=np.random.uniform(low=0.0, high=1.0, size=noItems)
print(p)
prob = [ [ 0 for i in range(noItems) ] for j in range(noItems) ]
for i in range(noItems):
    for j in range(noItems):
        if(i<j):
            prob[i][j] = round((p[i]/(p[i]+p[j])),2)
            prob[j][i] = round(1-prob[i][j],2)
print(prob)
fieldnames=[]
for i in range(noItems):
    fieldnames.append(i)
with open('BTL.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow
with open('BTL.csv', 'a', newline='') as file:
    fields = ['BTL Model']
    mywriter = csv.writer(file, delimiter=',')
    mywriter.writerows(prob)
dataset = pd.read_csv('BTL.csv',sep=',')
exactP = dataset.iloc[0:].values
print(exactP)
synthetic_data(noItems,4,3)
"""
