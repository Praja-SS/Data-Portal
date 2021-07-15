import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import itertools
import os
def borda_count(ds,ts,no,flag):
    trainsize=int(ts)
    n=int(no)
    dataset = pd.read_csv(ds,sep=',')
    X = dataset.iloc[:].values
    uniqueVal = dataset.iloc[:,:-1].values
#y = dataset.iloc[:, -1].values


    l=[]
    testsize=(100-trainsize)/100
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(X, test_size = testsize, random_state = 1)
#print(X_train)
#print(y_train)
#print(X_test)
#print(y_test)
    l=uniqueVal.tolist()
    l.sort()
    new_list = list(l for l,_ in itertools.groupby(l))
#print("New List", new_list)

    borda_matrix = [ [ 0 for i in range(n) ] for j in range(n) ]
    for i,j in new_list:
        count_x=0
        count_y=0
        for x,y,z in X_train:
            if(i==x and j==y and z==0):
                count_x+=1
            elif(i==x and j==y and z==1):
                count_y+=1
        borda_matrix[i][j]=round(count_x/(count_x+count_y),2)
        borda_matrix[j][i]=round(1-borda_matrix[i][j],2)
    #print(borda_matrix)
    score=[]
    sum = 0
    for i in range(n) :
        for j in range(n) :
            sum+= borda_matrix[i][j]
        score.append((round(sum,2),i))
        sum = 0
    #print(score)
    ls = sorted(score,reverse = True)
    #print(ls)
    rank=[]
    for x,y in ls:
        rank.append(y)
    #print(rank)
    sigma=[]
    for i in range(n):
        sigma.append(rank.index(i))


    error=0
    res_error=0
    if flag== 1:
        dataset = pd.read_csv('media/Model.csv',sep=',')
        exactP = dataset.iloc[:].values
        for i in range(n):
            for j in range(n):
                if(i!=j):
                    #a=rank.index(i)
                    #b=rank.index(j)
                    a=sigma[i]
                    b=sigma[j]
                    if(exactP[i][j]>0.5 and a<b ):
                        error+=1
        res_error= round(((error*2)/n*(n-1)),2)
    err=0
    for x,y,z in X_test:
        a=sigma[x]
        b=sigma[y]
        if((a>b and z==0) or (a<b and z==1)):
            err+=1
    accu=round((100-(err/len(X_test))*100),2)

    return sigma,res_error,accu

    #for i,j in new_list:
        #a=rank.index(i)
        #b=rank.index(j)
    #    a=sigma[i]
    #    b=sigma[j]
    #    if(exactP[i][j]>0.5 and a>b ):
    #                error+=1
#print(error)

#print("Pairwise - Disagreement Error : ", res_error)

#borda_count("dataset5.csv",80,5)
