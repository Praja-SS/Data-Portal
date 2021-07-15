import numpy as np
import math
import operator
from scipy.linalg import eig
from typing import List, Tuple, Dict, TypeVar
import pandas as pd
T = TypeVar('T')
def rowMaxSum(mat,N):
    maxSum = -1
    for i in range(0, N):
        sum = 0
        # calculate sum of row
        for j in range(0, N):
            sum += mat[i][j]
        if (sum > maxSum):
            maxSum = sum
    return maxSum
def rowSum(mat,row,n):
    sum=0
    for i in range(n):
        sum+=mat[row][i]
    return sum
def transpose(m):
    res = [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
    return res

def least_squares(data,ts,no):
    n=int(no)
    trainsize=int(ts)
    dataset = pd.read_csv(data,sep=',')
    X = dataset.iloc[:].values
    testsize=(100-trainsize)/100
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(X, test_size = testsize, random_state = 1)
    matches=[]
    for x,y,z in X_train:
        if(z == 0):
            matches.append((x,y))
        else:
            matches.append((y,x))

    ##print(matches)
    #matches = [('5','2'),('5','1'),('3','5'),('5','4'),('3','5'),('1','2')]
    list = [i for i in range(n)]
    item_to_index = {item: i for i, item in enumerate(list)}
    a = [ [ 0 for i in range(n) ] for j in range(n) ]
    y = [ [ 0 for i in range(n) ] for j in range(n) ]
    p = [ [ 0 for i in range(n) ] for j in range(n) ]
    #print(a)
    #print(A)
    #print(a)
    for (i,j) in matches:
        a[i][j] += 1

    for i in range(n):
       for j in range(n):
            f= a[i][j] + a[j][i]
            if(f!=0):
               a[i][j] /= f
    for i in range(n):
        for j in range(n):
            if(i!=j and i<j and (a[i][j]+a[i][j]!=0)):
                p[i][j] = a[i][j]/(a[i][j]+a[i][j])
                p[j][i] = 1 - a[i][j]/(a[i][j]+a[i][j])
    for i in range(n):
        for j in range(n):
            if (i!=j and (a[j][i]!=0 and a[i][j]!=0)):
                y[i][j]=math.log(a[i][j]/a[j][i])
            else:
                y[i][j]=0


    score=[]
    sum = 0
    for i in range(n) :
        for j in range(n) :
            sum+= y[i][j]
        sum= -(sum/n)
        score.append((round(sum,2),i))
        sum = 0
    #print(score)
    ls = sorted(score,reverse = True)
    rank=[]
    for x,y in ls:
        rank.append(y)
    sigma=[]
    for i in range(n):
        sigma.append(rank.index(i))
    sigma.reverse()
    #print(p)

    error=0
    res_error=0
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
##    print(accu)
#least_squares("Syntheticdataset1.csv",80,10)
#rank_centrality("dataset1.csv",80,5)
#rank_centrality("Syntheticdataset7.csv",80,10)
