import numpy as np
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

def extract_rc_scores(comparisons: List[Tuple[T, T]], no) -> Dict[T, float]:
    n=int(no)
    list = [i for i in range(n)]
    item_to_index = {item: i for i, item in enumerate(list)}
    a = [ [ 0 for i in range(n) ] for j in range(n) ]
    A = [ [ 0 for i in range(n) ] for j in range(n) ]
    p = [ [ 0 for i in range(n) ] for j in range(n) ]
    #print(a)
    #print(A)
    #print(a)
    for (i,j) in comparisons:
        a[i][j] += 1
    #print(a)

    for i in range(n):
       for j in range(n):
            f= a[i][j] + a[j][i]
            if(f!=0):
               a[i][j] /= f
    #print(a)
    """
    dataset = pd.read_csv('media/Model.csv',sep=',')
    exactP = dataset.iloc[:].values
    err=0
    for i in range(n):
        for j in range(n):
            if (i==j):
                A[i][j]=0;
            else:
                A[i][j]=exactP[j][i]
            """
    for i in range(n):
        for j in range(n):
            if(i!=j and i<j and (a[i][j]+a[i][j]!=0)):
                A[i][j] = a[i][j]/(a[i][j]+a[i][j])
                A[j][i] = 1 - a[i][j]/(a[i][j]+a[i][j])

    #print(A)
    #for i in range(n):
    #    for j in range(n):
    #       A[i][j]+=1

    d_max= rowMaxSum(A,n)
    #print(d_max)
    for i in range(n):
        for j in range(n):
            #if(i==j):
            #    p[i][j]=1-rowSum(A,i,n)
            #else:
                p[i][j] = A[i][j]/d_max
                #p[i][j]= (a[i][j])/((a[i][j]+a[j][i])*d_max)
    for i in range(n):
        for j in range(n):
            if(i==j):
                p[i][i]=1-rowSum(p,i,n)
#    print(p)
#    print(p)
    w, v = eig(p, left=True, right=False)

    max_eigv_i = np.argmax(w)
    scores = np.real(v[:, max_eigv_i])
    #print(scores)
    return {item: scores[index] for item, index in item_to_index.items()}


def rank_centrality(data,ts,no):
    n=int(no)
    trainsize=int(ts)
    dataset = pd.read_csv(data,sep=',')
    X = dataset.iloc[:].values
    testsize=(100-trainsize)/100
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(X, test_size = testsize, random_state = 1)
    matches=[]
    for x,y,z in X_train:
        if(z == 1):
            matches.append((x,y))
        else:
            matches.append((y,x))

    ##print(matches)
    #matches = [('5','2'),('5','1'),('3','5'),('5','4'),('3','5'),('1','2')]
    team_to_score = extract_rc_scores(matches,no)

    sorted_teams = sorted(team_to_score.items(), key=operator.itemgetter(1), reverse=True)
    rank=[]
    for team, score in sorted_teams:
        rank.append(team)
    #print(rank)
    sigma=[]
    for i in range(n):
        sigma.append(rank.index(i))
    #print(sigma)
        #print(team)
        #print('{} has a score of {!s}'.format(team, round(score, 3)))
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
#rank_centrality("Syntheticdataset111.csv",80,10)
#rank_centrality("dataset1.csv",80,5)
#rank_centrality("Syntheticdataset7.csv",80,10)
"""
import numpy as np
import operator
from scipy.linalg import eig
from typing import List, Tuple, Dict, TypeVar
import pandas as pd
T = TypeVar('T')
def extract_rc_scores(comparisons: List[Tuple[T, T]], regularized: bool = True) -> Dict[T, float]:
    winners, losers = zip(*comparisons)
    unique_items = np.hstack([np.unique(winners), np.unique(losers)])

    item_to_index = {item: i for i, item in enumerate(unique_items)}

    A = np.ones((len(unique_items), len(unique_items))) * regularized  # Initializing as ones results in the Beta prior

    for w, l in comparisons:
        A[item_to_index[l], item_to_index[w]] += 1

    A_sum = (A[np.triu_indices_from(A, 1)] + A[np.tril_indices_from(A, -1)]) + 1e-6  # to prevent division by zero

    A[np.triu_indices_from(A, 1)] /= A_sum
    A[np.tril_indices_from(A, -1)] /= A_sum

    d_max = np.max(np.sum(A, axis=1))
    A /= d_max

    w, v = eig(A, left=True, right=False)

    max_eigv_i = np.argmax(w)
    scores = np.real(v[:, max_eigv_i])

    return {item: scores[index] for item, index in item_to_index.items()}

def rank_centrality(data,ts,no):
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
    team_to_score = extract_rc_scores(matches)
    sorted_teams = sorted(team_to_score.items(), key=operator.itemgetter(1), reverse=True)
    rank=[]
    for team, score in sorted_teams:
        rank.append(team)
    print(rank)
    sigma=[]
    for i in range(n):
        sigma.append(rank.index(i))
    #print(sigma)
        #print(team)
        #print('{} has a score of {!s}'.format(team, round(score, 3)))
    error=0
    res_error=0
    dataset = pd.read_csv('media/Model.csv',sep=',')
    exactP = dataset.iloc[:].values
    err=0
    for i in range(n):
        for j in range(n):
            if(i!=j and i<j):
                #a=rank.index(i)
                #b=rank.index(j)
                a=sigma[i]
                b=sigma[j]
                if(exactP[i][j]>0.5 and a>b ):
                    error+=1
    print(err)
    res_error= round(((error*2)/n*(n-1)),2)
    print(res_error)
    err=0
    for x,y,z in X_test:
        a=sigma[x]
        b=sigma[y]
        if((a>b and z==0) or (a<b and z==1)):
            err+=1
    accu=round((100-(err/len(X_test))*100),2)
"""
"""
import numpy as np
import operator
from scipy.linalg import eig
from typing import List, Tuple, Dict, TypeVar
import pandas as pd
T = TypeVar('T')
def extract_rc_scores(comparisons: List[Tuple[T, T]], regularized: bool = True) -> Dict[T, float]:
    winners, losers = zip(*comparisons)
    unique_items = np.hstack([np.unique(winners), np.unique(losers)])

    item_to_index = {item: i for i, item in enumerate(unique_items)}

    A = np.ones((len(unique_items), len(unique_items))) * regularized  # Initializing as ones results in the Beta prior

    for w, l in comparisons:
        A[item_to_index[l], item_to_index[w]] += 1

    A_sum = (A[np.triu_indices_from(A, 1)] + A[np.tril_indices_from(A, -1)]) + 1e-6  # to prevent division by zero

    A[np.triu_indices_from(A, 1)] /= A_sum
    A[np.tril_indices_from(A, -1)] /= A_sum

    d_max = np.max(np.sum(A, axis=1))
    A /= d_max

    w, v = eig(A, left=True, right=False)

    max_eigv_i = np.argmax(w)
    scores = np.real(v[:, max_eigv_i])

    return {item: scores[index] for item, index in item_to_index.items()}

def rank_centrality(data,ts,no):
    n=int(no)
    trainsize=int(ts)
    dataset = pd.read_csv(data,sep=',')
    X = dataset.iloc[:].values
    testsize=(100-trainsize)/100
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(X, test_size = testsize, random_state = 1)
    matches=[]
    for x,y,z in X_train:
        if(z == 1):
            matches.append((x,y))
        else:
            matches.append((y,x))

    ##print(matches)
    #matches = [('5','2'),('5','1'),('3','5'),('5','4'),('3','5'),('1','2')]
    team_to_score = extract_rc_scores(matches)
    sorted_teams = sorted(team_to_score.items(), key=operator.itemgetter(1), reverse=True)
    rank=[]
    for team, score in sorted_teams:
        rank.append(team)
    print(rank)
    sigma=[]
    for i in range(n):
        sigma.append(rank.index(i))
    #print(sigma)
        #print(team)
        #print('{} has a score of {!s}'.format(team, round(score, 3)))
    error=0
    res_error=0
    dataset = pd.read_csv('media/Model.csv',sep=',')
    exactP = dataset.iloc[:].values
    err=0
    for i in range(n):
        for j in range(n):
            if(i!=j):
                #a=rank.index(i)
                #b=rank.index(j)
                a=sigma[i]
                b=sigma[j]
                if(exactP[i][j]>0.5 and a<b ):
                    error+=1
    print(err)
    res_error= round(((error*2)/n*(n-1)),2)
    print(res_error)
    err=0
    for x,y,z in X_test:
        a=sigma[x]
        b=sigma[y]
        if((a>b and z==0) or (a<b and z==1)):
            err+=1
    accu=round((100-(err/len(X_test))*100),2)
    print( sigma,res_error,accu)
    return sigma,res_error,accu
    #print(accu)
#rank_centrality("Syntheticdataset.csv",80,6)


#    print(accu)
#rank_centrality()
"""
