import math
import numpy as np
import description_length as MDL
import random as rand
from sklearn.linear_model import Ridge
import time

def smoother(lst,alpha):
    X = [lst[0]]
    for i in range(1,len(lst)):
        X.append(sum([lst[i-j]*alpha**j for j in range(i)])/sum([alpha**j for j in range(i)]))
    return X

def best_k_finder2(S):
    smooth90 = smoother(list(S.values()),0.9); smooth95 = smoother(list(S.values()),0.98)
    N = len(smooth90)
    caps = [i for i in range(N) if smooth95[i] <= smooth90[i] and i >= 10]
    if len(caps) > 1:
        cap = min(caps)
        search_dic = {i:S[i] for i in range(cap)}
        k_opt = min(search_dic, key = search_dic.get)
    else:
        k_opt = N
    return k_opt

def qrappend(Q,R,x):
   m = Q.shape[1] + 1
   r = np.matmul(np.transpose(Q),x)
   R = np.concatenate((R, r),axis=1)
   q = x - np.matmul(Q,r)
   f = np.linalg.norm(q)
   R = np.concatenate((R,np.zeros(m)),axis=0)
   R[m,m] = f 
   Q = np.concatenate((Q, q/f),axis=1)
   return Q,R 


def max_exclude(arr,indicies):
    maxval = 0 ; best_index = 0
    for i in range(len(arr)):
        if arr[i] > maxval and i not in indicies:
            maxval = arr[i]; best_index = i 
    return best_index
   

def fit_coefs(V,Y,alpha=0):
    # note that alpha used to be 10**(-9)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(V),V)+np.diag(np.ones(V.shape[1]))*alpha),
                                     np.transpose(V)),Y)
    #return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(V),V)),np.transpose(V)),Y)

def step1(Y,n,gamma):
    return {0:(n/2-1)*math.log(np.sum(np.square(Y))/n) + math.log(n/2)/2 + n*(1+math.log(math.pi*2))/2
            + 1/2 + math.log(gamma)}

def step2(X,Y,maxK):
    d = {i:abs(np.matmul(np.transpose(X[:,i]),Y)) for i in range(maxK)}
    B = [max(d, key=d.get)]
    V = np.transpose(np.array([X[:,b] for b in B]))
    coefs = fit_coefs(V, Y)
    e_b = Y - np.matmul(V,coefs)
    return B, e_b

def step3(X,B,e_b,maxK):
    mu = np.matmul(np.transpose(X),e_b)
    j = max_exclude(np.abs(mu), B)
    #j = np.argmax(np.abs(mu))
    #if j in B:
    #    print('J WAS ALREADY IN B')
    #    print('this is B:')
    #    print(B)
    #    print(type(B))
    #    print('here is mu: ')
    #    print(mu.shape)
    #    print('')
    #    time.sleep(1)
    #    some_other = rand.choice(list(set(list(range(maxK)))-set(B)))
    #    B_prime = B+[some_other]
    #    newcommer = some_other
    #else:
    #    print('All g')
    B_prime = B+[j]
    #    newcommer = j
    return B_prime, j#newcommer

def step4(X,Y,B_prime):
    V = np.transpose(np.array([X[:,b] for b in B_prime]))
    coefs = fit_coefs(V, Y) 
    #print('coefs are '+str(coefs))
    #print('argmin is '+str(np.argmin(np.abs(coefs))))
    return B_prime[np.argmin(np.abs(coefs))]

def reget_coefs(X,Y,B):
    V = np.transpose(np.array([X[:,b] for b in B]))
    coefs = fit_coefs(V, Y) 
    e_b = Y - np.matmul(V,coefs)
    return e_b, coefs

def get_MDL_of_B(X,Y,B):
    errors, coefs = reget_coefs(X, Y, B)
    #DL = MDL.D_length(errors, len(B), np.transpose(np.array([X[:,b] for b in B])))
    DL = MDL.D_length(errors, len(B), np.transpose(np.array([X[:,b] for b in B])), coefs, Y=Y)
    return DL, np.sum(np.square(errors))/len(errors), coefs, errors
    
def breath(X,Y,gamma=32, max_iterations = 10000, early_stop = False,verbose=False,min_size=10,
           cap='na'):
    n = len(Y); maxK = X.shape[1]
    # ----------------------- step 1
    S = step1(Y,n,gamma)
    Bs = {0:[]}
    Vars = {0:np.sum(np.square(Y))/n}
    Coefs_by_K = {}
    # ----------------------- step 2
    B,e_b = step2(X,Y,maxK)
    MDL, Var, cos, e_b = get_MDL_of_B(X, Y, B)
    #print("the B from step 2 was: "+str(B))
    # ----------------------- step 3
    K = 1
    while K < maxK:
        loop = True; counter = 0
        #print('---------------------------------------------------------- K = '+str(K))
        #print('at point 1, B is '+str(B))
        dicset = {}; MDLset = {}
        while loop:
            
            B_prime, newcommer = step3(X,B,e_b,maxK)
            #print('at point 2, B is '+str(B))
            # ----------------------- step 4
            #print('Bprime is '+str(B_prime))
            outgower = step4(X,Y,B_prime)
            #print('at point 3, B is '+str(B))
            #print('Bprime is '+str(B_prime))
            #print('Newcommer is '+str(newcommer))
            #print('outgower is '+str(outgower))
            #print('')
            if newcommer == outgower or K == maxK -1:
                MDL, Var, cos, e_b = get_MDL_of_B(X, Y, B)
                #print('MDL for K = '+str(K)+' was '+str(MDL))
                #print('with B = '+str([b+1 for b in B]))
                S[len(B)] = MDL
                Vars[K] = Var
                Bs[K] = [b for b in B]
                Coefs_by_K[K] = cos
                B.append(newcommer)
                break
            elif counter == max_iterations:
                print('hit max itterations')
                best_counter = min(MDLset, key = MDLset.get) #!
                B = dicset[best_counter]                     #!
                MDL, Var, cos, e_b = get_MDL_of_B(X, Y, B)
                #print('MDL for K = '+str(K)+' was '+str(MDL) + ' BUT LOOP WAS BROKEN BY COUNTER')
                #print('with B = '+str([b+1 for b in B]))
                S[len(B)] = MDL
                Vars[K] = Var
                Bs[K] = [b for b in B]
                Coefs_by_K[K] = cos
                B.append(newcommer)
                break
            else:
                e_b, coefs = reget_coefs(X, Y, B)           #!
                dicset[counter] = B                         #!
                MDLset[counter] = np.mean(np.square(e_b))   #!
                B.remove(outgower)
                B.append(newcommer)
                e_b, cos = reget_coefs(X, Y, B)
                
            counter += 1
            #print('counter = '+str(counter))
            #print('dicset:')
            #print(dicset)
            #time.sleep(1)
            #time.sleep(1)

        K += 1
        if verbose:
            print('k = '+str(K))
        if early_stop and K >=min_size:
            kopt = best_k_finder2(S)
            if kopt < K - 10:
                print('BROKE EARLY FROM BREATHING ALGORITHM')
                break
        if cap != 'na':
            if K >= cap:
                print('HIT CAP SIZE')
                break
        #else:
        #if K %10 == 0 and K >= 100:
        #    time.sleep(10)
    return S, Bs, Vars, Coefs_by_K
        
def dont_breath(X,Y,gamma=32):
    n = len(Y); maxK = X.shape[1]
    # ----------------------- step 1
    S = step1(Y,n,gamma)
    # ----------------------- step 2
    B,e_b = step2(X,Y,maxK)
    # ----------------------- step 3
    K = 1 
    Vars = {0:np.sum(np.square(Y))/n}
    while K < maxK:
        temp = []; temp2 = []
        for i in range(30):
            B = rand.sample(list(range(maxK)),k=K)
            MDL, var = get_MDL_of_B(X, Y, B)
            temp.append(MDL); temp2.append(var)
        S[K] = np.mean(temp); Vars[K] = np.mean(temp2)
        K += 1
    return S, Vars

    
def ridge_it(X,Y,gamma=32,alphas='auto'):
    n = len(Y); k = X.shape[1]
    if alphas == 'auto':
        alphas = [10**i for i in np.linspace(-9,0,91)]
    variances = {}; Coefs_by_a = {}
    for alpha in alphas:
        clf = Ridge(alpha=alpha)
        clf.fit(X,Y)
        coefs = clf.coef_
        Coefs_by_a[math.log10(alpha)] = coefs
        error = Y - np.matmul(X,coefs)
        n = len(error)
        variance = np.mean(np.square(error)) # the same as e^Te/n
        variances[math.log10(alpha)] = variance
    return variances, alphas, Coefs_by_a
        
            