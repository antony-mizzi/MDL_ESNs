import reservoirs as res
import numpy as np
import random as rand
import maps_and_systems as mapnsys
import matplotlib.pyplot as plt
import matplotlib as mpl
import breathing as breath
import genetic as gen
import math
import time
import pandas as pd
import networkx as nx
import phase_space_network_builder as PSNB
from scipy.linalg import fractional_matrix_power
import csv
import os

def measure(M,R):
    M = M/np.linalg.norm(M); R = R/np.linalg.norm(R)
    return np.dot(M,R)

def fit_coefs(V,Y,alpha=0):
    # note that alpha used to be 10**(-9)
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(V),V)+np.diag(np.ones(V.shape[1]))*alpha),
                                     np.transpose(V)),Y)

def log_data(details,fname):
    try:
        df1 = pd.read_csv('logs/'+fname)
        df2 = pd.DataFrame(details, index=[0])
        DF = pd.concat([df1, df2], ignore_index = True)
        DF.reset_index()
    except:
        DF = pd.DataFrame(details, index=[0])
    DF.to_csv('logs/'+fname, index=False)

def NET_counter(coefs,n,is_MDL=True,edges = True,triangles=True,squares=False):
    if is_MDL: # expect subset selection in the place of coefs
        N = len([c for c in coefs if c < n])
        if edges:
            E = len([c for c in coefs if c >= n and c < 2*n])
        else:
            E = 0
        if triangles:
            T = len([c for c in coefs if c >= 2*n and c < 3*n])
        else:
            T=0
        if squares:
            S = len([c for c in coefs if c >= 3*n])
        else:
            S=0 
    else:
        N = np.sum(np.square(coefs[:n]))
        if edges:
            E = np.sum(np.square(coefs[n:2*n]))
        else:
            E = 0
        if triangles:
            T = np.sum(np.square(coefs[2*n:3*n]))
        else:
            T = 0
        if squares:
            S = np.sum(np.square(coefs[3*n:]))
        else:
            S = 0
    tot = N+E+T+S
    proportions = {'Nodes':N/tot,'Edges':E/tot,'Triangles':T/tot,'Squares':S/tot}
    totals  = {'Nodes':N,'Edges':E,'Triangles':T,'Squares':S}
    return proportions,totals

def smoother(lst,alpha):
    X = [lst[0]]
    for i in range(1,len(lst)):
        X.append(sum([lst[i-j]*alpha**j for j in range(i)])/sum([alpha**j for j in range(i)]))
    return X

def best_k_finder(S):
    smooth90 = smoother(list(S.values()),0.9); smooth95 = smoother(list(S.values()),0.98)
    N = len(smooth90)
    caps = [i for i in range(N) if smooth95[i] <= smooth90[i] and i >= 10]
    if len(caps) > 1:
        cap = min(caps)
        search_dic = {i:S[i] for i in range(cap)}
        k_opt = min(search_dic, key = search_dic.get)
    else:
        k_opt = N-1
    return k_opt

def consistency_cap_old(X,X2):
    X = X-X.mean(axis=0, keepdims=True)
    row_means = np.linalg.norm(X, axis=0, keepdims=True)
    X = X / row_means
    X2 = (X2-X.mean(axis=0, keepdims=True))
    row_means2 = np.linalg.norm(X2, axis=0, keepdims=True)
    X2 = X2/row_means2
    C = np.matmul(np.transpose(X),X)
    vals,Q = np.linalg.eig(C)
    #sigma2 = np.diag(vals)+np.diag(10**(-6)*np.ones(len(vals)))
    sigminv = np.diag([1/(math.sqrt(abs(v))+10**(-7)) for v in vals])
    T = np.matmul(Q,np.matmul(sigminv,np.transpose(Q)))
    Xtild = np.matmul(X,T)
    Xtild2 = np.matmul(X2,T)
    Cc = np.matmul(np.transpose(Xtild),Xtild2)
    #Cc = np.matmul(np.transpose(np.matmul(X,T)),np.matmul(X2,T))
    gammas,Qc = np.linalg.eig(Cc)
    return sum([abs(G) for G in gammas])

def consistency_cap():
    df = pd.read_csv('logs/consistency/bigX.csv')
    leng = len(df['X_0'].values)
    ncomps = int(len(list(df.columns))/2)
    subset = rand.sample(list(range(ncomps)),k=ncomps)
    X = np.array([[df['X_'+str(s)].values[i] for s in subset] for i in range(leng)])
    X2 = np.array([[df['X2_'+str(s)].values[i] for s in subset] for i in range(leng)])
    del df
    X = X-X.mean(axis=0, keepdims=True)
    row_means = np.linalg.norm(X, axis=0, keepdims=True)
    X = X / row_means
    X2 = (X2-X.mean(axis=0, keepdims=True))
    row_means2 = np.linalg.norm(X2, axis=0, keepdims=True)
    X2 = X2/row_means2
    C = np.matmul(np.transpose(X),X)
    vals,Q = np.linalg.eig(C)
    del C, row_means, row_means2
    #sigma2 = np.diag(vals)+np.diag(10**(-6)*np.ones(len(vals)))
    sigminv = np.diag([1/(math.sqrt(abs(v))+10**(-7)) for v in vals])
    T = np.matmul(Q,np.matmul(sigminv,np.transpose(Q)))
    del vals, Q, sigminv
    Xtild = np.matmul(X,T)
    del X
    Xtild2 = np.matmul(X2,T)
    del X2, T
    Cc = np.matmul(np.transpose(Xtild),Xtild2)
    del Xtild, Xtild2
    #Cc = np.matmul(np.transpose(np.matmul(X,T)),np.matmul(X2,T))
    gammas,Qc = np.linalg.eig(Cc)
    #print('concap = '+str(sum([abs(G) for G in gammas])))
    print('concap = '+str(sum([G.real for G in gammas])))
    concomps = sorted([G.real for G in gammas],reverse=True)
    return concomps

    
def make_bigXs(net,TS_train,activation='tanh',steps_ahead=7,obs_noise='na',
                    leaky=1,edges=True,triangles=True,squares=True,terminate='na'):  
    transient_p = 500
    X_train1 = net.act(TS_train,transient_p,activation=activation,leaky=leaky,
                      edges=edges,triangles=triangles,squares=squares,
                      state='random')
    X_train2 = net.act(TS_train,transient_p,activation=activation,leaky=leaky,
                      edges=edges,triangles=triangles,squares=squares, 
                      state='random')
    row_means = np.linalg.norm(X_train1, axis=0, keepdims=True)
    X_train1 = X_train1 / row_means
    X_train2 = X_train2 / row_means
    #X_test = X_test / row_means
    X_train1 = X_train1[1000:-steps_ahead,:]; X_train2 = X_train2[1000:-steps_ahead,:]
    Y_train = TS_train[1000+steps_ahead+transient_p:]
    if obs_noise != 'na':
        Y_train += np.random.normal(loc=0.0,scale=10**obs_noise,size=len(Y_train))

    bigX = np.concatenate((X_train1[1000:],X_train2[1000:]),axis=1)

    all_comps = bigX.shape[1]//2

    col_names = ['X_'+str(i) for i in range(X_train1.shape[1])]+['X2_'+str(i) for i in range(X_train2.shape[1])]
    bigframe = pd.DataFrame(bigX,columns=col_names)
    bigframe.to_csv('logs/consistency/bigX.csv', index=False)
    
    

def plot_from_csv(csv_file='logs/concomps_power.csv',net_type='embedding',tail_cut = 0.01):
    df = pd.read_csv(csv_file)
    dfaug = df[df['aug']==True]; dfjn = df[df['aug']==False]
    Xaug = sorted(list(set([int(c) for c in dfaug.columns if c[0] != 'a'])))
    Xjn = sorted(list(set([int(c) for c in dfjn.columns if c[0] != 'a'])))
    cutpoint = len([i for i in Xjn if not math.isnan(np.mean(dfjn[str(i)]))]) -1
    Xjn = Xjn[:cutpoint]
    Yjn = [min(1,np.mean(dfjn[str(x)])) for x in Xjn]
    Yaug = [min(1,np.mean(dfaug[str(x)])) for x in Xaug]
    cutp2 = max([len([x for x in Yjn if x > tail_cut]),len([x for x in Yaug if x > tail_cut])])
    print('cutp2 = '+str(cutp2))
    print(Xaug[cutp2-1])
    Xjn = [1+x for x in Xjn[:min([len(Xjn),cutp2])]]; Xaug = [1+x for x in Xaug[:cutp2]]
    Yjn = Yjn[:min([len(Yjn),cutp2])]; Yaug = Yaug[:cutp2]
    fig,axs=plt.subplots(1,1,figsize=(10,7))
    axs.plot(np.array(Xjn),Yjn,color='k',label='Just Node States')
    axs.plot(np.array(Xaug),Yaug,color='k',linestyle='dashed',label='Augmented')
    axs.legend(fontsize=18)
    axs.tick_params(axis='both', labelsize=15)
    axs.set_xlabel('$n$',fontsize=25)
    axs.set_ylabel('$n^{\\text{th}}$ Component Size',fontsize=25)
    plt.tight_layout()
    plt.show()


        
def gather_data(net,activation='tanh',net_type='embedding',TS_length=15000,
            edges=False,triangles=False,squares=False,terminate='na'):
    TS_train = mapnsys.Rossler(TS_length+500, 0.02)
    
    if edges or triangles or squares:
        net.get_subgraphs(net.net_size,edges=edges,triangles=triangles,squares=squares)
    make_bigXs(net, TS_train, edges=edges, triangles=triangles, squares=squares, terminate = terminate)
    

def main(net,fname='concomps_power.csv',activation='tanh',edges=False,triangles=False,squares=False,TS_length=15000,
         terminate='na'):
    gather_data(net,activation=activation,edges=edges,triangles=triangles,
                squares = squares,TS_length=TS_length,terminate=terminate)
    print('Gathering Complete')
    details = {'aug':[edges]}
    concomps = consistency_cap()
    for i in range(len(concomps)):
        details[str(i)] = concomps[i]
    log_data(details,'concomps_power.csv')
    os.remove('logs/consistency/bigX.csv') 
    


for go in range(10):
    n=1000
    net = res.ESN(n,1)
    main(net)
    print('finished with node states')
    main(net,edges=True,triangles=True,squares=True)
    for tail_cut in [0.001,0.002,0.005,0.01,0.02,0.05]:
        plot_from_csv(tail_cut = tail_cut)
    os.remove('logs/concomps_power.csv') 
    time.sleep(30)
