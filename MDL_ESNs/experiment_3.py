# how does K_opt change with ESN size?
# how does the simiilarity between RR and MDL models change with ESN size?

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
import description_length as DL

def measure(M,R):
    M = M/np.linalg.norm(M); R = R/np.linalg.norm(R)
    return np.dot(M,R)

def log_data(details,fname):
    try:
        df1 = pd.read_csv('logs/'+fname)
        df2 = pd.DataFrame(details, index=[0])
        DF = pd.concat([df1, df2], ignore_index = True)
        DF.reset_index()
    except:
        DF = pd.DataFrame(details, index=[0])
    DF.to_csv('logs/'+fname, index=False)
    
def best_k_finder(dic):
    N = len(list(dic.keys())); vals = list(dic.values())
    if  N < 50:
        k_opt = min(dic, key = dic.get)
    else:
        cut = 20; counter = 20; keep_going = True
        while counter < N-20 and keep_going:
            if np.mean(vals[counter-cut:counter]) < np.mean(vals[counter:counter+cut]):
                counter += 1
            else:
                keep_going = False
        search_dic = {counter-cut+i:dic[counter-cut+i] for i in range(2*cut)}
        k_opt = min(search_dic, key = search_dic.get)
    return k_opt
    
def best_k_finder2(S):
    smooth90 = smoother(list(S.values()),0.9); smooth95 = smoother(list(S.values()),0.95)
    N = len(smooth90)
    caps = [i for i in range(N) if smooth95[i] <= smooth90[i] and i >= 10]
    if len(caps) > 1:
        cap = min(caps)
    else:
        cap = len(smooth90)
    search_dic = {i:S[i] for i in range(cap)}
    k_opt = min(search_dic, key = search_dic.get)
    return k_opt

def smoother(lst,alpha):
    X = [lst[0]]
    for i in range(1,len(lst)):
        X.append(sum([lst[i-j]*alpha**j for j in range(i)])/sum([alpha**j for j in range(i)]))
    return X

def double_sided_smoother(lst,alpha):
    X = [lst[0]]; N = len(lst)
    for i in range(1,len(lst)):
        X.append(sum([lst[j]*alpha**(abs(i-j)) for j in range(N)])/sum([alpha**(abs(i-j)) for j in range(N)]))
    return X
    

def get_data(activation='tanh',TS_length=2000,steps_ahead=1,obs_noise='na',leaky=1,
                  edges=False,triangles=False,squares=False,n=100):  
    transient_p = 500
    spec_rad = 0.8+0.2*rand.random();
    if edges:
        con_prob = 1/n + (1-1/n)*rand.random() 
    else:
        con_prob = (3+10*rand.random())/n #
    net = res.ESN(n, spec_rad, con_prob = con_prob, directed=False, self_loops=False)
    TS_train = mapnsys.Lorenz(TS_length+transient_p, 0.02)
    TS_test = mapnsys.Lorenz(10000+transient_p, 0.02)
    if edges or triangles or squares:
        net.get_subgraphs(n,edges=edges,triangles=triangles,squares=squares)
    X_train = net.act(TS_train,transient_p,activation=activation,leaky=leaky,
                      edges=edges,triangles=triangles,squares=squares)
    X_test = net.act(TS_test,transient_p,activation=activation,leaky=leaky,
                     edges=edges,triangles=triangles,squares=squares)
    row_means = np.linalg.norm(X_train, axis=0, keepdims=True)
    X_train = X_train / row_means
    X_test = X_test / row_means
    X_train = X_train[:-steps_ahead,:]; X_test = X_test[:-steps_ahead,:]
    Y_train = TS_train[steps_ahead+transient_p:]
    
    #coefs = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X_train),X_train)+
    #                                          np.diag(np.ones(X_train.shape[1]))*10**(-6)),
    #                                 np.transpose(X_train)),Y_train)
    
    #ers = Y_train - np.matmul(X_train,coefs)
    #length = DL.Length(ers, n, X_train)
    
    if obs_noise != 'na':
        Y_train += np.random.normal(loc=0.0,scale=10**obs_noise,size=len(Y_train))
    details = {'net_size':n,'activation':activation,'train_size':str(TS_length),'steps_ahead':steps_ahead,
               'noise':obs_noise,'leakage':leaky,'edges':edges,'triangles':triangles,'squares':squares} 
        
    S,Bs,Var,MDL_coefs = breath.breath(X_train, Y_train, early_stop=True)
    best_k = min(S, key = S.get)
    
    #plt.plot(list(S.keys()),[S[key] for key in S.keys()])
    #smooth90 = smoother(list(S.values()),0.9); smooth95 = smoother(list(S.values()),0.95)
    #dsmooth90 = double_sided_smoother(list(S.values()),0.9)
    #dsmooth95 = double_sided_smoother(list(S.values()),0.5)

    ##plt.plot(list(S.keys()),smooth90,label='a = 0.9',color='C1',alpha=0.3)
    #plt.plot(list(S.keys()),smooth95,0.95,label='a = 0.95',color='C2',alpha=0.3)
    #plt.plot(list(S.keys()),dsmooth90,color='C1',linestyle='dashed',alpha=0.3)
    #plt.plot(list(S.keys()),dsmooth95,color='C2',linestyle='dashed',alpha=0.3)
    #plt.legend()
    #plt.vlines(best_k_finder2(S),ymin=min(list(S.values())),ymax=max(list(S.values())),color='C3')
    #plt.vlines(dsmooth90.index(min(dsmooth90)),ymin=min(list(S.values())),ymax=max(list(S.values())),color='C1')
    #plt.vlines(dsmooth95.index(min(dsmooth95)),ymin=min(list(S.values())),ymax=max(list(S.values())),color='C2')
    other_best_k = best_k_finder2(S)
    #plt.vlines(other_best_k,ymin=min(list(S.values())),ymax=max(list(S.values())),color='C4')
    #plt.show()
    
    
    details['k_opt'] = other_best_k
    MDL_choices = Bs[other_best_k]

    M_vec = np.array([MDL_coefs[best_k][MDL_choices.index(i)] if i in MDL_choices else 0 for i in range(X_train.shape[1])])
    
        # c) ridge regression with all nodes for different penalty values 
        
    vars_ridge, alphas, coefs = breath.ridge_it(X_train, Y_train, alphas=[10**i for i in [-6,-5,-4,-3,-2,-1,0]])
    measures = {key: measure(M_vec,np.abs(coefs[key])) for key in coefs.keys()}
    for key in coefs.keys():
        details['a'+str(key)] = measures[key]
    for key in coefs.keys():
        details['s_90a'+str(key)] = effective_sparsity(coefs[key],0.1)
        details['s_99a'+str(key)] = effective_sparsity(coefs[key],0.01)
    
    if activation == 'tanh':
        fname = 'log3'
    elif activation == 'linear':
        fname = 'log3lin'
    if edges:
        fname += '_EnT'
    log_data(details, fname+'.csv')
    
def effective_sparsity(coefs,cut):
    c = np.sort(np.abs(coefs)/np.linalg.norm(coefs)); loop = True; count = 1
    while loop:
        #print('vector is: '+str(c[:count]))
        #print('power is: '+str(np.linalg.norm(c[:count])))
        if np.linalg.norm(c[:count]) < cut:
            count += 1
        else:
            loop = False
    return len(c) - count
    

def plot(csv_file='logs/log3.csv',train_size=2000):
    
    df = pd.read_csv(csv_file)
    df = df[df['train_size'] == train_size]
    df = df[df['net_size'] < 1001]
    Ns = list(set(list(df['net_size'])))
    Ns.sort()
    alphas = [c for c in df.columns if c[0]=='a' and c[1] != 'c']
    readout_sizes = [np.mean(np.log10(df[df['net_size']==n]['k_opt'])) for n in Ns]
    s99s = {a:[np.mean(np.log10(df[df['net_size']==n]['s_99'+a])) for n in Ns] for a in alphas}
    s90s = {a:[np.mean(np.log10(df[df['net_size']==n]['s_90'+a])) for n in Ns] for a in alphas}
    fig,axs = plt.subplots(1,1,figsize=(10,7))
    axs.plot(np.log10(Ns),[r for r in readout_sizes],color='k',label='MDL')
    axs.plot(np.log10(Ns),[r for r in readout_sizes],'X',color='k')
    counter = 0
    for key in ['a-5.0','a-4.0','a-3.0','a-2.0']:
        axs.plot(np.log10(Ns),[i for i in s90s[key]],linestyle='dashed',label='$\log_{10}(\\alpha) = $'+key[1:],
                 color='C'+str(counter))
        axs.plot(np.log10(Ns),[i for i in s99s[key]],linestyle='dotted',color='C'+str(counter))
        counter += 1
    axs.legend(fontsize=18)
    axs.set_xlabel('$\log_{10}$ Network Size',fontsize=25)
    axs.set_ylabel('$\log_{10}$ Readout Size',fontsize=25)
    axs.tick_params(axis='both', labelsize=15)
    #axs.set_title(str(train_size)+' Training Points',fontsize=18)
    plt.tight_layout()
    plt.show()
    
    #similarities = {a:[np.mean(df[df['net_size']==n][a]) for n in Ns] for a in alphas}
    #fig,axs = plt.subplots(1,1,figsize=(10,5))
    #for a in ['a-6.0','a-5.0','a-4.0','a-3.0']:
    #    axs.plot(np.log10(Ns),similarities[a],label='$\log_{10}(\\alpha) = $'+a[1:])
    #axs.legend(fontsize=12)
    #axs.set_xlabel('$\log_{10}$ Network Size',fontsize=15)
    #axs.set_ylabel('$\hat{m}\\cdot\hat{r}^{\\alpha}$',fontsize=15)
    #plt.tight_layout()
    #plt.show()
    


def main(TS_length,activation='tanh',edges=False,triangles=False):
    Ns = [int(10**i) for i in np.linspace(0,3,31)][6:]
    if edges:
        Ns = Ns[4:] 
    print(Ns)
    for n in Ns:
        #try:
        print('n = '+str(n))
        get_data(n=n,TS_length=TS_length,activation=activation,edges=edges,triangles=triangles)
        #except:
        #    print('Failed')
        #if n > 50:
        #    time.sleep(10)


plot(csv_file = 'logs/log3.csv',train_size=10000)
print(1/0)
for go in range(10):
    print('------------------------------------- go number '+str(go))
    #main(TS_length=500,activation='tanh')#,edges=True,triangles=True)
    #plot(csv_file = 'logs/log3_v2.csv',train_size=500)
    #time.sleep(30)
    #main(TS_length=2000,activation='tanh')#,edges=True,triangles=True)
    #plot(csv_file = 'logs/log3_v2.csv',train_size=2000)
    #time.sleep(30)
    main(TS_length=10000,activation='tanh',edges=False,triangles=False)
    plot(csv_file = 'logs/log3.csv',train_size=10000)
    time.sleep(60)

    
    
    









