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

def consistency_cap(subset,size='na'):
    df = pd.read_csv('logs/consistency/bigX.csv')
    leng = len(df['X_0'].values)
    if subset == 'random':
        ncomps = int(len(list(df.columns))/2)
        #print('ncomps = '+str(ncomps))
        subset = rand.sample(list(range(ncomps)),k=size)
    #else:
        #print('MDL subset is: ')
        #print(sorted(subset))
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
    return sum([G.real for G in gammas])

    
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
    
    S,Bs,Var,coefs = breath.breath(X_train1, Y_train, early_stop=False,verbose=True,max_iterations = 100, cap=terminate)
    best_k = min(S, key = S.get)
    
    Bs[max(Bs.keys())+1] = list(range(max(Bs.keys())+1))

    Bmax = max(list(Bs.keys()))
    extra_Bs = {}; last_selection = Bs[Bmax]
    not_selected = [i for i in range(all_comps) if i not in last_selection]
    for i in range(Bmax,all_comps):
        if i % 50 == 0 and i != all_comps:
            print('Getting extra B of size '+str(i))
            extra_Bs[i] = last_selection + rand.sample(not_selected,i-Bmax)
    extra_Bs[all_comps] = list(range(all_comps))

    data = [[best_k]]
    for k in Bs.keys():
        if k >= 1:
            data.append(Bs[k])
    for k in extra_Bs.keys():
        data.append(extra_Bs[k])

    with open('logs/consistency/subsets.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
        

def plot_from_csv(csv_file,net_type='embedding'):
    df = pd.read_csv(csv_file)
    #df = df.dropna(axis=1)
    dfaug = df[df['squares']==True]; dfjn = df[df['squares']==False]
    Ks = sorted(list(set([int(c[4:]) for c in dfjn.columns if c[:3] == 'MDL'])))
    Kopts = {'jn':dfjn['k_opt'].mean(),'aug':dfaug['k_opt'].mean()}
    Ks = [k for k in Ks if k%10==0]
    
    mdl = {'jn':[np.mean(dfjn['MDL_'+str(k)])/k for k in Ks],
           'aug':[np.mean(dfaug['MDL_'+str(k)])/k for k in Ks]}
    rands = {'jn':[np.mean(dfjn['rand_'+str(k)])/k for k in Ks],
             'aug':[np.mean(dfaug['rand_'+str(k)])/k for k in Ks]}
    cutpoint = len([i for i in mdl['jn'] if not math.isnan(i)]) -1
    mdl['jn'][cutpoint] = rands['jn'][cutpoint]
    mdl['aug'][-1] = rands['aug'][-1]
    
    #print(Kopts['jn'])
    #print(Kopts['aug'])
    #print(mdl['jn'][int(Kopts['jn'])])
    #print(mdl['jn'][cutpoint])
    #print(mdl['aug'][int(Kopts['aug'])])
    #print(mdl['aug'][-1])

    fig,axs = plt.subplots(1,1,figsize=(10,7))
    Ks = [math.log10(k) for k in Ks]
    axs.plot(Ks,rands['jn'],color='k',label='Just Node States')#'Random subsets')
    axs.plot(Ks,rands['aug'],color='k',linestyle='dashed',label='Augmented')
    axs.plot([Ks[cutpoint]],[rands['jn'][cutpoint]],'o',color='k',linestyle='dashed')
    axs.plot([Ks[-1]],[rands['aug'][-1]],'o',color='k',linestyle='dashed')
    axs.set_xlabel('$\log_{10}$ Readout Size',fontsize=25)
    axs.set_ylabel('Consistency',fontsize=25)
    axs.legend(fontsize=18)
    axs.tick_params(axis='both', labelsize=15)
    plt.tight_layout()
    plt.show()    
    
    fig,axs = plt.subplots(1,1,figsize=(10,7))
    maxi = max(mdl['jn']+mdl['aug']+rands['jn']+rands['aug'])
    mini = min(mdl['jn']+mdl['aug']+rands['jn']+rands['aug'])
    axs.plot(Ks,mdl['jn'],color='r',label='MDL Subsets')
    axs.plot(Ks,rands['jn'],color='k',label='Just Node States')#'Random subsets')
    axs.plot(Ks,rands['aug'],color='k',linestyle='dashed',label='Augmented')
    axs.plot(Ks,mdl['aug'],color='r',linestyle='dashed')
    axs.plot([Ks[-1]],[mdl['aug'][-1]],'o',color='r',linestyle='dashed')
    axs.plot([Ks[-1]],[rands['aug'][-1]],'o',color='k',linestyle='dashed')
    axs.plot([Ks[cutpoint]],[mdl['jn'][cutpoint]],'o',color='r',linestyle='dashed')
    axs.plot([Ks[cutpoint]],[rands['jn'][cutpoint]],'o',color='k',linestyle='dashed')


    axs.vlines(math.log10(Kopts['jn']),ymin=mini,ymax=maxi,color='r')
    axs.vlines(math.log10(Kopts['aug']),ymin=mini,ymax=maxi,color='r',linestyle='dashed')
    #axs.vlines(Kopts['jn'],ymin=mini,ymax=maxi,color='r')
    #axs.vlines(Kopts['aug'],ymin=mini,ymax=maxi,color='r',linestyle='dashed')
    axs.set_xlabel('$\log_{10}$ Readout Size',fontsize=25)
    axs.set_ylabel('Consistency',fontsize=25)
    axs.legend(fontsize=18)
    axs.tick_params(axis='both', labelsize=15)
    plt.tight_layout()
    plt.show()    


        
def gather_data(net,activation='tanh',net_type='embedding',TS_length=5000,
            edges=False,triangles=False,squares=False,terminate='na'):
    TS_train = mapnsys.Rossler(TS_length+500, 0.02)
    #TS_test = mapnsys.Lorenz(50000+500, 0.02) # this is actually redundant

    if edges or triangles or squares:
        net.get_subgraphs(net_size,edges=edges,triangles=triangles,squares=squares)
    make_bigXs(net, TS_train, edges=edges, triangles=triangles, squares=squares, terminate = terminate)
    
   
def conc_scatter(activation='tanh',steps_ahead=10,obs_noise='na',TS_length=3000, net_size = 400,specrad=1,
                    connectivity=6,leaky=1,edges=True,triangles=True,squares=True,terminate='na'):  
    transient_p = 500
    TS_train = mapnsys.Rossler(TS_length+500, 0.02)
    TS_test = mapnsys.Rossler(30000, 0.02)
    net = res.ESN(net_size, spec_rad=specrad, connectivity = connectivity, directed=False, self_loops=False)
    if edges or triangles or squares:
        net.get_subgraphs(net_size,edges=edges,triangles=triangles,squares=squares)
    X_train1 = net.act(TS_train,transient_p,activation=activation,leaky=leaky,
                      edges=edges,triangles=triangles,squares=squares,
                      state='random')
    X_train2 = net.act(TS_train,transient_p,activation=activation,leaky=leaky,
                      edges=edges,triangles=triangles,squares=squares, 
                      state='random')
    X_test = net.act(TS_test,transient_p,activation=activation,leaky=leaky,
                      edges=edges,triangles=triangles,squares=squares,
                      state='random')
    row_means = np.linalg.norm(X_train1, axis=0, keepdims=True)
    X_train1 = X_train1 / row_means
    X_train2 = X_train2 / row_means
    X_test = X_test / row_means
    #X_test = X_test / row_means
    X_train1 = X_train1[1000:-steps_ahead,:]; X_train2 = X_train2[1000:-steps_ahead,:]
    X_test = X_test[1000:-steps_ahead,:]
    Y_train = TS_train[1000+steps_ahead+transient_p:]
    if obs_noise != 'na':
        Y_train += np.random.normal(loc=0.0,scale=10**obs_noise,size=len(Y_train))
    Y_test = TS_test[1000+steps_ahead+transient_p:]
    
    bigX = np.concatenate((X_train1[1000:],X_train2[1000:]),axis=1)
    print(bigX.shape)
    col_names = ['X_'+str(i) for i in range(X_train1.shape[1])]+['X2_'+str(i) for i in range(X_train2.shape[1])]
    bigframe = pd.DataFrame(bigX,columns=col_names)
    bigframe.to_csv('logs/consistency/bigX.csv', index=False)
    
    MSEs = []; Concs = []; sizes = []; MSE_trains = []; options = list(range(X_train1.shape[1]))
    for point in range(50):
        print('Gathering point '+str(point+1))
        subset = rand.sample(options,k=100)#k=rand.randint(1,len(options)-1))
        V = np.transpose(np.array([X_train1[:,b] for b in subset]))
        Vtest = np.transpose(np.array([X_test[:,b] for b in subset]))
        coefs = fit_coefs(V, Y_train)
        e_b = Y_test - np.matmul(Vtest,coefs)
        MSEs.append(np.mean(np.square(e_b)))
        e_t = Y_train - np.matmul(V,coefs)
        MSE_trains.append(np.mean(np.square(e_t)))
        Concs.append(consistency_cap(subset))
        sizes.append(len(subset))
    MSEs = np.array(MSEs); Concs = np.array(Concs); sizes = np.array(sizes)
    MSE_trains = np.array(MSE_trains)
    quick_plot(Concs/sizes, np.log10(MSE_trains/MSEs), Xlab='Consistency', Ylab='Performance Loss')
    # negative Performance loss is bad
    MSEs = np.log10(MSEs)
    quick_plot(Concs, MSEs, Xlab='Consistent Components', Ylab='$\log_{10}$ MSE')
    quick_plot(Concs/sizes, MSEs, Xlab='Consistency', Ylab='$\log_{10}$ MSE')
    
    
    

def quick_plot(X,Y,title='',Xlab='',Ylab='',msize=3):
    fig,axs = plt.subplots(1,1,figsize=(10,7))
    axs.plot(X,Y,'o',color='k',markersize=msize)
    axs.set_xlabel(Xlab,fontsize=25)
    axs.set_ylabel(Ylab,fontsize=25)
    axs.set_title(title,fontsize=25)
    plt.tight_layout()
    plt.show()
    
   
    
def process_run(net_type = 'lorenz', system = 'rossler', activation = 'tanh', edges = False, triangles = False, 
                squares = False,fname='concomps.csv',terminate='na'):
    subsets = {}; bestk = 0
    with open('logs/consistency/subsets.csv','r') as reader:
        print('READING SUBSETS.CSV')
        for line in reader.readlines():
            if bestk == 0:
                bestk = int(line)
            lst = [int(i) for i in line.split(',')]
            subsets[len(lst)] = [int(i) for i in line.split(',')]

    #subsets[len(subsets)] = list(range(len(subsets)))
    #print('appended this to subsets: ')
    #print(subsets[-1])
    print('len(subsets) = '+str(len(subsets)))
    if terminate == 'na':
        terminate = len(subsets)
    for K in subsets.keys():#,10):
        print('--------------------- PROCESSING K = '+str(K))
        print('MDL')
        #print('(subset is '+str(subsets[K])+')')
        concap = consistency_cap(subsets[K])
        details = {'net_type':[net_type],'system':[system],'activation':[activation],'edges':[edges],
                   'triangles':[triangles],'squares':[squares],'subset':['MDL'],'k_opt':[bestk],
                   'size':[K],'comps':[concap]}
        log_data(details, 'consistency/temp.csv')
        for go in range(1):
            print('random # '+str(go))
            #random_set = rand.sample(list(range(len(subsets)-1)),k=K)
            concap = consistency_cap('random',size=K)
            details = {'net_type':[net_type],'system':[system],'activation':[activation],'edges':[edges],
                       'triangles':[triangles],'squares':[squares],'subset':['random'],'k_opt':[bestk],
                       'size':[K],'comps':[concap]}
            log_data(details, 'consistency/temp.csv')
        if K >= 100 and K % 2 == 0:
            time.sleep(10)
    details = {'net_type':[net_type],'system':[system],'activation':[activation],'edges':[edges],
               'triangles':[triangles],'squares':[squares],'subset':['random'],'k_opt':[bestk],
               'size':[K],'comps':[concap]}
    df = pd.read_csv('logs/consistency/temp.csv')
    dfM = df[df['subset']=='MDL']
    dfR = df[df['subset']=='random']
    sizes = list(set(list(df['size'])))
    for s in sizes:
        details['MDL_'+str(s)] = np.mean(dfM[dfM['size']==s]['comps'])
        details['rand_'+str(s)] = np.mean(dfR[dfR['size']==s]['comps'])
    log_data(details, fname)

    os.remove('logs/consistency/subsets.csv') 
    os.remove('logs/consistency/temp.csv') 
    os.remove('logs/consistency/bigX.csv') 



def main(net,fname,activation='tanh',edges=False,triangles=False,squares=False,TS_length=5000,terminate='na'):
    gather_data(net,activation=activation,edges=edges,triangles=triangles,
                squares = squares,TS_length=TS_length,terminate=terminate)
    print('Gathering Complete')
    #time.sleep(10)
    process_run(activation=activation,net_type='WS',edges=edges,triangles=triangles,
                squares = squares,system='Lorenz',fname=fname,terminate=terminate)



#conc_scatter(edges=True,triangles=True,squares=True)
#print(1/0)
plot_from_csv('logs/concomps3.csv')
print(1/0)


net_size=1000; connectivity=6; specrad = 1
net = res.ESN(net_size, spec_rad=specrad, connectivity = connectivity, directed=False, self_loops=False)
main(net,'concomps3.csv')
time.sleep(180)
main(net,'concomps3.csv',edges=True,triangles=True,squares=True,terminate=net_size+1)


plot_from_csv('logs/concomps3.csv')
print(1/0)

plot_from_csv('logs/consistency.csv')
print(1/0)
net_size = 1000
TS_length = 5000; system='lorenz'
activation='tanh';net_type='lorenz';s=1;edges=False;triangles=False;squares=False

gather_data(net_size=net_size,activation=activation,net_type=net_type,edges=edges,triangles=triangles,
            squares = squares,TS_length=TS_length,s=s)#,triangles=True)
print('Gathering Complete')
time.sleep(60)
process_run(activation=activation,net_type=net_type,edges=edges,triangles=triangles,
            squares = squares,system=system)
plot_from_csv('logs/consistency.csv')