# --------------------------------- MSE vs forecasting horizon -----------------------------------------------
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
#import phase_space_network_builder as PSNB

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
    return {'N':N/tot,'E':E/tot,'T':T/tot,'S':S/tot}

        
    
def best_k_finder(dic):
    N = len(list(dic.keys())); vals = list(dic.values())
    if  N < 50:
        search_dic = {i:dic[i] for i in range(1,N)}
    else:
        cut = 20; counter = 20; keep_going = True
        while counter < N-20 and keep_going:
            if np.mean(vals[counter-cut:counter]) < np.mean(vals[counter:counter+cut]):
                counter += 1
            else:
                keep_going = False
        search_dic = {counter-cut+i:dic[counter-cut+i] for i in range(1,2*cut)}
    k_opt = max([1,min(search_dic, key = search_dic.get)])
    return k_opt
    
def best_k_finder2(S):
    smooth90 = smoother(list(S.values()),0.9); smooth95 = smoother(list(S.values()),0.98)
    N = len(smooth90)
    caps = [i for i in range(N) if smooth95[i] <= smooth90[i] and i >= 10]
    if len(caps) > 1:
        cap = min(caps)
    else:
        cap = len(smooth90)
    search_dic = {i:S[i] for i in range(cap)}
    k_opt = max([1,min(search_dic, key = search_dic.get)])
    return k_opt

def smoother(lst,alpha):
    X = [lst[0]]
    for i in range(1,len(lst)):
        X.append(sum([lst[i-j]*alpha**j for j in range(i)])/sum([alpha**j for j in range(i)]))
    return X
    

def get_data(net,activation='linear',TS_length=5000,steps_ahead=1,obs_noise='na',leaky=1,
                  edges=False,triangles=False,squares=False,n=1000,system='Lorenz'):  
    
    transient_p = 500
    #obs_noise = -3 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change this later
    if system == 'Lorenz':
        TS_train = mapnsys.Lorenz(TS_length+transient_p)#mapnsys.Lorenz(TS_length+transient_p, 0.02)
        TS_test = mapnsys.Lorenz(25000)#mapnsys.Lorenz(50000+transient_p, 0.02)
    elif system == 'Rossler':
        TS_train = mapnsys.Rossler(TS_length+transient_p)#mapnsys.Lorenz(TS_length+transient_p, 0.02)
        TS_test = mapnsys.Rossler(25000)
    elif system == 'Thomas':
        TS_train = mapnsys.Thomas(TS_length+transient_p)#mapnsys.Lorenz(TS_length+transient_p, 0.02)
        TS_test = mapnsys.Thomas(25000)
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
    Y_train = TS_train[steps_ahead+transient_p:]; Y_test = TS_test[steps_ahead+transient_p:]
    if obs_noise != 'na':
        Y_train += np.random.normal(loc=0.0,scale=10**obs_noise,size=len(Y_train))
    
    details = {'net_size':n,'activation':activation,'train_size':str(TS_length),'steps_ahead':steps_ahead,
               'noise':obs_noise,'leakage':leaky,'edges':edges,'triangles':triangles,'squares':squares} 
      
    S,Bs,Var,MDL_coefs = breath.breath(X_train, Y_train, early_stop=True)
    if TS_length <= 2001:
        best_k = best_k_finder(S)
    else:
        best_k = best_k_finder2(S)
    V = np.transpose(np.array([X_test[:,b] for b in Bs[best_k]]))
    errs = Y_test - np.matmul(V,MDL_coefs[best_k])
    details['MDL'] = np.mean(np.square(errs))
    details['k_opt'] = best_k
    subgs = NET_counter(MDL_coefs[best_k], n, edges=edges, triangles=triangles, squares=squares)
    for key in subgs.keys():
        details[key+'_MDL'] = subgs[key]


    #plt.plot(list(S.keys()),[S[key] for key in S.keys()])
    #smooth90 = smoother(list(S.values()),0.9); smooth95 = smoother(list(S.values()),0.95)
   
    #plt.plot(list(S.keys()),smooth90,label='a = 0.9',color='C1',alpha=0.3)
    #plt.plot(list(S.keys()),smooth95,0.95,label='a = 0.95',color='C2',alpha=0.3)
    #plt.legend()
    #plt.vlines(best_k,ymin=min(list(S.values())),ymax=max(list(S.values())),color='C3')
    #plt.show()


    vars_ridge, alphas, coefs = breath.ridge_it(X_train, Y_train, 
                                                alphas=[10**i for i in 
                                                        [-9,-8,-7,-6,-5,-4,-3,-2,-1,0]])
    for key in coefs.keys():
        errs = Y_test - np.matmul(X_test,coefs[key])
        details['a'+str(key)] = np.mean(np.square(errs))
    for key in coefs.keys():
        subgs = NET_counter(coefs[key], n, is_MDL=False, edges=edges, triangles=triangles, squares = squares)
        for g in subgs.keys():
            details[g+'_a'+str(key)] = subgs[g]
    
    coefs = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X_train),X_train)),
                                     np.transpose(X_train)),Y_train)
    
    errs = Y_test - np.matmul(X_test,coefs)
    details['a0'] = np.mean(np.square(errs))
    subgs = NET_counter(coefs, n, is_MDL=False, edges=edges, triangles=triangles, squares = squares)
    for g in subgs.keys():
        details[g+'_a0'] = subgs[g]
    filename = 'log4_'+system+'.csv'
    log_data(details, filename)
    
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
    
def cont_plot_data_collector(csv_file='logs/log4.csv',activation='tanh',draw_squares=True,noise='-2'):
    df = pd.read_csv(csv_file)
    df = df[df['noise'] == noise]
    df = df[df['activation'] == activation]
    dfN = df[df['edges'] == False]
    dfS = df[df['squares'] == True]
    Hs = list(set(list(dfN['steps_ahead'])))
    Hs.sort()
    alphas = [c for c in df.columns if c[0]=='a' and c[1] != 'c']

    MDL = {'N':[np.mean(np.log10(dfN[dfN['steps_ahead']==h]['MDL'])) for h in Hs],
           'S':[np.mean(np.log10(dfS[dfS['steps_ahead']==h]['MDL'])) for h in Hs]}
    RR = {}
    
    for alpha in alphas:
        RR[alpha] = {'N':[np.mean(np.log10(dfN[dfN['steps_ahead']==h][alpha])) for h in Hs],
                     'S':[np.mean(np.log10(dfS[dfS['steps_ahead']==h][alpha])) for h in Hs]}
    return MDL, RR, Hs


def contrast_plot(csv_file='logs/log4.csv',activation='linear',draw_squares=True,noise='-2'):
    linestyles = {'N':'solid','S':'dashed'}
    labels = {'N':'Nodes Only','S':'Augmentented'}
    for log in [False,True]:
        fig,axs=plt.subplots(1,2,figsize=(10,4))  
        for i in range(2):
            if i == 0:
                MDL,RR,Hs = cont_plot_data_collector(csv_file=csv_file,activation=activation,noise='na')
                if log == False:
                    MDL['N'] = [10**k for k in MDL['N']]; MDL['S'] = [10**k for k in MDL['S']]
                    for alpha in RR.keys():
                        RR[alpha]['N'] = [10**k for k in RR[alpha]['N']] 
                        RR[alpha]['S'] = [10**k for k in RR[alpha]['S']]
            elif i == 1:
                MDL,RR,Hs = cont_plot_data_collector(csv_file=csv_file,activation=activation,noise=noise)
                if log == False:
                    MDL['N'] = [10**k for k in MDL['N']]; MDL['S'] = [10**k for k in MDL['S']]
                    for alpha in RR.keys():
                        RR[alpha]['N'] = [10**k for k in RR[alpha]['N']] 
                        RR[alpha]['S'] = [10**k for k in RR[alpha]['S']]
            for subg in ['N','S']:
                axs[i].plot(Hs,MDL[subg],color='k',linestyle=linestyles[subg],label=labels[subg])
                c = 0
                for alpha in ['a-9.0','a-5.0','a-2.0']:#['a-9.0','a-6.0','a-3.0']:
                    if subg == 'N':
                        axs[i].plot(Hs, RR[alpha][subg],color='C'+str(c),linestyle=linestyles[subg],
                                    label='$log_{10}(\\alpha) = $'+alpha[1:])
                    else:
                        axs[i].plot(Hs,RR[alpha][subg],color='C'+str(c),linestyle=linestyles[subg])
                    c += 1
        
                
        axs[0].legend(fontsize=12)
        #axs[1].legend(fontsize=12)
        axs[0].set_xlabel('Forecasting Horizon (times $\delta=0.02$)',fontsize=12)
        axs[1].set_xlabel('Forecasting Horizon (times $\delta=0.02$)',fontsize=12)
        axs[1].sharey(axs[0])
        if log == False:
            axs[0].set_ylabel('MSE',fontsize=12)
        else:
            axs[0].set_ylabel('$log_{10}$(MSE)',fontsize=12)
        axs[0].set_title('No added noise',fontsize=12)
        axs[1].set_title('Noise with $\sigma = 10^{'+noise+'}$',fontsize=12)
        plt.tight_layout()
        plt.show()

def plot_all(aug='N'):
    Lyap_scales = {'Lorenz':0.02*0.9056,'Rossler':0.2*0.0674,'Thomas':0.2*0.0406}
    counter = 0;  alpha_list = ['a-9.0','a-8.0','a-7.0','a-6.0','a-5.0','a-3.0']
    fig,axs = plt.subplots(1,3,figsize=(18,6))
    for system in ['Lorenz','Rossler','Thomas']:
        csv_file='logs/log4_'+system+'.csv'
        df = pd.read_csv(csv_file)
        if aug == 'N':
            df = df[df['edges'] == False]
        else:
            df = df[df['squares'] == True]
        Hs = list(set(list(df['steps_ahead'])))
        Hs.sort()
        alphas = [c for c in df.columns if c[0]=='a' and c[1] != 'c']
        MDL = [np.mean(np.log10(df[df['steps_ahead']==h]['MDL'])) for h in Hs]
        RR = {a:[np.mean(np.log10(df[df['steps_ahead']==h][a])) for h in Hs] for a in alphas}
        Lys = [H*Lyap_scales[system] for H in Hs]
        axs[counter].plot(Lys,MDL,color='k',label='MDL', linewidth=2.5)
        for alpha in alpha_list:
            axs[counter].plot(Lys,RR[alpha],label='$log_{10}(\\alpha) = $'+alpha[1:], linewidth=2.5)
        axs[counter].set_xlabel('Forecasting Horizon',fontsize=25)
        axs[counter].set_title(system,fontsize=25)
        axs[counter].tick_params(axis='both', labelsize=15)
        counter += 1
    axs[0].set_ylabel('$log_{10}$ MSE',fontsize=25)
    axs[0].legend(fontsize=18)
    plt.tight_layout()
    plt.show()
    
def plot_all_comparisson():
    Lyap_scales = {'Lorenz':0.02*0.9056,'Rossler':0.2*0.0674,'Thomas':0.2*0.0406}
    counter = 0;  alpha_list = ['a-9.0','a-8.0','a-7.0','a-6.0','a-5.0','a-3.0']
    fig,axs = plt.subplots(3,2,figsize=(12,18))
    for system in ['Lorenz','Rossler','Thomas']:
        csv_file='logs/log4_'+system+'.csv'
        df = pd.read_csv(csv_file)
        dfaug = df[df['edges'] == True]
        dfjn = df[df['squares'] == False]
        Hs = list(set(list(dfjn['steps_ahead'])))
        Hs.sort()
        alphas = [c for c in dfjn.columns if c[0]=='a' and c[1] != 'c']
        print(alphas)
        MDLjn = [np.mean(np.log10(dfjn[dfjn['steps_ahead']==h]['MDL'])) for h in Hs]
        RRjn = {a:[np.mean(np.log10(dfjn[dfjn['steps_ahead']==h][a])) for h in Hs] for a in alphas}
        MDLaug = [np.mean(np.log10(dfaug[dfaug['steps_ahead']==h]['MDL'])) for h in Hs]
        RRaug = {a:[np.mean(np.log10(dfaug[dfaug['steps_ahead']==h][a])) for h in Hs] for a in alphas}
        mini = min([min(MDLaug),min(MDLjn)]+[min(RRaug[a]) for a in RRaug.keys()]+[min(RRjn[a]) for a in RRjn.keys()])
        maxi = max(RRaug['a-9.0'])
        mini = mini - 0.05*(maxi-mini); maxi = maxi + 0.05*(maxi-mini)
        Lys = [H*Lyap_scales[system] for H in Hs]
        axs[counter,0].plot(Lys,MDLjn,color='k',label='MDL', linewidth=2.5)
        axs[counter,1].plot(Lys,MDLaug,color='k',label='MDL', linewidth=2.5)
        for alpha in alpha_list:
            axs[counter,0].plot(Lys,RRjn[alpha],label='$log_{10}(\\alpha) = $'+alpha[1:], linewidth=2.5)
            axs[counter,1].plot(Lys,RRaug[alpha], linewidth=2.5)
        axs[counter,0].set_ylim(mini,maxi)
        axs[counter,1].set_ylim(mini,maxi)
        #axs[counter].set_title(system,fontsize=25)
        axs[counter,0].tick_params(axis='both', labelsize=15)
        axs[counter,1].tick_params(axis='both', labelsize=15)
        counter += 1
    axs[0,0].set_title('Without Augmentation',fontsize=30)
    axs[0,1].set_title('With Augmentation',fontsize=30)
    axs[0,0].set_ylabel('$log_{10}$ MSE (Lorenz)',fontsize=30)
    axs[1,0].set_ylabel('$log_{10}$ MSE (Rossler)',fontsize=30)
    axs[2,0].set_ylabel('$log_{10}$ MSE (Thomas)',fontsize=30)
    axs[2,0].set_xlabel('Forecasting Horizon',fontsize=30)
    axs[2,1].set_xlabel('Forecasting Horizon',fontsize=30)
    axs[0,0].legend(fontsize=18)
    plt.tight_layout()
    plt.show()
        
def plot(system='Lorenz',activation='tanh',draw_squares=True,noise='na',TS_length=1000):
    csv_file='logs/log4_'+system+'.csv'
    Lyap_scales = {'Lorenz':0.02*0.9056,'Rossler':0.2*0.0674,'Thomas':0.2*0.0406}
    df = pd.read_csv(csv_file)
    df = df[df['noise'] == noise]
    df = df[df['activation'] == activation]
    #df = df[df['train_size'] == TS_length]
    dfN = df[df['edges'] == False]
    dfE = df[(df['edges'] == True) & (df['triangles'] == False)]
    dfT = df[(df['triangles'] == True) & (df['squares'] == False)]
    dfS = df[df['squares'] == True]
    if draw_squares:
        df = dfS
    else:
        df = dfT

    Hs = list(set(list(dfN['steps_ahead'])))

    Hs.sort()
    alphas = [c for c in df.columns if c[0]=='a' and c[1] != 'c']

    MDL = {'N':[np.mean(np.log10(dfN[dfN['steps_ahead']==h]['MDL'])) for h in Hs],
           'E':[np.mean(np.log10(dfE[dfE['steps_ahead']==h]['MDL'])) for h in Hs],
           'T':[np.mean(np.log10(dfT[dfT['steps_ahead']==h]['MDL'])) for h in Hs],
           'nodes':[np.mean(df[df['steps_ahead']==h]['N_MDL']) for h in Hs],
           'edges':[np.mean(df[df['steps_ahead']==h]['E_MDL']) for h in Hs],
           'triangles':[np.mean(df[df['steps_ahead']==h]['T_MDL']) for h in Hs]}

    RR = {}
    for alpha in alphas:
        RR[alpha] = {'N':[np.mean(np.log10(dfN[dfN['steps_ahead']==h][alpha])) for h in Hs],
               'E':[np.mean(np.log10(dfE[dfE['steps_ahead']==h][alpha])) for h in Hs],
               'T':[np.mean(np.log10(dfT[dfT['steps_ahead']==h][alpha])) for h in Hs],
               'nodes':[np.mean(df[df['steps_ahead']==h]['N_'+alpha]) for h in Hs],
               'edges':[np.mean(df[df['steps_ahead']==h]['E_'+alpha]) for h in Hs],
               'triangles':[np.mean(df[df['steps_ahead']==h]['T_'+alpha]) for h in Hs]}
        if draw_squares:
            RR[alpha]['S'] = [np.mean(np.log10(dfS[dfS['steps_ahead']==h][alpha])) for h in Hs]
            RR[alpha]['squares'] = [np.mean(dfS[dfS['steps_ahead']==h]['S_'+alpha]) for h in Hs]
    if draw_squares:
        MDL['S'] = [np.mean(np.log10(dfS[dfS['steps_ahead']==h]['MDL'])) for h in Hs]
        MDL['squares'] =[np.mean(df[df['steps_ahead']==h]['S_MDL']) for h in Hs]
        
        
    linestyles = {'N':'solid','E':'dashdot','T':'dotted','S':'dashed'}
    #linestyles = {'N':'solid','E':'dashdot','T':'dotted','S':'solid'}
    labels = {'N':'$N$','E':'$N\cup E$', 'T':'$N\cup E \cup T$','S':'$N\cup E \cup T \cup S$'}
    labels = {'N':'MDL','E':'$N\cup E$', 'T':'$N\cup E \cup T$','S':'Augmented'}
    #labels = {'N':'MDL','E':'$N\cup E$', 'T':'$N\cup E \cup T$','S':'MDL'}
    # ------------------------------------------------------- plot 1
    fig,axs=plt.subplots(1,1,figsize=(10,7))
    alpha_list = ['a-9.0','a-8.0','a-7.0','a-6.0','a-5.0','a-3.0']
    #alpha_list = ['a-6.0','a-5.0','a-4.0','a-3.0','a-2.0']
    Lys = [H*Lyap_scales[system] for H in Hs]
    for subg in ['N','S']:#,'E','T']:
        axs.plot(Lys,[10 ** i for i in MDL[subg]],color='k',linestyle=linestyles[subg],label=labels[subg], 
                 linewidth=2.5)
    for subg in ['N','S']:    
        c = 0
        for alpha in alpha_list:#['a-9.0','a-6.0','a-3.0']:
            if subg == 'N':
                axs.plot(Lys,[10 ** i for i in RR[alpha][subg]],color='C'+str(c),linestyle=linestyles[subg],
                            label='$log_{10}(\\alpha) = $'+alpha[1:], linewidth=2.5)
            else:
                axs.plot(Lys,[10 ** i for i in RR[alpha][subg]],color='C'+str(c),linestyle=linestyles[subg],
                         linewidth=2.5)
            c += 1
    if False:#draw_squares:
        c = 0
        axs.plot(Hs,[10**i for i in MDL['S']],color='k',linestyle=linestyles['S'],label=labels['S'])
        for alpha in ['a-8.0','a-5.0','a-2.0']:
            axs.plot(Hs,[10 ** i for i in RR[alpha]['S']],color='C'+str(c),linestyle=linestyles['S'])
            c += 1
            
    axs.legend(fontsize=17)
    #axs[1].legend(fontsize=12)
    axs.set_xlabel('Forecasting Horizon (Lyapunov time)',fontsize=25)
    #axs[1].set_xlabel('Forecasting Horizon (times $\delta=0.02$)',fontsize=12)
    axs.set_ylabel('MSE',fontsize=25)
    axs.tick_params(axis='both', labelsize=15)
    #axs.set_title('Noise with $\sigma = 10^{'+noise+'}$')
    plt.tight_layout()
    plt.show()
    
    # ----------------------------------------------------- plot 1 logged
    fig,axs=plt.subplots(1,1,figsize=(10,7))
    alpha_list = ['a-9.0','a-8.0','a-7.0','a-6.0','a-5.0','a-3.0']
    #alpha_list = ['a-6.0','a-5.0','a-4.0','a-3.0','a-2.0']
    #labels={'N':'MDL','S':'MDL'}
    #linestyles = {'N':'solid','S':'solid'}
    Lys = [H*Lyap_scales[system] for H in Hs]
    for subg in ['N','S']:#,'E','T']:
        axs.plot(Lys,MDL[subg],color='k',linestyle=linestyles[subg],label=labels[subg], linewidth=2.5)
    for subg in ['N','S']:    
        c = 0
        for alpha in alpha_list:#['a-9.0','a-6.0','a-3.0']:
            if subg == 'N':
                axs.plot(Lys,RR[alpha][subg],color='C'+str(c),linestyle=linestyles[subg],
                            label='$log_{10}(\\alpha) = $'+alpha[1:], linewidth=2.5)
            else:
                axs.plot(Lys,RR[alpha][subg],color='C'+str(c),linestyle=linestyles[subg], linewidth=2.5)
            c += 1
            
    axs.legend(fontsize=17)
    axs.set_xlabel('Forecasting Horizon (Lyapunov time)',fontsize=25)
    axs.set_ylabel('$log_{10}$'+' MSE',fontsize=25)
    axs.tick_params(axis='both', labelsize=15)
    plt.tight_layout()
    plt.show()
    
    # -------------------------------------------------------- plot 3
    if False:
        fig,axs = plt.subplots(1,2,figsize=(10,4),sharey=True)
        c = 0
        for subg in ['nodes','edges','triangles']:
            axs[0].plot(Hs,MDL[subg],color='C'+str(c),label=subg)
            axs[1].plot(Hs,RR['a-9.0'][subg],color='C'+str(c))
            c += 1
        if draw_squares:
            axs[0].plot(Hs,MDL['squares'],color='C'+str(c),label='squares')
            axs[1].plot(Hs,RR['a0']['squares'],color='C'+str(c))
        axs[0].legend()
        plt.tight_layout()
        plt.show()


def plot_heatmaps(csv_file='logs/log4.csv',alphas = [-9,-6,-3],augmented=False):   
    df = pd.read_csv(csv_file)
    if augmented:
        df = df[df['squares'] == True]
    else:
        df = df[df['edges'] == False]
    noise_vals = list(set(list(df['noise'])))
    noise_vals.remove('na')
    noise_vals = sorted(noise_vals)
    print(noise_vals)
    forecast_lengths = list(set(list(df['steps_ahead'])))
    forecast_lengths = sorted(forecast_lengths)
    keylist = ['MDL']+['a'+str(a)+'.0' for a in alphas]; data = {}
    for key in keylist:
        data[key] = np.array([[0.0 for i in range(len(forecast_lengths))] for j in range(len(noise_vals))])
        #data[key] = {f:{} for f in forecast_lengths}
    for key in keylist:
        for f in range(len(forecast_lengths)):
            for n in range(len(noise_vals)):
                #print(df[(df['steps_ahead'] == forecast_lengths[f]) & 
                #                           (df['noise'] == noise_vals[n])][key].values)
                #print(np.mean(df[(df['steps_ahead'] == forecast_lengths[f]) & 
                #                            (df['noise'] == noise_vals[n])][key].values))
                #time.sleep(2)
                data[key][n,f] += math.log10(np.mean(df[(df['steps_ahead'] == forecast_lengths[f]) & 
                                            (df['noise'] == noise_vals[n])][key].values))
    print(data['MDL'])
    mini = min([np.min(data[key]) for key in keylist])
    maxi = max([np.max(data[key]) for key in keylist])
    fig,axs = plt.subplots(1,4,figsize=(12,3))
    for i in range(4):
        #im = axs[i].imshow(np.flip(data[keylist[i]],axis=0))
        im = axs[i].imshow(data[keylist[i]],vmin=mini,vmax=maxi)
        axs[i].set_title(keylist[i])
        axs[i].set_yticks([0,3,6,9,12,15])
        axs[i].set_xticks([2,7,13])
        axs[i].set_yticklabels([-1,-2,-3,-4,-5,-6])
        axs[i].set_xticklabels([5,15,25])
    plt.tight_layout()
    plt.colorbar(im,ax=axs.ravel().tolist(),shrink=0.82)
    plt.show()


def plot_all_heatmaps(csv_file='logs/log4.csv',alphas = [-9,-6,-3],augmented=False):   
    fig,axs = plt.subplots(3,4,figsize=(12,9))
    ims = {}; csv_files = ['logs/log4_'+end for end in ['Lorenz.csv','Rossler.csv','Thomas.csv']]
    titles = {'MDL':'MDL'}; colour= 'Greys'#'gist_rainbow'
    Lyap_scales = {0:0.02*0.9056,1:0.2*0.0674,2:0.2*0.0406}
    for a in alphas:
        titles['a'+str(a)+'.0'] = '$log_{10}(\\alpha)=$'+str(a)
    for k in range(3):
        df = pd.read_csv(csv_files[k])
        lyap_scaler=Lyap_scales[k]
        if augmented:
            df = df[df['squares'] == True]
        else:
            df = df[df['edges'] == False]
        noise_vals = list(set(list(df['noise'])))
        noise_vals.remove('na')
        noise_vals = sorted(noise_vals)

        forecast_lengths = list(set(list(df['steps_ahead'])))
        forecast_lengths = sorted(forecast_lengths)
        keylist = ['MDL']+['a'+str(a)+'.0' for a in alphas]; data = {}
        for key in keylist:
            data[key] = np.array([[0.0 for i in range(len(forecast_lengths))] for j in range(len(noise_vals))])
            #data[key] = {f:{} for f in forecast_lengths}
        for key in keylist:
            for f in range(len(forecast_lengths)):
                for n in range(len(noise_vals)):
                    #print(df[(df['steps_ahead'] == forecast_lengths[f]) & 
                    #                           (df['noise'] == noise_vals[n])][key].values)
                    #print(np.mean(df[(df['steps_ahead'] == forecast_lengths[f]) & 
                    #                            (df['noise'] == noise_vals[n])][key].values))
                    #time.sleep(2)
                    data[key][n,f] += math.log10(np.mean(df[(df['steps_ahead'] == forecast_lengths[f]) & 
                                                (df['noise'] == noise_vals[n])][key].values))

        mini = -10.2#min([np.min(data[key]) for key in keylist])
        maxi = 2#max([np.max(data[key]) for key in keylist])
        
        for i in range(4):
            #im = axs[i].imshow(np.flip(data[keylist[i]],axis=0))
            ims[k] = axs[k,i].imshow(data[keylist[i]],vmin=mini,vmax=maxi,cmap=colour)
            if k == 0:
                axs[k,i].set_title(titles[keylist[i]],fontsize=18)
            if i == 0:
                axs[k,i].set_yticks([0,3,6,9,12,15])
                axs[k,i].set_yticklabels([-1,-2,-3,-4,-5,-6])
            else:
                axs[k,i].set_yticks([])
            #if k == 2:
            axs[k,i].set_xticks([2,13])
            axs[k,i].set_xticklabels([round(h*lyap_scaler,3) for h in [5,25]])
            #else:
            #    axs[k,i].set_xticks([])
            axs[k,i].tick_params(axis='both', labelsize=15)
    plt.tight_layout(pad=3)
    #plt.gcf().subplots_adjust(bottom=0.1,right=0.95,left=0.05,top=1,wspace=0.1,hspace=0.1)
    #fig.supxlabel('Forecasting Horizon',fontsize=25)
    #fig.supylabel('$log_{10}\sqrt{\sigma_{noise}^2}$',fontsize=25)
    cbar = plt.colorbar(ims[0],ax=axs.ravel().tolist(),cmap='seismic',shrink=1)
    cbar.ax.tick_params(labelsize=15)
    plt.show()

    
def main(TS_length=10000,activation='tanh',edges=False,triangles=False,squares=False,n=200,noise='na',
         system='Lorenz'):
    spec_rad = 1#0.8+0.2*rand.random();
    if activation == 'linear':
        spec_rad = 0.9
    net_size=1000
    #TS_builder = mapnsys.Lorenz(net_size, 0.02, just_x=False)['X']
    #G,pos = PSNB.build(TS_builder, 5)
    #net = res.ESN(net_size, spec_rad, con_prob = con_prob, directed=False, self_loops=False,from_graph=G,pos=pos)
    net = res.ESN(net_size, spec_rad, connectivity = 6, directed=False, self_loops=False)
    horizons = list(range(1,31,2))
    print(horizons)
    
    for horizon in horizons:
        print('------ Forecast horizon = '+str(horizon)+':')
        #print('just_nodes')
        get_data(net,n=n,TS_length=TS_length,activation=activation,edges=False,triangles=False,
                 steps_ahead = horizon,obs_noise=noise,system=system)
        #time.sleep(15)
        #print('nodes + edges')
        #get_data(net,n=n,TS_length=TS_length,activation=activation,edges=True,triangles=False,
        #         steps_ahead = horizon,obs_noise=noise)
        #time.sleep(15)
        #print('triangles + nodes and edges')
        #try:
        #    get_data(net,n=n,TS_length=TS_length,activation=activation,edges=True,triangles=True,
        #             steps_ahead = horizon,obs_noise=noise)
        #    #time.sleep(30)
        #except:
        #    print('COULDNT FIND SQUARES')
        #print('also squares')
        try:
            get_data(net,n=n,TS_length=TS_length,activation=activation,edges=True,triangles=True,squares=True,
                     steps_ahead = horizon,obs_noise=noise,system=system)
        
        except:
            print('COULDNT FIND TRIANGLES')
        
        time.sleep(15)
    plot(system=system,noise=str(noise),TS_length=TS_length,activation=activation)
        

plot(activation='linear')
plot_all()
plot_all_comparisson()
plot_all_heatmaps('logs/log4_Thomas.csv')
plot_all_heatmaps('logs/log4_Thomas.csv',augmented=True)
print(1/0)
#for system in ['Lorenz','Rossler','Thomas']:
#    plot(system=system,noise='na')
#plot_all_heatmaps('logs/log4_Thomas.csv')
#plot_all_heatmaps('logs/log4_Thomas.csv',augmented=True)
#noises = np.linspace(-6,-1,16).tolist()
#noises.append('na')
#print(noises)
#print(1/0)
for go in range(10):
    for noise in ['na']:
        print('------------------------ noise = : '+str(noise))
        #main(TS_length=5000,noise=noise,activation='tanh',system='Rossler')
        #time.sleep(60)
        main(TS_length=5000,noise=noise,activation='linear')
        time.sleep(60)
        #main(TS_length=5000,noise=noise,activation='tanh',system='Thomas')
        #time.sleep(120)
    #main(TS_length=5000,noise=-1)
    #time.sleep(30)
    #main(TS_length=5000,noise=-5)
    #time.sleep(30)
    #main(TS_length=5000,noise=-3,activation='tanh')
    #time.sleep(30)
    #main(TS_length=5000,noise=-2)
    #time.sleep(30)



