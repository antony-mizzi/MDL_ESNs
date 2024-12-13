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
    
def log_datav2(details,fname):
    try:
        df1 = pd.read_csv('logs/'+fname)
        df2 = pd.DataFrame(details)
        DF = pd.concat([df1, df2], ignore_index = True)
        DF.reset_index()
    except:
        DF = pd.DataFrame(details)
    DF.to_csv('logs/'+fname, index=False)

def NET_counter(coefs,net,n,is_MDL=True,edges = True,triangles=True,squares=False):
    if is_MDL: # expect subset selection in the place of coefs
        N = len([c for c in coefs if c < n])
        if edges:
            E = len([c for c in coefs if c >= n and c < n+net.NOE])
        else:
            E = 0
        if triangles:
            T = len([c for c in coefs if c >= n+net.NOE and c < n+net.NOE+net.NOT])
        else:
            T=0
        if squares:
            S = len([c for c in coefs if c >= n+net.NOE+net.NOT])
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

def make_plot(activation='tanh',TS_length=5000,steps_ahead=1,obs_noise='na',leaky=1,
                  edges=False,triangles=False,squares=False,n=1000):  
    transient_p = 500
    spec_rad = 0.8+0.2*rand.random()
    #TS_builder = mapnsys.Lorenz(n, 0.02, just_x=False)['X']
    #G,pos = PSNB.build(TS_builder, 5)
    net = res.ESN(n, spec_rad, connectivity = 6, directed=False, self_loops=False)#,from_graph=G,pos=pos)
    TS_train = mapnsys.Lorenz(TS_length+transient_p, 0.02)
    TS_test = mapnsys.Lorenz(50000+transient_p, 0.02)
    if edges or triangles or squares:
        net.get_subgraphs(n,edges=edges,triangles=triangles,squares=squares)
    X_train = net.act(TS_train,transient_p,activation=activation,leaky=leaky,
                      edges=edges,triangles=triangles,squares=squares)
    X_test = net.act(TS_test,transient_p,activation=activation,leaky=leaky,
                     edges=edges,triangles=triangles,squares=squares)
    row_means = np.abs(X_train).mean(axis=0, keepdims=True)
    X_train = X_train / row_means
    X_test = X_test / row_means
    X_train = X_train[:-steps_ahead,:]; X_test = X_test[:-steps_ahead,:]
    Y_train = TS_train[steps_ahead+transient_p:]; Y_test = TS_test[steps_ahead+transient_p:]
    if obs_noise != 'na':
        Y_train += np.random.normal(loc=0.0,scale=10**obs_noise,size=len(Y_train))
    MSE = {'activation':activation,'train_size':str(TS_length),'steps_ahead':steps_ahead,'noise':obs_noise,
           'leakage':leaky,'edges':edges,'triangles':triangles,'squares':squares,'net_size':n}
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Normalise the rows in X 
    # 2) get MSE on X_test for readouts trained the following ways:
        
        # a) MDL with breathing algorithm for each value of K >= 1   
        
    S,Bs,Var,coefs = breath.breath(X_train, Y_train, early_stop=True)
    best_k = min(S, key = S.get)
    MDL_choices = Bs[best_k]
    M_vec = np.array([1 if i in MDL_choices else 0 for i in range(n)])
    #print(M_vec)
    #print('')
    M_vec2 = np.array([coefs[best_k][MDL_choices.index(i)] if i in MDL_choices else 0 for i in range(n)])
    print(M_vec2)
    MSE['k_opt'] = best_k
    
        # c) ridge regression with all nodes for different penalty values 
        
    vars_ridge, alphas, coefs = breath.ridge_it(X_train, Y_train)
    measures = [measure(M_vec,np.abs(coefs[key])) for key in coefs.keys()]
    fig,axs=plt.subplots(1,1,figsize=(10,7))
    axs.set_xlabel('$log_{10}(\\alpha)$')
    axs.set_ylabel('$\hat{m}\cdot\hat{r}^{\\alpha}$')
    axs.plot(np.log10(alphas),measures)
    axs.tick_params(axis='both', labelsize=15)
    plt.show()
    
    measures = [measure(M_vec2,coefs[key]) for key in coefs.keys()]
    fig,axs=plt.subplots(1,1,figsize=(10,7))
    axs.set_xlabel('$log_{10}(\\alpha)$',fontsize=25)
    axs.set_ylabel('$\hat{m}\cdot\hat{r}^{\\alpha}$',fontsize=25)
    axs.plot(np.log10(alphas),measures,color='k')
    axs.plot(np.log10(alphas),measures,'X',color='k')
    axs.tick_params(axis='both', labelsize=15)
    plt.show()
    
    coefs = {k:np.log10(np.abs(coefs[k])) for k in coefs.keys()}
    
    
    fig,ax = plt.subplots(1,2,figsize=(10,5),layout='constrained')
    #node_size (by MDL choices)
    axs = [ax[0],ax[1]]; penalties = [-5.0,-2.0]
    titles = ['$\\alpha = 10^{'+str(int(p))+'}$' for p in penalties]
    vmin = -3; vmax = 1

    for i in range(2):
        axs[i].set_title(titles[i],fontsize=25)
        not_chosen = [i for i in range(n) if i not in MDL_choices]
        colours = np.array([coefs[penalties[i]][j] for j in not_chosen])
        nx.draw_networkx_nodes(net.net,pos=net.node_positions,nodelist = not_chosen, node_size=100,
                               node_color=colours,cmap='seismic',vmin=vmin,vmax=vmax,ax=axs[i])#,with_labels=False)
        colours = np.array([coefs[penalties[i]][j] for j in MDL_choices])
        nx.draw_networkx_nodes(net.net,pos=net.node_positions,nodelist=MDL_choices,node_size=200,node_color=colours,
                         cmap='seismic',vmin=vmin,vmax=vmax,ax=axs[i],node_shape='*')#,with_labels=False)
        nx.draw_networkx_edges(net.net,pos=net.node_positions,ax=axs[i],width=0.3,alpha=0.5)
    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #plt.colorbar()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='seismic', norm=norm)
    cbar = fig.colorbar(sm,ax=ax[1])#,shrink=0.8)
    cbar.set_ticks([])
    #plt.tight_layout()
    plt.show()
    
def make_plot_type2(net,TS_train,TS_test,activation='linear',TS_length=10000,steps_ahead=1,obs_noise='na',
                    leaky=1,edges=True,triangles=True,squares=True,n=1000):  
    transient_p = 500
    
    if edges or triangles or squares:
        net.get_subgraphs(n,edges=edges,triangles=triangles,squares=squares,method='cycle_basis')#'sample')
    X_train = net.act(TS_train,transient_p,activation=activation,leaky=leaky,
                      edges=edges,triangles=triangles,squares=squares)
    #X_test = net.act(TS_test,transient_p,activation=activation,leaky=leaky,
    #                 edges=edges,triangles=triangles,squares=squares)
    row_means = np.linalg.norm(X_train, axis=0, keepdims=True)
    X_train = X_train / row_means
    #X_test = X_test / row_means
    X_train = X_train[:-steps_ahead,:]#; X_test = X_test[:-steps_ahead,:]
    Y_train = TS_train[steps_ahead+transient_p:]; Y_test = TS_test[steps_ahead+transient_p:]
    if obs_noise != 'na':
        Y_train += np.random.normal(loc=0.0,scale=10**obs_noise,size=len(Y_train))

    S,Bs,Var,coefs = breath.breath(X_train, Y_train, early_stop=True)
    best_k = min(S, key = S.get)
    MDL_choices = Bs[best_k]
    if True: #steps_ahead % 1 == 0:
        net.draw_selected_components(MDL_choices)
    return NET_counter(MDL_choices,net,n,squares=True)

def make_4plot(net,TS_train,TS_test,activation='linear',TS_length=10000,steps_ahead=1,obs_noise='na',
                    leaky=1,edges=True,triangles=True,squares=True,n=1000,method='cycle_basis'):  
    transient_p = 500
    
    if edges or triangles or squares:
        net.get_subgraphs(n,edges=edges,triangles=triangles,squares=squares,method='cycle_basis')
    X_train = net.act(TS_train,transient_p,activation=activation,leaky=leaky,
                      edges=edges,triangles=triangles,squares=squares)
    #X_test = net.act(TS_test,transient_p,activation=activation,leaky=leaky,
    #                 edges=edges,triangles=triangles,squares=squares)
    row_means = np.linalg.norm(X_train, axis=0, keepdims=True)
    X_train = X_train / row_means
    #X_test = X_test / row_means
    X_train = X_train[:-steps_ahead,:]#; X_test = X_test[:-steps_ahead,:]
    Y_train = TS_train[steps_ahead+transient_p:]; Y_test = TS_test[steps_ahead+transient_p:]
    if obs_noise != 'na':
        Y_train += np.random.normal(loc=0.0,scale=10**obs_noise,size=len(Y_train))
    
    deets = {}; keylist = ['N','E','T','S']; cuts = [n,n+net.NOE,n+net.NOE+net.NOT,n+net.NOE+net.NOT+net.NOS]
    for i in range(4):
        print('breathing '+keylist[i])
        S,Bs,Var,coefs = breath.breath(X_train[:,:cuts[i]], Y_train, early_stop=True,verbose=False)
        best_k = min(S, key = S.get)
        deets[keylist[i]] = Bs[best_k]
    net.four_plot(deets)
    
def explore_node_choices(activation='linear',TS_length=10000,steps_ahead=1,obs_noise='na',
                    leaky=1,edges=False,triangles=False,squares=False,n=1000,method='cycle_basis',net_size=1000,
                    spec_rad=0.9,con_prob='na',system='Lorenz'):  
    transient_p = 500
    if system=='Lorenz':
        TS_train = mapnsys.Lorenz(TS_length+500, 0.02)
        TS_builder = mapnsys.Lorenz(net_size, 0.02, just_x=False,transient=5000)['X']
        NPClog = 'node_choices/NPC.csv'; NPTlog = '/node_choices/NPT.csv'
    elif system == 'Rossler':
        TS_train = mapnsys.Rossler(TS_length+500, 0.02)
        TS_builder = mapnsys.Rossler(net_size, 0.04, just_x=False,transient=5000)['X']
        NPClog = 'node_choices/NPC_rossler.csv'; NPTlog = '/node_choices/NPT_rossler.csv'
    G,pos = PSNB.build(TS_builder, 5, depict=True)
    net = res.ESN(net_size, spec_rad, con_prob = con_prob, directed=False, self_loops=False,from_graph=G,pos=pos)
    #if edges or triangles or squares:
    net.get_subgraphs(n,edges=edges,triangles=triangles,squares=squares,method='sample')
    X_train = net.act(TS_train,transient_p,activation=activation,leaky=leaky,
                      edges=edges,triangles=triangles,squares=squares)
    #X_test = net.act(TS_test,transient_p,activation=activation,leaky=leaky,
    #                 edges=edges,triangles=triangles,squares=squares)
    row_means = np.abs(X_train).mean(axis=0, keepdims=True)
    X_train = X_train / row_means
    #X_test = X_test / row_means
    X_train = X_train[:-steps_ahead,:]#; X_test = X_test[:-steps_ahead,:]
    Y_train = TS_train[steps_ahead+transient_p:]#; Y_test = TS_test[steps_ahead+transient_p:]
    S,Bs,Var,coefs = breath.breath(X_train, Y_train, early_stop=True,verbose=False,min_size=51,cap=52)
    #best_k = min(S, key = S.get)
    MDL_choices = Bs[50]
    #net.draw_selected_components(MDL_choices)
    chosen_positions = {'x':[net.node_positions[i][0] for i in MDL_choices],
                        'y':[net.node_positions[i][1] for i in MDL_choices]}
    log_datav2(chosen_positions, NPClog)
    x_positions = [net.node_positions[i][0] for i in MDL_choices]
    all_xs = [t[0] for t in TS_builder]
    chosen_times = {'T_'+str(i):[all_xs.index(x_positions[i])] for i in range(len(MDL_choices))}
    log_data(chosen_times, NPTlog)
    

def plot_from_csv(csv_file,net_type='lorenz',system='lorenz'):
    df = pd.read_csv(csv_file)
    df = df[df['net_type']==net_type]
    df = df[df['system']==system]
    dfL = df[df['activation']=='linear']
    dfT = df[df['activation']=='tanh'] #!!!!!!!!!!!! just to avoid errors for the moment
    Hs = list(set(list(df['steps_ahead'])))
    Hs.sort() 
    ylabs = {'_tot':'Number Selected','_prop':'Proportion Selected'}
    for suffix in ['_tot','_prop']:
        fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for key in ['Nodes','Edges','Triangles','Squares']:
            axs[0].plot(Hs,[np.mean(dfL[dfL['steps_ahead']==h][key+suffix]) for h in Hs],label=key)
            axs[1].plot(Hs,[np.mean(dfT[dfT['steps_ahead']==h][key+suffix]) for h in Hs])
        axs[0].legend(fontsize=10)
        axs[0].set_xlabel('Forecasting Horizon (times $\delta=0.02$)',fontsize=12)
        axs[1].set_xlabel('Forecasting Horizon (times $\delta=0.02$)',fontsize=12)
        axs[0].set_title('Linear Activation ('+system+')',fontsize=15)
        axs[1].set_title('Tanh Activation ('+system+')',fontsize=15)
        axs[0].set_ylabel(ylabs[suffix],fontsize=12)
        plt.tight_layout()
        plt.show()
    
    for suffix in ['_tot','_prop']:
        fig,axs=plt.subplots(1,1,figsize=(10,7))
        for key in ['Nodes','Edges','Triangles','Squares']:
            axs.plot(Hs,[np.mean(dfL[dfL['steps_ahead']==h][key+suffix]) for h in Hs],label=key)
        axs.legend(fontsize=18)
        axs.set_xlabel('Forecasting Horizon (times $\delta=0.02$)',fontsize=25)
        axs.set_ylabel(ylabs[suffix],fontsize=25)
        plt.tight_layout()
        plt.show()
    pass

def gather_data(net_size=1000,activation='linear',net_type='lorenz',TS_length=10000,
                system='lorenz'):
    spec_rad = 0.8+0.2*rand.random(); con_prob = 0.01+0.49*rand.random()#(3+6*rand.random())/net_size
    if activation == 'tanh':
        spec_rad = 1
    else:
        spec_rad = 0.9
    if system == 'lorenz':
        TS_train = mapnsys.Lorenz(TS_length+500, 0.02)
        TS_test = mapnsys.Lorenz(50000+500, 0.02) # this is actually redundant
    elif system == 'rossler':
        TS_train = mapnsys.Rossler(TS_length+500, 0.04)
        TS_test = mapnsys.Rossler(50000+500, 0.04)
    if net_type == 'lorenz':
        TS_builder = mapnsys.Lorenz(net_size, 0.02, just_x=False)['X']
        G,pos = PSNB.build(TS_builder, 5)
        net = res.ESN(net_size, spec_rad, con_prob = con_prob, directed=False, self_loops=False,from_graph=G,pos=pos)
    elif net_type == 'rossler':
        TS_builder = mapnsys.Rossler(net_size, 0.04, just_x=False)['X']
        G,pos = PSNB.build(TS_builder, 5)
        net = res.ESN(net_size, spec_rad, con_prob = con_prob, directed=False, self_loops=False,from_graph=G,pos=pos)
    elif net_type == 'ER':
        net = res.ESN(net_size,spec_rad,con_prob=con_prob)
    for s in range(1,26):   
        if system == 'rossler':
            s = int(s*4) - 3
        #make_4plot(net,TS_train,TS_test,activation=activation,steps_ahead=s)
        props,tots = make_plot_type2(net,TS_train,TS_test,activation=activation,steps_ahead=s) 
        details = {'net_type':net_type,'system':system,'activation':activation,'steps_ahead':s}
        for key in props.keys():
            details[key+'_prop'] = props[key]
            details[key+'_tot'] = tots[key]
        #log_data(details,'choices_rossler.csv')
        #time.sleep(20)

def quick_test(net_size=1000,activation='linear',net_type='lorenz',TS_length=10000,
                system='lorenz',spec_rad=0.9,con_prob=None,leaky=0.2,edges=True,triangles=True,
                squares = True,transient_p=500,steps_ahead=5,obs_noise='na'):
    n = net_size
    TS_train = mapnsys.Lorenz(TS_length+500, 0.02, just_x=False)
    TS_train = TS_train['X'][:,:2]
    print('this is TS_train:')
    print(TS_train)
    TS_builder = mapnsys.Lorenz(net_size, 0.02, just_x=False)['X']
    G,pos = PSNB.build(TS_builder, 5)
    net = res.ESN(net_size, spec_rad, con_prob = con_prob, directed=False, self_loops=False,from_graph=G,pos=pos)
    if edges or triangles or squares:
        net.get_subgraphs(net_size,edges=edges,triangles=triangles,squares=squares,method='cycle_basis')
    for go in range(2):
        if go == 1:
            print('---------------- running with static Win')
            X_train = net.act(TS_train,transient_p,activation=activation,leaky=leaky,
                              edges=edges,triangles=triangles,squares=squares)
        else:
            print('---------------- running with dynamic Win')
            X_train = net.act(TS_train,transient_p,activation=activation,leaky=leaky,
                              edges=edges,triangles=triangles,squares=squares,dynamic_Win=True)
        #X_test = net.act(TS_test,transient_p,activation=activation,leaky=leaky,
        #                 edges=edges,triangles=triangles,squares=squares)
        row_means = np.abs(X_train).mean(axis=0, keepdims=True)
        X_train = X_train / row_means
        #X_test = X_test / row_means
        X_train = X_train[:-steps_ahead,:]#; X_test = X_test[:-steps_ahead,:]
        #Y_train = TS_train[steps_ahead+transient_p:]#; Y_test = TS_test[steps_ahead+transient_p:]
        Y_train = TS_train[steps_ahead+transient_p:,0]
        if obs_noise != 'na':
            Y_train += np.random.normal(loc=0.0,scale=10**obs_noise,size=len(Y_train))
        
        deets = {}; keylist = ['N','E','T','S']; cuts = [n,n+net.NOE,n+net.NOE+net.NOT,n+net.NOE+net.NOT+net.NOS]
        for i in range(4):
            print('breathing '+keylist[i])
            S,Bs,Var,coefs = breath.breath(X_train[:,:cuts[i]], Y_train, early_stop=True,verbose=False,max_iterations = 10)
            best_k = min(S, key = S.get)
            deets[keylist[i]] = Bs[best_k]
            plt.title(keylist[i])
            plt.plot(Var.keys(),[math.log(v) for v in Var.values()])
            plt.show()
            time.sleep(15)
        net.four_plot(deets)

def NPC_NPT_plot(number_of_cuts=11,system='Lorenz'):
    if system == 'Lorenz':
        NPClog = 'logs/node_choices/NPC.csv'; NPTlog = 'logs/node_choices/NPT.csv'
    elif system == 'Rossler':
        NPClog = 'logs/node_choices/NPC_rossler.csv'; NPTlog = 'logs/node_choices/NPT_rossler.csv'
    dfC = pd.read_csv(NPClog)
    dfT = pd.read_csv(NPTlog)
    poiss_rats = []; time_gaps = []; cuts = [int(i) for i in np.linspace(0,1000,number_of_cuts)]
    bins = []
    print('cuts = '+str(cuts))
    for i in range(len(dfT.index)):
        temp_bin = [[] for i in range(number_of_cuts - 1)]
        row = dfT.iloc[i]
        row = row.dropna().values
        for i in range(len(row)):
            slot = len([c for c in cuts[1:] if row[i] > c])
            temp_bin[slot].append(row[i])
        bins.append([len(b) for b in temp_bin])
        row = row*0.02
        poiss_rats.append(np.var(row)/np.mean(row))
        sorted_row = sorted(row.tolist())
        time_gaps += [sorted_row[i+1]-sorted_row[i] for i in range(len(sorted_row)-1)]
    fig,axs=plt.subplots(1,2,figsize=(10,5))
    axs[0].plot(dfC['x'].values,dfC['y'].values,'o',color='k',markersize=2,alpha=0.2)
    axs[1].hist(time_gaps,bins=60)
    plt.tight_layout()
    plt.show()
    
    max_obs = min([50,int(1000/(number_of_cuts-1))])
    
    observations = {i:0 for i in range(max_obs+1)}
    for B in bins:
        for k in B:
            observations[k] += 1
            
    biggest_obs = max([i for i in range(max_obs) if observations[i] != 0])+1
    num_obs = sum([observations[o] for o in observations])
    lamda = sum([o*observations[o] for o in observations])/num_obs
    theory = [lamda**i*math.exp(-1*lamda)/math.factorial(i) for i in range(max_obs+1)]
    print('sum of theoreotical p is '+str(sum(theory)))
    chi_square = 0
    for i in range(max_obs+1):
        chi_square += ((observations[i]/num_obs)-theory[i])**2/theory[i]
    print('Chi square value is '+str(chi_square))
    plt.title('Chi squared = '+str(round(chi_square,3)))
    plt.plot(list(observations.keys())[:biggest_obs],list(observations.values())[:biggest_obs])
    plt.plot(list(range(len(theory)))[:biggest_obs],[num_obs*t for t in theory][:biggest_obs],color='r')
    plt.show()
    
    


make_plot(n=1000)
print(1/0)
#for NOC in [5,6,9,11,21,26,41,51,101,126]:
#    NPC_NPT_plot(number_of_cuts=NOC)
#print(1/0)
for go in range(100):
    print('----------------- go '+str(go))
    explore_node_choices(activation='tanh',system='Rossler')
    NPC_NPT_plot(system='Rossler')
make_plot()
plot_from_csv('logs/choices_rossler.csv',system='lorenz')
plot_from_csv('logs/choices_rossler.csv',system='rossler')
#while True:
#    gather_data(TS_length=5000,activation='tanh')
#    time.sleep(120)
#    gather_data(TS_length=5000)
#    time.sleep(120)
#    gather_data(TS_length=5000,system='rossler',activation='tanh')
#    time.sleep(120)
#    gather_data(TS_length=5000,system='rossler')
#    plot_from_csv('logs/choices_rossler.csv',system='lorenz')
#    plot_from_csv('logs/choices_rossler.csv',system='rossler')
#    time.sleep(180)
#print(1/0)
#gather_data(activation='linear')
#time.sleep(120)
#gather_data(activation='tanh')
#time.sleep(120)
#plot_from_csv('logs/choices_log.csv',net_type='ER')
#gather_data(activation='linear',net_type='ER')
#time.sleep(120)
#gather_data(activation='tanh',net_type='ER')
