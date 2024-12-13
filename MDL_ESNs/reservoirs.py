import numpy as np
import random as rand
import maps_and_systems as mapnsys
import description_length as MDL
import breathing as breath
import matplotlib.pyplot as plt
import math 
import networkx as nx
from itertools import combinations
import time

def find_triangles(G,n):
    triangles = []; nodes = list(G.nodes()); counter = 0
    triangle_sets = []
    while len(triangles) < n:
        u = rand.choice(nodes)
        neighbors = list(G.neighbors(u))
        if len(neighbors) > 1:
            vw = rand.sample(neighbors, k=2)
            if G.has_edge(vw[0], vw[1]):
                if set((u, vw[0], vw[1])) not in triangle_sets:
                    triangle_sets.append(set((u, vw[0], vw[1])))
                    triangles.append((u, vw[0], vw[1]))
                    counter = 0
        counter += 1
        if counter > 10000000:
            print(1/0)
            break
    return triangles

def flatten(xss):
    return [x for xs in xss for x in xs]

def find_squares(G,n):
    squares = []; nodes = list(G.nodes()); counter = 0
    square_sets = []
    while len(squares) < n:
        u = rand.choice(nodes)
        neighbors = list(G.neighbors(u))
        if len(neighbors) > 1:
            vw = rand.sample(neighbors, k=2)
            v_neighs = set(list(G.neighbors(vw[0])))
            w_neighs = set(list(G.neighbors(vw[1])))
            common = v_neighs.intersection(w_neighs)
            common = list(common.difference(set([u])))
            if len(common) >= 1:
                x = rand.choice(common)
                if set((u, vw[0], vw[1], x)) not in square_sets:
                    square_sets.append(set((u, vw[0], vw[1], x)))
                    squares.append((u, vw[0], vw[1], x))
                    counter = 0
        counter += 1
        if counter > 10000000:
            print(1/0)
            break
    return squares

def softmax(vector):
 #e = np.exp(vector)
 return vector/vector.sum()#e / e.sum()

def get_dynamic_Win(pos,point):
    distances = np.linalg.norm(np.array(pos)-point,axis=1)
    k = int(math.sqrt(len(distances)))
    result = np.argpartition(distances, k)[:k]
    #print('distances:')
    #print(distances)
    #print('')
    #print('Win:')
    Win = np.array([0.5 if i in result else 0 for i in range(len(distances))])
    #Win = math.sqrt(len(distances))*softmax(np.reciprocal(distances+np.ones(len(distances))*10**(-6)))
    #print('max weight is '+str(np.max(Win))+', sum is '+str(Win.sum()))
    
    #time.sleep(0.2)
    return Win

class ESN:
    def __init__(self,n,spec_rad,connectivity=6,rewire=0.01,directed=True,self_loops=False,from_graph='no',
                 pos='na'):
        #Win = np.array([1-2*rand.random() for i in range(n)])
        Win = np.array([rand.gauss(0,0.3) for i in range(n)])
        self.Win = Win#/np.linalg.norm(Win)
        if from_graph == 'no':
            #net = nx.erdos_renyi_graph(n,connectivity/n)
            net = nx.watts_strogatz_graph(n,connectivity,rewire)
            #net = nx.barabasi_albert_graph(n,4)
            self.node_positions = nx.spring_layout(net)
        else:
            net = from_graph
            self.node_positions = pos
        self.net = net

        self.net_size = n
        M = nx.to_numpy_array(net)
        if from_graph == 'no':
            for i in range(n-1):
                for j in range(i,n):
                    if M[i,j] == 1:
                        new_val = rand.gauss(0,1)
                        M[i,j] += new_val-1
                        M[j,i] += new_val-1
        eigens = [abs(i) for i in np.linalg.eigvals(M)]
        self.M = (spec_rad/max(eigens))*M
    
    def get_subgraphs(self,n,edges=True,triangles=True,squares=True,method ='sample'):
        G = nx.from_numpy_array(self.M)
        self.net = G
        if method == 'sample':
            if edges:
                self.edges = rand.sample(list(G.edges()),n)
                self.NOE = n
            else:
                self.edges = 'na'
                self.NOE = 0
            if triangles:
                self.triangles = find_triangles(G, n)
                self.NOT = n
            else:
                self.triangles = 'na'
                self.NOT = 0
            if squares:
                self.squares = find_squares(G, n)
                self.NOS = n
            else:
                self.squares = 'na'
                self. NOS = 0
        elif method == 'cycle_basis':
            print('Finding Cycle Basis')
            cycles = nx.cycle_basis(G)
            print('Found Basis')
            self.edges = list(G.edges)
            self.NOE = len(self.edges)
            if triangles:
                self.triangles = [c for c in cycles if len(c)==3]
                self.NOT = len(self.triangles)
                print('found '+str(self.NOT)+' triangles')
            else:
                self.triangles = 'na'
                self.NOT = 0
            if squares:
               self.squares = [c for c in cycles if len(c)==4]
               self.NOS = len(self.squares)
               print('found '+str(self.NOS)+' squares')
            else:
                self.squares = 'na'
                self.NOS = 0
           
        
    def act(self,TS,transient_p,leaky=1,edges=False,triangles=False,squares=False,activation='tanh',
            state='na',dynamic_Win=False, attractor=None):
        if state == 'na':
            state = np.zeros(len(self.Win))
        elif state == 'random':
            state=np.array([1-2*rand.random() for i in range(len(self.Win))])
        X = []; Win = self.Win
        # drop y components of TS if they are present 
        try:
            TS = TS[:,0]
        except:
            pass
        counter = 0
        #if not dynamic_Win:
        #xpos = np.array([x[0] for x in self.node_positions])
        for point in TS:
            if dynamic_Win:
                attractor_point = attractor[counter,:]
                #print('TS point is '+str(point))
                #print('Attractor point is '+str(attractor_point))
                Win = get_dynamic_Win(self.node_positions, attractor_point)
                state = (1-leaky)*state + leaky*(np.matmul(self.M,state) + Win) #(point-xpos)*
                counter += 1
            elif activation == 'tanh':
                state = (1-leaky)*state + leaky*np.tanh(np.matmul(self.M,state) + point*Win)
            elif activation == 'linear':
                state = (1-leaky)*state + leaky*(np.matmul(self.M,state) + point*Win)
            to_record = state.tolist()
            if edges:
                #to_record += [math.copysign(abs(state[e[0]]*state[e[1]])**(1/2),state[e[0]]*state[e[1]]) 
                #              for e in self.edges]
                to_record += [state[e[0]]*state[e[1]] for e in self.edges]
            if triangles:
                #to_record += [math.copysign(abs(state[t[0]]*state[t[1]]*state[t[2]])**(1/3),
                #                            state[t[0]]*state[t[1]]*state[t[2]])
                #              for t in self.triangles]
                to_record += [state[t[0]]*state[t[1]]*state[t[2]] for t in self.triangles]
            if squares:
                to_record += [state[s[0]]*state[s[1]]*state[s[2]]*state[s[3]] for s in self.squares]
            X.append(to_record)
        #self.X = np.transpose(np.insert(np.array(X[transient_p:]),0,1,axis=1))
        X = np.array(X[transient_p:])
        self.state = state
        return X
    
    def fit(self,Y,steps_ahead=1,transient_p=100):
        X = self.act(Y,transient_p)[:-steps_ahead,:]
        Y = Y[steps_ahead+transient_p:]
        self.coefs = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),
                                         np.transpose(X)),Y)
        pred = self.predict(Y, steps_ahead)
        error = np.matmul(X,self.coefs) - Y
        return {'DL':MDL.Length(error, X.shape[0], X)}
    
    def step(self,inp,state,leaky=1,edges=False,triangles=False,squares=False,activation='tanh',
            dynamic_Win=False, attractor=None, coefs = ['na'], subset = ['na'], xnorm=['na'], ynorm=1):

        if coefs[0] == 'na':
            coefs = np.ones(len(state))
        if subset[0] == 'na':
            subset = list(range(len(coefs)))
        if xnorm[0] == 'na':
            xnorm = np.ones(len(coefs))
        if activation == 'tanh':
            state = (1-leaky)*state + leaky*np.tanh(np.matmul(self.M,state) + inp/ynorm*self.Win)
        elif activation == 'linear':
            state = (1-leaky)*state + leaky*(np.matmul(self.M,state) + inp/ynorm*self.Win)
        to_act = state.tolist()
        if edges:
            to_act += [state[e[0]]*state[e[1]] for e in self.edges]
        if triangles:
            to_act += [state[t[0]]*state[t[1]]*state[t[2]] for t in self.triangles]
        if squares:
            to_act += [state[s[0]]*state[s[1]]*state[s[2]]*state[s[3]] for s in self.squares]
        return state, np.matmul(np.array([to_act[i]/xnorm[i] for i in subset]),coefs)
    
    def free_step(self,coefs,state,activation='tanh',leaky=1,subset='na'):
        if subset=='na':
            inp = np.matmul(state,coefs)
        else:
            inp = np.matmul(np.transpose(np.array([state[b] for b in subset])),coefs)
        if activation == 'tanh':
            state = (1-leaky)*state + leaky*np.tanh(np.matmul(self.M,state) + inp*self.Win)
        elif activation == 'linear':
            state = (1-leaky)*state + leaky*(np.matmul(self.M,state) + inp*self.Win)
        return state,inp
    
    def increment(self,state,inp,activation='tanh',leaky=1):
        if activation == 'tanh':
            state = (1-leaky)*state + leaky*np.tanh(np.matmul(self.M,state) + inp*self.Win)
        elif activation == 'linear':
            state = (1-leaky)*state + leaky*(np.matmul(self.M,state) + inp*self.Win)
        return state
        
    def predict(self,TS,steps_ahead):
        state = self.state; preds = []
        for point in TS[:-steps_ahead]:
            state = np.tanh(np.matmul(self.M,state) + point*self.Win)
            preds.append(np.matmul(state,np.transpose(self.coefs)))
        prediction = np.array(preds); desired = TS[steps_ahead:]
        return {'prediction':prediction,'desired':TS[steps_ahead:],
                'corr':np.corrcoef(prediction,desired)[0,1]}
    
    def draw_selected_components(self,subset):
        #subset needs to be a list or array of the indicies of rows selected from X
        selected_nodes = [i for i in subset if i < self.net_size]
        selected_edges = [self.edges[i-self.net_size] for i in subset if self.net_size <= i < self.net_size+self.NOE]
        selected_traingles = [self.triangles[i-self.net_size-self.NOE]
                              for i in subset if self.net_size+self.NOE <= i < self.net_size+self.NOE+self.NOT]
        selected_squares = [self.squares[i-self.net_size-self.NOE-self.NOT]
                            for i in subset if self.net_size+self.NOE+self.NOT <= i]
        if False:
            fig,axs = plt.subplots(1,2,figsize=(10,4))
            nx.draw_networkx_nodes(self.net,pos=self.node_positions,ax=axs[0],node_size=10,node_color='k',alpha=1)
            nx.draw_networkx_edges(self.net,pos=self.node_positions,ax=axs[0],width=0.2,edge_color='k',alpha=0.4)
            nx.draw_networkx_nodes(self.net,pos=self.node_positions,ax=axs[1],node_size=10,node_color='k',alpha=0.2)
            nx.draw_networkx_edges(self.net,pos=self.node_positions,ax=axs[1],width=0.2,edge_color='k',alpha=0.4)
            nx.draw_networkx_nodes(self.net, pos=self.node_positions,nodelist=selected_nodes,
                                   node_size=30,node_color='C0',ax=axs[1])
            edges = {}
            if len(selected_edges) > 0:
                edges['edges'] = [(e[0],e[1]) for e in selected_edges]
                nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = edges['edges'], 
                                       edge_color = 'C1', ax = axs[1])
            if len(selected_traingles) > 0:
                edges['triangles'] = [[(t[0],t[1]),(t[1],t[2]),(t[2],t[0])] for t in selected_traingles]
                nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = flatten(edges['triangles']), 
                                       edge_color = 'C2', ax = axs[1])
            if len(selected_squares) > 0:
                edges['squares'] = [[(t[0],t[1]),(t[1],t[2]),(t[2],t[3]),(t[3],t[0])] for t in selected_squares]
                nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = flatten(edges['squares']),
                                       edge_color = 'C3', ax = axs[1])
            
            plt.tight_layout()
            plt.show()
        
        fig,axs = plt.subplots(1,1,figsize=(8,6))
        nx.draw_networkx_nodes(self.net,pos=self.node_positions,ax=axs,node_size=10,node_color='k',alpha=0.2)
        nx.draw_networkx_edges(self.net,pos=self.node_positions,ax=axs,width=0.2,edge_color='k',alpha=0.4)
        nx.draw_networkx_nodes(self.net, pos=self.node_positions,nodelist=selected_nodes,
                               node_size=30,node_color='C0',ax=axs)

        edges = {}
        if len(selected_edges) > 0:
            edges['edges'] = [(e[0],e[1]) for e in selected_edges]
            nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = edges['edges'], 
                                   edge_color = 'C1', ax = axs)
        if len(selected_traingles) > 0:
            edges['triangles'] = [[(t[0],t[1]),(t[1],t[2]),(t[2],t[0])] for t in selected_traingles]
            nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = flatten(edges['triangles']), 
                                   edge_color = 'C2', ax = axs)
        if len(selected_squares) > 0:
            edges['squares'] = [[(t[0],t[1]),(t[1],t[2]),(t[2],t[3]),(t[3],t[0])] for t in selected_squares]
            nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = flatten(edges['squares']),
                                   edge_color = 'C3', ax = axs)
        
        plt.tight_layout()
        plt.show()
        
        if False:
            fig,axs = plt.subplots(2,4,figsize=(10,4))
            nx.draw_networkx_nodes(self.net,pos=self.node_positions,ax=axs[0,0],node_size=10,node_color='k',alpha=1)
            edges = [(e[0],e[1]) for e in self.edges]
            nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = edges, 
                                   edge_color = 'k', ax = axs[0,1])
            triangles = [[(t[0],t[1]),(t[1],t[2]),(t[2],t[0])] for t in self.triangles]
            nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = flatten(triangles), 
                                   edge_color = 'k', ax = axs[0,2])
            if self.NOS > 0:
                squares = [[(t[0],t[1]),(t[1],t[2]),(t[2],t[3]),(t[3],t[0])] for t in self.squares]
                nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = flatten(squares),
                                       edge_color = 'k', ax = axs[0,3])
            
            nx.draw_networkx_nodes(self.net, pos=self.node_positions,nodelist=selected_nodes,
                                   node_size=10,node_color='k',ax=axs[1,0])
            edges = {}
            if len(selected_edges) > 0:
                edges['edges'] = [(e[0],e[1]) for e in selected_edges]
                nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = edges['edges'], 
                                       edge_color = 'k', ax = axs[1,1])
            if len(selected_traingles) > 0:
                edges['triangles'] = [[(t[0],t[1]),(t[1],t[2]),(t[2],t[0])] for t in selected_traingles]
                nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = flatten(edges['triangles']), 
                                       edge_color = 'k', ax = axs[1,2])
            if len(selected_squares) > 0:
                edges['squares'] = [[(t[0],t[1]),(t[1],t[2]),(t[2],t[3]),(t[3],t[0])] for t in selected_squares]
                nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = flatten(edges['squares']),
                                       edge_color = 'k', ax = axs[1,3])
            
            plt.tight_layout()
            plt.show()
            
            
            fig,axs = plt.subplots(1,4,figsize=(12,3))
            nx.draw_networkx_nodes(self.net,pos=self.node_positions,ax=axs[0],node_size=10,node_color='k',alpha=0.3)
            edges = [(e[0],e[1]) for e in self.edges]
            nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = edges, 
                                   edge_color = 'k', ax = axs[1], alpha=0.3)
            triangles = [[(t[0],t[1]),(t[1],t[2]),(t[2],t[0])] for t in self.triangles]
            nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = flatten(triangles), 
                                   edge_color = 'k', ax = axs[2], alpha=0.3)
            squares = [[(t[0],t[1]),(t[1],t[2]),(t[2],t[3]),(t[3],t[0])] for t in self.squares]
            nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = flatten(squares),
                                   edge_color = 'k', ax = axs[3], alpha=0.3)
            
            nx.draw_networkx_nodes(self.net, pos=self.node_positions,nodelist=selected_nodes,
                                   node_size=10,node_color='r',ax=axs[0])
            edges = {}
            if len(selected_edges) > 0:
                edges['edges'] = [(e[0],e[1]) for e in selected_edges]
                nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = edges['edges'], 
                                       edge_color = 'r', ax = axs[1])
            if len(selected_traingles) > 0:
                edges['triangles'] = [[(t[0],t[1]),(t[1],t[2]),(t[2],t[0])] for t in selected_traingles]
                nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = flatten(edges['triangles']), 
                                       edge_color = 'r', ax = axs[2])
            if len(selected_squares) > 0:
                edges['squares'] = [[(t[0],t[1]),(t[1],t[2]),(t[2],t[3]),(t[3],t[0])] for t in selected_squares]
                nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = flatten(edges['squares']),
                                       edge_color = 'r', ax = axs[3])
            
            plt.tight_layout()
            plt.show()
        
    def four_plot(self,dic):
        fig,axs=plt.subplots(2,2,figsize=(12,10))
        ax = [axs[0,0],axs[0,1],axs[1,0],axs[1,1]]; keylist = ['N','E','T','S']
        for p in range(4):
            subset = dic[keylist[p]]
            selected_nodes = [i for i in subset if i < self.net_size]
            selected_edges = [self.edges[i-self.net_size] for i in subset if self.net_size <= i < self.net_size+self.NOE]
            selected_traingles = [self.triangles[i-self.net_size-self.NOE]
                                  for i in subset if self.net_size+self.NOE <= i < self.net_size+self.NOE+self.NOT]
            selected_squares = [self.squares[i-self.net_size-self.NOE-self.NOT]
                                for i in subset if self.net_size+self.NOE+self.NOT <= i]
            nx.draw_networkx_nodes(self.net,pos=self.node_positions,ax=ax[p],node_size=10,node_color='k',alpha=0.2)
            nx.draw_networkx_edges(self.net,pos=self.node_positions,ax=ax[p],width=0.2,edge_color='k',alpha=0.4)
            nx.draw_networkx_nodes(self.net, pos=self.node_positions,nodelist=selected_nodes,
                                   node_size=30,node_color='C0',ax=ax[p])
            edges = {}
            if len(selected_edges) > 0:
                edges['edges'] = [(e[0],e[1]) for e in selected_edges]
                nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = edges['edges'], 
                                       edge_color = 'C1', ax = ax[p])
            if len(selected_traingles) > 0:
                edges['triangles'] = [[(t[0],t[1]),(t[1],t[2]),(t[2],t[0])] for t in selected_traingles]
                nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = flatten(edges['triangles']), 
                                       edge_color = 'C2', ax = ax[p])
            if len(selected_squares) > 0:
                edges['squares'] = [[(t[0],t[1]),(t[1],t[2]),(t[2],t[3]),(t[3],t[0])] for t in selected_squares]
                nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = flatten(edges['squares']),
                                       edge_color = 'C3', ax = ax[p])
            
        plt.tight_layout()
        plt.show()
        
        fig,axs=plt.subplots(1,4,figsize=(16,4))
        ax = [axs[0],axs[1],axs[2],axs[3]]; keylist = ['N','E','T','S']
        for p in range(4):
            subset = dic[keylist[p]]
            selected_nodes = [i for i in subset if i < self.net_size]
            selected_edges = [self.edges[i-self.net_size] for i in subset if self.net_size <= i < self.net_size+self.NOE]
            selected_traingles = [self.triangles[i-self.net_size-self.NOE]
                                  for i in subset if self.net_size+self.NOE <= i < self.net_size+self.NOE+self.NOT]
            selected_squares = [self.squares[i-self.net_size-self.NOE-self.NOT]
                                for i in subset if self.net_size+self.NOE+self.NOT <= i]
            nx.draw_networkx_nodes(self.net,pos=self.node_positions,ax=ax[p],node_size=10,node_color='k',alpha=0.2)
            nx.draw_networkx_edges(self.net,pos=self.node_positions,ax=ax[p],width=0.2,edge_color='k',alpha=0.4)
            nx.draw_networkx_nodes(self.net, pos=self.node_positions,nodelist=selected_nodes,
                                   node_size=30,node_color='C0',ax=ax[p])
            edges = {}
            if len(selected_edges) > 0:
                edges['edges'] = [(e[0],e[1]) for e in selected_edges]
                nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = edges['edges'], 
                                       edge_color = 'C1', ax = ax[p])
            if len(selected_traingles) > 0:
                edges['triangles'] = [[(t[0],t[1]),(t[1],t[2]),(t[2],t[0])] for t in selected_traingles]
                nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = flatten(edges['triangles']), 
                                       edge_color = 'C2', ax = ax[p])
            if len(selected_squares) > 0:
                edges['squares'] = [[(t[0],t[1]),(t[1],t[2]),(t[2],t[3]),(t[3],t[0])] for t in selected_squares]
                nx.draw_networkx_edges(self.net, pos=self.node_positions,edgelist = flatten(edges['squares']),
                                       edge_color = 'C3', ax = ax[p])
            
        plt.tight_layout()
        plt.show()
        
        
