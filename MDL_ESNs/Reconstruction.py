import numpy as np 
import networkx as nx
import random as rand
import maps_and_systems as mapnsys
import breathing as breath
import time
import math
import attractor_comp as comper
#import graphing as grap

def fit_coefs(V,Y,alpha=0):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(V),V)+np.diag(np.ones(V.shape[1]))*alpha),
                                     np.transpose(V)),Y)
    
    
class ESN:
    def __init__(self,n,specrad,connectivity,rewire=0.01,winstrength=0.3):
        self.state = np.zeros(n)
        net = nx.barabasi_albert_graph(n,4)
        M = nx.to_numpy_array(net)
        for i in range(n-1):
            for j in range(i,n):
                if M[i,j] == 1:
                    new_val = rand.gauss(0,1)
                    M[i,j] += new_val-1
                    M[j,i] += new_val-1
        eigens = [abs(i) for i in np.linalg.eigvals(M)]
        self.M = (specrad/max(eigens))*M
        self.Win = np.array([rand.gauss(0,winstrength) for i in range(n)])
    def act(self,leak,u):
        self.state = leak*np.tanh(u*self.Win+np.matmul(self.M,self.state)) + (1-leak)*self.state
    def add_noise(self,sigma):
        self.state += np.random.normal(loc=0,scale=sigma,size=len(self.state))
    def euler_increment(self,leak,u,h=4):
        self.state = (1-1/h)*self.state + (1/h)*(leak*np.tanh(u*self.Win+np.matmul(self.M,self.state)) +
                                           (1-leak)*self.state) 
    def predict(self,coeffs,subset=['na']):
        if subset[0] == 'na':
            pred = np.matmul(self.state,np.transpose(coeffs))
        else:
            V = np.transpose(np.array([self.state[b] for b in subset]))
            pred = np.matmul(V,np.transpose(coeffs))
        return pred
    def autonomous_recon(self,coeffs,subset=['na'],npoints=10000,leak=1,h=4,noise=0):
        output = []
        for fo in range(npoints):
            for inc in range(h):
                next_inp = self.predict(coeffs,subset=subset)
                if inc == 0:
                    output.append(next_inp)
                #self.act(leak,next_inp)
                self.euler_increment(leak,next_inp,h=h)
            #self.add_noise(noise)
        return output


def run(netsize=100, leaky=1, specrad=1, winstrength=0.3,h=1,noise=0):
    TSS = [mapnsys.runge_Lorenz(11000),mapnsys.runge_Rossler(11000,Sr=0.1),
           mapnsys.runge_Thomas(rand.choice([11000,21000,31000,51000]),Sr=0.4)]
    # lag should be 40 for thomas
    counter = 0; srate = 0.1; lag = 10
    sysname = {0:'Lorenz: ',1:'Rossler: ',2:'Thomas: '}
    for TS in TSS:
        #mapnsys.embedding_plot(TS, 4)
        if counter == 3:
            counter = 1; srate = 0.2; lag = 10
        res = ESN(netsize,specrad[counter],6,winstrength=winstrength[counter])
        X = []
        for u in TS:
            res.act(leaky, u)
            #res.add_noise(10**(-3))
            #X.append(res.state)
            X.append(res.state+np.random.normal(loc=0,scale=noise,size=len(res.state)))
        X = np.array(X)
        X = X[1000:-1,:]; Y = TS[1001:]
        S,Bs,Var,MDL_coefs = breath.breath(X, Y, early_stop=True, max_iterations = 30)
        #grap.Description_Length_Plot(S)
        best_k = min(S, key = S.get)
        MDL_choices = Bs[best_k]#; coeffs = MDL_coefs[best_k]
        #MDL_choices = ['na']#list(range(netsize)) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for alpha in [0]:
            coeffs = fit_coefs(np.transpose(np.array([X[:,b] for b in MDL_choices])), Y,alpha=alpha)
            #coeffs = fit_coefs(X, Y, alpha=alpha)
            to_plot = res.autonomous_recon(coeffs,subset=MDL_choices,leak=leaky,h=h)
            recon_score = comper.compare(to_plot,TS)
            #print('RECON SCORE IS '+str(recon_score))
            if recon_score < 2.5:
                img_code = 'recon_'+''.join([str(rand.randint(0,9)) for i in range(6)])
            else:
                img_code = 'na'
            if noise == 0:
                noise_label = '0'
            else:
                noise_label = str(round(math.log10(noise),1))
            if counter == 0:
                mapnsys.embedding_plot(to_plot, 4, to_also_plot = TS[:10000],save_name=img_code)
                mapnsys.embedding_plot(to_plot, 4, title = 'Score = '+str(round(recon_score,2))+', Specrad = '+str(round(specrad[counter],3))+
                                       ', Win str = '+str(round(winstrength[counter],3))+', noise = '+noise_label+
                                       ', h = '+str(round(h,3))+', leak = '+str(round(leaky,3)),to_also_plot = TS,
                                       save_name=img_code)
            else:
                mapnsys.embedding_plot(to_plot, lag, to_also_plot = TS[:10000],save_name=img_code)
                mapnsys.embedding_plot(to_plot, lag, title = 'Specrad = '+str(round(specrad[counter],3))+
                                       ', Win str = '+str(round(winstrength[counter],3))+', Noise = '+noise_label+
                                       ', h = '+str(round(h,3))+', leak = '+str(round(leaky,3)),to_also_plot = TS,
                                       save_name=img_code)
        counter += 1
        #time.sleep(30)
        
        
for go in range(200):
    specrad = {0:0.9+0.3*rand.random(),1: 0.9+0.3*rand.random(),2: 0.8+0.2*rand.random()}
    winstrength = {0: 0.8*rand.random(),1: 0.5*rand.random(),2: 0.2*rand.random()}#**2
    leaky = rand.choice([1,10**(-4*rand.random())])
    h = 1#rand.choice([1,rand.randint(2,20)])
    noise = rand.choice([0,10**(-rand.randint(2,6))])
    run(netsize=1000,leaky=leaky,specrad=specrad,winstrength=winstrength,h=h,noise=noise)
    time.sleep(60)
    
#NOTES: 
    # BEST LORENZ RECONSTRUCTIONS OCCUR WHEN SPECRAD ~ 1, WIN STRENGTH ~ 0.3 (1000 nodes alpha = 0)
    # BEST THOMAS RECONSTRUCTION OCCURED WITH SPECRAD = 1.084, WIN STRENGTH = 0.023, Sample rate = 0.01 
    #(400 nodes, alpha = 10**(-6))
    # BEST ROSSLER RECONSTRUCTION OCCURED WITH SPECRAD = 1.185, WIN STRENGTH = 0.075, Sample rate = 0.01
    #(400 nodes, alpha = 10**(-6))
    
# TO TRY:
    # CHANGE THE TIME SERIES LENGTH
