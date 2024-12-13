import numpy as np
import matplotlib.pyplot as plt
import random as rand
import math
import time


def embedding_plot(TS,lag,title='na',to_also_plot=['na'],save_name='na'):
    #if len(TS) > 1000:
    #    cap = 1000
    #else:
    cap = len(TS) - lag - 1
    fig,axs=plt.subplots(1,1,figsize=(8,8))
    axs.plot(TS[:cap],TS[lag:cap+lag],linewidth=1,alpha=0.5,color='k')
    if to_also_plot[0] != 'na':
        axs.plot(to_also_plot[:cap],to_also_plot[lag:cap+lag],linewidth=1,alpha=0.2,color='b')
    if title != 'na':
        axs.set_title(title,fontsize=15)
        axs.plot([TS[0]],[TS[lag]],'X',color='r')
        if save_name != 'na':
            save_name += '_deets'
    else:
        axs.set_xticks([])
        axs.set_yticks([])
    axs.set_xlabel('$x(t)$',fontsize=60)
    axs.set_ylabel('$x(t+\\tau)$',fontsize=60)
    plt.tight_layout()
    if save_name != 'na':
        plt.savefig('recon_pics/'+save_name+'.png')
        print('SAVED A RECON')
    plt.show()
    
def Lorenz(n,dt=0.0005,Sr=0.02,sigma=10,rho=28,beta=8/3,scale=True,transient=5000,just_x=True,start_state=['na']):
    if start_state[0] == 'na':
        x = np.array([rand.gauss(0,1) for i in range(3)])
    else:
        x = start_state
    X = []
    for i in range(transient+int(n*Sr/dt)):
        dx = sigma*(x[1]-x[0]); dy = x[0]*(rho-x[2]) - x[1]; dz = x[0]*x[1] - beta*x[2]
        x += dt*np.array([dx,dy,dz])
        X.append(x/1)
    X = X[transient:]
    if just_x:
        out = np.array([X[i][0]/7.34 for i in range(len(X)) if i%(Sr/dt)==0])
    else:
        out = {'x':np.array([X[i][0]/7.34 for i in range(len(X)) if i%(Sr/dt)==0]),
               'X':np.array([X[i] for i in range(len(X)) if i%(Sr/dt)==0])}
    return out


def Rossler(n,dt=0.01,Sr=0.2,a=0.1,b=0.1,c=14,scale=True,transient=5000,just_x=True,start_state=['na']):
    Sr = 0.05 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
    if start_state[0] == 'na':
        x = np.array([rand.gauss(0,1) for i in range(3)])
    else:
        x = start_state
    X = []
    for i in range(transient+int(n*Sr/dt)):
        dx = -x[1]-x[2]; dy = x[0] + a*x[1]; dz = b + x[2]*(x[0]-c)
        x += dt*np.array([dx,dy,dz])
        X.append(x/1)
    X = X[transient:]
    if just_x:
        out = np.array([X[i][0]/9.17 for i in range(len(X)) if i%(Sr/dt)==0])
    else:
        out = {'x':np.array([X[i][0]/9.17 for i in range(len(X)) if i%(Sr/dt)==0]),
               'X':np.array([X[i] for i in range(len(X)) if i%(Sr/dt)==0])}
    return out


def Thomas(n,dt=0.01,Sr=0.2,b=0.208186,scale=True,transient=5000,just_x=True,start_state=['na']):
    Sr = 0.05 #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
    if start_state[0] == 'na':
        x = np.array([1.1,1.1,-0.01])+np.array([rand.gauss(0,0.001) for i in range(3)])
    else:
        x = start_state
    X = [];
    for i in range(transient+int(n*Sr/dt)):
        dx = math.sin(x[1])-b*x[0]; dy = math.sin(x[2]) - b*x[1]; dz = math.sin(x[0]) - b*x[2]
        x += dt*np.array([dx,dy,dz])
        X.append(x/1)
    X = X[transient:]
    if just_x:
        out = np.array([X[i][0]-1.87 for i in range(len(X)) if i%(Sr/dt)==0])
    else:
        out = {'x':np.array([X[i][0] for i in range(len(X)) if i%(Sr/dt)==0]),
               'X':np.array([X[i] for i in range(len(X)) if i%(Sr/dt)==0])}
    return out

def Langford(n,dt=0.01,a=0.95,b=0.7,c=0.6,d=3.5,e=0.25,f=0.1):
    x = np.array([0.1,1,0.01]); X = []
    for i in range(500+n):
        dx = (x[2]-b)*x[0]-d*x[1]; dy = d*x[0] + (x[2]-b)*x[1]
        dz = c + a*x[2] - (x[2]**3)/3 -(x[0]**2 + x[1]**2)*(1+e*x[2]) + f*x[2]*x[0]**3
        x += dt*np.array([dx,dy,dz])
        X.append(x[1]/0.27)
        #time.sleep(1)
    return np.array(X[500:])

def Lorenz_step(x,dt=0.0005,sigma=10,rho=28,beta=8/3):
    dx = sigma*(x[1]-x[0]); dy = x[0]*(rho-x[2]) - x[1]; dz = x[0]*x[1] - beta*x[2]
    x += dt*np.array([dx,dy,dz])
    return x


def Rossler_step(x,dt=0.01,a=0.1,b=0.1,c=14):
    dx = -x[1]-x[2]; dy = x[0] + a*x[1]; dz = b + x[2]*(x[0]-c)
    x += dt*np.array([dx,dy,dz])
    return x


def Thomas_step(x,dt=0.01,b=0.208186,scale=True):
    dx = math.sin(x[1])-b*x[0]; dy = math.sin(x[2]) - b*x[1]; dz = math.sin(x[0]) - b*x[2]
    x += dt*np.array([dx,dy,dz])
    return x
        
def lorenz_inc_function(x,sigma=10,rho=28,beta=8/3):
    return [sigma*(x[1]-x[0]),x[0]*(rho-x[2]) - x[1], x[0]*x[1] - beta*x[2]]

def rossler_inc_function(x,a=0.1,b=0.1,c=14):
    return [-x[1]-x[2],x[0] + a*x[1], b + x[2]*(x[0]-c)]

def thomas_inc_function(x,b=0.208186):
    return [math.sin(x[1])-b*x[0], math.sin(x[2]) - b*x[1], math.sin(x[0]) - b*x[2]]

def Lorenz_runge_step(x,dt=0.001,sigma=10,rho=28,beta=8/3):
    k1 = lorenz_inc_function(x,sigma=sigma,rho=rho,beta=beta)
    k2 = lorenz_inc_function([x[0]+0.5*dt*k1[0],x[1]+0.5*dt*k1[1],x[2]+0.5*dt*k1[2]],sigma=sigma,rho=rho,beta=beta)
    k3 = lorenz_inc_function([x[0]+0.5*dt*k2[0],x[1]+0.5*dt*k2[1],x[2]+0.5*dt*k2[2]],sigma=sigma,rho=rho,beta=beta)
    k4 = lorenz_inc_function([x[0]+0.5*dt*k3[0],x[1]+0.5*dt*k3[1],x[2]+0.5*dt*k3[2]],sigma=sigma,rho=rho,beta=beta)
    return [x[0]+dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,x[1]+dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6,
            x[2]+dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6]

def Rossler_runge_step(x,dt=0.001,a=0.1,b=0.1,c=14):
    k1 = rossler_inc_function(x,a=a,b=b,c=c)
    k2 = rossler_inc_function([x[0]+0.5*dt*k1[0],x[1]+0.5*dt*k1[1],x[2]+0.5*dt*k1[2]],a=a,b=b,c=c)
    k3 = rossler_inc_function([x[0]+0.5*dt*k2[0],x[1]+0.5*dt*k2[1],x[2]+0.5*dt*k2[2]],a=a,b=b,c=c)
    k4 = rossler_inc_function([x[0]+0.5*dt*k3[0],x[1]+0.5*dt*k3[1],x[2]+0.5*dt*k3[2]],a=a,b=b,c=c)
    return [x[0]+dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,x[1]+dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6,
            x[2]+dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6]

def Thomas_runge_step(x,dt=0.001,b=0.2):
    k1 = thomas_inc_function(x,b=b)
    k2 = thomas_inc_function([x[0]+0.5*dt*k1[0],x[1]+0.5*dt*k1[1],x[2]+0.5*dt*k1[2]],b=b)
    k3 = thomas_inc_function([x[0]+0.5*dt*k2[0],x[1]+0.5*dt*k2[1],x[2]+0.5*dt*k2[2]],b=b)
    k4 = thomas_inc_function([x[0]+0.5*dt*k3[0],x[1]+0.5*dt*k3[1],x[2]+0.5*dt*k3[2]],b=b)
    return [x[0]+dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6,x[1]+dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6,
            x[2]+dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6]

def runge_Lorenz(n,dt=0.001,Sr=0.02,sigma=10,rho=28,beta=8/3,scale=True,transient=5000,just_x=True,start_state=['na']):
    if start_state[0] == 'na':
        x = np.array([rand.gauss(0,1) for i in range(3)])
    else:
        x = start_state
    X = []
    for i in range(transient+int(n*Sr/dt)):
        #print(Lorenz_runge_step(x,sigma=sigma,rho=rho,beta=beta))
        x = np.array(Lorenz_runge_step(x,dt=dt,sigma=sigma,rho=rho,beta=beta))
        X.append(x/1)
    X = X[transient:]
    if just_x:
        out = np.array([X[i][0]/7.34 for i in range(len(X)) if i%(Sr/dt)==0])
    else:
        out = {'x':np.array([X[i][0]/7.34 for i in range(len(X)) if i%(Sr/dt)==0]),
               'X':np.array([X[i] for i in range(len(X)) if i%(Sr/dt)==0])}
    return out

def runge_Rossler(n,dt=0.01,Sr=0.2,a=0.1,b=0.1,c=14,scale=True,transient=5000,just_x=True,start_state=['na']):
    if start_state[0] == 'na':
        x = np.array([rand.gauss(0,1) for i in range(3)])
        x = 10*x/np.linalg.norm(x)
    else:
        x = start_state
    X = []
    for i in range(transient+int(n*Sr/dt)):
        #print(Lorenz_runge_step(x,sigma=sigma,rho=rho,beta=beta))
        x = np.array(Rossler_runge_step(x,dt=dt,a=a,b=b,c=c))
        X.append(x/1)
    X = X[transient:]
    if just_x:
        out = np.array([X[i][0]/10.2 for i in range(len(X)) if i%(Sr/dt)==0])
    else:
        out = {'x':np.array([X[i][0]/10.2 for i in range(len(X)) if i%(Sr/dt)==0]),
               'X':np.array([X[i] for i in range(len(X)) if i%(Sr/dt)==0])}
    return out

def runge_Thomas(n,dt=0.01,Sr=0.2,b=0.208186,scale=True,transient=5000,just_x=True,start_state=['na']):
    if start_state[0] == 'na':
        x = np.array([1.1,1.1,-0.01])+np.array([rand.gauss(0,0.001) for i in range(3)])
    else:
        x = start_state
    X = []
    for i in range(transient+int(n*Sr/dt)):
        #print(Lorenz_runge_step(x,sigma=sigma,rho=rho,beta=beta))
        x = np.array(Thomas_runge_step(x,dt=dt,b=b))
        X.append(x/1)
    X = X[transient:]
    if just_x:
        out = np.array([X[i][0]  for i in range(len(X)) if i%(Sr/dt)==0])
        out -= np.mean(out)
        out = out/np.std(out)
        
    else:
        out = {'x':np.array([X[i][0]-1.87 for i in range(len(X)) if i%(Sr/dt)==0]),
               'X':np.array([X[i] for i in range(len(X)) if i%(Sr/dt)==0])}
    return out


def box_it_up(TS1,TS2,cuts): #Takes time series, returns box coordinates of each point in embedding space
    coords = []; 
    counts = [[10**(-3) for i in range(len(cuts)+1)] for i in range(len(cuts)+1)]
    #print('length of TS is :' +str(len(TS)))
    for t in range(len(TS1)):
        p = [len([c for c in cuts if TS1[t] < c]),len([c for c in cuts if TS2[t] < c])]
        counts[p[0]][p[1]] += 1
        coords.append(p)
    counts = np.array(counts)/(len(TS1))
    return counts

def line_it_up(TS,cuts):
    counts = [10**(-3) for i in range(len(cuts)+1)] 
    for t in range(len(TS)):
        p = len([c for c in cuts if TS[t] < c])
        counts[p] += 1
    counts = np.array(counts)/(len(TS))
    return counts


def mutual_info(TS1,TS2,nbins=51):
    cuts = np.linspace(min(TS1),max(TS1),nbins)
    pXY = box_it_up(TS1,TS2,cuts)
    pX = line_it_up(TS1,cuts)
    I = 0
    for x in range(len(cuts)+1):
        for y in range(len(cuts)+1):
            I += pXY[x,y]*math.log(pXY[x,y]/(pX[x]*pX[y]))
    return I 

def first_min_of_MI(TS):
    lags = []; infos = []; lag = 1; keep_going = True; counter = 'na'
    while keep_going:
        #info = abs(np.corrcoef(TS[lag:],TS[:-lag])[0,1]) # not actually mutal information 
        info = mutual_info(TS[lag:],TS[:-lag])
        if len(infos) >= 1:
            if info > infos[-1] and counter =='na':
                counter = 0 ; first_min = lag
        infos.append(info) ; lags.append(lag)
        if counter != 'na':
            counter += 1 
        if counter == 10:
            keep_going = False
        lag += 1
        print(lag)
    plt.plot(lags,infos)
    plt.show()
    return first_min

#TS = Thomas(20000)
#print(np.mean(np.abs(TS)))
#print(np.mean(TS))
#embedding_plot(TS, 10)
