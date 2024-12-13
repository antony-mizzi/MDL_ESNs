import numpy as np 
#import reservoirs as res 
import maps_and_systems as mapnsys
import random as rand
import time
import math




def gram_schmidt(A): 
    n = len(A); omegas = []
    for j in range(n):
        for k in range(j):
            A[j] = A[j] - np.dot(A[k],A[j])*A[k]
        omegas.append(np.linalg.norm(A[j]))
        A[j] = A[j]/np.linalg.norm(A[j])
    return A,omegas


def difference_eq(V,x0,tau,epsilon=10**(-8),dt=0.01):
    U = []
    #print('going into difference equation, this is x0: '+str(x0))
    #print('this is V: '+str(V))
    #print('')
    for v in V:
        #print('for the pertubation '+str(V)+':')
        x = x0+epsilon*v
        #print('starting with x = '+str(x))
        for t in range(tau):
            x = mapnsys.Lorenz_step(x,dt=dt)
            #print(x)
            #time.sleep(1)
        U.append(x)
        #print('')
    #print('U is '+str(U))
    xfinal = x0
    for t in range(tau):
        xfinal = mapnsys.Lorenz_step(xfinal,dt=dt)
    #print('xfinal is '+str(xfinal))
    #print('differences are: '+str([(u-xfinal) for u in U]))
    #time.sleep(10)
    return [(u-xfinal)/epsilon for u in U], xfinal

def get_spectrum(n,reps=2000,epsilon=10**(-2),tau=6,dt=0.01):
    U = np.diag(np.ones(n)).tolist(); Omegas = {i:[] for i in range(n)}
    #x = np.array([1-2*rand.random() for i in range(n)])

    x = mapnsys.Lorenz(1000,just_x=False)['X'][-1,:]
    for r in range(reps):
        #print('r = '+str(r))
        #print(U)
        #print('')
        V,omegas = gram_schmidt(U)
        for i in range(n):
            Omegas[i].append(omegas[i])
        U,x = difference_eq(V, x, tau, dt=dt)
    exponents = {i:np.sum([math.log(w) for w in Omegas[i][1000:]])/(len(Omegas[i][1000:])*dt*tau) for i in Omegas.keys()}
    return exponents

def bootstrap(X):
    means = [np.mean(rand.choices(X,k=len(X))) for i in range(100)]
    return np.std(means)
    
#n=3; lyaps = []
#for go in range(100):
#    # for L: 0.0005; for R: 0.0005
#    exponents = get_spectrum(n,dt=0.0005)
#    lyaps.append(exponents[0])
#    print('go '+str(go)+' LE = '+str(round(exponents[0],3)))
#    if go % 10 == 0 and go !=0:
#        mean = np.mean(lyaps); err = bootstrap(lyaps)
#        print('so far, '+str(round(mean-3*err,4))+' < LE < '+str(round(mean+3*err,4))+ '( est = '+str(round(mean,4))+')')
#        time.sleep(15)
        


#mean = np.mean(lyaps); err = bootstrap(lyaps)
#print(str(round(mean-3*err,4))+' < LE < '+str(round(mean+3*err,4))+ '( est = '+str(round(mean,4))+')')
#print(str(round(3*err,4)))
#print(1/mean)

#Lor est: 
#Ros est: 
#Tho est: 0.0382