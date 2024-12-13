import numpy as np 
import math
import time

def stay_over_1(x):
    return max([1,x])

def order_of_mag(x):
    return max([abs(x),1/abs(x)])

def stay_positive(element):
    return max([10**(-6),element])
    
def lstar_outer(coefs,precis):
    order_func = np.vectorize(order_of_mag)
    bigl1 = lstar_inner(2*np.ceil(np.abs(coefs)/np.abs(precis)))
    bigl2 = lstar_inner(np.ceil(np.abs(2*np.log(order_func(coefs)))))
    return bigl1 + bigl2

def lstar_inner(x):
    x = np.abs(x)
    bigl = math.log(2.865)
    x = 1 + np.ceil(x) 
    vfunc = np.vectorize(stay_over_1)
    while np.max(x) > 1:
        x = np.log(vfunc(x))
        bigl += np.sum(x)
    return bigl

def newton_estimate_deltas(Q,variance):
    k = (Q.shape[0])
    if k == 1:
        output = np.array([math.sqrt(variance/Q[0][0])])
    else:
        Vfunc = np.vectorize(stay_positive)
        V = np.array([math.sqrt(variance/Q[i][i]) for i in range(k)])
        V = Vfunc(V)
        best_approx = [V,10**6]; go = 0; X = []
        V_recip = np.reciprocal(V)
        while go < 100:    
            F = np.matmul(Q,V)-variance*V_recip
            J = Q + variance * np.diag(np.square(V_recip))
            V = V - np.matmul(np.linalg.inv(J),F)
            V = Vfunc(V) # keeps deltas above 10 ^ -6
            V_recip = np.reciprocal(V)
            MAE = np.mean(np.abs(np.matmul(Q,V)-variance*V_recip))
            X.append(math.log10(MAE))
            if MAE < best_approx[1]:
                best_approx = [V,MAE]
            go += 1
            if MAE < 10**(-4):#0.001:
                go = 100
        output = best_approx[0]
        #plt.title('k = '+str(k))
        #plt.plot(X)
        #plt.show()
        #time.sleep(1)
    return output

def D_length(error,k,V,coefs,gamma=32,Y=None):
    n = len(error)
    variance = np.sum(np.square(error))/n
    Q = np.matmul(np.transpose(V),V)
    #print('there are '+str(k)+' coeffiecents')
    ####################################################################################3
    ## note that this is not quite accurate since Y has not been normalised but V and coefs have
    #temp = np.linalg.inv(np.matmul(np.transpose(V),V))
    #R = np.matmul(np.matmul(np.matmul(temp,temp),np.transpose(V)),Y)
    #numerator = np.dot(np.abs(coefs),R)
    #print('numerator = '+str(numerator))
    #denominator = np.matmul(np.matmul(np.transpose(R),Q),R)
    #print('denom = '+str(denominator))
    #print('In theory the best penalty is '+str(numerator/denominator))
    #print('')
    #time.sleep(2)
    #####################################################################################
    deltas = newton_estimate_deltas(Q, variance)
    return 0.5*n*(1+math.log(2*math.pi*variance)) + lstar_outer(coefs,deltas)

