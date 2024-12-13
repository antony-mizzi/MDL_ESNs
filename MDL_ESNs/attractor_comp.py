import numpy as np
import math 
import matplotlib.pyplot as plt
import maps_and_systems as mapnsys
import time


class box:
    def __init__(self):
        self.box = []
    def split(self,dim):
        self.box.append([box() for i in range(int(2**dim))])
            
    
class grid:
    def __init__(self,dim,depth):
        self.boxes = box()
        for level in range(1,depth):
            self.boxes.split(dim)


def box_it_up(TS,delays,cuts): #Takes time series, returns box coordinates of each point in embedding space
    coords = []; 
    counts = [[10**(-3) for i in range(len(cuts)+1)] for i in range(len(cuts)+1)]
    #print('length of TS is :' +str(len(TS)))
    for t in range(max(delays),len(TS)):
        p = [len([c for c in cuts if TS[t-delays[i]] < c]) for i in range(len(delays))]
        counts[p[0]][p[1]] += 1
        coords.append(p)
    counts = np.array(counts)/(len(TS)-max(delays))
    return coords, cuts, counts

def Kul_div(C1,C2):
    width = C1.shape[0]
    div = 0
    for i in range(width):
        for j in range(width):
            div -= C2[i,j]*math.log(C1[i,j]/C2[i,j])
    return div

def compare(TS1,TS2,Ncuts=51,delays=[0,8]):
    std = np.std(TS2)
    cuts = np.linspace(-5*std,5*std,Ncuts).tolist()
    coords,cuts,counts1 = box_it_up(TS1, delays, cuts)
    coords,cuts,counts2 = box_it_up(TS2, delays, cuts)
    return Kul_div(counts1, counts2)

def quicktest():
    TS = mapnsys.Lorenz(30000)
    width = 4; Ncuts = 101
    coords,cuts,counts = box_it_up(TS, [0,4], Ncuts=Ncuts, width=width)
    X = [cuts[c[0]] for c in coords]; Y = [cuts[c[1]] for c in coords]
    plt.plot(X,Y,'o',markersize=1,alpha=0.1)
    plt.show()
    for rho in [28,30,35,45,80]:
        TS = mapnsys.Lorenz(30000,rho=rho)
        width = 4; Ncuts = 101
        coords,cuts,counts2 = box_it_up(TS, [0,4], Ncuts=Ncuts, width=width)
        X = [cuts[c[0]] for c in coords]; Y = [cuts[c[1]] for c in coords]
        plt.plot(X,Y,'o',markersize=1,alpha=0.1,color='k')
        plt.show()
        print(Kul_div(counts, counts2))
