import math
import torch

def pk(S1):
    m = torch.mean(S1)
    return m

def pkk(S1,S2):
    N = S1.shape[0]
    s = 1/N*torch.sum(S1*S2)
    return s
    
def MI(S1, S2):
    mi = 0
    K1 = S1.shape[1]
    K2 = S2.shape[1]
    for k in range(K1):
        for km in range(K2):
            in_log = pkk(S1[:,k],S2[:,km])/(pk(S1[:,k])*pk(S2[:,km]))
            if in_log > 0:
                mi += pkk(S1[:,k],S2[:,km])*math.log(in_log)
            #else:
             #   print(f'could not compute MI, in_log value:{in_log}')
    return mi
            
def NMI(S1, S2):
    nmi = 2*MI(S1,S2)/(MI(S1,S1) + MI(S2,S2))
    return nmi