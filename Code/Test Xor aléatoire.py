# -*- coding: cp1252 -*-
import random
import math
import numpy as np
import PerceptronSparse as per
import time
import copy

a=np.loadtxt("XORSample.txt")
inputs=a[0:100,0:2]
outputs=a[0:100,2]
n=5000
errRiem=np.zeros((n+1,10))
errNat=np.zeros((n+1,10))
tRiem=np.zeros((n+1,10))
tNat=np.zeros((n+1,10))
for i in range(10):
    print("Itération "+str(i))
    res=per.perceptron([2,5,3,1],[3,2,1])
    res1=copy.deepcopy(res)
    w0=res.weights
    ins0=res.ins
    outs0=res.outs
    resNat=res.optim(inputs,outputs,0.001,0,fp=1.001,fm=1.001,display=5000,nMax=n)
    errNat[:,i]=resNat[1]
    tNat[:,i]=resNat[0]
    
    resRiem=res1.unitNatGradAl(inputs,outputs,0.001,0,fp=1.001,fm=1.001,display=5000,nMax=n)
    errRiem[:,i]=resRiem[1]
    tRiem[:,i]=resRiem[0]

np.savetxt("errRiem1.txt",errRiem,fmt='%d')
np.savetxt("tRiem1.txt",tRiem,fmt='%f')
np.savetxt("errNat1.txt",errNat,fmt='%d')
np.savetxt("tNat1.txt",tNat,fmt='%f')


