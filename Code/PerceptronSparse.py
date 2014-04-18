# -*- coding: cp1252 -*-
import random
import math
import numpy as np


#Fonction d'activation
def activation(net):
    if net<-700:
        return 1/(1+math.exp(700))
    else:
        return 1/(1+math.exp(-net))
vactivation = np.vectorize(activation, otypes=[float])


#Fonction de perte Bernouilli
def pertBern(y,aOut):
    n=len(aOut)
    return np.array([(y-aOut[i])/(aOut[i]*(1-aOut[i])) for i in range(n)])

#Module de Fisher associ� � la perte de Bernouilli
def fishModBern(percept,jRate=None):
    if jRate is None:
        jRate=percept.transRate()
    nOut=len(jRate)
    nc=len(percept.n_couche)
    aOut=percept.couches[nc-1]
    mod=[]
    #Calculs pour les couches cach�es
    for k in range(nc-2):
        modCou=[]
        for i in range(percept.n_couche[k+1]):
            modCou.append(sum([jRate[j][k][i]**2/(aOut[j]*(1-aOut[j])) for j in range(nOut)]))
        mod.append(modCou)
    #Calcul pour la couche de sortie
    modCou=[1/(aOut[j]*(1-aOut[j])) for j in range(nOut)]
    mod.append(modCou)
    return mod

#Fonction de perte pour la classification 2
def pertClas2(y,aOut):
    n=len(aOut)
    som=sum([aOut[i]**2 for i in range(n)])
    return np.array([2*(i==y)/aOut[i]-2*aOut[i]/som for i in range(n)])

def fishModClas2(percept,jRate=None):
    if jRate is None:
        jRate=percept.transRate()
    nOut=len(jRate)
    nc=len(percept.n_couche)
    aOut=percept.couches[nc-1]
    S=np.sum(aOut**2)
    mod=[]
    #Calculs pour les couches cach�es
    for k in range(nc-2):
        modCou=[]
        for i in range(percept.n_couche[k+1]):
            A=4/S*sum([jRate[j][k][i]**2 for j in range(nOut)])
            B=4/S**2*sum([jRate[j][k][i]*aOut[j] for j in range(nOut)])**2
            modCou.append(A-B)
        mod.append(modCou)
    #Calcul pour la couche de sortie
    modCou=[4/S-4/S**2*aOut[j]**2 for j in range(nOut)]
    mod.append(modCou)
    return mod

class fonctionPerte:
    def __init__(self,nom):
        if nom=='Bernouilli':
            self.perte=pertBern
            self.fishMod=fishModBern
##            self.backMod=
        elif nom=='Clas2':
            self.perte=pertClas2
            self.fishMod=fishModClas2

#Fonction de perte pour la classification 1
def pertClas1(y,aOut):
    n=len(aOut)
    som=sum([math.exp(aOut[i]) for i in range(n)])
    return [(i==y)-math.exp(aOut[i])/som for i in range(n)]



                            
    

#Fonction pour initialiser les poids    
def iniWeights(perc):
    n_couche=perc.n_couche
    n_con=perc.n_con
    outs=perc.outs
    nc=len(n_couche)
    weights=[]
    for i in range(nc-1):
        weightc=np.zeros((n_couche[i]+1,n_couche[i+1]))
        biais=(i!=(nc-2))
        for j in range(n_couche[i]):
            weightc[j+1,outs[i][j]-biais]=np.random.normal(0,2,size=n_con[i])
        weightc[0,:]=-0.5*np.sum(weightc[1:,:], axis=0)
        weights.append(weightc)
    return weights

def iniCon(perc):
    nc=len(perc.n_couche)
    ins=[]
    outs=[]
    for i in range(nc-1):
        inc=[np.zeros(1,dtype='int') for j in range(perc.n_couche[i+1])]
        outc=[]
        biais=(i!=(nc-2))
        for j in range(perc.n_couche[i]):
            con=random.sample(np.arange(perc.n_couche[i+1])+biais,perc.n_con[i])
            outc.append(np.array(con))
            for k in con:
                inc[k-biais]=np.hstack((inc[k-biais],np.array([j+1])))
        ins.append(inc)
        outs.append(outc)
    return [ins,outs]


#Classe perceptron
class perceptron :
####Atributs :
    #   n_couche : liste du nombre de neurones pour chaque couche en excluant le biais
    #   n_con    : liste du nombre de connexions des neurones de chaque couche (�videmment sauf la derni�re)
    #   couches  : liste de couches, o� une couche est une liste de valeurs de neurones, avec le biais 1 comme premier �l�ment sauf pour la derni�re couche
    #   weights  : liste de matrices de poids par couche, de taille (n_couche[i]+1,n_couche[i+1])
    #   ins      : liste de liste de connexions incidentes par couche, ins[i][j] correspond au tableau des num�ros de neurones de la couche i incident sur le neurone j+1 (j pour la derni�re couche) de la couche i+1.
    #   outs     : liste de liste de connexions sortantes par couche, outs[i][j] correspond au tableau des num�ros des neurones de la couche i+1 vers lesquels le neurone j+1 (j pour la derni�re couche) de la couche i communique.
    def __init__ (self, n_couche,n_con=None,couches=None,weights=None) :
        self.n_couche=n_couche
        if n_con is None:
            self.n_con=n_couche[1:]
        else:
            self.n_con=n_con
        insOuts=iniCon(self)
        self.ins=insOuts[0]
        self.outs=insOuts[1]
        if weights is not None:
            self.weights=weights
        else:
            self.weights=iniWeights(self)
        if couches is not None:
            self.couches=couches
        else:
            self.couches=[0 for i in range(len(n_couche))]
            self.maj(np.zeros(n_couche[0]))

    
         
        
####M�thodes

    #Met � jour le r�seau de neurones en fonction de inputs et revoie la derni�re couche
    def maj(self,inputs):
        n=len(self.n_couche)
        self.couches[0]=np.hstack((np.ones(1),inputs))
        for i in range(n-1):
            net=np.dot(self.couches[i],self.weights[i])
            if i!=n-2:
                self.couches[i+1]=np.hstack((np.ones(1),vactivation(net)))
            else:
                self.couches[i+1]=vactivation(net)
        return self.couches[n-1]

    #Renvoie une liste de m�me taille que la derni�re couche (index�e par k_out)
    #o� chaque �l�ment est une liste de couches des taux de transferts J_k^k_out associ� au neurone k 
    def transRate(self):
        nc=len(self.n_couche)
        nOut=self.n_couche[nc-1]
        jOut=[]
        for kOut in range(nOut):
            auxJ=[np.zeros(self.n_couche[i]) for i in range (1,nc-1)]
            auxJ.append([i==kOut for i in range(nOut)])
            inc=self.ins[nc-2][kOut][1:]
            auxJ[nc-3][inc-1]=self.weights[nc-2][inc,kOut]*self.couches[nc-1][kOut]*(1-self.couches[nc-1][kOut])
            if nc>3:
                for iCouche in range(nc-4,-1,-1):
                    inc2=np.array([],dtype='int')
                    for k in inc:
                        inc2=np.hstack((inc2,self.ins[iCouche+1][k-1]))
                    inc2=np.unique(inc2)[1:]
                    a=auxJ[iCouche+1][inc-1]*self.couches[iCouche+2][inc]
                    a*=1-self.couches[iCouche+2][inc]
                    b=self.weights[iCouche+1][inc2,:]
                    b=b[:,inc-1]
                    b=b.transpose()
                    
                    auxJ[iCouche][inc2-1]=np.dot(a,b)
                    inc=inc2
            jOut.append(auxJ)
        return jOut

     #Fait une descente de gradient selon les entr�es inputss, sorties outs, le pas de descente pas et la fonction de perte fPerte
     #Retourne la somme des erreurs au carr� sur l'�chantillon
    def optIter(self,inputss,outs,pas,fPerte=pertClas2):
        nc=len(self.n_couche)
        nOut=self.n_couche[nc-1]
        ni=len(inputss)
        wTemp=iniWeights(self.n_couche,zero=True)
        somErr=0
        for i in range(ni):
            sortie=self.maj(inputss[i])
            jTrans=self.transRate()
            bOut=fPerte(outs[i],sortie)
            for j in range(nc-1):
                #Couches cach�es
                if j<nc-2:
                    b=[sum([jTrans[k][j][l]*bOut[k] for k in range(nOut)]) for l in range(self.n_couche[j+1])]
                    a=np.array(self.couches[j]).reshape(self.n_couche[j]+1,1)
                    ba=np.array(b)*(self.couches[j+1][1:])
                    ba*=np.ones(self.n_couche[j+1])-self.couches[j+1][1:]
                    ba=ba.reshape(1,self.n_couche[j+1])
                #Couche de sortie
                else:
                    b=bOut
                    a=np.array(self.couches[j]).reshape(self.n_couche[j]+1,1)
                    ba=np.array(b)*self.couches[j+1]
                    ba*=np.ones(self.n_couche[j+1])-self.couches[j+1]
                    ba=ba.reshape(1,self.n_couche[j+1])

                wTemp[j]+=np.dot(a,ba)
            somErr+=np.sum(np.array(bOut)**2)
        for j in range(nc-1):
            self.weights[j]=self.weights[j]+pas*wTemp[j]/ni
        return somErr

    def unitNatGrad(self,inputss,outs,pas,fPerte=fonctionPerte('Bernouilli')):
        nc=len(self.n_couche)
        nOut=self.n_couche[nc-1]
        ni=len(inputss)
        gTemp=[[np.zeros((1,len(self.ins[j][i]))) for i in range(self.n_couche[j+1])] for j in range(nc-1)]
        fTemp=[[np.zeros((len(self.ins[j][i]),len(self.ins[j][i]))) for i in range(self.n_couche[j+1])] for j in range(nc-1)]
        somErr=0
        for i in range(ni):
            sortie=self.maj(inputss[i])
            jTrans=self.transRate()
            fishMod=fPerte.fishMod(self,jTrans)
            bOut=fPerte.perte(outs[i],sortie)
            somErr+=np.sum(bOut**2)
            for j in range(nc-1):
                biais=(j!=(nc-2))
                nCon=self.n_con[j]
                for k in range(self.n_couche[j+1]):
                    inc=self.ins[j][k]
                    ak=self.couches[j+1][biais+k]
                    rak=ak*(1-ak)
                    couche=self.couches[j][inc].reshape(1,len(inc))
                    #Calcul de F non moyenn�
                    mat=np.dot(couche.transpose(),couche)
                    fTemp[j][k]+=(fishMod[j][k]*rak**2)*mat
                    #Calcul de G non moyenn�
                    bk=sum([jTrans[kOut][j][k]*bOut[kOut] for kOut in range(nOut)])
                    gTemp[j][k]+=rak*bk*couche
        for j in range(nc-1):
            for k in range(self.n_couche[j+1]):
                n=fTemp[j][k].shape[1]
                invF=np.linalg.inv(fTemp[j][k]+0.0001*np.diag([1 for i in range(n)]))
                dwk=np.dot(invF,gTemp[j][k].transpose())
                inc=self.ins[j][k]
                self.weights[j][inc,k]=(self.weights[j][inc,k]+(pas*dwk).transpose()).reshape(len(inc),)
        return somErr
                    

    def unitNatGradAl(self,inputss,outs,pas,eps,fPerte=fonctionPerte('Clas2'),fp=1.1,fm=1.01,display=1000,nMax=10000):
        ite=0
        err0=self.unitNatGrad(inputss,outs,pas,fPerte)
        err1=err0
        err2=err1+1
        while err1>eps and ite<nMax:
            if ite%display==0:
                print(ite)
                print(err1)
                print(pas)
                print(self.couches[len(self.n_couche)-1][0])
            #Ajustement du pas (� bien calibrer pour que �a fonctionne)
            if err2<err1:
                pas/=fm
            else:
                pas*=fp
            err2=err1
            err1=self.unitNatGrad(inputss,outs,pas,fPerte)
            ite+=1
        return [err0,err1]
    
    
    #Fait une descente de gradient jusqu'� nMax it�rations � moins que la somme des carr�s de l'�chantillon soit inf�rieur � eps
    #Renvoie l'erreur au d�part de l'optimisation et l'erreur de fin
    def optim(self,inputss,outs,pas,eps,fPerte=pertClas2,nMax=10000):
        ite=0
        err0=self.optIter(inputss,outs,pas,fPerte)
        err1=err0
        err2=err1+1
        while err1>eps and ite<nMax:
            if ite%1000==0:
                print(ite)
                print(err1)
                print(pas)
                
            #Ajustement du pas (� bien calibrer pour que �a fonctionne)
##            if err2<err1:
##                pas/=1.1
##            else:
##                pas*=1.01
            err2=err1
            err1=self.optIter(inputss,outs,pas,fPerte)
            ite+=1
        return [err0,err1]


##test=perceptron([2,10,3,2],[5,3,2])
##print(test.weights[0])
##print(np.sum(test.weights[0],0))
####print(test.ins[2])
####print(test.outs[2])
####print(test.maj([1,0,1,0,1,0,1,1,1,0]))
####print(test.transRate()[0])
####print(test.transRate()[1])
####
##inputs = [[0,0],[0,1],[1,0],[1,1]]
##desiredOuts = [0,1,1,0]
####
####print(test.unitNatGrad(inputs,desiredOuts,0.001))
##print(test.unitNatGradAl(inputs,desiredOuts,0.001,0.00000001))
##print(test.maj([1,0]))
##print(test.maj([0,0]))
##print(test.maj([1,1]))
##print(test.maj([0,1]))
