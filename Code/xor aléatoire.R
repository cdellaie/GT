# setwd("C:/Users/Clément/Documents/GitHub/GT/Code")
#.libPaths("C:/Users/Clément/Documents/R-library")
install.packages("kernlab")
install.packages("neuralnet")

#xor aléatoire
n=1000
s=0.4

#Construction d'un mélange de gaussiennes
U <- runif(n,0,1)
V <- runif(n,0,1)
W <- runif(n,0,1)
X <- rnorm(n,0,1)
Y <- rnorm(n,0,1)

u <- 1*(U<0.5)*(1*(V<0.5)*(s*X+1)+1*(V>0.5)*(s*X-1)) +
		1*(U>0.5)*(1*(W<0.5)*(s*X-1)+1*(W>0.5)*(s*X+1))
v <- 1*(U<0.5)*(1*(V<0.5)*(s*Y+1)+1*(V>0.5)*(s*Y-1)) +
		1*(U>0.5)*(1*(W<0.5)*(s*Y+1)+1*(W>0.5)*(s*Y-1))

x=cbind(u,v)
y <- 1*(U<0.5)
x=cbind(x,y)
colnames(x)=c("X","Y","Output")    	
write.table(x,"XORSample.txt",row.names=F,col.names=F)


couleur <- rep('red',n)
couleur[y==1]<-'blue'
plot(u[1:100],v[1:100], col=couleur[1:100])
title(main='Version aléatoire du Xor')

#méthodes par noyaux
library(kernlab)

svp <- ksvm(x,y,type="C-svc",kernel="rbfdot")
ypred=predict(svp,x[1:100,1:2])
sum(ypred==y[1:100])
plot(svp,data=x[1:100,1:2])
plot(svp,data=x)
title("SVM classification plot")

svp <- ksvm(x,y,type="C-svc",kernel="vanilladot")
plot(svp,data=x)

svp <- ksvm(x[1:100,1:2],y[1:100],type="C-svc",kernel=polydot(degree=2))
ypred=predict(svp,x[1:100,1:2])
sum(ypred==y[1:100])
plot(svp,data=x[1:100,1:2])

#RN
#Régression logistique sur le xor aléatoire

library("neuralnet")

data=cbind(x,y)
net.xor <- neuralnet(y~u+v,data, hidden=10,act.fct="logistic", threshold=0.01)
print(net.xor)
plot(net.xor)
prediction(net.xor)

#testons le RN sur des exemples
test<-  matrix(c(1,1,-1,1,1,-1,-1,-1),ncol=2,byrow=TRUE)
net.results <-  compute(net.xor,as.data.frame(test))
ls(net.results)
print(net.results$net.result)




####Résultats python

nIter=5001

errRiem=read.table("errRiem1.txt")
errOrdR=rowMeans(errRiem)

tRiem=as.matrix(read.table("tRiem1.txt"))
tRiem=tRiem-rep(tRiem[1,],nIter)
tAbsR=rowMeans(tRiem)

errNat=read.table("errNat1.txt")
errOrdN=rowMeans(errNat)

tNat=as.matrix(read.table("tNat1.txt"))
tNat=tNat-rep(tNat[1,],nIter)
tAbsN=rowMeans(tNat)

ymax=max(cbind(errOrdN,errOrdR))
ymin=min(cbind(errOrdN,errOrdR))

plot(tAbsR,errOrdR,type="l",col="red",ylim=c(ymin,ymax))
lines(tAbsN,errOrdN,col="blue")



