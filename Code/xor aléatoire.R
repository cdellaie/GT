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
colnames(x)=c("Input","Output")    	
y <- 1*(U<0.5)   

couleur <- rep('red',n)
couleur[y==1]<-'blue'
plot(u,v, col=couleur)
title(main='Version aléatoire du Xor')

#méthodes par noyaux
library(kernlab)

svp <- ksvm(x,y,type="C-svc",kernel="rbfdot")
plot(svp,data=x)
title("SVM classification plot")

svp <- ksvm(x,y,type="C-svc",kernel="vanilladot")
plot(svp,data=x)

svp <- ksvm(x,y,type="C-svc",kernel=polydot(degree=2))
plot(svp,data=x)

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













