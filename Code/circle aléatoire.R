#xor aléatoire
n=1000
s=0.4

#Construction d'un mélange de gaussiennes
u <- runif(n,0,1)
v <- runif(n,0,1)
w <- runif(n,0,1)
z <- runif(n,0,1)
x <- rnorm(n,0,1)
y <- rnorm(n,0,1)




x=cbind(u,v)
colnames(x)=c("Input","Output")    	
y <- 1*(u*u+v*v<2)   

1*(2*2+0.5*0.5<2)

couleur <- rep('red',n)
couleur[y==1]<-'blue'
plot(u,v, col=couleur)
title(main='points dans le cercle')

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
net.circle <- neuralnet(y~u+v,data, hidden=10,act.fct="logistic", threshold=0.01)

print(net.circle)
plot(net.circle)
prediction(net.circle)

#testons le RN sur des exemples
test<-  matrix(c(0,0,-1/2,sqrt(3)/2,1,-1,-1,-1),ncol=2,byrow=TRUE)
net.resultsc <-  compute(net.circle,as.data.frame(test))
print(net.resultsc$net.result)
