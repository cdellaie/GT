function y=layer(W,f,x)
    y=f(W*x);
endfunction

function y=sigm(x)
    y=(1+exp(-x))^(-1);
endfunction

W=zeros(2,2,10);
for i=1:10
    W(:,:,i)=ones(2,2);
end

// essaie avec un perceptron : pas de couche cachée
//génération des exemples ou étiquettes
N=100;  // nombre d'exemples
h=0.1; //pas d'apprentissage
//un modèle simple
X=grand(1,N,'nor',0,1);
T=3*X;
M=W(:,:,1);

for j=1:N
    x=X(i);
    y=layer(M,sigm,x*ones(2,1));
    M=M+h*x*(1-y)*y*(T(j)-y);
end


