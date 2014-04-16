//fonction d'activation
function y=sigm(u)
    y=1/(1+exp(u));
endfunction

//ondelettes
function y=phi(u)
    y=1*((u<0.5)&(u>0))-1*((u>0.5)&(u<1));
endfunction
t=(-1):0.01:2;

scf(0)
for j=1:2
    for k=0:1
        plot2d2(t,sqrt(2)^j*phi((2^j)*t-k),j+k);
    end
end
title('Ondelettes');


