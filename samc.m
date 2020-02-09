tic
Gamma=[0,1,0,0,0;0.5,0,0.5,0,0;0,0.5,0,0.5,0;0,0,0.5,0,0.5;0,0,0,1,0];
pi=[0.2,0.2,0.2,0.2,0.2];
N=10000000;
L=1;
zeta=zeros(5,1);
Tfrac=[1.4,1.4065,1.413,1.4195,1.426];
Currentx=unidrnd(10,20,20)-1;
beta=0.8;
t0=40000;
for t=1:N
    j=find(cumsum(Gamma(L,:))>rand,1,'first');
    Currentu=CalculateU(Currentx);
    q_jx=exp(-Currentu*Tfrac(j));
    q_lx=exp(-Currentu*Tfrac(L));
    p_jx=pi(j)/exp(zeta(j))*q_jx;
    p_lx=pi(L)/exp(zeta(L))*q_lx;
    temp=min(1,Gamma(j,L)/Gamma(L,j)*p_jx/p_lx);
    t_rand=rand;
    L=(temp>t_rand)*j+(temp<=t_rand)*L;
    tempx=Currentx;
    si=unidrnd(400);
    s=tempx(si);
    a=unidrnd(9)-1;
    s_new=(a<s)*a+(a>=s)*(a+1);
    tempx(si)=s_new;
    tempu=CalculateU(tempx);
    MHratio=min(1,exp(-tempu*Tfrac(L))/exp(-Currentu*Tfrac(L)));
    if(MHratio>rand)
        Currentx=tempx; 
    end
    gainfactor=eye(5);
    for i=1:5
        gainfactor(i,i)=min(pi(i),t^(-beta))*(t<=t0)+(t>t0)*min(pi(i),(t-t0+t0^beta)^(-1));
    end
    zeta=zeta+gainfactor*[(L==1)/pi(1),(L==2)/pi(2),(L==3)/pi(3),(L==4)/pi(4),(L==5)/pi(5)]';
    zeta=zeta-zeta(1);
end
toc