clear all, close all, clc;
%% Potts model:Canonical ensemble simulation
tic
N1 = 20000000;
label = 1;
T = [1.4,1.4065,1.413,1.4195,1.426];
%{
trans = [0,1,0,0,0;...
        0.5,0,0.5,0,0;...
        0,0.5,0,0.5,0;...
        0,0,0.5,0,0.5;...
        0,0,0,1,0];
%}
zeta = [0,0,0,0,0];
pi = [0.2,0.2,0.2,0.2,0.2];
node = zeros(N1,5);
s_u1 = zeros(5,801);
t_u1 = zeros(20,20);
t_u2 = zeros(20,20);
t_u3 = zeros(20,20);
t_u4 = zeros(20,20);
t_u5 = zeros(20,20);
    
%% part 1: Self-adjusted mixture sampling

cur_k = unidrnd(10,20,20);
cur_u = 0;
%Calculate u(x)  
for m = 1: 19
    for n = 1: 19
        if(cur_k(m,n) == cur_k(m+1, n))
            cur_u = cur_u-1;    
        end 
        if(cur_k(m,n) == cur_k(m, n+1))
            cur_u = cur_u-1;
        end
    end
end

for m = 1: 19
    if(cur_k(m,20) == cur_k(m+1,20))
        cur_u = cur_u-1;
    end
    if(cur_k(m,20) == cur_k(m,1))
        cur_u = cur_u-1;
    end
    if(cur_k(20,m) == cur_k(20,m+1))
        cur_u = cur_u-1;
    end
    if(cur_k(20,m) == cur_k(1,m))
        cur_u = cur_u-1;
    end
end

if(cur_k(20,20) == cur_k(1,20))
    cur_u = cur_u-1;
end
if(cur_k(20,20) == cur_k(20,1))
    cur_u = cur_u-1;
end    

%Iteration to convergence
for i = 1: N1
    
    if(rem(i-1,400) == 0)
        
        %Global jump
        w = zeros(1,5);
        sum_w = 0;
        for m = 1:5
            sum_w = sum_w+pi(m)*exp(-zeta(m))*exp(-cur_u*(T(m)-1.4));
        end
        for j = 1:5 
            w(j) = pi(j)*exp(-zeta(j))*exp(-cur_u*(T(j)-1.4))/sum_w;
        end
        label = find(cumsum(w)>rand,1,'first');
        %{
        %New label
        next_label = find(cumsum(trans(label,:))>rand,1,'first');

        %Local jump: Determine whether to accept new label
        q_jx = exp(-cur_u*T(next_label));
        p_jx = pi(next_label)/exp(zeta(next_label))*q_jx;
        q_lx = exp(-cur_u*T(label));
        p_lx = pi(label)/exp(zeta(label))*q_lx;
        temp = min(1,trans(next_label,label)/trans(label,next_label)*p_jx/p_lx);
        if(temp > rand)
            label = next_label;
        end
        %}
    end
    
    %Update u
    %Change the value of one point in turn
    r_x = rem(floor((i-1)/20),20)+1;
    r_y = rem(i-1,20)+1;
    
    %Record the value of the points around the change point
    round_u = zeros(4);
    if(r_x == 1)
        round_u(1) = cur_k(20,r_y);
    else
        round_u(1) = cur_k(r_x-1,r_y);
    end
    if(r_x == 20)
        round_u(2) = cur_k(1,r_y);
    else
        round_u(2) = cur_k(r_x+1,r_y);
    end
    if(r_y == 1)
        round_u(3) = cur_k(r_x,20);
    else
        round_u(3) = cur_k(r_x,r_y-1);
    end
    if(r_y == 20)
        round_u(4) = cur_k(r_x,1);
    else
        round_u(4) = cur_k(r_x,r_y+1);
    end

    %Calculate the probability of moving to other values
    u1 = zeros(1,10);
    pi_u1 = zeros(1,10);
    sum_pi = 0;
    for m = 1: 10
        for n = 1: 4
            if(m == round_u(n))
                u1(m) = u1(m)-1;
            end
        end
        pi_u1(m) = exp(-u1(m)*T(label));
        sum_pi = sum_pi+pi_u1(m);
    end
    pi_u1 = pi_u1/sum_pi;
    
    %Update u(x)
    u_x = find(cumsum(pi_u1)>rand,1,'first');
    for k = 1: 4
        if(u_x == round_u(k))
            cur_u = cur_u-1;
        end
        if(cur_k(r_x,r_y) == round_u(k))
            cur_u = cur_u+1;
        end
    end
    cur_k(r_x,r_y) = u_x;
    
    %Update zeta
    zeta=zeta+1/i*[w(1)/pi(1),w(2)/pi(2),w(3)/pi(3),w(4)/pi(4),w(5)/pi(5)];
    zeta=zeta-zeta(1);
    
    %Record zeta & typical samples
    node(i,:) = zeta;
    s_u1(label,801+cur_u) = s_u1(label,801+cur_u)+1;
    %Get the peak as a typical sample
    if(i >= N1/2)
        if(i == N1/2)
            max_u1 = find(s_u1(1,:) == max(s_u1(1,:)));
            max_u2 = find(s_u1(2,:) == max(s_u1(2,:)));
            max_u3 = find(s_u1(3,:) == max(s_u1(3,:)));
            max_u4 = find(s_u1(4,:) == max(s_u1(4,:)));
            max_u5 = find(s_u1(5,:) == max(s_u1(5,:)));
        else
            if(cur_u == max_u1-801 && label == 1)
                t_u1 = cur_k;
            end
            if(cur_u == max_u2-801 && label == 2)
                t_u2 = cur_k;
            end
            if(cur_u == max_u3-801 && label == 3)
                t_u3 = cur_k;
            end
            if(cur_u == max_u4-801 && label == 4)
                t_u4 = cur_k;
            end
            if(cur_u == max_u5-801 && label == 5)
                t_u5 = cur_k;
            end
        end
    end
end

x = 1:N1;
x_u = linspace(-2,0,801);
figure(1),plot(x,node(:,1),x,node(:,2),x,node(:,3),x,node(:,4),x,node(:,5));
title('Zeta');
figure(2),subplot(2,3,1),bar(x_u,s_u1(1,:));
xlabel('T = 1.4');
figure(2),subplot(2,3,2),bar(x_u,s_u1(2,:));
xlabel('T = 1.4065');
figure(2),subplot(2,3,3),bar(x_u,s_u1(3,:));
xlabel('T = 1.4130');
figure(2),subplot(2,3,4),bar(x_u,s_u1(4,:));
xlabel('T = 1.4195');
figure(2),subplot(2,3,5),bar(x_u,s_u1(5,:));
xlabel('T = 1.4260');
figure(3),subplot(2,3,1),stem3(t_u1,'MarkerSize',4,'LineStyle',':');
xlabel('T = 1.4');
figure(3),subplot(2,3,2),stem3(t_u2,'MarkerSize',4,'LineStyle',':');
xlabel('T = 1.4065');
figure(3),subplot(2,3,3),stem3(t_u3,'MarkerSize',4,'LineStyle',':');
xlabel('T = 1.4130');
figure(3),subplot(2,3,4),stem3(t_u4,'MarkerSize',4,'LineStyle',':');
xlabel('T = 1.4195');
figure(3),subplot(2,3,5),stem3(t_u5,'MarkerSize',4,'LineStyle',':');
xlabel('T = 1.4260');

toc