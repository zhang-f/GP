close all, clear all, clc;

%% Gaussian joint distribution
%{
N: number of point
cur_node: The current node
mu: mean value
sigma: mean variance matrix
%}
N = 100000;
cur_node = [5,10];
mu = [5,10];
sigma = [1,1;1,4];
example = mvnrnd(mu,sigma,floor(N/2.5));
figure(1),subplot(1,2,1), scatter(example(:,1),example(:,2),2);
axis([0 10 0 20]);
title('理论结果');

tic;
num = 1;
next_node = zeros(1,2);
node_x = zeros(1,N);
node_y = zeros(1,N);
node_x(1) = cur_node(1);
node_y(1) = cur_node(2);
coffe = zeros(1,N);
for i = 2: N
    u = rand();
    %{
    next_node = [10*rand(),20*rand()];
    buff = mvnpdf(next_node,mu,sigma)/mvnpdf(cur_node,mu,sigma);
    %}
    next_node = normrnd(cur_node,[1,4]);
    buff = normpdf(next_node,cur_node,[1,4])*mvnpdf(next_node,mu,sigma)/(mvnpdf(cur_node,mu,sigma)*normpdf(cur_node,next_node,[1,4]));
    
    buff = min(1,buff);
    %Decide whether to accept
    if u < buff
        node_x(i) = next_node(1);
        node_y(i) = next_node(2);
        num = num+1;
    else
        node_x(i) = cur_node(1);
        node_y(i) = cur_node(2);
    end
    cur_node = [node_x(i),node_y(i)];
    %corr = corrcoef(node_x,node_y);
    %coffe(i) = corr(1,2);
end

figure(1),subplot(1,2,2), scatter(node_x,node_y,2);
axis([0 10 0 20]);
title('MH采样结果');
%figure(3),plot(coffe);
accept = num/N;
toc;