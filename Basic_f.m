tic
close all, clear all, clc;

%% load data
data_2010 = load('data2010.mat');
data_2011 = load('data2011.mat');
data_2012 = load('data2012.mat');
data_2013 = load('data2013.mat');
data_2014 = load('data2014.mat');
data_2015 = load('data2015.mat');
data_2016 = load('data2016.mat');

train_data_2010 = data_2010.data2010.Score((data_2010.data2010.TargetScore1 ~= -1),:);
train_label_2010 = data_2010.data2010.TargetScore1((data_2010.data2010.TargetScore1 ~= -1),:);
test_data_2010 = data_2010.data2010.Score((data_2010.data2010.TargetScore1 == -1),:);
test_label_2010 = data_2010.data2010.TargetScore1((data_2010.data2010.TargetScore1 == -1),:);

train_data_2011 = data_2011.data2011.Score((data_2011.data2011.TargetScore1 ~= -1),:);
train_label_2011 = data_2011.data2011.TargetScore1((data_2011.data2011.TargetScore1 ~= -1),:);
test_data_2011 = data_2011.data2011.Score((data_2011.data2011.TargetScore1 == -1),:);
test_label_2011 = data_2011.data2011.TargetScore1((data_2011.data2011.TargetScore1 == -1),:);

train_data_2012 = data_2012.data2012.Score((data_2012.data2012.TargetScore1 ~= -1),:);
train_label_2012 = data_2012.data2012.TargetScore1((data_2012.data2012.TargetScore1 ~= -1),:);
test_data_2012 = data_2012.data2012.Score((data_2012.data2012.TargetScore1 == -1),:);
test_label_2012 = data_2012.data2012.TargetScore1((data_2012.data2012.TargetScore1 == -1),:);

train_data_2013 = data_2013.data2013.Score((data_2013.data2013.TargetScore1 ~= -1),:);
train_label_2013 = data_2013.data2013.TargetScore1((data_2013.data2013.TargetScore1 ~= -1),:);
test_data_2013 = data_2013.data2013.Score((data_2013.data2013.TargetScore1 == -1),:);
test_label_2013 = data_2013.data2013.TargetScore1((data_2013.data2013.TargetScore1 == -1),:);

train_data_2014 = data_2014.data2014.Score((data_2014.data2014.TargetScore1 ~= -1),:);
train_label_2014 = data_2014.data2014.TargetScore1((data_2014.data2014.TargetScore1 ~= -1),:);
test_data_2014 = data_2014.data2014.Score((data_2014.data2014.TargetScore1 == -1),:);
test_label_2014 = data_2014.data2014.TargetScore1((data_2014.data2014.TargetScore1 == -1),:);

train_data_2015 = data_2015.data2015.Score((data_2015.data2015.TargetScore1 ~= -1),:);
train_label_2015 = data_2015.data2015.TargetScore1((data_2015.data2015.TargetScore1 ~= -1),:);
test_data_2015 = data_2015.data2015.Score((data_2015.data2015.TargetScore1 == -1),:);
test_label_2015 = data_2015.data2015.TargetScore1((data_2015.data2015.TargetScore1 == -1),:);

train_data_2016 = data_2016.data2016.Score((data_2016.data2016.TargetScore1 ~= -1),:);
train_label_2016 = data_2016.data2016.TargetScore1((data_2016.data2016.TargetScore1 ~= -1),:);
test_data_2016 = data_2016.data2016.Score((data_2016.data2016.TargetScore1 == -1),:);
test_label_2016 = data_2016.data2016.TargetScore1((data_2016.data2016.TargetScore1 == -1),:);

%% A: Standard linear model
a = zeros(374,1);
w = (train_data_2010'*train_label_2010)'/(train_data_2010'*train_data_2010);
%w = (train_data_2010(1:100,:)'*train_label_2010(1:100,:))'/(train_data_2010(1:100,:)'*train_data_2010(1:100,:));
%m = train_data_2010(101:133,:)*w';
%error = MSE(m,train_label_2010(101:133,:));
a(1:60,1) = test_data_2010*w';
w = (train_data_2011'*train_label_2011)'/(train_data_2011'*train_data_2011);
a(61:110,1) = test_data_2011*w';
%w = (train_data_2011(1:90,:)'*train_label_2011(1:90,:))'/(train_data_2011(1:90,:)'*train_data_2011(1:90,:));
%m = train_data_2011(91:107,:)*w';
%error = MSE(m,train_label_2010(101:133,:));
w = (train_data_2012'*train_label_2012)'/(train_data_2012'*train_data_2012);
a(111:164,1) = test_data_2012*w';
w = (train_data_2013'*train_label_2013)'/(train_data_2013'*train_data_2013);
a(165:214,1) = test_data_2013*w';
w = (train_data_2014'*train_label_2014)'/(train_data_2014'*train_data_2014);
a(215:264,1) = test_data_2014*w';
w = (train_data_2015'*train_label_2015)'/(train_data_2015'*train_data_2015);
a(265:307,1) = test_data_2015*w';
w = (train_data_2016'*train_label_2016)'/(train_data_2016'*train_data_2016);
a(308:end,1) = test_data_2016*w';


%% B: Probabilistic analysis of standard linear models
b = zeros(374,1);
sigma_n = 1;
sigma_p = [1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2;...
           0.2, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2;...
           0.2, 0.2, 1, 0.2, 0.2, 0.2, 0.2, 0.2;...
           0.2, 0.2, 0.2, 1, 0.2, 0.2, 0.2, 0.2;...
           0.2, 0.2, 0.2, 0.2, 1, 0.2, 0.2, 0.2;...
           0.2, 0.2, 0.2, 0.2, 0.2, 1, 0.2, 0.2;...
           0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1, 0.2;...
           0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1;];
A = 1/sigma_n^2.*train_data_2010'*train_data_2010+inv(sigma_p);
w_ave = (train_data_2010'*train_label_2010)'/A;
w = mvnrnd(w_ave',inv(A),1);
%m = train_data_2010(101:133,:)*w';
%error = MSE(m,train_label_2010(101:133,:));
b(1:60,1) = test_data_2010*w';
A = 1/sigma_n^2.*train_data_2011'*train_data_2011+inv(sigma_p);
w_ave = (train_data_2011'*train_label_2011)'/A;
w = mvnrnd(w_ave',inv(A),1);
b(61:110,1) = test_data_2011*w';
A = 1/sigma_n^2.*train_data_2012'*train_data_2012+inv(sigma_p);
w_ave = (train_data_2012'*train_label_2012)'/A;
w = mvnrnd(w_ave',inv(A),1);
b(111:164,1) = test_data_2012*w';
A = 1/sigma_n^2.*train_data_2013'*train_data_2013+inv(sigma_p);
w_ave = (train_data_2013'*train_label_2013)'/A;
w = mvnrnd(w_ave',inv(A),1);
b(165:214,1) = test_data_2013*w';
A = 1/sigma_n^2.*train_data_2014'*train_data_2014+inv(sigma_p);
w_ave = (train_data_2014'*train_label_2014)'/A;
w = mvnrnd(w_ave',inv(A),1);
b(215:264,1) = test_data_2014*w';
A = 1/sigma_n^2.*train_data_2015'*train_data_2015+inv(sigma_p);
w_ave = (train_data_2015'*train_label_2015)'/A;
w = mvnrnd(w_ave',inv(A),1);
b(265:307,1) = test_data_2015*w';
A = 1/sigma_n^2.*train_data_2016'*train_data_2016+inv(sigma_p);
w_ave = (train_data_2016'*train_label_2016)'/A;
w = mvnrnd(w_ave',inv(A),1);
b(308:end,1) = test_data_2016*w';

%% C: Polynomial basis function
% Probability method
c = zeros(374,1);
c(1:60,1) = C_Pro(train_data_2010,train_label_2010,test_data_2010);
c(61:110,1) = C_Pro(train_data_2011,train_label_2011,test_data_2011);
c(111:164,1) = C_Pro(train_data_2012,train_label_2012,test_data_2012);
c(165:214,1) = C_Pro(train_data_2013,train_label_2013,test_data_2013);
c(215:264,1) = C_Pro(train_data_2014,train_label_2014,test_data_2014);
c(265:307,1) = C_Pro(train_data_2015,train_label_2015,test_data_2015);
c(308:end,1) = C_Pro(train_data_2016,train_label_2016,test_data_2016);
%m = C_Pro(train_data_2010(1:100,:),train_label_2010(1:100,:),train_data_2010(101:133,:));
%error = MSE(m,train_label_2010(101:133));

%% D: GPR 
d = zeros(374,1);
d(1:60,1) = GPR_exp(train_data_2010,train_label_2010,test_data_2010,100,10,200000);
%error = MSE(train_label_2010(101:133),e);
d(61:110,1) = GPR_exp(train_data_2011,train_label_2011,test_data_2011,100,10,50000);
%error = MSE(test_label_2011,train_label_2011);
d(111:164,1) = GPR_exp(train_data_2012,train_label_2012,test_data_2012,50,10,30000);
%error = MSE(test_label_2012,train_label_2012(101:125,:));
d(165:214,1) = GPR_exp(train_data_2013,train_label_2013,test_data_2013,60,10,500000);
%error = MSE(test_label_2013,train_label_2013(101:137,:));
d(215:264,1) = GPR_exp(train_data_2014,train_label_2014,test_data_2014,60,10,60000);
%error = MSE(test_label_2014,train_label_2014(81:108));
d(265:307,1) = GPR_exp(train_data_2015,train_label_2015,test_data_2015,100,10,100000);
%error = MSE(test_label_2015,train_label_2015(81:100));
d(308:end,1) = GPR_exp(train_data_2016,train_label_2016,test_data_2016,50,10,200000);
%error = MSE(test_label_2016,train_label_2016);

%dir = ('D:\class_doc\2019_Autumn\Ëæ»ú¹ý³Ì\project2_GPR_2019\GPR.mat');
%save(dir,'a','b','c','d');
toc
% Method C function
function [test_label] = C_Pro(train_data, train_label,test_data)
    sigma_n = 1;
    sigma_p = 0.5*ones(16);
    for k = 1: 16
        sigma_p(k,k) = 1;
    end
    ex_train_data = ex_poly(train_data);
    ex_test_data = ex_poly(test_data);
    A = 1/sigma_n^2.*ex_train_data*ex_train_data'+inv(sigma_p);
    w_ave = (ex_train_data*train_label)'/A;
    w = mvnrnd(w_ave',inv(A),1);
    test_label = ex_test_data'*w';
end
% Base function extension
function [x] = ex_poly(y)
    x(1:8,:) = y';
    x(9:16,:) = y'.^2;
end
function [test_label] = GPR_exp(train_data,train_label,test_data,sigma_f,sigma_n,I)
    global sn;
    global s_f;
    global si;
    sn = [];
    s_f = [];
    si = [];
    [len wid] = size(train_data);
    K = zeros(len);
    delta_kI = zeros(len);
    delta_kn = zeros(len);
    delta_kf = zeros(len);
    step = 1;
    num = 0;
    while(1)
        for m = 1: len
            for n = 1: len
                temp = (train_data(m,:)-train_data(n,:))*(train_data(m,:)-train_data(n,:))';
                K(m,n) = sigma_f^2*exp(-1/2/I*temp)+sigma_n^2*(m==n);
            end
        end
        for m = 1: len
            for n = 1: len
                temp = (train_data(m,:)-train_data(n,:))*(train_data(m,:)-train_data(n,:))';
                delta_kI(m,n) = sigma_f^2*(1/2)*temp*I^(-2)*exp(-1/2/I*temp);
                delta_kf(m,n) = 2*sigma_f*exp(-1/2/I*temp);
                delta_kn(m,n) = 2*sigma_n*(m==n);
            end
        end
        alpha_k = inv(K)*train_label;
        delta_f = 1/2*trace((alpha_k*alpha_k'-inv(K))*delta_kf);
        delta_I = 1/2*trace((alpha_k*alpha_k'-inv(K))*delta_kI);
        delta_n = 1/2*trace((alpha_k*alpha_k'-inv(K))*delta_kn);
        sigma_f = sigma_f + sigma_f*step*delta_f;
        sigma_n = sigma_n + step*delta_n;
        I = I + 5000*I*step*delta_I;
        num = num + 1;
        sn = [sn,sigma_n];
        si = [si,I];
        s_f = [s_f,sigma_f];
        if(num > 15)
            step = step/1.1;
        end
        if((abs(delta_I) < 10^(-5) && abs(delta_n) < 0.5 && abs(delta_f) < 0.5 && num > 100)||num > 500)
            break;
        end
    end
    k_t = zeros(len,1);
    for g = 1: size(test_data,1)
        for p = 1: len
            k_t(p,1) = sigma_f^2*exp(-1/2/I*(train_data(p,:)-test_data(g,:))*(train_data(p,:)-test_data(g,:))');
        end
        test_mu = k_t'/K*train_label;
        test_conv = sigma_f^2-k_t'/K*k_t;
        %test_label(g,1) = mvnrnd(test_mu,test_conv,1);
        test_label(g,1) = test_mu;
    end
end
% Mean Squared Error
function [error] = MSE(train_label,test_label)
    error = 0;
    for m = 1: length(train_label) 
        error = error + (train_label(m,1)-test_label(m,1))^2;
    end
    error = error/length(train_label);
end
