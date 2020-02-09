close all, clear all, clc;
tic
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
%% �ж����������
R = zeros(7,8);
for m = 1: 8
    r = corrcoef(train_data_2010(:,m),train_label_2010);
    R(1,m) = r(1,2);
    r = corrcoef(train_data_2011(:,m),train_label_2011);
    R(2,m) = r(1,2);
    r = corrcoef(train_data_2012(:,m),train_label_2012);
    R(3,m) = r(1,2);
    r = corrcoef(train_data_2013(:,m),train_label_2013);
    R(4,m) = r(1,2);
    r = corrcoef(train_data_2014(:,m),train_label_2014);
    R(5,m) = r(1,2);
    r = corrcoef(train_data_2015(:,m),train_label_2015);
    R(6,m) = r(1,2);
    r = corrcoef(train_data_2016(:,m),train_label_2016);
    R(7,m) = r(1,2);
end
Max_R = zeros(7,4);
for n = 1: 7
    for m = 1: 4
        pos = find(abs(R(n,:)) == max(abs(R(n,:))));
        Max_R(n,m) = pos;
        R(n,pos) = 0;
    end
end
%% ����ѵ������
train_2010 = train_data_2010(:,Max_R(1,:));
train_2011 = train_data_2011(:,Max_R(2,:));
train_2012 = train_data_2012(:,Max_R(3,:));
train_2013 = train_data_2013(:,Max_R(4,:));
train_2014 = train_data_2014(:,Max_R(5,:));
train_2015 = train_data_2015(:,Max_R(6,:));
train_2016 = train_data_2016(:,Max_R(7,:));
test_2010 = test_data_2010(:,Max_R(1,:));
test_2011 = test_data_2011(:,Max_R(2,:));
test_2012 = test_data_2012(:,Max_R(3,:));
test_2013 = test_data_2013(:,Max_R(4,:));
test_2014 = test_data_2014(:,Max_R(5,:));
test_2015 = test_data_2015(:,Max_R(6,:));
test_2016 = test_data_2016(:,Max_R(7,:));
%% ���и�������
e2 = zeros(374,1);
e2(1:60,1) = GPR_exp(train_2010,train_label_2010,test_2010,100,10,200000);
%error = MSE(train_label_2010(101:133),e);
e2(61:110,1) = GPR_exp(train_2011,train_label_2011,test_2011,100,10,50000);
e2(111:164,1) = GPR_exp(train_2012,train_label_2012,test_2012,50,10,30000);
e2(165:214,1) = GPR_exp(train_2013,train_label_2013,test_2013,60,10,500000);
e2(215:264,1) = GPR_exp(train_2014,train_label_2014,test_2014,60,10,60000);
e2(265:307,1) = GPR_exp(train_2015,train_label_2015,test_2015,100,10,100000);
e2(308:end,1) = GPR_exp(train_2016,train_label_2016,test_2016,50,10,200000);
optimal = e2;
%dir = ('D:\class_doc\2019_Autumn\�������\project2_GPR_2019\GPR.mat');
%save(dir,'e2','optimal','-append');
toc
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

function [error] = MSE(train_label,test_label)
    error = 0;
    for m = 1: length(train_label) 
        error = error + (train_label(m,1)-test_label(m,1))^2;
    end
    error = error/length(train_label);
end