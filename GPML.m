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
%% GPML toolbox
e1 = zeros(374,1);
meanfunc = [];
covfunc = @covSEiso;
likfunc = @likGauss;
hyp = struct('mean', [], 'cov', [5,4], 'lik', 2);
hyp2 = minimize(hyp, @gp, -200, @infGaussLik, meanfunc, covfunc, likfunc, train_data_2010, train_label_2010);
[e1(1:60,1) ys] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, train_data_2010, train_label_2010, test_data_2010);
%error = MSE(e,train_label_2010(101:133,:));
hyp = struct('mean', [], 'cov', [5 4], 'lik', 2.5);
hyp2 = minimize(hyp, @gp, -200, @infGaussLik, meanfunc, covfunc, likfunc, train_data_2011, train_label_2011);
[e1(61:110,1) ys] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, train_data_2011, train_label_2011, test_data_2011);
hyp = struct('mean', [], 'cov', [6 4], 'lik', 2.5);
hyp2 = minimize(hyp, @gp, -200, @infGaussLik, meanfunc, covfunc, likfunc, train_data_2012, train_label_2012);
[e1(111:164,1) ys] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, train_data_2012, train_label_2012, test_data_2012);
hyp = struct('mean', [], 'cov', [19 19], 'lik', 2.5);
hyp2 = minimize(hyp, @gp, -200, @infGaussLik, meanfunc, covfunc, likfunc, train_data_2013, train_label_2013);
[e1(165:214,1) ys] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, train_data_2013, train_label_2013, test_data_2013);
hyp = struct('mean', [], 'cov', [6 4], 'lik', 2.5);
hyp2 = minimize(hyp, @gp, -200, @infGaussLik, meanfunc, covfunc, likfunc, train_data_2014, train_label_2014);
[e1(215:264,1) ys] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, train_data_2014, train_label_2014, test_data_2014);
hyp2 = minimize(hyp, @gp, -200, @infGaussLik, meanfunc, covfunc, likfunc, train_data_2015, train_label_2015);
[e1(265:307,1) ys] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, train_data_2015, train_label_2015, test_data_2015);
hyp2 = minimize(hyp, @gp, -200, @infGaussLik, meanfunc, covfunc, likfunc, train_data_2016, train_label_2016);
[e1(308:end,1) ys] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, train_data_2016, train_label_2016, test_data_2016);
%dir = ('D:\class_doc\2019_Autumn\Ëæ»ú¹ý³Ì\project2_GPR_2019\GPR.mat');
%save(dir,'e1','-append');
toc

function [error] = MSE(train_label,test_label)
    error = 0;
    for m = 1: length(train_label) 
        error = error + (train_label(m,1)-test_label(m,1))^2;
    end
    error = error/length(train_label);
end