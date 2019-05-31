%% Importing and formatting data
clear all
close all
clc

% import csv file
data = readtable('AdultDataHeader.csv');

%% catergoricals to nominals
names = data.Properties.VariableNames;

[nrows, ncols] = size(data);
category = false(1,ncols);
for i = 1:ncols
    if isa(data.(names{i}),'cell') || isa(data.(names{i}),'nominal')
        category(i) = true;
        data.(names{i}) = grp2idx(data.(names{i}));
    end
end

%feature selection
data = horzcat(data(:,1:3),data(:,5:14),data(:,16));

%transforming table to array
datamain = table2array(data);

%% BASIC STATS
disp('Frequency of label class:')
tabulate(data.earningClassBinary)

%freqs of categorical predictor features
% disp('Frequency of Workclass:')
% tabulate(data.Workclass)
% disp('Frequency of education_num:')
% tabulate(data.education_num)
% disp('Frequency of matrial_status:')
% tabulate(data.matrial_status)
% disp('Frequency of occupation:')
% tabulate(data.occupation)
% disp('Frequency of relationship:')
% tabulate(data.relationship)
% disp('Frequency of race:')
% tabulate(data.race)
% disp('Frequency of sex:')
% tabulate(data.sex)
% disp('Frequency of native_country:')
% tabulate(data.native_country)

%Correlation of predictor features
covmat = corrcoef(double(datamain));
figure
x = size(datamain, 2);
imagesc(covmat);
set(gca,'XTick',1:x);
set(gca,'YTick',1:x);
set(gca,'XTickLabel',data.Properties.VariableNames);
set(gca,'YTickLabel',data.Properties.VariableNames);
axis([0 x+1 0 x+1]);
grid;
colorbar;

%% BALANCING DATA
% Find out which rows are 1.
ZERORows = find(~data.earningClassBinary);
% Take only the first 5000:
ZERORows  = ZERORows(1:5000);
% Find out which rows are 0's - we want to keep all those.
ONERows = find(data.earningClassBinary);
% Combine the 0 and 1 rows into one list of indexes.
rowsToExtract = sort([ONERows; ZERORows]);
% Now extract only the first 5000 1's, but all the false.
datamain = datamain(rowsToExtract,:);

%Checking class balance
data2 = array2table(datamain);
tabulate(data2.datamain14)

%% creating test and training set
%Separating preditors and lables
X = double(datamain(:,1:13));
Y = double(datamain(:,14));
disp('Size of predictor matrix')
size(X)
disp('Size of class matrix')
size(Y)

%% create training 90% and testing 10% sets
c1 = cvpartition(Y,'Holdout',.1); 
Ytraining = Y(training(c1),:);
Xtraining = X(training(c1),:);
Ytesting = Y(test(c1),:);
Xtesting = X(test(c1),:);

%% Partioning training set for k-fold cross val
fold = 10; 
c = cvpartition(Ytraining,'KFold',fold); 

%% MLP training net with gradient descent and momentum

rng ('default');
disp('Starting MLP');

%GRID SEARCH
MSEMatrix = []; %set empty matrix for grid search results
row = {};
tic
%loop over various num of epochs 
for e = [50 100 200]
    epochs = e;
    %loop over various values for momentum
    for m = 0.1:0.2:0.9
        momentum = m;
        %loop over various learning rates 
        for lr = 0.1:0.2:0.9
            learnRate = lr;
            %loop over various neuron amounts
            for n = 20:10:30
                numneurons = n; 
                AccCounter = 0;
                for i = 1:fold
                    %Create training and validation sets for predictors and labels
                    X_Train = Xtraining(training(c,i),:);
                    Y_Train = Ytraining(training(c,i));
                    X_Test = Xtraining(test(c,i),:);
                    Y_Test = Ytraining(test(c,i),:);  
                    %setting hyperparameters
                    net = feedforwardnet(numneurons,'traingdm');
                    net.trainParam.lr = learnRate;
                    net.trainParam.mc = momentum;
                    net.trainParam.epochs = epochs;
                    net.numLayers = 2;
                    net.performFcn = 'mse';
                    %train the net
                    [net,tr] = train(net,X_Train',Y_Train');
                    %test the net
                    testnet = net(X_Test');
                    testnet = testnet>0.5; %sigmoid funtion activation
                    testnet2 = double(testnet);
                    %Confusion matrix - predicted class against actual class
                    [MLPcon, classorder] = confusionmat(Y_Test,testnet2);  
                    %Confusion matrix for each class as a percentage of the true class
                    MLPconper = bsxfun(@rdivide,MLPcon,sum(MLPcon,2)) * 100;
                    %Classification rate - accuracy 
                    MLPAcc = (trace(MLPcon)/sum(MLPcon(:)));
                    AccCounter =  AccCounter +  MLPAcc; %counter    
                end
                %Mean accuracy across all folds
                AccMean = AccCounter/fold;                
                %fprintf('Mean accuracy for %g neurons, lr = %g, m = %g, ep = %g: %g \n',numneurons,learnRate,momentum,epochs,AccMean)
                %adding all mean accuracies to a matrix
                row = [AccMean,numneurons,learnRate,momentum,epochs];
                MSEMatrix = [MSEMatrix;row];                
            end    
        end
    end    
end    
toc

%% max element in matrix - best accuracy and it's parameters
[M,I] = max(MSEMatrix,[],1);
V = MSEMatrix(I(1,1),:);
z= V(1,1);
trainacc = 100*z;
fprintf('Training MLP: Best accuracy %f%% obtained with parameters: %g neurons, lr = %g, m = %g, ep = %g \n',trainacc,V(1,2),V(1,3),V(1,4),V(1,5));

%% Use best parameters on testing set
net2 = feedforwardnet(V(1,2),'traingdm');
net2.trainParam.lr = V(1,3);
net2.trainParam.mc = V(1,4);
net2.trainParam.epochs = V(1,5);
net2.numLayers = 2;
net2.performFcn = 'mse';
% train the net
[net2,tr] = train(net2,Xtesting',Ytesting');
% test the net
testnetMLP = net2(Xtesting');
testnetMLP = testnetMLP>0.5; %sigmoid funtion activation
testnetMLP2 = double(testnetMLP);
[MLPcon2, classorder] = confusionmat(Ytesting,testnetMLP2);
MLPconper2 = bsxfun(@rdivide,MLPcon2,sum(MLPcon2,2)) * 100;
MLPAcc2 = (trace(MLPcon2)/sum(MLPcon2(:)));
%printf('Testing: Percentage Incorrect Classification   : %f%%\n', 100*(1-MLPAcc2));
fprintf('Testing: Percentage Correct Classification : %f%%\n', 100*MLPAcc2);
%plot confusion matrix
plotconfusion(Ytesting',testnetMLP2)
%plot roc curve
%plotroc(Ytesting',testnetMLP)


%% SVM

rng ('default');
disp('Starting SVM');

%Grid search
SVMMatrix = []; %set empty matrix for grid search results
row2 = {};
tic;
%loop over box contraint
for box = 0.1:0.2:0.9
    boxcons = box;
    %loop over kernalscale
    for kern = 0.5:0.2:0.9
        kernscale = kern; 
        AccCounterSVM = 0;
        for i = 1:fold
            %Create training and validation sets for predictors and labels
            X_Train = Xtraining(training(c,i),:);
            Y_Train = Ytraining(training(c,i));
            X_Test = Xtraining(test(c,i),:);
            Y_Test = Ytraining(test(c,i),:);        
            %training svm
            SVMModel = fitcsvm(X_Train,Y_Train,'Standardize',true,'KernelFunction','RBF','BoxConstraint',boxcons,'KernelScale',kernscale);
            SVMPred = predict(SVMModel,X_Test);
            %Confusion matrix - predicted class against actual class
            [SVMcon, classorder] = confusionmat(Y_Test,SVMPred);
            %Confusion matrix for each class as a percentage of the true class
            SVMconper = bsxfun(@rdivide,SVMcon,sum(SVMcon,2)) * 100;
            %Classification rate - accuracy 
            SVMAcc = (trace(SVMcon)/sum(SVMcon(:)));
            AccCounterSVM =  AccCounterSVM +  SVMAcc;
        end
        %Mean accuracy across all folds
        AccSVMMean = AccCounterSVM/fold;                
        %fprintf('Mean accuracy for box constraint = %g, kernal scale = %g: %g \n',boxcons,kernscale,AccSVMMean)
        %adding all mean accuracies to a matrix
        row2 = [AccSVMMean,boxcons,kernscale];
        SVMMatrix = [SVMMatrix;row2];           
    end
end
toc;

%% max element in matrix - best accuracy and it's parameters
[M,I] = max(SVMMatrix,[],1);
V2 = SVMMatrix(I(1,1),:);
z2= V2(1,1);
trainacc2 = 100*z2;
fprintf('Training SVM: Best accuracy %f%% obtained with parameters: box constraint = %g, kernal scale = %g \n',trainacc2,V2(1,2),V2(1,3));


%% Use best parameters on test set
boxcons = V2(1,2);
kernscale = V2(1,3); 
SVMModel2 = fitcsvm(Xtesting,Ytesting,'Standardize',true,'KernelScale','rbf','BoxConstraint',boxcons,'KernelScale',kernscale);
SVMPred2 = predict(SVMModel2,Xtesting);
[SVMcon2, classorder] = confusionmat(Ytesting,SVMPred2);
SVMconper2 = bsxfun(@rdivide,SVMcon2,sum(SVMcon2,2)) * 100;
SVMAcc2 = (trace(SVMcon2)/sum(SVMcon2(:)));
%fprintf('Percentage Incorrect EClassification   : %f%%\n', 100*(1-SVMAcc2));
fprintf('Testing SVM: Percentage Correct Classification : %f%%\n', 100*SVMAcc2);
%plot confusion matrix
plotconfusion(Ytesting',SVMPred2')
%plot roc curve
%plotroc(Ytesting',SVMPred2)

           

