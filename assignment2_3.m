%% clear workspace, remove figures and load data
clc
clear
close all

turkishData = csvread('turkish-se-SP500vsMSCI.csv');
carsData = csvread('mtcarsdata-4features.csv',1,1);

%define data set percentage
percentage = 5;

%define number of repetitions
repetitions = 10;

%compute subsets size
subsetSizeTurkish = round(size(turkishData,1)*percentage/100);
subsetSizeCars = round(size(carsData,1)*percentage/100);

%initialize J_MSE arrays, which contains objective of subset and objective
%of remaining set
J_MSE1 = zeros(repetitions,2);
J_MSE3 = zeros(repetitions,2);
J_MSE4 = zeros(repetitions,2);
%% 1 - rerun 1

for i=1:repetitions
    %get subset
    randomIndexes = randperm(size(turkishData,1),subsetSizeTurkish);
    turkishXSubset = turkishData(randomIndexes,1);
    turkishTSubset = turkishData(randomIndexes,2);
    
    %get remaining set, using setdiff
    turkishXRemaining = turkishData(setdiff(1:end,randomIndexes),1);
    turkishTRemaining = turkishData(setdiff(1:end,randomIndexes),2);
    
    
    %compute W of the two subsets
    wSub = sum(turkishXSubset.*turkishTSubset)/sum(turkishXSubset.^2);
    wRemaining = sum(turkishXRemaining.*turkishTRemaining)/sum(turkishXRemaining.^2);
    
    %Compute the objective (mean square error) on the two sets
    J_MSE1(i,1) = mean((turkishXSubset*wSub - turkishTSubset).^2);
    J_MSE1(i,2) = mean((turkishXRemaining*wRemaining - turkishTRemaining).^2);
end
%% 1 - rerun 3

for i=1:repetitions
    %get subset
    randomIndexes = randperm(size(carsData,1),subsetSizeCars);
    carsXSubset = carsData(randomIndexes,4);
    carsTSubset = carsData(randomIndexes,1);
    
    %get remaining set, using setdiff
    carsXRemaining= carsData(setdiff(1:end,randomIndexes),4);
    carsTRemaining = carsData(setdiff(1:end,randomIndexes),1);
    
    %compute W1 and W0 of the two subsets
    w1Subset = sum((carsXSubset-mean(carsXSubset)).*(carsTSubset-mean(carsTSubset)))/sum((carsXSubset-mean(carsXSubset)).^2);
    w0Subset = mean(carsTSubset)-(w1Subset*mean(carsXSubset));
    
    w1Remaining = sum((carsXRemaining-mean(carsXRemaining)).*(carsTRemaining-mean(carsTRemaining)))/sum((carsXRemaining-mean(carsXRemaining)).^2);
    w0Remaining = mean(carsTRemaining)-(w1Remaining*mean(carsXRemaining));
    
    %Compute the objective (mean square error) on the two sets
    carsYSubset = carsXSubset*w1Subset + w0Subset;
    carsYRemaining = carsXRemaining*w1Remaining + w0Remaining;
    J_MSE3(i,1) = mean((carsYSubset - carsTSubset).^2);
    J_MSE3(i,2) = mean((carsYRemaining - carsTRemaining).^2);
end

%% 1 - rerun 4
for i=1:repetitions
    %get subset
    randomIndexes = randperm(size(carsData,1),subsetSizeCars);
    carsTSubset = carsData(randomIndexes,1);
    carsXSubset = carsData(randomIndexes,2:4);
    
    %get remaining set, using setdiff
    carsTRemaining= carsData(setdiff(1:end,randomIndexes),1);
    carsXRemaining = carsData(setdiff(1:end,randomIndexes),2:4);
    
    % incorporate additive parameters w0 by adding one constant column in each
    % set.
    carsXSubset = [ones(size(carsXSubset,1),1),carsXSubset];
    carsXRemaining = [ones(size(carsXRemaining,1),1),carsXRemaining];
    
    %use pinv function to get Moore-Penrose pseudoinverse matrices
    multiWSubset = pinv(carsXSubset)*carsTSubset;
    multiWRemaining = pinv(carsXRemaining)*carsTRemaining;
    
    %Compute the objective (mean square error) on the training data
    carsYSubset = carsXSubset*multiWSubset;
    carsYRemaining = carsXRemaining*multiWRemaining;
    J_MSE4(i,1) = mean((carsYSubset - carsTSubset).^2);
    J_MSE4(i,2) = mean((carsYRemaining - carsTRemaining).^2);
end

%% Plot bar graph for given data

labels{1} = strcat(num2str(percentage),'% subset');
labels{2} = 'remaining set';

subplot(3,1,1);
bar(J_MSE1);
title('Objective on turkish data');
ylabel('J_M_S_E Task 1');
xlabel('random subsets with 5% of the total data');
legend(labels);

subplot(3,1,2);
bar(J_MSE3);
title('Objective on Cars Data (One-dimensional)');
ylabel('J_M_S_E Task 3');
xlabel('random subsets with 5% of the total data');
legend(labels);

subplot(3,1,3);
bar(J_MSE4);
title('Objective on Cars Data (Multi-dimensional)');
ylabel('J_M_S_E Task 4');
xlabel('random subsets with 5% of the total data');
legend(labels);
