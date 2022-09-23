%% clear workspace, remove figures and load data
clc
clear
close all

turkishData = csvread('turkish-se-SP500vsMSCI.csv');
carsData = csvread('mtcarsdata-4features.csv',1,1);



%% 1 - One-dimensional problem without intercept on the Turkish stock exchange data

turkishX=turkishData(:,1);
turkishT=turkishData(:,2);

%compute W using given formula. It corresponds to:
%(summation of each x term multiplied by t term) /
%(summation of each  x term squared)
w = sum(turkishX.*turkishT)/sum(turkishX.^2);

%create plot
hold on;
grid on;
%set lables as seen in slides
xlabel ("Standard and Poor's 500 return index");
ylabel ("MSCI Europe index");
title('Indexes: The least squares solution');
%plot result
scatter(turkishX,turkishT,'x');
%plot solution line, using two given points
xCoordW = [-0.055, 0.075];
yCoordW = w*xCoordW;
plot (xCoordW, yCoordW,'lineWidth',1.5);

%% 2 - Compare graphically the solution obtained on different random subsets (10%) of the whole data set
turkishX=turkishData(:,1);
turkishT=turkishData(:,2);
%create new plot
figure
hold on;
grid on;
%set lables as seen in slides
xlabel ("Standard and Poor's 500 return index");
ylabel ("MSCI Europe index");
title('Indexes: The least squares solution with two different subset');
%plot entire set
scatter(turkishX,turkishT,'x');

for i=1:2
    %get random subset
    subsetSize = round(size(turkishData,1)/10);
    randomTurkishIndexes = randperm(size(turkishData,1),subsetSize);
    randomTurkishSubsetX = turkishX(randomTurkishIndexes);
    randomTurkishSubsetT = turkishT(randomTurkishIndexes);
    
    %compute again W
    wSubset = sum(randomTurkishSubsetX.*randomTurkishSubsetT)/sum(randomTurkishSubsetX.^2);
    
    %plot solution line, using two given points
    xCoordW = [-0.055, 0.075];
    yCoordW = wSubset*xCoordW;
    plot (xCoordW, yCoordW,'lineWidth',1.5);
    
end

%% 3 - One-dimensional problem with intercept on the Motor Trends car data, using columns mpg and weight
carsX=carsData(:,4);
carsT=carsData(:,1);

%compute W1
w1 = sum((carsX-mean(carsX)).*(carsT-mean(carsT)))/sum((carsX-mean(carsX)).^2);
%compute W0
w0 = mean(carsT)-(w1*mean(carsX));

%create new plot
figure
hold on;
grid on;
%set lables as seen in slides
xlabel ("Car Weight (lbs/1000)");
ylabel ("Fuel efficency (mpg)");
title('Motor Trends survey: The least squares solution');
%plot entire set
scatter(carsX,carsT,'x');

%plot solution line, using two given points
xCoordW = [1.5, 5.5];
yCoordW = w1*xCoordW + w0;
plot (xCoordW, yCoordW,'lineWidth',1.5);

%% 4 - Multi-dimensional problem on the complete MTcars data, using all four columns (predict mpg with the other three columns)

carsT=carsData(:,1);
carsX=carsData(:,2:4);

% incorporate additive parameter w0 by adding one constant column.
carsX = [ones(size(carsX,1),1),carsX];

%use pinv function to get Moore-Penrose pseudoinverse matrix of cars
%and display result
disp("found 4 W with this set:");
multiW = pinv(carsX)*carsT;
disp(multiW);
