% % LR
% clc
% clear
% close all
% N = 80;
% Data1 = rand(N,2);
% Data2 = rand(N,2) + 0.8;
% 
% X = [Data1;Data2];
% 
% Y = [-ones(N,1);ones(N,1)];
% 
% PP = LR(500);
% PP.train(X,Y);
% 
% [~,T] = PP.predict(X,true);
% T(T>=0.5) = 1;
% T(T<0.5)  = -1;
% disp(['Train error: ',num2str(sum(Y~=T)/(2*N))]);
% plot(X(1:N,1),X(1:N,2),'or');
% hold on
% plot(X(N+1:end,1),X(N+1:end,2),'+b');
% hold off
% 
% N = 20;
% Data1 = rand(N,2);
% Data2 = rand(N,2) + 0.8;
% 
% X = [Data1;Data2];
% Y = [-ones(N,1);ones(N,1)];
% [~,T] = PP.predict(X,false);
% T(T>=0.5) = 1;
% T(T<0.5)  = -1;
% disp(['Test error: ',num2str(sum(Y~=T)/(2*N))]);


% % Perception
% clc
% clear
% N = 80;
% Data1 = rand(N,2);
% Data2 = rand(N,2)+0.7;
% X = [Data1;Data2];
% 
% Y = [-ones(N,1);ones(N,1)];
% % 
% PP = Perception(40);
% PP.train(X,Y);
% figure;
% plot(X(Y==1,1),X(Y==1,2),'ob')
% hold on;
% plot(X(Y~=1,1),X(Y~=1,2),'+r')
% 
% [~,T] = PP.predict(X,true);
% hold off;
% 
% T(T>=0) = 1;
% T(T<0)  = -1;
% disp(['Train error: ',num2str(sum(Y~=T)/(2*N))]);
% 
% N = 20;
% Data1 = rand(N,2);
% Data2 = rand(N,2)+0.7;
% X = [Data1;Data2];
% Y = [-ones(N,1);ones(N,1)];
% [~,T] = PP.predict(X,false);
% T(T>=0) = 1;
% T(T<0)  = -1;
% disp(['Test error: ',num2str(sum(Y~=T)/(2*N))]);

% % CNN
% close all;
% clear
% %% plane.jpg, matlab.jpg, dog.jpg, testCNN.jpg
% imdata = imread('Data/testCNN.jpg');
% CN = CNN(32,2,4,'maxPooling',2,'1');
% CN.CNNconv(imdata);

% %% RNN
% % 1 xor 0 = 1, 1 xor 1 = 0, and 0 xor 0 = 0
% % Sample: [[1,0,...,0,0],[0,0,...,1,1]] label: [1,0,...,1,1]
% N = 256;
% T = 8;
% halfData1 = dec2bin(0:N-1)-'0';
% halfData2 = halfData1(randperm(N),:);
% dataSet  = zeros(N,T,2);
% dataSet(:,:,1) = halfData1;
% dataSet(:,:,2) = halfData2;
% Label =  bitxor(halfData1,halfData2);
% 
% Model = binaryRNN(16,2000);
% Model.train(dataSet,Label)
% 
% N = 256;
% T = 8;
% halfData1 = dec2bin(0:N-1)-'0';
% halfData2 = halfData1(randperm(N),:);
% dataSet  = zeros(N,T,2);
% dataSet(:,:,1) = halfData1;
% dataSet(:,:,2) = halfData2;
% Label =  bitxor(halfData1,halfData2);
% preY  = Model.predict(dataSet);
% disp(['test mse: ',num2str(mse(Label-preY))]);

% % BP
% % mnist_dataset
% clear;
% load('/Users/1995spring/Documents/神经网络/finalsubmite/dbn-master/mnist_uint8.mat');
% train_x = im2double(train_x);
% train_y = im2double(train_y);
% 
% test_x  = im2double(test_x);
% test_y  = im2double(test_y);
% 
% % train_x = train_x/255.0;
% % train_y = train_y/1.0;
% % 
% % test_x  = test_x/255.0;
% % test_y  = test_y/1.0;
% bpModel = BP(16,500,100);
% train_size = 600;
% train_sample  = randperm(60000,train_size);
% bpModel.train(train_x(train_sample,:),train_y(train_sample,:));
% test_size = 100;
% test_sample  = randperm(10000,test_size);
% [Z,Y] = bpModel.predict(test_x(test_sample,:));
% mse(Z-test_y(test_sample,:))

% % CNN filter
% 
% clear;
% load('/Users/1995spring/Documents/神经网络/finalsubmite/dbn-master/mnist_uint8.mat');
% cnnModel = CNNfc(2,2,2,100,'1',2,'meanPooling');
% 
% imdata = imread('Data/dog2.jpg');
% figure;
% imshow(imdata)
% cnnModel.CNNfilter(imdata);



