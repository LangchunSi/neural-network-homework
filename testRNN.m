%% RNN
% 1 xor 0 = 1, 1 xor 1 = 0, and 0 xor 0 = 0
% Sample: [[1,0,...,0,0],[0,0,...,1,1]] label: [1,0,...,1,1]
N = 256;
T = 8;
halfData1 = dec2bin(0:N-1)-'0';
halfData2 = halfData1(randperm(N),:);
dataSet  = zeros(N,T,2);
dataSet(:,:,1) = halfData1;
dataSet(:,:,2) = halfData2;
Label =  bitxor(halfData1,halfData2);

Model = binaryRNN(16,2000);
Model.train(dataSet,Label)

N = 256;
T = 8;
halfData1 = dec2bin(0:N-1)-'0';
halfData2 = halfData1(randperm(N),:);
dataSet  = zeros(N,T,2);
dataSet(:,:,1) = halfData1;
dataSet(:,:,2) = halfData2;
Label =  bitxor(halfData1,halfData2);
preY  = Model.predict(dataSet);
disp(['test mse: ',num2str(mse(Label-preY))]);