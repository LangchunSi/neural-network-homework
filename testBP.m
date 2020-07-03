% BP
% mnist_dataset
clear;
load('/Users/1995spring/Documents/Éñ¾­ÍøÂç/finalsubmite/dbn-master/mnist_uint8.mat');
train_x = im2double(train_x);
train_y = im2double(train_y);

test_x  = im2double(test_x);
test_y  = im2double(test_y);

% train_x = train_x/255.0;
% train_y = train_y/1.0;
% 
% test_x  = test_x/255.0;
% test_y  = test_y/1.0;
bpModel = BP(16,500,100);
train_size = 600;
train_sample  = randperm(60000,train_size);
bpModel.train(train_x(train_sample,:),train_y(train_sample,:));
test_size = 100;
test_sample  = randperm(10000,test_size);
[Z,Y] = bpModel.predict(test_x(test_sample,:));
mse(Z-test_y(test_sample,:))