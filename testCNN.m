% CNN filter

clear;
load('/Users/1995spring/Documents/Éñ¾­ÍøÂç/finalsubmite/dbn-master/mnist_uint8.mat');
cnnModel = CNNfc(2,2,2,100,'1',2,'meanPooling');

imdata = imread('Data/dog2.jpg');
figure;
imshow(imdata)
cnnModel.CNNfilter(imdata);