% load('mnist_uint8.mat')
% Zero = train_x(train_y(:,1)==1,:);
% One = train_x(train_y(:,2)==1,:);
% Two = train_x(train_y(:,3)==1,:);

% TT = load('Data/One.mat');
% train_x = TT.One;
TT = load('Data/Zero.mat');
train_x = TT.Zero;
NN = length(train_x);
N_sample = 1000;p = 784;
x = train_x(randperm(NN,N_sample),:);
x = double(x);

model = DBN([p 300 100 50]);
model = model.train(x , 3000);
% generate
model2=model.predict(100);

Im = zeros(280 , 280);
for i = 1 : size(model.rbmList(1).testData_x , 1)
    i0 = floor((i-1)/10);j0 = mod(i-1,10);
    Im(i0*28+1:(i0+1)*28,j0*28+1:(j0+1)*28) = reshape(model.rbmList(1).testData_x(i , :) , 28 , 28);
end
imshow([Im(1:140,:) Im(141:280,:)]);
