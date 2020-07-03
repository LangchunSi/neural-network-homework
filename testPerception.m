% Perception
clc
clear
N = 80;
Data1 = rand(N,2);
Data2 = rand(N,2)+0.7;
X = [Data1;Data2];

Y = [-ones(N,1);ones(N,1)];
% 
PP = Perception(40);
PP.train(X,Y);
figure;
plot(X(Y==1,1),X(Y==1,2),'ob')
hold on;
plot(X(Y~=1,1),X(Y~=1,2),'+r')

[~,T] = PP.predict(X,true);
hold off;

T(T>=0) = 1;
T(T<0)  = -1;
disp(['Train error: ',num2str(sum(Y~=T)/(2*N))]);

N = 20;
Data1 = rand(N,2);
Data2 = rand(N,2)+0.7;
X = [Data1;Data2];
Y = [-ones(N,1);ones(N,1)];
[~,T] = PP.predict(X,false);
T(T>=0) = 1;
T(T<0)  = -1;
disp(['Test error: ',num2str(sum(Y~=T)/(2*N))]);