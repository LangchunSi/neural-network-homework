% LR
clc
clear
close all
N = 80;
Data1 = rand(N,2);
Data2 = rand(N,2) + 0.8;

X = [Data1;Data2];

Y = [-ones(N,1);ones(N,1)];

PP = LR(500);
PP.train(X,Y);

[~,T] = PP.predict(X,true);
T(T>=0.5) = 1;
T(T<0.5)  = -1;
disp(['Train error: ',num2str(sum(Y~=T)/(2*N))]);
plot(X(1:N,1),X(1:N,2),'or');
hold on
plot(X(N+1:end,1),X(N+1:end,2),'+b');
hold off

N = 20;
Data1 = rand(N,2);
Data2 = rand(N,2) + 0.8;

X = [Data1;Data2];
Y = [-ones(N,1);ones(N,1)];
[~,T] = PP.predict(X,false);
T(T>=0.5) = 1;
T(T<0.5)  = -1;
disp(['Test error: ',num2str(sum(Y~=T)/(2*N))]);