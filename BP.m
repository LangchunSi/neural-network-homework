classdef BP < handle
% Back Propagation
% Construction: input layer -> hidden layer -> output layer

% This code is based on <<Machine Learning>> written by Zhihua Zhou(P101-104)

    properties(SetAccess = private)
        nHidden = 0;  % number of hidden nodes
        nIter   = 0;  % number of iterations
        nBatch  = 0;  % batch size
        WA      = []; % weights between input layer and hidden layer
        theta   = []; % bias of hidden layer
        WB      = []; % weights between hidden layer and output layer
        gamma   = []; % bias of output layer
    end
    
    methods
        % constructure
        function Obj = BP(nHidden,nIter,nBatch)
            Obj.nHidden = nHidden;
            Obj.nIter   = nIter;
            Obj.nBatch  = nBatch;
        end
         
        % train
        function train(Obj,X,T)
            [N,D] = size(X);
            L     = size(T,2); % number of output nodes
            if isempty(Obj.WA) || isempty(Obj.WB) || isempty(Obj.theta) || isempty(Obj.gamma)
                Obj.WA    = randn(Obj.nHidden,D);
                Obj.theta = randn(Obj.nHidden,1);
                Obj.WB    = randn(L,Obj.nHidden);
                Obj.gamma = randn(L,1);
            end
            
            eta = 1e-1; % learn rate
            MSE = zeros(1,Obj.nIter);
            
            for i = 1 : Obj.nIter
                select = randperm(N,Obj.nBatch);
                MSEt   = 0;
                for j = 1 : Obj.nBatch
                    [Z,Y] = Obj.predict(X(select(j),:));
                    % update WB and gamma
                    g         = (T(select(j),:)-Z).*Z.*(1-Z);
                    Obj.WB    = Obj.WB + eta*g'*Y';
                    Obj.gamma = Obj.gamma - eta*g';
                    % update WA and theta
                    e         =  Obj.WB'*g'.*Y.*(1-Y);
                    Obj.WA    = Obj.WA + eta*e*X(select(j),:);
                    Obj.theta = Obj.theta - eta*e;
                    
                    MSEt      = MSEt + mse(Z-T(select(j),:));
                end
                MSE(i) = MSEt/Obj.nBatch;
                
            end
            figure;
            plot(1:Obj.nIter,MSE,'-r','LineWidth',1.4);
            xlabel('Epoch');
            ylabel('MSE');
            title('BP');
        end
        
        % predict
        function [Z,Y] = predict(Obj,X)
            Y = 1./(1+exp(-Obj.WA*X'+Obj.theta));
            Z = 1./(1+exp(-Obj.WB*Y+Obj.gamma))';
        end
    end
end