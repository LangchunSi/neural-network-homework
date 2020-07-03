classdef LR < handle
% Logistic Regression

% This code is based on <<Neural Network and Deep Learning>> written by Xipeng Qiu(Section 3)

    properties(SetAccess = private)
        nIter = 0;  % number of iterations
        theta = 0;  % bias
        W     = []; % weights between input layer and output layer 
    end
    
    methods
        % constructure
        function Obj = LR(nIter)
            Obj.nIter = nIter;
        end
        
        % train
        function train(Obj,X,T)
            [N,D] = size(X);
            if isempty(Obj.W)
                Obj.W = ones(1,D);
            end
            Loss = zeros(1,Obj.nIter);
            eta = 0.01; % learn rate;
            for i = 1 : Obj.nIter
                loss = 0;
                for j = 1 : N
                    [Z,Y]     = Obj.predict(X(j,:),false);
                    Obj.W     = Obj.W - eta*X(j,:)*(T(j) - Y);
                    Obj.theta = Obj.theta - eta*(Y - T(j));
                    loss      = loss + (Y-T(j))^2;
                end
                Loss(i) = sqrt(loss)/N;
            end
            figure;
            plot(1:Obj.nIter,Loss,'-r','LineWidth',1.4)
            xlabel('Epoch');
            ylabel('Loss');
            title('LR');
        end
        
        % preduct
        function [Z,Y] = predict(Obj,X,disp)
            Z = X*Obj.W' + Obj.theta;
            Y = 1./(1+exp(-Z));
            if disp % D = 2
                figure;
                xX = 0:0.1:2;
                plot(xX,-(Obj.W(1).*xX + Obj.theta)./Obj.W(2),'-');
                hold on;
            end
        end
    end

end