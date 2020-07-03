classdef Perception < handle
% Perception
% Construction: input -> signal function -> output

% This code is based on <<Neural Network and Deep Learning>> written by Xipeng Qiu

    properties(SetAccess = private)
        nIter = 0;  % number of iterations
        theta = 0;  % bias
        W     = []; % weights
    end
    
    methods
        % constructure
        function Obj = Perception(nIter)
            Obj.nIter = nIter;
        end
        
        % train
        function train(Obj,X,T)
            [N,D] = size(X);
            if isempty(Obj.W)
                Obj.W = zeros(1,D);
            end
            Loss = zeros(1,Obj.nIter);
            for i = 1 : Obj.nIter
                loss = 0;
                for j = i : N
                    [Z,Y] = Obj.predict(X(j,:),false);
                    if T(j)*Z <= 0
                        Obj.W     = Obj.W + T(j)*X(j,:);
                        Obj.theta = Obj.theta + T(j);
                    end
                    loss      = loss + (Y-T(j))^2;
                end
                Loss(i) = sqrt(loss)/N;
            end
            figure;
            plot(1:Obj.nIter,Loss,'-r','LineWidth',1.4);
            xlabel('Epoch');
            ylabel('Loss');
            title('Perception');
        end
        
        % predict
        function [Z,Y] = predict(Obj,X,disp)
            Z = X*Obj.W' + Obj.theta;
            Y = ones(size(Z));
            Y(Z<=0) = -1;
            if disp % D = 2
%                 figure;
                xX = 0:0.1:2;
                plot(xX,-(Obj.W(1).*xX + Obj.theta)./Obj.W(2),'-');
%                 hold on;
%                 plot(X(Y==1,1),X(Y==1,2),'ob')
%                 plot(X(Y~=1,1),X(Y~=1,2),'+r')
%                 hold off;
            end
        end
    end
    
end