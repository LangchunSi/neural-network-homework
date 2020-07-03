classdef binaryRNN < handle
% RNN for binary operation

% note:
% 1) This work is inspired by RNN for binary addition: https://blog.csdn.net/zzukun/article/details/49968129
    properties(SetAccess = private)
        nHidden = 0;  % number of hidden nodes
        nIter   = 0;  % number of iterations
        W       = []; % weight between hidden layer and output layer  
        U       = []; % last output of hidden layer  
        V       = []; % weight between input layer and hidden layer
    end
    
    methods
        % Constructure
        function Obj = binaryRNN(nHidden,nIter)
            Obj.nHidden = nHidden;
            Obj.nIter   = nIter;
        end
        
        function train(Obj,X,T)
            [N,D] = size(X(:,:,1)); % X size NxDx2, like zeros(N,D,2)
            % Initilize weights
            Obj.W = 2*rand(Obj.nHidden,1)-1;
            Obj.U = 2*rand(Obj.nHidden,Obj.nHidden)-1;
            Obj.V = 2*rand(2,Obj.nHidden)-1;
            
            % Temp parameter
            dW = zeros(Obj.nHidden,1);
            dU = zeros(Obj.nHidden,Obj.nHidden);
            dV = zeros(2,Obj.nHidden);
            
            
            %% Training
            allError = zeros(1,Obj.nIter);
            for iter = 1 : Obj.nIter
                Error = 0;
                for i = 1 : N
                    a     = X(i,:,1);
                    b     = X(i,:,2);
                    label = T(i,:);
                    y     = zeros(1,D);
                    preH  = zeros(1,Obj.nHidden);
                    hX    = zeros(D,Obj.nHidden);
                    
                    % Forward propagation
                    for t = D : -1 : 1
                        x   = [a(t),b(t)]  ; 
                        
                        h   = Obj.sigmoid(x*Obj.V+preH*Obj.U); 
                        y(t)= Obj.sigmoid(h*Obj.W);    
                        hX(t,:) = h; 
                        preH = h; 
                    end
                    % Error
                    E     = y - label;
                    Error = Error+norm(E,2)/2;
                    ndh   = zeros(1,Obj.nHidden);
                    
                    % Back propagation
                    for t = 1 : D
                        dy = E(t).*Obj.sigmoidOutput2d(y(t)); 
                        dh = (dy*Obj.W'+ndh*Obj.U').*Obj.sigmoidOutput2d(hX(t,:));
                        dW = dW+hX(t,:)'*dy; 
                        if t < T 
                             dU=dU+hX(t+1,:)'*dh; 
                        end 
                        dV  = dV+[a(t),b(t)]'*dh; 
                        ndh = dh;  
                    end
                    
                    % Update weights
                    lr = 0.01; % learn rate
                    Obj.W = Obj.W - lr*dW; 
                    Obj.U = Obj.U - lr*dU;
                    Obj.V = Obj.V - lr*dV;
                    
                    dW = zeros(Obj.nHidden,1);
                    dU = zeros(Obj.nHidden,Obj.nHidden);
                    dV = zeros(2,Obj.nHidden);
                end
                
                allError(iter) = Error/N;
            end
            % show training process
            figure;
            plot(1:Obj.nIter,allError,'-r','LineWidth',1.4);
            xlabel('Epoch');
            ylabel('MSE');
            title('RNN');
            
        end
        
        function Y = sigmoid(Obj,X)
            Y = 1./(1+exp(-X)); 
        end
        
        function DX = sigmoidOutput2d(Obj,Y) 
            DX = Y.*(1-Y); 
        end
        
        function Y = predict(Obj,X)
            [N,D,~] = size(X);
            Y       = zeros(N,D);
            for i = 1 : N
                preH  = zeros(1,Obj.nHidden);
                hX    = zeros(D,Obj.nHidden);
                a     = X(i,:,1);
                b     = X(i,:,2);
                y     = zeros(1,D);
                for t = D : -1 : 1
                    x   = [a(t),b(t)];
                    h   = Obj.sigmoid(x*Obj.V+preH*Obj.U); 
                    y(t)= Obj.sigmoid(h*Obj.W);    
                    hX(t,:) = h; 
                    preH = h; 
                end
                Y(i,:) = y;
            end
        end
    end
end