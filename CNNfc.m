classdef CNNfc < handle
% Convolutional Neural Network
% Construction: input layer -> conv layer -> ReLU -> pooling layer
% Note: Grayscale image, without full connect layer
% 我只是个没有感情的滤波器
    properties(SetAccess=private)
        % conv
        filterSize = 2;   % size of filter
        numFilter  = 1;   % number of filters
        stride     = 1;   % size of step
        padSize    = 0;   % padding size
        padType    = '0'; % padding type: '0' padding or '1' padding
        filter     = [];  % weight of filter
        bias       = [];
        % pooling
        poolSize   = 2;            % size of pooling matrix, max pooling
        poolType   = 'maxPooling'; % pooling type: max pooling or mean pooling
        
    end
    
    methods
        % Constructure
        function Obj = CNNfc(filterSize,numFilter,stride,padSize,padType,poolSize,poolType)
            Obj.filterSize = filterSize;
            Obj.numFilter  = numFilter;
            Obj.stride     = stride;
            Obj.padSize    = padSize;
            Obj.padType    = padType;
            Obj.poolSize   = poolSize;
            Obj.poolType   = poolType;
        end
        
        % filter
        function CNNfilter(Obj,image)
            % rgb to gray
            Image = Obj.grayIm(image);
            
            % set weights of all filters
            Obj.filter = ones(Obj.filterSize,Obj.filterSize,Obj.numFilter);
            for i = 1 : Obj.numFilter
                Obj.filter(:,:,i) = randi([-1,1],Obj.filterSize,Obj.filterSize);
            end
            Obj.filter = int8(Obj.filter);
            Obj.bias   = randi([0,1],1,Obj.numFilter);
            [H,W]      = size(Image);
            
            % padding
            if Obj.padSize > 0
                pad = 1; % padding type
                if Obj.padType == '0'
                    pad = 0;
                end
                Image = padarray(Image,[Obj.padSize,Obj.padSize],pad);
            end
            hNew = floor((H + 2*Obj.padSize - Obj.filterSize)/Obj.stride) + 1;
            wNew = floor((W + 2*Obj.padSize - Obj.filterSize)/Obj.stride) + 1;
            s    = Obj.stride;
            
            % conv process
            convOut  = zeros(hNew,wNew,Obj.numFilter);
            for f = 1 : Obj.numFilter
                for h = 0 : hNew-1
                    for w = 0 : wNew-1
                        convOut(h+1,w+1,f) = sum(sum(Image(h*s+1:(Obj.filterSize + h*s),...
                            w*s+1:(Obj.filterSize + w*s)).*Obj.filter(:,:,f) + Obj.bias(f)));
                    end
                end
            end
            
            reluOut = Obj.ReLU(convOut);
            poolOut = Obj.maxPooling(reluOut);
            
            % show results
            Obj.resultShow(convOut,reluOut,poolOut);
        end
        
        function Image = grayIm(Obj,image)
            Image = image;
            if ndims(image) == 3
                Image = rgb2gray(image);
            end
            Image = int8(Image);
        end
        
        function relu = ReLU(Obj,image)
            relu = max(image,0);
        end
        % max pool filter with poolSize*poolSize filters and stride
        % poolSize
        function maxpool = maxPooling(Obj,image)
            [H,W,F] = size(image);
            hNew    = floor((H - Obj.poolSize)/Obj.poolSize) + 1;
            wNew    = floor((W - Obj.poolSize)/Obj.poolSize) + 1;
            maxpool = zeros(hNew,wNew,F);
            s       = Obj.poolSize;
            for f = 1 : F
                for h = 0 : hNew-1
                    for w = 0 : wNew-1
                        window = image(h*s+1:(Obj.poolSize + h*s),w*s+1:(Obj.poolSize + w*s),f);
                        if strcmp(Obj.poolType,'maxPooling')
                            maxpool(h+1,w+1,f) = max(window(:));
                        else
                            maxpool(h+1,w+1,f) = mean(window(:));
                        end
                    end
                end
            end
        end
        
        function resultShow(Obj,convOut,reluOut,poolOut)
            figure;
            hold on;
            for f = 1 : Obj.numFilter
                subplot(3,Obj.numFilter,f);
                imshow(convOut(:,:,f));
                title(['Conv ',num2str(f)]);
            end
            
            for f = Obj.numFilter + 1 : 2*Obj.numFilter
                subplot(3,Obj.numFilter,f);
                imshow(reluOut(:,:,f - Obj.numFilter));
                title(['ReLU ',num2str(f - Obj.numFilter)]);
            end
            
            for f = 2*Obj.numFilter + 1 : 3*Obj.numFilter
                subplot(3,Obj.numFilter,f);
                imshow(poolOut(:,:,f - 2*Obj.numFilter));
                title([Obj.poolType,' Pooling ',num2str(f - 2*Obj.numFilter)]);
            end
            
            hold off;
        end
    end
end