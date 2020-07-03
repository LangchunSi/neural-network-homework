classdef RBM
    properties
        trainData_x;        %ѵ������ x
        testData_x;         %�����ع�����
        testData_h;         %�����ع�����
        v;                  %�ɼ��㵥Ԫ     v
        h;                  %���ز㵥Ԫ     h
        v_num;              %����     v��Ԫ�ĸ���
        h_num;              %����     h��Ԫ�ĸ���
        w;                  %Ȩ��     w
        b;                  %Ȩ��     b
        c;                  %Ȩ��     c
        N_sample;           %�������� N_sample
        iter_train;         %��ǰ��������
        IterMax;            %����������
        batchSize;          %���δ�С
        numIterOfTrain;     %ѵ��ʱ�ļ���˹������������
        numIterOfPredict;   %Ԥ��ʱ�ļ���˹������������
    end
    methods
        function obj = RBM(v_num , h_num)
            if(nargin == 0)
                v_num = 1;
                h_num = 1;
            end
            obj.v_num = v_num;
            obj.h_num = h_num;
            obj.v = zeros(obj.v_num , 1);
            obj.h = zeros(obj.h_num , 1);
            obj.w = rand(obj.v_num , obj.h_num);
            obj.b = ones(obj.v_num , 1);
            obj.c = ones(obj.h_num , 1);
            obj.iter_train = 0;
            obj.IterMax = 3000;
            obj.batchSize = 100;
            obj.numIterOfTrain=15;
            obj.numIterOfPredict=50;
        end
        function obj = train(obj , trainData_x , Iter)            %�ݶ��½�ѵ��
            if(nargin == 2)
                Iter = obj.IterMax;
            end
            obj.trainData_x = trainData_x;
            obj.N_sample = size(obj.trainData_x,1);
            batch_size = obj.batchSize;
            alpha = .1 / batch_size;
%             gradient_w = rand(obj.v_num , obj.h_num);
            while obj.iter_train < Iter %sum(abs(gradient_w)) > 1e-18 
                obj.iter_train = obj.iter_train + 1;
                trainData_x_batch = obj.trainData_x(randperm(obj.N_sample , batch_size) , :);
                %���ݼ����֣����ࣩ
                gradient_w = trainData_x_batch' * sigmoid(repmat(obj.c' , batch_size , 1) + ...
                    trainData_x_batch * obj.w);
                gradient_b = sum(trainData_x_batch)';
                gradient_c = sum(sigmoid(repmat(obj.c' , batch_size , 1) + ...
                    trainData_x_batch * obj.w))';
                %����˹�������֣����ࣩ
                k=obj.numIterOfTrain;
                for i = 1 : k
                    h_batch = double(rand(batch_size , obj.h_num) < sigmoid(repmat(obj.c' , batch_size , 1) +...
                        trainData_x_batch * obj.w));
                    trainData_x_batch = double(rand(batch_size , obj.v_num) < sigmoid(repmat(obj.b' , batch_size , 1) + ...
                        h_batch * obj.w'));
                end
                gradient_w = gradient_w - trainData_x_batch' * sigmoid(repmat(obj.c' , batch_size , 1) + ...
                    trainData_x_batch * obj.w);
                gradient_b = gradient_b - sum(trainData_x_batch)';
                gradient_c = gradient_c - sum(sigmoid(repmat(obj.c' , batch_size , 1) + ...
                    trainData_x_batch * obj.w))';
                %���²���
                obj.w = obj.w + alpha * gradient_w;
                obj.b = obj.b + alpha * gradient_b;
                obj.c = obj.c + alpha * gradient_c;
                %�ڴ˻���Ȩ�ز�����̬ͼ
            end
        end
        function obj = predict(obj , num_batch , testData_h_batch)    %�ع�
            if(nargin == 3)
                testData_x_batch = double(rand(num_batch , obj.v_num) < sigmoid(repmat(obj.b' , num_batch , 1) + ...
                    testData_h_batch * obj.w'));
            else
                testData_x_batch = obj.trainData_x(randperm(obj.N_sample , num_batch) , :);
            end
            k=obj.numIterOfPredict;
            for i = 1 : k
                h_batch = double(rand(num_batch , obj.h_num) < sigmoid(repmat(obj.c' , num_batch , 1) +...
                    testData_x_batch * obj.w));
                testData_x_batch = double(rand(num_batch , obj.v_num) < sigmoid(repmat(obj.b' , num_batch , 1) + ...
                    h_batch * obj.w'));
            end
            obj.testData_x = testData_x_batch;
            obj.testData_h = h_batch;
        end
    end
end

function output = sigmoid(x)
output =1 ./ (1 + exp(-x));
end

                %% ���ƶ�̬ͼ
%                 if (mod(obj.iter_train , 50) == 0)
%                     pause(.001);disp(obj.iter_train)
% %                     plot(obj.c);%axis([1 , obj.h_num , -10 , 10]);
%                     set(gcf,'Units','normalized','Position',[0 0 1 1]);
%                     image(reshape(trainData_x_batch(1 , :) , 28 , 28) , 'CDataMapping' , 'scaled');
%                 end