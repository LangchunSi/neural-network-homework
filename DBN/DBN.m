classdef DBN
    properties
        trainData_x;        %ѵ������ x
        testData_x;         %�����ع�����
        v_num;              %����     v��Ԫ�ĸ���
        h_num;              %����     ÿ�����ز㵥Ԫ�ĸ���
        N_layer;            %����     ���ز�ĸ���
        N_sample;           %�������� N_sample
        rbmList;            %�������ز��װ���б�
        IterMax;            %����������
    end
    methods
        function obj = DBN(h_num)
            obj.v_num = h_num(1);
            obj.N_sample = size(obj.trainData_x,1);
            obj.h_num=h_num;
            obj.N_layer=length(h_num);
            % ���������ز�,num_layerΪһ�����飬�ֱ�ȷ��0,1,2��...��ĵ�Ԫ����
            rbmList(1:obj.N_layer-1) = pfor(@RBM , h_num(1:end-1) , h_num(2:end));
            obj.rbmList = rbmList;
            obj.IterMax = 3000;
        end
        function obj = train(obj , trainData_x , Iter)            %�ݶ��½�ѵ��
            if(nargin == 2)
                Iter = obj.IterMax;
            end
            obj.trainData_x = trainData_x;
            obj.N_sample = size(obj.trainData_x,1);
            obj.rbmList(1) = obj.rbmList(1).train(trainData_x , Iter);
            obj.rbmList(1) = obj.rbmList(1).predict(obj.rbmList(1).N_sample);
            for i = 2 : obj.N_layer-1
                obj.rbmList(i) = obj.rbmList(i).train(obj.rbmList(i-1).testData_h , Iter);
                obj.rbmList(i) = obj.rbmList(i).predict(obj.rbmList(i).N_sample);
            end
        end
        function obj = predict(obj , num_batch)    %�ع�
            testData_x_batch = obj.rbmList(obj.N_layer-1).testData_x(randperm...
                (obj.rbmList(obj.N_layer-1).N_sample , num_batch) , :);
            for i = obj.N_layer-2 : -1 : 1
                obj.rbmList(i) = obj.rbmList(i).predict(num_batch , testData_x_batch);
                testData_x_batch = obj.rbmList(i).testData_x;
            end
            obj.testData_x = testData_x_batch;
        end
    end
end

function [output]=pfor(fun , varargin)
    Iter = length(varargin{1});
    nar = nargin(fun);
    arginList = cell(Iter , nar);
    for i = 1 : Iter
        for j = 1 : nar
            arginList{i , j} = varargin{j}(i);
        end
    end
    output(1 : Iter) = fun(arginList{i , :});
    for i = 1 : Iter
        output(i) = fun(arginList{i , :});
    end
end