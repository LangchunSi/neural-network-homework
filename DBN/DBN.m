classdef DBN
    properties
        trainData_x;        %训练数据 x
        testData_x;         %测试重构数据
        v_num;              %参数     v单元的个数
        h_num;              %参数     每个隐藏层单元的个数
        N_layer;            %参数     隐藏层的个数
        N_sample;           %样本总数 N_sample
        rbmList;            %将各隐藏层封装成列表
        IterMax;            %最大迭代次数
    end
    methods
        function obj = DBN(h_num)
            obj.v_num = h_num(1);
            obj.N_sample = size(obj.trainData_x,1);
            obj.h_num=h_num;
            obj.N_layer=length(h_num);
            % 创建各隐藏层,num_layer为一个数组，分别确定0,1,2，...层的单元个数
            rbmList(1:obj.N_layer-1) = pfor(@RBM , h_num(1:end-1) , h_num(2:end));
            obj.rbmList = rbmList;
            obj.IterMax = 3000;
        end
        function obj = train(obj , trainData_x , Iter)            %梯度下降训练
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
        function obj = predict(obj , num_batch)    %重构
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