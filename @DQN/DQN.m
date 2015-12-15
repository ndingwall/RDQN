classdef DQN
    properties (SetAccess = immutable)
        % all arhcitetural variables (Assume all hidden layers same size)
        inputSize
        outputSize
        actions
        actionDim
        actionSize
        filterArch
        outputArch
        N           % effective number of parameters
    end
    properties 
        lambda      % Tradeoff between reward and disparity error
        lambdaR     % Regularizer factor
        gamma       % Discount factor
        numMoves
        % Sampling
        samples
        samplesSigma
        seed
        % Matrix form of parameters
        filterB     % Bias of the filtered dist
        filterW     % Cell array of filtered weight matricies
        outputW     % Cell array of output weight matricies
        nodeFunc    % The non linearity function
        nodeFuncGrad % The gradient of the non-linearity function, given the outputs, e.g. x .* (1-x) for sigmoid and 
        isTerminal
    end
    properties (Dependent)
        w       % vector representation of the parameters
    end
    methods
        % For optimisation packages
        function [err,grad] = evaluateDataAt(obj,w,games,gradOn)
            obj.w = w;
            [err,grad] = obj.evaluateData(games,gradOn);
            if(gradOn == 1)
                grad = grad.w;
            else 
                grad = 0;
            end
        end
        % Constructor
        function obj = DQN(inputSize,outputSize,actions,filterArch, outputArch)
            obj.inputSize = inputSize;
            obj.outputSize = outputSize;
            obj.actions = actions;
            [obj.actionDim, obj.actionSize] = size(obj.actions);
            obj.filterArch = filterArch;
            obj.outputArch = outputArch;
            % NB - whnever there is a +1 it is for bias
            % Filter params
            obj.N = obj.filterArch(1) * (obj.inputSize + 1);
            obj.N = obj.N + obj.filterArch(1) * (obj.inputSize + obj.actionDim +  obj.filterArch(end) + 1);
            for i=2:size(obj.filterArch,2)
                obj.N = obj.N + obj.filterArch(i) * (obj.filterArch(i-1) + 1);
            end
            % Output params
            obj.N = obj.N + obj.outputArch(1) * (obj.filterArch(end) + 1);
            for i=2:size(obj.outputArch,2)
                obj.N = obj.N + obj.outputArch(i) * (obj.outputArch(i-1) + 1);
            end
            obj.N = obj.N + obj.actionSize * (obj.outputArch(end) + 1);
            obj.isTerminal = 0;
            obj.samples = 100;
            obj.samplesSigma = 1;
            obj.seed = 0;
            obj.lambda = 0;
            obj.lambdaR = 0;
            obj.gamma = 0.95;
            obj.nodeFunc = @(x) 1.5930 .* tanh(x);
            obj.nodeFuncGrad = @(x) 1.5930  - x.^2/1.5930;
            obj.w = zeros(obj.N,1);
            obj.filterB = sqrt(3) * 2*(rand(size(obj.filterB)) -0.5)/ sqrt(size(obj.filterB,2));
            obj.filterB(:,end) = 0;
            for i=1:size(obj.filterW,2)
                obj.filterW{i} = sqrt(3) * 2*(rand(size(obj.filterW{i})) - 0.5) / sqrt(size(obj.filterW{i},2));
                obj.filterW{i}(:,end) = 0;
            end
            for i=1:size(obj.outputW,2)
                obj.outputW{i} = sqrt(3) * 2*(rand(size(obj.outputW{i})) - 0.5) / sqrt(size(obj.outputW{i},2));
                obj.outputW{i}(:,end) = 0;
            end
        end
        % Setter methods to keep w consistent with the Matrix parameters
        function obj = set.w(obj,val)
            if(~all([obj.N,1] == size(val)))
                error('Can not assign inconsistent size vector![Expected:(%d,%d),was (%d,%d)',[obj.N,1],size(val));
            end
            % NB - whnever there is a +1 it is for bias
            index = 1;
            % filterB
            obj.filterB = reshape(val(index:(index - 1 + obj.filterArch(1) * (obj.inputSize + 1))),[obj.filterArch(1), (obj.inputSize + 1)]);
            index = index + obj.filterArch(1) * (obj.inputSize + 1);
            % filterW
            obj.filterW{1} = reshape(val(index:(index - 1 + obj.filterArch(1) * (obj.inputSize + obj.actionDim + obj.filterArch(end) + 1))),...
                [obj.filterArch(1), (obj.inputSize + obj.actionDim +  obj.filterArch(end) + 1)]);
            index = index + obj.filterArch(1) * (obj.inputSize + obj.actionDim + obj.filterArch(end) + 1);
            for i=2:size(obj.filterArch,2)
                obj.filterW{i} = reshape(val(index:(index - 1 + obj.filterArch(i) * (obj.filterArch(i-1) + 1))),[obj.filterArch(i), (obj.filterArch(i-1) + 1)]);
                index = index + obj.filterArch(i) * (obj.filterArch(i-1) + 1);
            end
            % outputW 
            obj.outputW{1} = reshape(val(index:(index - 1 + obj.outputArch(1) * (obj.filterArch(end) + 1))),[obj.outputArch(1), (obj.filterArch(end) + 1)]);
            index = index + obj.outputArch(1) * (obj.filterArch(end) + 1);
            for i=2:size(obj.outputArch,2)
                obj.outputW{i} = reshape(val(index:(index - 1 + obj.outputArch(i) * (obj.outputArch(i-1) + 1))),[obj.outputArch(i), (obj.outputArch(i-1) + 1)]);
                index = index + obj.outputArch(i) * (obj.outputArch(i-1) + 1);
            end
            obj.outputW{size(obj.outputArch,2)+1} = reshape(val(index:(index - 1 + obj.actionSize*(obj.outputArch(end) + 1))),[obj.actionSize, (obj.outputArch(end)+1)]);
        end
        function w = get.w(obj)
            w = zeros(obj.N,1);
            index = 1;
            [D1,D2,D3] = size(obj.filterB);
            % filterB
            w(index:(index+D1*D2*D3-1),:) = reshape(obj.filterB,[D1*D2*D3,1]);
            index = index + D1*D2*D3;
            % filterW
            for i=1:size(obj.filterW,2)
                [D1,D2,D3] = size(obj.filterW{i});
                w(index:(index+D1*D2*D3-1),:) = reshape(obj.filterW{i},[D1*D2*D3,1]);
                index = index + D1*D2*D3;
            end
            % outputW
            for i=1:size(obj.outputW,2)
                [D1,D2,D3] = size(obj.outputW{i});
                w(index:(index+D1*D2*D3-1),:) = reshape(obj.outputW{i},[D1*D2*D3,1]);
                index = index + D1*D2*D3;
            end
        end
    end
end