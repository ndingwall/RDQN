function [err,varargout] = evaluateSample(dqn,w,wOld,history, k)
% Evaluates the error and the gradient if requested for applying the
% predictive features to the data chain.
% Keeps two networks - one with old parameters for the target update,
% and one with new parameters that get continually optimised.
% 
cost = 0;

%rng(5);
if k > length(history)
    error('Sample size is larger than total data points.');
end

dqn.w = w ;
dqnOld = dqn ; % Initialise network with old values
dqnOld.w = wOld ; % Set params in old network
if(nargout > 3)
    error('No more than two output arguments are supported - error and gradient.');
end

err = dqn.lambdaR * dqn.w' * dqn.w / 2;
mistakes = 0;
N = size(history,2);

% Choose subset of data
idx = randperm(N,k);
sample = history(idx);
% sample = history;

% Initialise grad
if nargout > 1
    grad = dqn; % Make a new DQN to store the gradient
    grad.w = dqn.lambdaR*dqn.w; % Initialise its gradient
end
filterLen = size(dqn.filterArch,2);
outputLen = size(dqn.outputArch,2)+1;
hsize = dqn.filterArch(end);
% Main loop
for thisSamp = 1:k
    % Extract info from this sample
    s = sample{thisSamp}.s;
    r = sample{thisSamp}.r;
    a = sample{thisSamp}.a;
    s2 = sample{thisSamp}.s2;
    terminal = sample{thisSamp}.t;
    time = sample{thisSamp}.time;
    % initialise arrays
    optA = sample{thisSamp}.optA;
    filterS = cell(1,filterLen);
    oldFilterS = cell(1,filterLen);
    outCur = cell(1,outputLen);
    outFut = cell(1,outputLen);
    % -----------------------------------------------------------------
    % ---------------------Filter calculations-------------------------
    % -----------------------------------------------------------------
    %
    % (Forward pass as far as bottleneck layer through filterB and filterW)
    filterS{1} = dqn.nodeFunc(dqn.filterB * [s; 1]);
    for i = 2:filterLen-1
        filterS{i} = dqn.nodeFunc(dqn.filterW{i} * [filterS{i-1}; 1]);
    end
    filterS{filterLen} = dqn.filterW{filterLen} * [filterS{filterLen - 1}; 1];
    %
    % (Forward pass using old parameters)
    oldFilterS{1} = dqnOld.nodeFunc(dqnOld.filterB * [s2; 1]);
    for i = 2:filterLen-1
        oldFilterS{i} = dqnOld.nodeFunc(dqnOld.filterW{i} * [oldFilterS{i-1}; 1]);
    end
    oldFilterS{filterLen} = dqnOld.filterW{filterLen} * [oldFilterS{filterLen - 1}; 1];

    % --------------------------------------------------------------------
    % --------------- GET Q VALUES FOR CURRENT STATE ---------------------
    % --------------------------------------------------------------------
    % Forward pass from bottleneck to output (one output per action)
    outCur{1} = dqn.nodeFunc(dqn.outputW{1} * [filterS{filterLen}; 1]) ;
    for i = 2:outputLen-1
        outCur{i} = dqn.nodeFunc(dqn.outputW{i} * [outCur{i-1}; 1]) ;
    end
    % Check if network agrees with optimal action
    if nargout > 2 % i.e. we want to record the number of mistakes the network makes
        qValues = dqn.outputW{outputLen} * [outCur{outputLen - 1}; 1] ;
        [~, mmm] = max(qValues);
        if(mmm ~= optA)
            mistakes = mistakes + 1/k ;
        end
        outCur{outputLen} = qValues(a); % store value of action taken
    else % if we don't want to check for mistakes, compute value of action taken
        outCur{outputLen} = dqn.outputW{outputLen}(a,:) * [outCur{outputLen-1}; 1] ;
    end
    % --------------------------------------------------------------------
    % ------------------ Calculating Q target ----------------------------
    % --------------------------------------------------------------------
    % I.e. forward pass from bottleneck through filterS and outputW
    % This uses the OLD network
    outFut{1} = dqnOld.nodeFunc(dqnOld.outputW{1} * [oldFilterS{filterLen}; 1]) ;
    for i = 2:outputLen-1
        outFut{i} = dqnOld.nodeFunc(dqnOld.outputW{i} * [outFut{i-1}; 1]);
    end
    outFut{outputLen} = dqnOld.outputW{outputLen} * [outFut{outputLen-1}; 1];
    [~,mi] = max(outFut{outputLen}); % Choose the best action according to old network
    if terminal == 1
        Qtarget = r - time*cost*abs(a-2);
        %fprintf('State [%2d, %2d], action: %d\n ---> [%2d, %2d], reward: %2d, Q target: %d\n\n',s(1),s(2),a,s2(1),s2(2),r, Qtarget);
    else
        Qtarget = r - time*cost*abs(a-2) + dqnOld.gamma * outFut{outputLen}(mi) ;
        %fprintf('State [%2d, %2d], action: %d\n ---> [%2d, %2d], reward: %2d, Q target: %1.2f\n\n',s(1),s(2),a,s2(1),s2(2),r, Qtarget);
    end
    err = err + (outCur{outputLen} - Qtarget).^2 / k ;

    % -------------------------------------------------------------
    % ---------------------BACKPROP--------------------------------
    % -------------------------------------------------------------
    if(nargout > 1)
        % Output Errors from current Q value
        outCurrE = zeros(dqn.actionSize,1);
        outCurrE(a) = 2 * (outCur{outputLen} - Qtarget) / k; % compute delta
        for i=outputLen:-1:2
            grad.outputW{i} = grad.outputW{i} + outCurrE * [outCur{i-1}; 1]';
            outCurrE = (dqn.outputW{i}(:,1:end-1)' * outCurrE) .* dqn.nodeFuncGrad(outCur{i-1});
        end
        grad.outputW{1} = grad.outputW{1} + outCurrE * [filterS{filterLen}; 1]';
        filterE = zeros(size(filterS{filterLen}));
        filterE = filterE + dqn.outputW{1}(:,1:end-1)' * outCurrE;
        
        % filterE(:,t) = (obj.filterW{1}(:,1:hsize)'*tempE + filterE(:,t));
        grad.filterW{filterLen} = grad.filterW{filterLen} + filterE * [filterS{filterLen-1}; 1]';
        tempE = (dqn.filterW{filterLen}(:,1:end-1)' * filterE) .* dqn.nodeFuncGrad(filterS{filterLen-1});
        for i=filterLen-1:-1:2
            grad.filterW{i} = grad.filterW{i} + tempE * [filterS{i-1}; 1]';
            tempE = (dqn.filterW{i}(:,1:end-1)' * tempE ).* dqn.nodeFuncGrad(filterS{i-1});
        end
        grad.filterB = grad.filterB + tempE * [s; 1]';
    end
end
if(nargout > 1)
    varargout{1} = grad.w;
end
if(nargout > 2)
    varargout{2} = mistakes;
end
end