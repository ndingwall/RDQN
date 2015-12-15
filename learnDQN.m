function [ w1, AS ] = learnDQN( dqn,w,history,innerIters,outerIters, k , AS, varargin)
% LEARNDQN Learns Q values using a DQN. Manages two networks: one that it 
% updates continually, and a target network that only gets updated 
% periodically.
%
% learnDQN( dqn,w,history,innerIters,outerIters, k , lr , parallel)
%
% Note this uses its own implementation of Adam.
%   INPUTS
%   history     - experience in the form of a cell array, each containing
%               s (state), a (action), r (reward), s2 (next state), and 
%               optA (optimal action).
%   innerIters  target network update frequency (performs this number of
%               iterations of Adam algorithm before updating target
%               network.
%   outerIters  number of updates to the target distribution. (Therefore
%               total number of iterations is innerIters * outerIters).
%   k           size of minibatch
%   AS          hyperparameters related to Adam implementation to retain
%               per-parameter learning rates etc between loops
%   varargin    first element gives the number of previous weight vectors
%               to store
%
%   OUTPUTS
%   w1          The optimal w produced after each outer loop

if isempty(varargin)
    memLen = 1000000;
else
    memLen = varargin{1};
end
    
wOld = w ;
w1 = w ;

for i = 1:outerIters
    tic
    fprintf('Outer loop %d\n',i)
    % Do inner loop
    if strcmp(class(dqn),'DQN') || strcmp(class(dqn),'DQNrelu')
        ff = @(w, hist) dqn.evaluateSample(w, wOld, hist, k);
        [wOpt,AS] = adamSampled(ff, w1(:,end), history, innerIters, 1, AS);
    elseif strcmp(class(dqn),'DQRNN')
        ff = @(w, games) dqn.evaluateSample(w, wOld, games, k);
        [wOpt,AS] = adamSampled(ff, w1(:,end), history, innerIters, 1, AS);
    else
        error('DQN type is not recognised');
    end
    wOld = wOpt(:,end) ;
    w1 = [w1, wOld] ;
    w1 = w1(:, max(1, size(w1,2) - memLen):end);
    toc
end

function [w,AS,varargout] = adamSampled(ff,w1,history,iter,K,AS)
% This is the same as usual, except it doesn't report back on every
% iteration. It also skips the check for the value of K (to avoid confusion
% with the sampling regime in evaluateSample).
    N = size(history,2);
    %     f = @(x) ff(x,games);
    e = zeros(1,iter);
    w = zeros(size(w1,1),iter);
    dw = zeros(size(w1));
    if(nargout > 2)
        e2 = zeros(1,iter);
    end
    for t=1:iter
        if(K>1)
            if(mod(t + AS.steps,K) == 1)
                ind = randperm(N);
                j = 1;
            else
                j = j+1;
            end
            f = @(x) ff(x,history(ind(1+(j-1)*N/K:j*N/K)));
        else
            f = @(x) ff(x,history);
        end
        AS.bt = AS.b1 * AS.lambda^(AS.steps + t - 1);
        if(t > 1)
            if(nargout < 3)
                [e(t),dw] = f(w(:,t-1));
            elseif(nargout == 3)
                [e(t),dw,e2(t)] = f(w(:,t-1));
            end
        else
            if(nargout < 3)
                [e(t),dw] = f(w1);
            elseif(nargout == 3)
                [e(t),dw,e2(t)] = f(w1);
            end
        end
        AS.mt = AS.bt .* AS.mt + (1 - AS.bt) .* dw;
        AS.vt = AS.b2 .* AS.vt + (1 - AS.b2) .* (dw.^2);
        mhat = AS.mt / (1 - AS.b1^(t+AS.steps));
        vhat = AS.vt / (1 - AS.b2^(t+AS.steps));
        if(t > 1)
            w(:,t) = w(:,t-1) - AS.alpha.*mhat ./ (sqrt(vhat)+10^-8);
        else
            w(:,t) = w1 - AS.alpha.*mhat ./ (sqrt(vhat)+10^-8);
        end
        if(nargout < 3)
           fprintf('Iteration[%d] - Error: %.5f, MaxGrad - %.5f\n', t, e(t), max(abs(dw)));
        elseif(nargout == 3)
           fprintf('Iteration[%d] - Error: %.5f, MaxGrad - %.5f, Mistakes - %.5f\n', t, e(t), max(abs(dw)),e2(t));
        end
        
    end
    AS.steps = AS.steps + iter;
    if(nargout > 2)
        varargout{1} = e;
    end
    if(nargout > 3)
        varargout{2} = e2;
    end
end


end
