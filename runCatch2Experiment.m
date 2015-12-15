function id = runCatch2Experiment(minibatchSize, nInnerLoops, learningRate, dqnType, nHours, saveW, wStart)
% Trains either a DQN or a DQRNN to play the Catch game, either using just
% experience replay or epsilon greedy.
% INPUT
%   minibatchSize   number of frames/games to use for each gradient step
%   nInnerLoops     number of interations to complete with a fixed target
%                   network
%   learningRate    0.001 works well
%   dqnType         1 = DQN (tanh); 2 = DQN (rectified linear); 
%                   3 = DQRNN (tanh); 4 = DQRNN (rectified linear)
%   saveW           save the weight vector
%   wStart          provide an initial weight vector

startTime = tic;
prob = 0; % Probability that paddle goes opposite direction
dataName = 'catch2Data';
gm.H = 12; gm.W = 3; % gm contains some properties of the game. Here we specify the height and width of the grid
numExperiments = 1; % Number of times to repeat the whole experiment
nFuncEvals = 50000000; % number of function evaluations desired 
nInitTrans = 25000; % number of transitions to generate with random policy
nNewTrans = 2500; % number of transitions to generate in each session
initEpsilon = 1.0; % initial epsilon value for epsilon-greedy policy
minEpsilon = 0.1; % minimum value of epsilon
%nInnerLoops = 10; % number of updates to Q network to make before updating target network
nOuterLoops = 1; % number of times to update target network before generating new transitions
%minibatchSize = 32; % number of transitions to use for SGD step
memoryLength = nInitTrans; % number of transitions to remember
%learningRate = 1/1000; % learning rate
saveEvery = ceil(1000/nInnerLoops); % How often to compute wincount and save w.
if any(dqnType == [1 3])
    nodeFn = 'tanh';
else
    nodeFn = 'relu';
end
if any(dqnType == [1 2])
    architecture = 'DQN';
else
    architecture = 'DQRNN';
end
filterArch = [250 100];
outputArch = [200 100 50];
if any(dqnType == [3,4])
    nSessions = ceil(nFuncEvals / (minibatchSize * gm.H * nInnerLoops)) % Compute number of sessions to play
    nNewTrans = ceil(nNewTrans / gm.H);
    memoryLength = ceil(memoryLength / gm.H);
else
    nSessions = ceil(nFuncEvals / (minibatchSize * nInnerLoops)) % Compute number of sessions to play
end
epsilonAnneal = (1 - 0.1) / nSessions; % amount to reduce epsilon by after each session

% %% Save settings
formatOut = 'yy-mm-dd--HH-MM-SS';
id = datestr(now,formatOut);
% fileID = fopen(strcat('results/catch_',id,'_Settings.csv'),'w');
% fprintf(fileID,'nSessions,%d\nnInitTrans,%d\nnNewTrans,%d\ninitEpsilon,%1.2f\nminEpsilon,%1.2f\nepsilonAnneal,%1.7f\nnInnerLoops,%d\nnOuterLoops,%d\nminibatchSize,%d\nmemoryLength,%d\nlearningRate,%1.7f\nfilterArch,%s\noutputArch,%s\nnodeFn,%s\narchitecture,%s\n',...
%     nSessions,nInitTrans,nNewTrans,initEpsilon,minEpsilon,epsilonAnneal,nInnerLoops,nOuterLoops,...
%     minibatchSize,memoryLength,learningRate,num2str(filterArch),num2str(outputArch),nodeFn,architecture);
% fclose(fileID);
% settingsID = strcat('results/catch_',id,'_settings.mat');
% save(settingsID, 'id', 'nSessions', 'nInitTrans', 'nNewTrans', 'initEpsilon', 'minEpsilon', 'epsilonAnneal', 'nInnerLoops', 'nOuterLoops', 'minibatchSize', 'memoryLength', 'learningRate', 'filterArch', 'outputArch', 'nodeFn', 'architecture', 'gm');

%% Work out how to normalise games
inputDim = (2*gm.H+1)*(2*gm.W+1);
sss = zeros(inputDim,1);
sss(1:2) = 1;
gm.mean = mean(sss);
gm.var = var(sss);

%% train for nSessions, and repeat numIters times
winCount = zeros(numExperiments,nSessions);
timings = zeros(numExperiments,nSessions);
wChange = zeros(numExperiments,nSessions);
wMaxChange = zeros(numExperiments,nSessions);
SSE = zeros(numExperiments, nSessions);
% Make file names
weightsID = strcat('results/catch_',id,'_wHist','.mat');
resultsID = strcat('results/catch_',id,'_results','.mat');

prematureTermination = 0;

for expt = 1:numExperiments
    fprintf('\n-------------------------------------------------');
    fprintf('\n---------------Experiment %d---------------------',expt);
    fprintf('\n-------------------------------------------------\n\n');
    % Generate some initial transitions using random policy
    load(dataName);
    % Check if games are normalised
    if any(games{1}.frame(:) == 0)
        normalise = 1;
    else
        normalise = 0;
    end

    if isfield(games{1},'r')
        for i = 1:length(games)
            games{i}.reward = games{i}.r;
            games{i}.moves = games{i}.actions;
        end
    else
        for i = 1:length(games)
            games{i}.r = games{i}.reward;
            games{i}.actions = games{i}.moves;
            games{i}.optmoves = games{i}.optA;  
        end
    end

    if any(dqnType == [1,2])
        history = catchToHistory(games, gm, normalise);
        clear games;
    elseif any(dqnType == [3,4])
        for i = 1:length(games)
             games{i}.frame = (games{i}.frame - gm.mean) / sqrt(gm.var);
        end
    else
        error('DQN type not recognised');
    end
    
    % Initialise a DQN
    if dqnType == 1
        dqn = DQN(inputDim*2, 3, [1 2 3], filterArch,outputArch);
    elseif dqnType == 2
        dqn = DQN(inputDim*2, 3, [1 2 3], filterArch,outputArch);
        dqn.nodeFunc = @(x) sqrt(2) * max(0,x);
        dqn.nodeFuncGrad = @(x) (x>0) / sqrt(2);
    elseif dqnType == 3
        dqn = DQRNN(inputDim, [1 2 3], filterArch, outputArch);
    elseif dqnType == 4
        dqn = DQRNN(inputDim, [1 2 3], filterArch, outputArch);
        dqn.nodeFunc = @(x) sqrt(2) * max(0,x);
        dqn.nodeFuncGrad = @(x) (x>0) / sqrt(2);
        dqn.useIdentity();
    else
        error('dqnType must be a number from 1 to 4 (1: tanh, 2: relu, 3: RNN w/ tanh, 4: RNN w/ relu.')
    end
    
    % Use initial weight vector if provided
    if sum(wStart) ~= 0
        dqn.w = wStart;
        fprintf('Set start weights');
    end

    % Set parameters
    epsilon = initEpsilon + epsilonAnneal ; % This forces the first epsilon value to be initEpsilon
    wOld = dqn.w;

    AS = initaliseAdam(learningRate);        

    % Train for nSessions
    for session = 1:nSessions
        fprintf('\n---- Session %d : total interations so far = %d ----\n',session,AS.steps);
        startThisSession = tic;
        % train on current data
        if any(dqnType == [1,2])
            [wNew, AS] = learnDQN(dqn, wOld, history(max(1,(end-memoryLength+1)):end), ...
                nInnerLoops, nOuterLoops, minibatchSize, AS, 1);
        elseif any(dqnType == [3,4])
            [wNew, AS] = learnDQN(dqn, wOld, games, ...
                nInnerLoops, nOuterLoops, minibatchSize, AS, 1);
        else
            error('DQN type not recognised');
        end
        timings(expt,session) = toc(startThisSession);
        % Generate some more transitions using this policy
        epsilon = minEpsilon + (1 - ( toc(startTime) / (( nHours - 0.5) * 3600))) * (initEpsilon - minEpsilon);
        fprintf('Epsilon: %1.5f\n', epsilon);
        if (nNewTrans > 0) && (dqnType < 3)
            % Compute new epsilon
            %epsilon = max(epsilon - epsilonAnneal, minEpsilon);
            newTrans = [history, followCatch2Policy(dqn, wNew(:,end),epsilon, nNewTrans, gm, prob )];
            history = newTrans(max(1,(end - memoryLength + 1)) : end);
        end
        if (nNewTrans > 0) && (dqnType >= 3)
            %epsilon = max(epsilon - epsilonAnneal, minEpsilon);
            [~, newTrans] = followCatch2Policy(dqn, wNew(:,end), epsilon, nNewTrans, gm, prob);
            games = [games, newTrans];
            games = games(max(1,(end - memoryLength + 1)) : end);
        end
        if mod(session, saveEvery) == 0
            [winCount(expt,session), SSE(expt,session), ~] = testCatch2Policy(gm, dqn, wNew(:,end), prob);
            fprintf('Win count is now %2.4f. SSE is %5.2f\n',winCount(expt,session), SSE(expt,session));
        end
        % Store the distance between old and new weight vector
        wChange(expt,session) = norm(wOld - wNew(:,end));
        wMaxChange(expt,session) = max(abs(wOld - wNew(:,end)));
        wOld = wNew(:,end);
        wFinal = wNew(:,end);
        if saveW == 1
            save(weightsID,'wFinal');
        elseif saveW ~= 0
            error('saveW must be 0 (do not save) or 1 (save weights).')
        end
        if toc(startTime) > (nHours-0.5)*60*60
            save('earlyTermination','expt', 'session');
            fprintf('Terminated prematurely after %d experiments and %d sessions (out of %d:%d)\n',...
                expt, session, numExperiments, nSessions);
            prematureTermination = 1;
            break
        end
    end
    if prematureTermination == 1
        break
    end
end
    
    % Save results for this experiment
if (nNewTrans == 0)
    save(resultsID,'winCount','wChange','wMaxChange','timings','SSE');
elseif any(dqnType == [1 2])
    save(resultsID,'history','winCount','wChange','wMaxChange','timings','SSE');
elseif any(dqnType == [3 4])
    save(resultsID,'games','winCount','wChange','wMaxChange','timings','SSE');
end

end

function [AS] = initaliseAdam(lr)
    AS.mt = 0;
    AS.vt = 0;
    AS.b1 = 0.9;
    AS.b2 = 0.999;
    AS.alpha = lr;
    AS.lambda = 1 - 10^-8;
    AS.steps = 0;
end
