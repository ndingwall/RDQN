function [ newTrans, varargout ] = followCatchPolicy( dqn, w, epsilon, nNewTrans, gm, prob )
%FOLLOWCATCHPOLICY Plays some more games of catch with an epsilon-greedy
%policy
% INPUTS
%   dqn
%   w
%   epsilon
%   nNewTrans
%   H
%   W
%   prob
% OUTPUTS
%   newTrans
%   (optional): games

em = CatchEmulator();
counter = 0;

for game = 1:ceil(nNewTrans / (gm.H-1))
    time = 0;
    [o,r,f] = em.start(gm.H, gm.W, prob);
    if strcmp(class(dqn),'DQRNN')
        bottleNeckState = [];
        t = 0;
        mmm = 2;
    end
    a2 = [0;1;0]; % Always stay still first
    [o2, r2, f2] = em.act(a2);
    time = time + 1;
    if nargout > 1
        frame = [o(:), o2(:)];
        moves = [2];
        reward = [r, r2];
    end
    obs = ([reshape(o,[numel(o),1]); reshape(o2,[numel(o2),1])]  - gm.mean) / sqrt(gm.var) ;
    while f == 0
        counter = counter + 1;
        % ------------------Choose action------------------------
        % generate random num for epsilon greedy policy
        threshold = rand(1);
        if threshold < epsilon
            mmm = randi(3);
        else
            if strcmp(class(dqn),'DQN') || strcmp(class(dqn),'DQNrelu')
                [~, mmm] = max(forwardPass(dqn, w, obs));
            elseif strcmp(class(dqn),'DQRNN')
                t = t + 1;
                obs = (reshape(o,[numel(o),1]) - gm.mean) / sqrt(gm.var) ;
                [q, bottleNeckState] = dqn.forwardPass(w, obs, mmm, t, bottleNeckState);
                [~, mmm] = max(q);
            else
                error('DQN type not recognised');
            end    
        end
        if nargout > 1
            moves = [moves, mmm];
        end
        a2 = zeros(3,1);
        a2(mmm) = 1;
        newTrans{counter}.s = obs;
        newTrans{counter}.a = mmm;
        newTrans{counter}.time = time;
        o = o2;
        % ----------------Perform action ------------------------
        [o2, r2, f2] = em.act(a2);
        obs = ([reshape(o,[numel(o),1]); reshape(o2,[numel(o2),1])]  - gm.mean) / sqrt(gm.var) ;
        time = time + 1;
        % ----------------Store s2, r and t----------------------
        newTrans{counter}.r = r2;
        newTrans{counter}.s2 = obs;
        newTrans{counter}.t = f2;
        newTrans{counter}.optA = 2;
        f = f2;
        if nargout > 1
            frame = [frame, o2(:)];
            reward = [reward, r2];
        end
    end
    if nargout > 1
        newGames{game}.frame = (frame - gm.mean) / sqrt(gm.var);
        newGames{game}.reward = reward;
        newGames{game}.moves = [moves, 2];
        newGames{game}.optA = ones(1,length(reward));
        newGames{game}.terminal = 1;
        newGames{game}.length = length(reward);
    end
end

if nargout > 1
    varargout{1} = newGames;
end
