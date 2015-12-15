function [totalR, history, games, bnecks, frames] = testTManPolicy(dqn, w, epsilon, gmInfo)



    % Tranposeter man - POMDP with hidden state whether we have the water
    % Have to transport in continouse 2D space from bottom line to top
    % Always start in the middle
    % Actions are 1 - nothing, 2-up,3-down, 4-left, 5-right
    % Size of the square is from -N to N
    sigma = 0;
    actions = [0 0 0 -1 1;0 1 -1 0 0];
    T = gmInfo.L;
    S = gmInfo.S;
    reporting = 0;
    randstart = 1; % Randomise start location?
    frames = [];
    bnecks = [];
    xstart = [-1,0,1];
    ystart = [-2,-1,0,1,2];
    totalR = zeros(1, length(xstart) * length(ystart));
    games = cell(1, length(xstart) * length(ystart));
    gNum = 0;
    
    for xs = xstart
        for ys = ystart
            gNum = gNum + 1;
            gm.frame = zeros((2*S+1)*(2*S+1),T+1);
            %gm.frame(2*S*S+2*S+1,1) = 1;
            gm.reward = zeros([1,T+1]);
            gm.moves = randi(5,[1,T+1]);
            gm.optmoves = ones(1,T+1)*2;
            gm.condframe = zeros([3,T+1]);
            gm.condframe(1,1) = ys; 
            gm.condframe(2,1) = xs;
            gm.frame(:,1) = coordsToFrame(gm.condframe(1:2,1), S);
            %%
            gm.t = zeros(1,T+1);
            gm.t(end) = 1;
            gm.time = 0:T;
            % Choose first move
            if strcmp(class(dqn),'DQN')
                mmm = 1; % Stay still on first move
            elseif strcmp(class(dqn), 'DQRNN')
                bottleNeckState = [];
                obs = (gm.frame(:,1) - gmInfo.mean) / sqrt(gmInfo.var);
                %fprintf('i=%d | j=%2d | t=%2d | btlLen = %d\n', i, j, gm.time(j+1), length(bottleNeckState));
                [q, bottleNeckState] = dqn.forwardPass(w,obs,1,1,bottleNeckState);
                [~, mmm] = max(q);
                frames = [frames; [gm.condframe(:,1)' 0]];
                bnecks = [bnecks; bottleNeckState'];
            end
            threshold = rand(1);
            if threshold > epsilon
                gm.moves(1) = mmm;
            end
            %
            %
            for j=1:T
                % Do first move
                gm.condframe(1:2,j+1) = min(max(gm.condframe(1:2,j) + actions(:,gm.moves(j)) + randn(2,1)*sigma,-S),S);
                % Make frame
                gm.frame(:,j+1) = coordsToFrame(gm.condframe(1:2,j+1),S);
                % Add to / remove from / keep in basket!
                if(gm.condframe(3,j) == 0 && gm.condframe(2,j+1) == -S)
                    gm.condframe(3,j+1) = 1;
                elseif(gm.condframe(3,j) == 1 && gm.condframe(2,j+1) == S)
                    gm.condframe(3,j+1) = 0;
                else
                    gm.condframe(3,j+1) = gm.condframe(3,j);
                end
                % Assign rewards
                if(gm.condframe(3,j+1) == 0 && gm.condframe(3,j) == 1)
                    gm.reward(j+1) = (gm.condframe(1,j+1) + S + 1);%5*S/2 ;%
                    %fprintf('%1.1f  ',gm.reward(j+1));
                elseif(gm.condframe(3,j+1) == 0)
                    gm.reward(j+1) = 0;%-1;
                else
                    gm.reward(j+1) = 0;%gm.condframe(2,j+1)/(2*S)+0.5;
                end
                % Use a random strategy (gm.moves initialised to random)
                if(rand < 0.5)
                    gm.moves(j+1) = gm.moves(j);
                end
                if strcmp(class(dqn),'DQN')
                    obs = [(gm.frame(:,j) - gmInfo.mean) / sqrt(gmInfo.var); (gm.frame(:,j+1) - gmInfo.mean) / sqrt(gmInfo.var)];
                    q = dqn.forwardPass(w, obs);
                elseif strcmp(class(dqn),'DQRNN')
                    obs = (gm.frame(:,j+1) - gmInfo.mean) / sqrt(gmInfo.var);
                    %fprintf('i=%d | j=%2d | t=%2d | btlLen = %d\n', i, j, gm.time(j+1), length(bottleNeckState));
                    [q, bottleNeckState] = dqn.forwardPass(w,obs,gm.moves(j),gm.time(j+1),bottleNeckState);
                    frames = [frames; [gm.condframe(:,j+1)' gm.moves(j)]];
                    bnecks = [bnecks; bottleNeckState'];
                    if reporting; fprintf('% d, % d, % d ||| S%1.3f, L%1.3f, R%1.3f, D%1.3f, U%1.3f\n',...
                        gm.condframe(1,j+1),gm.condframe(2,j+1),gm.condframe(3,j+1),...
                        q(1),q(2),q(3),q(4),q(5)); end
                end
                % Do optimal movement is epsilon exceeded
                threshold = rand(1);
                if threshold > epsilon
                    [~,mmm] = max(q);
                    gm.moves(j+1) = mmm;
                end
                if(gm.condframe(3,j+1) == 0)
                    gm.optmoves(j+1) = 2;
                else
                    gm.optmoves(j+1) = 3;
                end
            end
            gm.length = T+1;
            gm.reward = gm.reward / (2*S+1);
            totalR(gNum) = sum(gm.reward) / T;
            games{gNum} = gm;
        end
    end
    history = catchToHistory(games, gmInfo, 1);
end

function f = coordsToFrame(condf,S)
image = zeros(2*S+1);
image(round(condf(1) + S + 1), round(condf(2) + S + 1)) = 1;
f = reshape(image,[(2*S+1)*(2*S+1),1]);
end
