function [winCount, SSE, wins] = testCatchPolicy(gm, obj, w, prob)
% TESTCATCHPOLICY Computes the number of wins by deterministically trying
% all possible games.
%
% testCatchPolicy(H,W,dqn,w)
%   H       height of grid
%   W       width of grid
%   dqn     identifier of DQN objec
%   w       weight vector to use
%   prob    probability of hand moving the opposite direction
%
% OUTPUTS
%   winCount    number of wins (out of 2*(2*W + 1))
%   wins        wins for each (horizontal) final position of the ball

if prob == 0
    n = 1;
else
    n = 10;
    %fprintf('Averaging over %d trials.',n);
end
animate = 0;

em = CatchEmulator();

%% Load best (0-error) DQN
if (gm.H == 10 && gm.W == 3)
    dqn = DQN(294, 3, [1 2 3], [200 50], [50 25]);
    dqn.nodeFunc = @(x) max(0,x);
    dqn.nodeFuncGrad = @(x) (x>0);
    load wSmallerZeroError;
    dqn.w = wFinal;
    clear wFinal;
end
SSE = 0;
%%

r_hist = zeros(n, (2*(2*gm.W+1)));
final_pos = zeros(n, (2*(2*gm.W+1)));
if animate; figure; hold on; end
for run = 1:n
    game = 0;
    for dir = -1:2:1
        for start_pos = 1:(2*gm.W + 1)
            game = game + 1;
            %fprintf('\n\n----------- Game %d ----------------\n',game);
            [o,r,f] = em.forceStart(gm.H, gm.W, prob, dir, start_pos);
            if strcmp(class(obj),'DQRNN')
                obs = (reshape(o,[numel(o),1]) - gm.mean) / sqrt(gm.var);
                [qEst, bottleNeckState] = obj.forwardPass(w, obs, 0, 1, []);
                t = 1;
            end
            if animate; em.displayMe; end
            mmm = 2;
            a2 = [0;1;0];
            while f == 0
                [o2, r2, f2] = em.act(a2);
                if animate; pause; em.displayMe; end
                if strcmp(class(obj),'DQN') || strcmp(class(obj),'DQNrelu')
                    obs = ([reshape(o,[numel(o),1]); reshape(o2,[numel(o2),1])] - gm.mean) / sqrt(gm.var) ;
                    q = obj.forwardPass(w,obs);
                    [~, mmm] = max(q);
                    if (gm.H == 10 && gm.W == 3)
                        obs = ([reshape(o,[numel(o),1]); reshape(o2,[numel(o2),1])] - gm.mean) / sqrt(gm.var) ;
                        qTrue = dqn.forwardPass(dqn.w, obs); 
                        SSE = SSE + sum((q - qTrue).^2);
                    end
                    if animate; fprintf('% 2.3f | % 2.3f | % 2.3f ||| %d\n', q(1),q(2),q(3),mmm); end
                elseif strcmp(class(obj),'DQRNN')
                    t = t + 1;
                    obs = (reshape(o2,[numel(o2),1]) - gm.mean) / sqrt(gm.var);
                    [q, bottleNeckState] = obj.forwardPass(w, obs, mmm, t, bottleNeckState);
                    obs2 = ([reshape(o,[numel(o),1]); reshape(o2,[numel(o2),1])] - gm.mean) / sqrt(gm.var) ;
                    qTrue = dqn.forwardPass(dqn.w, obs2); 
                    SSE = SSE + sum((q - qTrue).^2);
                    [~, mmm] = max(q);
                    if animate; fprintf('% 2.3f | % 2.3f | % 2.3f ||| %d\n', q(1),q(2),q(3),mmm); end
                else
                    error('DQN type not recognised');
                end
                a2 = zeros(3,1);
                a2(mmm) = 1;
                o = o2;
                r = r2;
                f = f2;
            end
            r_hist(run, game) = r;
            final_pos(run, game) = em.b_loc(2);
            if animate; pause(1); end
        end
    end
end

winCount = sum(r_hist(:) == 1) / n;

wins = zeros(max(final_pos(:)),2);
for i = 1:max(final_pos(:))
    wins(i,1) = sum(final_pos(:) == i);
    wins(i,2) = sum((r_hist(:) == 1) .* (final_pos(:) == i));
end
end