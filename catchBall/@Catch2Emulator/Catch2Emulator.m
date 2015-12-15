classdef Catch2Emulator < AbstractEmulator
    % CatchEmulator - simulates a catch game -
    % simpler version of the pong game
    properties (SetAccess = private, Hidden = true)
        prob % Probability to do opposite action, 0 - deterministic
        Vs % Vertical speed
        Hs % Horizontal speed
        H % height of the screen si 2*H + 1
        W % width of the screen is 2*W + 1
        loc % your current location
        b_loc % location of the ball [H;W]
        b_dir % current direction of the ball (-1 left, 1 right)
        locHistory
        b_locHistory
        b_dirHistory
    end
    properties
        actions = eye(3);
        numActions = 3;
    end
    methods
        function [o,r,f] = start(this, H, W, prob)
            this.H = H;
            this.W = W;
            this.prob = prob;
            this.Vs = randsample(1:3,1); %randi(2);
            this.Hs = randsample(0:3,1); %randi(4) - 1;
            this.loc = W+1;
            this.b_loc = [1; randsample(1:(2*W + 1), 1)]; % randi(2*W+1)];
            this.b_dir = randsample(-1:2:1,1);%randi(2)*2 - 3;
            this.locHistory = this.loc;
            this.b_locHistory = this.b_loc;
            this.b_dirHistory = this.b_dir;
            [o,r,f] = generateObservations(this);
%             fprintf('dir: %d | Vs: %d | Hs: %d | b_loc: %d\n',...
%                 this.b_dir, this.Vs, this.Hs, this.b_loc(2));
        end
        function [o,r,f] = reset(this)
            [o,r,f] = this.start(this.H, this.W, this.prob);
        end
        function [o,r,f] = act(this,action)
            if this.b_loc(1) >= 2*this.H+1
                error('Game has ended');
            end
            action = logical(action);
            if ~all(size(action) == [3,1]) || sum(action) ~= 1
                error('The action should be a vector of size [3,1] being an indicator of [left, stay, right] actions');
            end
            if rand < this.prob
                action = flipud(action);
            end
            this.loc = min(max(-action(1) + action(3)+this.loc,1),2*this.W+1);
            this.b_loc = this.b_loc + [this.Vs; this.Hs*this.b_dir];
            if this.b_loc(2) < 2
                this.b_dir = 1;
                this.b_loc(2) = 2 - this.b_loc(2);
            elseif this.b_loc(2) > 2*this.W
                this.b_dir = -1;
                this.b_loc(2) = 4*this.W+2 - this.b_loc(2);
            end
            this.locHistory = [this.locHistory, this.loc];
            this.b_locHistory = [this.b_locHistory, this.b_loc];
            this.b_dirHistory  = [this.b_dirHistory , this.b_dir];
            [o,r,f] = generateObservations(this);
        end
        function [o,r,f] = forceStart(this, H, W, prob, Vs, Hs, dir, loc)
            % dir should be +/- 1
            % loc should be integer in range [1, 2*W+1]
            this.H = H;
            this.W = W;
            this.prob = prob;
            this.Vs = Vs;
            this.Hs = Hs;
            this.loc = W+1;
            this.b_loc = [1; loc];
            this.b_dir = dir;
            this.locHistory = this.loc;
            this.b_locHistory = this.b_loc;
            this_b_dirHistory = this.b_dir;
            [o,r,f] = generateObservations(this);
        end
        function displayMe(this)
            [o,~,~] = generateObservations(this);
            imagesc(o);
            axis equal tight;
        end
    end
    methods (Access = private)
        function [o,r,f] = generateObservations(this)
            o = zeros(2*this.H+1, 2*this.W+1);
            o(end,this.loc) = 1;
            o(this.b_loc(1),this.b_loc(2)) = 1;
            if this.b_loc(1) >= 2*this.H+1
                f = 1;
                if this.b_loc(2) == this.loc
                    r = 1;
                else
                    r = -1;
                end
            else
                f = 0;
                r = 0;
            end
        end
    end
end