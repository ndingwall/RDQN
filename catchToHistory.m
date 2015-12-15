function [ history ] = catchToHistory( games, varargin )
%GAMES2HISTORY Converts a series of games to a single history struct.
% Each element in history contains:
%   s - current state
%   a - action taken
%   r - reward received
%   s2 - next state
%   optA - optimal move
if ~isempty(varargin)
    gm = varargin{1};
    normalise = varargin{2};
else
    normalise = 0;
end

if ~isfield(games{1},'r')
    for i = 1:length(games)
        games{i}.r = games{i}.reward;
        games{i}.actions = games{i}.moves;
        %games{i}.optmoves = games{i}.optA;
    end
end

i = 0;
for thisGame = 1:length(games)
    g = games{thisGame};
    for state = 1:(length(g.r) - 2)
        i = i + 1;
        history{i}.s = ([g.frame(:,state); g.frame(:,state + 1)]);
        if normalise
            history{i}.s = (history{i}.s - gm.mean) / sqrt(gm.var);
        end
        history{i}.a = g.actions(state+1);
        history{i}.r = g.r(state+2);
        history{i}.s2 = ([g.frame(:,state+1); g.frame(:,state+2)]);
        if normalise
            history{i}.s2 = (history{i}.s2 - gm.mean) / sqrt(gm.var);
        end
        history{i}.optA = g.optmoves(state+1);
        if state == length(g.r) - 2
            history{i}.t = 1;
        else
            history{i}.t = 0;
        end
        history{i}.time = state;
        if isfield(g, 'b_loc')
            history{i}.s_b_loc = g.b_loc(:,state+1);
            history{i}.s2_b_loc = g.b_loc(:,state+2);
            history{i}.s_g_loc = g.g_loc(state+1);
            history{i}.s2_g_loc = g.g_loc(state+2);
        end
    end
end