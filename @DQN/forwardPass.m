function [ qValues ] = forwardPass( dqn, w, s, varargin)
%FORWARDPASS Performs a forward pass through a DQN.
% Input:
%	w 	weight vector
%	s 	state
% Output:
%	qValues	for each available action

if ~isempty(varargin)
    f = varargin{1};
    s = f(s);
end

dqn.w = w ;
filterLen = size(dqn.filterArch,2);
outputLen = size(dqn.outputArch,2)+1;
filterS = cell(1,filterLen);
outCur = cell(1,outputLen);

filterS{1} = dqn.nodeFunc(dqn.filterB * [s; 1]);
for i = 2:filterLen-1
    filterS{i} = dqn.nodeFunc(dqn.filterW{i} * [filterS{i-1}; 1]);
end
filterS{filterLen} = dqn.filterW{filterLen} * [filterS{filterLen - 1}; 1];

outCur{1} = dqn.nodeFunc(dqn.outputW{1} * [filterS{filterLen}; 1]) ;
for i = 2:outputLen-1
    outCur{i} = dqn.nodeFunc(dqn.outputW{i} * [outCur{i-1}; 1]) ;
end
qValues = dqn.outputW{outputLen} * [outCur{outputLen - 1}; 1] ;

end

