function confM = getConfusionMatrix(obj,games)
% Returns all prediction made of the outputs and a confusion matrix for
% this data set (FOR NOW ONLY FOR 1D outputs)
N = size(games,2);
confM = zeros(2,2,3);
if(obj.outputSize == 1)
    errf = @(x,y) log(1*(((2*y-1).*x)>=-50) + exp(-x.*(2*y-1) .* (((2*y-1).*x)>=-50))) - (2*y-1).*x .* (((2*y-1).*x) < -50);
    grad_errf = @(x,y) -(2*y-1) ./ (1 + exp((2*y-1) .* x));
    rewardf = @(x) x;
    % rewardf = @(x) x;
    % reward_grad = @(x,er) er;
else
    errf = @(x,y) - log(sum(x.*y));
    grad_errf = @(x,y) x - y;
    rewardf = @(x) exp(x) ./ repmat(sum(exp(x)),[size(x,1),1]);
    %reward_grad = @(x,er) (-x*x' + diag(x))*er;
    %reward_grad = @(x,y) x - y;
end
% x - vertically softmax vector
% y - binary vector which reward
% softmax = @(x) exp(x) ./ repmat(sum(exp(x)),[size(x,1),1]);
% softmax_grad = @(x,er) (-x*x' + diag(x))*er;
% Main loop
N = size(games,2);
s = zeros(obj.phiSize,1000);
ss = zeros(obj.phiSize,10000);
sp = zeros(obj.phiSize,10000);
rss = zeros(1,10000);
rs = zeros(1,10000);
ys = rs;
for game=1:N
    T = games{game}.length;
    x = games{game}.frame;
    % Make y's to be +/-1
    %y = games{game}.reward(1:T,:)'*2-1;
    y = games{game}.reward;
    m = games{game}.moves(1:T-1);
    % Filtered layers
    filterH = zeros(obj.hiddenSize,T);
    filterPhi = zeros(obj.phiSize,T);
    filterH(:,1) = obj.nodeFunc(obj.filterBin * [x(:,1); 1]);
    filterPhi(:,1) = obj.nodeFunc(obj.filterBh * [filterH(:,1); 1]);
    for i=2:T
        filterH(:,i) = obj.nodeFunc(obj.filterWin(:,:,m(i-1)) * [filterPhi(:,i-1); x(:,i); 1]);
        filterPhi(:,i) = obj.nodeFunc(obj.filterWh(:,:,m(i-1)) * [filterH(:,i); 1]);
    end
    % Solipsistic predictions
    solipH = zeros(obj.hiddenSize,T-1);
    solipPhi = zeros(obj.phiSize,T-1);
    for i=1:T-1
        solipH(:,i) = obj.nodeFunc(obj.solipWin(:,:,m(i)) * [filterPhi(:,i); 1]);
        solipPhi(:,i) = obj.nodeFunc(obj.solipWh(:,:,m(i)) * [solipH(:,i); 1]);
    end
    rewardsH = obj.nodeFunc(obj.rewardWin * [solipPhi; ones([1,T-1])]);
    rewards = rewardf(obj.rewardWh * [rewardsH; ones([1,T-1])]);
    for k=1:3
        for i=1:size(rewards,2)
            cords = getCorrectIndex(rewards(k,i)>0.5,y(k,i+1));
            confM(cords(1),cords(2),k) = confM(cords(1),cords(2),k) + 1;
        end
    end
end
end
function ind = getCorrectIndex(pred,target)
    if(pred > 0 && target > 0)
        % True positive
        ind = [1,1];
    elseif(pred == 0 && target > 0)
        % False negative
        ind = [2,1];
    elseif(pred > 0 && target == 0)
        % False positive
        ind = [1,2];
    elseif(pred == 0 && target == 0)
        ind = [2,2];
    end
end

    

