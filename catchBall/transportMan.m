function games = transportMan(N,T,S,sigma)
    % Tranposeter man - POMDP with hidden state whether we have the water
    % Have to transport in continous 2D space from bottom line to top
    % Always start in the middle
    % Actions are 1 - nothing, 2-up,3-down, 4-left, 5-right
    % Size of the square is from -N to N
    actions = [0 0 0 -1 1;0 1 -1 0 0];
    
    for i=1:N
        games{i}.frame = zeros(2,T+1);
        games{i}.reward = zeros([1,T+1]);
        games{i}.moves = randi(5,[1,T+1]);
        games{i}.optmoves = ones(1,T+1)*3;
        games{i}.condframe = zeros([1,T+1]);
        for j=1:T
            games{i}.frame(:,j+1) = min(max(games{i}.frame(:,j) + actions(:,games{i}.moves(j)) + randn(2,1)*sigma,-S),S);
            if(games{i}.condframe(j) == 0 && games{i}.frame(2,j+1) == -S)
                games{i}.condframe(j+1) = 1;
            elseif(games{i}.condframe(j) == 1 && games{i}.frame(2,j+1) == S)
                games{i}.condframe(j+1) = 0;
            else
                games{i}.condframe(j+1) = games{i}.condframe(j);
            end
            if(games{i}.condframe(j+1) == 0 && games{i}.condframe(j) == 1)
                games{i}.reward(j+1) = 5*S;
            elseif(games{i}.condframe(j+1) == 0)
                games{i}.reward(j+1) = -1;
            else
                games{i}.reward(j+1) = games{i}.frame(2,j+1)/(2*S)+0.5;
            end
            if(rand < 0.5)
                games{i}.moves(j+1) = games{i}.moves(j);
            end
            if(games{i}.condframe(j+1) == 0)
                games{i}.optmoves(j+1) = 3;
            else
                games{i}.optmoves(j+1) = 2;
            end
        end
        games{i}.length = T+1;
        games{i}.reward = games{i}.reward * 10;
    end
