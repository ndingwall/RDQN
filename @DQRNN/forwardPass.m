function [q, bottleNeckState] = forwardPass(dqn, w, state, prev_action, t, bottleNeckState)
    % Evaluates the error and the gradient if requested for applying the
    % predictive features to the data chain
    
    % Optionally specify the bottleneck state
    %if ~isempty(varargin)
%         dqn.bottleNeckState = varargin{1};
%         fprintf('%d, updated bottleNeckState.\n',t);
    %end
    
    dqn.w = w;

    filterLen = size(dqn.filterArch,2);
    outputLen = size(dqn.outputArch,2)+1;
%    hsize = dqn.filterArch(end);
    
    filterS = cell(1,filterLen);
    outCur = cell(1,outputLen);
    % Get first hidden state
    if t == 1
        filterS{1} = dqn.nodeFunc(dqn.filterB * [state; 1]);
    else
        filterS{1} = dqn.nodeFunc(dqn.filterW{1} * [bottleNeckState; state; prev_action; 1]); % filterS{end}(:,t-1)
    end
    % Propogate to the bottleneck
    for i=2:filterLen-1
        filterS{i} = dqn.nodeFunc(dqn.filterW{i} * [filterS{i-1}; 1]);
    end
    filterS{filterLen} = dqn.filterW{filterLen} * [filterS{filterLen-1}; 1];
    bottleNeckState = filterS{filterLen};

    outCur{1} = dqn.nodeFunc(dqn.outputW{1} * [filterS{filterLen}; 1]);
    for i=2:outputLen-1
        outCur{i} = dqn.nodeFunc(dqn.outputW{i} * [outCur{i-1}; 1]);
    end
%     if nargout > 2
%         qValues = dqn.outputW{outputLen} * [outCur{outputLen-1}; 1];
%         [~,mmm] = max(qValues);
%         if(mmm ~= games{game}.optmoves(t))
%             err2 = err2 + 1/(k*(T-1));
%         end
%     end
%    outCur{outputLen} = dqn.outputW{outputLen}(m(t),:) * [outCur{outputLen-1}; 1];
    q = dqn.outputW{outputLen} * [outCur{outputLen-1}; 1];

    
    
    
%     
%     % Main loop
% %     for game = sample
%         T = games{game}.length;
%         x = games{game}.frame;
%         r = games{game}.reward;
%         m = games{game}.moves;
%         v = zeros(dqn.actionSize,T);
%         % initialise array
%         filterS = cell(1,filterLen);
%         oldFilterS = cell(1,filterLen);
%         outCur = cell(1,outputLen);
%         outFut = cell(1,outputLen);
%         % -----------------------------------------------------------------
%         % ---------------------Filter calculations-------------------------
%         % -----------------------------------------------------------------
%         %
%         % (Forward pass as far as bottleneck layer through filterB and filterW)        
%         for t=1:T
%             if(t==1)
%                 filterS{1}(:,t) = dqn.nodeFunc(dqn.filterB * [x(:,t); 1]);
%             else
%                 filterS{1}(:,t) = dqn.nodeFunc(dqn.filterW{1} * [dqn.bottleNeckState; x(:,t); dqn.actions(:,m(t-1)); 1]); % filterS{end}(:,t-1)
%             end
%             for i=2:filterLen-1
%                 filterS{i}(:,t) = dqn.nodeFunc(dqn.filterW{i} * [filterS{i-1}(:,t); 1]);
%             end
%             filterS{filterLen}(:,t) = dqn.filterW{filterLen} * [filterS{filterLen-1}(:,t); 1];
%             dqn.bottleNeckState = filterS{filterLen}(:,t);
%         filterE = zeros(size(filterS{filterLen}));
%         % --------------------------------------------------------------------
%         % --------------- GET Q VALUES FOR CURRENT STATE ---------------------
%         % --------------------------------------------------------------------
%         % Forward pass from bottleneck to output (one output per action)
%         for t=1:T-1
%             % Calculating current Q_t
%             outCur{1} = dqn.nodeFunc(dqn.outputW{1} * [filterS{filterLen}(:,t); 1]);
%             for i=2:outputLen-1
%                 outCur{i} = dqn.nodeFunc(dqn.outputW{i} * [outCur{i-1}; 1]);
%             end
%             if nargout > 2
%                 qValues = dqn.outputW{outputLen} * [outCur{outputLen-1}; 1];
%                 [~,mmm] = max(qValues);
%                 if(mmm ~= games{game}.optmoves(t))
%                     err2 = err2 + 1/(k*(T-1));
%                 end
%             end
%             outCur{outputLen} = dqn.outputW{outputLen}(m(t),:) * [outCur{outputLen-1}; 1];
%             v(:,t) = dqn.outputW{outputLen} * [outCur{outputLen-1}; 1];
%             % --------------------------------------------------------------------
%             % ------------------ Calculating Q target ----------------------------
%             % --------------------------------------------------------------------
%             % I.e. forward pass from bottleneck through outputW
%             % This uses the OLD network
%             outFut{1} = dqnOld.nodeFunc(dqnOld.outputW{1} * [oldFilterS{filterLen}(:,t+1);1]);
%             for i=2:outputLen-1
%                 outFut{i} = dqnOld.nodeFunc(dqnOld.outputW{i} * [outFut{i-1}; 1]);
%             end
%             outFut{outputLen} = dqnOld.outputW{outputLen} * [outFut{outputLen-1}; 1];
%             [~,mi] = max(outFut{outputLen});
%             % Make Q target (ignore last 
%             if t == T-1
%                 Qtarget = r(t+1);
%             else
%                 Qtarget = r(t+1) + dqn.gamma * outFut{outputLen}(mi);
%             end
%             err = err + (outCur{outputLen} - Qtarget).^2 / (k*(T-1));
%             if(nargout > 1)
%                 % Output Errors from current Q value
%                 outCurrE = zeros(dqn.actionSize,1);
%                 outCurrE(m(t)) = 2 * (outCur{outputLen} - Qtarget) / (k*(T-1));
%                 for i=outputLen:-1:2
%                     grad.outputW{i} = grad.outputW{i} + outCurrE * [outCur{i-1}; 1]';
%                     outCurrE = (dqn.outputW{i}(:,1:end-1)' * outCurrE) .* dqn.nodeFuncGrad(outCur{i-1});
%                 end
%                 grad.outputW{1} = grad.outputW{1} + outCurrE * [filterS{filterLen}(:,t); 1]';
%                 filterE(:,t) = filterE(:,t) + dqn.outputW{1}(:,1:end-1)' * outCurrE; %  ...
% %                 % Output Errors from future Q value
% %                 outFutE = zeros(dqnOld.actionSize,1);
% %                 outFutE(mi,:) = - dqnOld.gamma* 2 * (outCur{outputLen} - Qtarget) / (k*(T-1));
% %                 for i=outputLen:-1:2
% % %                     grad.outputW{i} = grad.outputW{i} + outFutE * [outFut{i-1}; 1]';
% %                     outFutE = (dqnOld.outputW{i}(:,1:end-1)' * outFutE) .* dqnOld.nodeFuncGrad(outFut{i-1});
% %                 end
% % %                 grad.outputW{1} = grad.outputW{1} + outFutE * [filterS{filterLen}(:,t+1);1]';
% %                 % Accumulating filter errors
% %                  filterE(:,t+1) = filterE(:,t+1) + dqn.outputW{1}(:,1:end-1)' * outFutE;  % 
%             end
%         end
%         if(nargout > 1)
%             tempE = zeros(size(filterS{1}(:,1)));
%             for t=T:-1:1
%                 filterE(:,t) = (dqn.filterW{1}(:,1:hsize)'*tempE + filterE(:,t));
%                 grad.filterW{filterLen} = grad.filterW{filterLen} + filterE(:,t) * [filterS{filterLen-1}(:,t); 1]';
%                 tempE = (dqn.filterW{filterLen}(:,1:end-1)'*filterE(:,t)) .* dqn.nodeFuncGrad(filterS{filterLen-1}(:,t));
%                 for i=filterLen-1:-1:2
%                     grad.filterW{i} = grad.filterW{i} + tempE * [filterS{i-1}(:,t); 1]';
%                     tempE = (dqn.filterW{i}(:,1:end-1)'*tempE ).* dqn.nodeFuncGrad(filterS{i-1}(:,t));
%                 end
%                 if t > 1
%                     grad.filterW{1} = grad.filterW{1} + tempE * [filterS{end}(:,t-1); x(:,t); dqn.actions(:,m(t-1)); 1]';
%                 else
%                     grad.filterB = grad.filterB + tempE * [x(:,t); 1]';
%                 end
%             end
%         end
% %     end
%     if(nargout > 1)
%         varargout{1} = grad.w;
%     end
%     if(nargout > 2)
%         varargout{2} = err2;
%     end
% end