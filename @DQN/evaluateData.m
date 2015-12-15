function [err,varargout] = evaluateData(obj,w,games)
    % Evaluates the error and the gradient if requested for applying the
    % predictive features to the data chain
    obj.w = w;
    if(nargout > 3)
        error('No more than two output argumetns are supported - error and gradient.');
    elseif(nargout > 1)
        grad = obj;
        grad.w = obj.lambdaR*obj.w;
        %    grad2 = grad;
    end
    err = obj.lambdaR*obj.w'*obj.w/2;
    err2 = 0;
    N = size(games,2);
    filterL = size(obj.filterArch,2);
    outputL = size(obj.outputArch,2)+1;
    hsize = obj.filterArch(end);
    % Main loop
    for game = 1:N
        T = games{game}.length;
        x = games{game}.frame;
        r = games{game}.reward;
        m = games{game}.moves;
        v = zeros(obj.actionSize,T);
        filterS = cell(1,filterL);
        outCur = cell(1,outputL);
        outFut = cell(1,outputL);
        % Filter calculations
        for t=1:T
            filterS{1}(:,t) = obj.nodeFunc(obj.filterB * [x(:,t); 1]);
            for i=2:filterL-1
                filterS{i}(:,t) = obj.nodeFunc(obj.filterW{i} * [filterS{i-1}(:,t); 1]);
            end
            filterS{filterL}(:,t) = obj.filterW{filterL} * [filterS{filterL-1}(:,t); 1];
        end
        filterE = zeros(size(filterS{filterL}));
        for t=1:T-1
            % Calculating current Q_t
            outCur{1} = obj.nodeFunc(obj.outputW{1} * [filterS{filterL}(:,t); 1]);
            for i=2:outputL-1
                outCur{i} = obj.nodeFunc(obj.outputW{i} * [outCur{i-1}; 1]);
            end
            if nargout > 2
                qValues = obj.outputW{outputL} * [outCur{outputL-1}; 1];
                [~,mmm] = max(qValues);
                if(mmm ~= games{game}.optmoves(t))
                    err2 = err2 + 1/(N*(T-1));
                end
            end
            outCur{outputL} = obj.outputW{outputL}(m(t),:) * [outCur{outputL-1}; 1];
            v(:,t) = obj.outputW{outputL} * [outCur{outputL-1}; 1];
            % Calculating predictive Q_t+1
            outFut{1} = obj.nodeFunc(obj.outputW{1} * [filterS{filterL}(:,t+1);1]);
            for i=2:outputL-1
                outFut{i} = obj.nodeFunc(obj.outputW{i} * [outFut{i-1}; 1]);
            end
            outFut{outputL} = obj.outputW{outputL} * [outFut{outputL-1}; 1];
            [~,mi] = max(outFut{outputL});
            Qtarget = r(t+1) + obj.gamma * outFut{outputL}(mi);
            err = err + (outCur{outputL} - Qtarget).^2 / (N*(T-1));
            if(nargout > 1)
                % Output Errors from current Q value
                outCurrE = zeros(obj.actionSize,1);
                outCurrE(m(t)) = 2 * (outCur{outputL} - Qtarget) / (N*(T-1));
                for i=outputL:-1:2
                    grad.outputW{i} = grad.outputW{i} + outCurrE * [outCur{i-1}; 1]';
                    outCurrE = (obj.outputW{i}(:,1:end-1)' * outCurrE) .* obj.nodeFuncGrad(outCur{i-1});
                end
                grad.outputW{1} = grad.outputW{1} + outCurrE * [filterS{filterL}(:,t); 1]';
                filterE(:,t) = filterE(:,t) + obj.outputW{1}(:,1:end-1)' * outCurrE;
                %             % Output Errors from future Q value
                %             outFutE = zeros(obj.actionSize,1);
                %             outFutE(mi,:) = - obj.gamma* 2 * (outCur{outputL} - Qtarget) / (N*(T-1));
                %             for i=outputL:-1:2
                %                 grad.outputW{i} = grad.outputW{i} + outFutE * [outFut{i-1}; 1]';
                %                 outFutE = (obj.outputW{i}(:,1:end-1)' * outFutE) .* obj.nodeFuncGrad(outFut{i-1});
                %             end
                %             grad.outputW{1} = grad.outputW{1} + outFutE * [filterS{filterL}(:,t+1);1]';
                % Accumulating filter errors
                %             filterE(:,t+1) = filterE(:,t+1) + obj.outputW{1}(:,1:end-1)' * outFutE;
            end
        end
        if(nargout > 1)
            for t=T:-1:1
                % filterE(:,t) = (obj.filterW{1}(:,1:hsize)'*tempE + filterE(:,t));
                grad.filterW{filterL} = grad.filterW{filterL} + filterE(:,t) * [filterS{filterL-1}(:,t); 1]';
                tempE = (obj.filterW{filterL}(:,1:end-1)'*filterE(:,t)) .* obj.nodeFuncGrad(filterS{filterL-1}(:,t));
                for i=filterL-1:-1:2
                    grad.filterW{i} = grad.filterW{i} + tempE * [filterS{i-1}(:,t); 1]';
                    tempE = (obj.filterW{i}(:,1:end-1)'*tempE ).* obj.nodeFuncGrad(filterS{i-1}(:,t));
                end
                grad.filterB = grad.filterB + tempE * [x(:,t); 1]';
            end
        end
    end
    if(nargout > 1)
        varargout{1} = grad.w;
    end
    if(nargout > 2)
        varargout{2} = err2;
    end
end