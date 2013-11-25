function [fidx val max_ig] = dt_choose_feature_multi(X, Z, Xrange, colidx)
% DT_CHOOSE_FEATURE_MULTI - Selects feature with maximum multi-class IG.
%
% Usage:
% 
%   [FIDX FVAL MAX_IG] = dt_choose_feature(X, Z, XRANGE, COLIDX)
%
% Given N x D data X and N x K indicator labels Z, where X(:,j) can take on values in XRANGE{j}, chooses
% the split X(:,FIDX) <= VAL to maximize information gain MAX_IG. I.e., FIDX is
% the index (chosen from COLIDX) of the feature to split on with value
% FVAL. MAX_IG is the corresponding information gain of the feature split.
%
% Note: The relationship between Y and Z is that Y(i) = find(Z(i,:)).
% Z is the categorical representation of Y: Z(i,:) is a vector of all zeros
% except for a one in the Y(i)'th column.
% 
% Hint: It is easier to compute entropy, etc. when using Z instead of Y.
%
% SEE ALSO
%    DT_TRAIN_MULTI

%compute the H for Z
pz = sum(Z, 1)/size(Z, 1);
H = multi_entropy(pz');                                                                      %mean(Y) is the propability for output to be zero

% Compute conditional entropy for each feature.
ig = zeros(numel(Xrange), 1);                                                                      %Xrange is a cell array. numel(Xrange) == D
split_vals = zeros(numel(Xrange), 1);

% Compute the IG of the best split with each feature. This is vectorized
% so that, for each feature, we compute the best split without a second for
% loop. Note that if we were guaranteed binary features, we could vectorize
% this entire loop with the same procedure.
t = CTimeleft(numel(colidx));
fprintf('Evaluating features on %d examples: ', size(Z, 1));
for i = colidx
    t.timeleft();

    % Check for constant values. Only one split feature in Xi
    if numel(Xrange{i}) == 1
        ig(i) = 0; split_vals(i) = 0;
        continue;
    end
    
    % Compute up to 10 possible splits of the feature.
    r = linspace(double(Xrange{i}(1)), double(Xrange{i}(end)), min(10, numel(Xrange{i})));
    split_f = bsxfun(@le, X(:,i), r(1:end-1));                                                     %N*numel(Xrange{i}) 
    
    % Compute conditional entropy of all possible splits.
    px = mean(split_f);                                                                            %mean(split_f) is the P for feature X(i) drop in less than that range
    nx = sum(split_f);
    nnotx = sum(~split_f);
    z_given_x = zeros(size(r,2)-1, size(Z, 2));
    z_given_notx = zeros(size(r,2)-1, size(Z, 2));
    for j = 1:(size(r,2)-1)
        tmp_x = bsxfun(@and, Z, split_f(:, j));
        z_given_x(j, :) = sum(tmp_x, 1)/nx(j);
    
        tmp_notx = bsxfun(@and, Z, ~split_f(:,j));
        z_given_notx(j,:) = sum(tmp_notx, 1)/nnotx(j);
    end
    cond_H = px.*multi_entropy(z_given_x') + (1-px).*multi_entropy(z_given_notx');
    
    % Choose split with best IG, and record the value split on.
    [ig(i) best_split] = max(H-cond_H);                                                            %from cond_H select the split which can maximize the IG 
    split_vals(i) = r(best_split);
end

% Choose feature with best split.                                                                  %all of the IG select the feature which get max IG 
[max_ig fidx] = max(ig);
val = split_vals(fidx);