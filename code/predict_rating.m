function rates = predict_rating(Xt_counts, Xq_counts, Xt_additional_features,...
                                Xq_additional_features, Yt)
% Returns the predicted ratings, given wordcounts and additional features.
%
% Usage:
%
%   RATES = PREDICT_RATING(XT_COUNTS, XQ_COUNTS, XT_ADDITIONAL_FEATURES, ...
%                         XQ_ADDITIONAL_FEATURES, YT);
%
% This is the function that we will use for checkpoint evaluations and the
% final test. It takes a set of wordcount and additional features and produces a
% ranking matrix as explained in the project overview.
%
% This function SHOULD NOT DO ANY TRAINING. This code needs to run in under
% 10 minutes. Therefore, you should train your model BEFORE submission, save
% it in a .mat file, and load it here.

% YOUR CODE GOES HERE
% THIS IS JUST AN EXAMPLE

%{
% nb block
load ./nb.mat

N = size(Xq_counts, 1);

%rates = int8(ones(N,1));
rates = predict(nb, Xq_counts);
%}

% svm block
load ./model2.mat

Ktest = kernel_intersection(Xt_counts, Xq_counts);
unknown = zeros(5000, 1);
[rates, ~, ~] = svmpredict(unknown, [(1:size(Ktest,1))' Ktest], model2);
end
