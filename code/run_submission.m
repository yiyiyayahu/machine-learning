clear;
load ../data/review_dataset.mat
load ./index.mat

Xt_counts = train.counts(:, index(1:5000));
Yt = train.labels;
Xq_counts = quiz.counts(:, index(1:5000));
X1 = Xt_counts(1:1000, :);

initialize_additional_features;
%% Run algorithm
rates = predict_rating(Xt_counts, Xq_counts, Xt_additional_features,...
                       Xq_additional_features, Yt);

%% Save results to a text file for submission
dlmwrite('submit.txt',rates,'precision','%d');