%%
clear;
load ../data/review_dataset.mat
load ./index.mat

Xt_counts = train.counts(:, index(1:5000));
%Xq_counts = quiz.counts(:, index(1:5000));
%Xt_counts = sparse(Xt_counts);
Yt = train.labels;
X1 = Xt_counts(1:1000, :);
Y1 = Yt(1:1000);

%t = CTimeleft(2);
%t.timeleft();
K = kernel_intersection(Xt_counts, Xt_counts);
%Ktest = kernel_intersection(Xq_counts, X1);

%t.timeleft();
model2 = svmtrain(Yt, [(1:size(K,1))' K], '-t 4');


%model2 = svmtrain(Yt, Xt_counts, '-t 0');
save('model2.mat', 'model2');
