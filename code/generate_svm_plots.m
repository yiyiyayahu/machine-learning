%% Plots/submission for SVM portion, Question 1.

%% Put your written answers here.
clear all
answers{1} = ['The intersection kernal works best. '...
    'Because after mapping the feature into dictionary words feature space, the feature space has 46785 features with each word as a feature dimension and the frequency of occurence as value.'...
    'So in intersection kernal, documents with similar types of words will have higher kernal score and predicted as the same group label'];

save('problem_1_answers.mat', 'answers');

%% Load and process the data.

load ../data/windows_vs_mac.mat;
[X Y] = make_sparse(traindata, vocab);
[Xtest Ytest] = make_sparse(testdata, vocab);

%% Bar Plot - comparing error rates of different kernels

% INSTRUCTIONS: Use the KERNEL_LIBSVM function to evaluate each of the
% kernels you mentioned. Then run the line below to save the results to a
% .mat file.

kl = @(x,x2) kernel_poly(x, x2, 1);
kq = @(x,x2) kernel_poly(x, x2, 4);
kc = @(x,x2) kernel_poly(x, x2, 3);
kg = @(x,x2) kernel_gaussian(x, x2, 20);
ki = @(x,x2) kernel_intersection(x, x2);

[e1 i1] = kernel_libsvm(X, Y, Xtest, Ytest, kl);
[e2 i2] = kernel_libsvm(X, Y, Xtest, Ytest, kq);
[e3 i3] = kernel_libsvm(X, Y, Xtest, Ytest, kc);
[e4 i4] = kernel_libsvm(X, Y, Xtest, Ytest, kg);
[e5 i5] = kernel_libsvm(X, Y, Xtest, Ytest, ki);

results.linear = e1 % ERROR RATE OF LINEAR KERNEL GOES HERE
results.quadratic = e2 % ERROR RATE OF QUADRATIC KERNEL GOES HERE
results.cubic = e3 % ERROR RATE OF CUBIC KERNEL GOES HERE
results.gaussian = e4 % ERROR RATE OF GAUSSIAN (SIGMA=20) GOES HERE
results.intersect = e5 % ERROR RATE OF INTERSECTION KERNEL GOES HERE

% Makes a bar chart showing the errors of the different algorithms.
algs = fieldnames(results);
for i = 1:numel(algs)
    y(i) = results.(algs{i});
end
bar(y);
set(gca,'XTickLabel', algs);
xlabel('Kernel');
ylabel('Test Error');
title('Kernel Comparisons');

print -djpeg -r72 plot_1.jpg;
