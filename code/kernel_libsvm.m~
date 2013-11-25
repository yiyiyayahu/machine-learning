function model = kernel_libsvm(Xtrain, Ytrain, kernel)
% Trains a SVM using libsvm and evaluates on test data.
%
% Usage:
%
%   MODEL = KERNEL_LIBSVM(XTRAIN, YTAIN, KERNEL)
%
% Runs training a SVM with the given kernel function, using
% cross validation to choose regularization parameter C. X, Y should 
% be created using SPARSE. KERNEL is a FUNCTION HANDLE to
% the appropriate KERNEL function, which must take ONLY TWO PARAMETERS
% K(X,X2).
% return a model for the trainning data that can be used in svmpredict
% function.
%
% EXAMPLES:
%
% Compute error using a poly kernel with P=2:
%
% >> k = @(x,x2) kernel_poly(x, x2, 1);
% >> model = kernel_libsvm(Xtrain, Ytrain, k)
%

% Compute kernel matrices for training.
%Xt = sparse(Xtrain);
K = kernel(Xtrain, Xtrain);

% Use built-in libsvm cross validation to choose the C regularization
% parameter.
crange = 10.^(-10:2:3);
acc = zeros(1, numel(crange));
for i = 1:numel(crange)
    acc(i) = svmtrain(Ytrain, [(1:size(K,1))' K], sprintf('-t 4 -v 10 -c %g', crange(i)));
end
[~, bestc] = max(acc);
fprintf('Cross-val chose best C = %g\n', crange(bestc));

% Train and evaluate SVM classifier using libsvm
model = svmtrain(Ytrain, [(1:size(K,1))' K], sprintf('-t 4 -c %g', crange(bestc)));







    
    
