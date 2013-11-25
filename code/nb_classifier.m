function nb = nb_classifier( Xtrain, Ytrain )
%NB_CLASSIFIER Summary of this function goes here
%   Detailed explanation goes here
nb = NaiveBayes.fit(Xtrain, Ytrain, 'Distribution', 'mn');
end

