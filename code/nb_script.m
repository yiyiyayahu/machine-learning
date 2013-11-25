load ../data/review_dataset.mat

Xt_counts = train.counts;
Yt = train.labels;

nb = nb_classifier( Xt_counts, Yt );
save('nb.mat', 'nb');