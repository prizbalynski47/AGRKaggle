Dummy metrics:
    Most freq: 0.96757, 0.96757
    Linear: 0.72839, 0.48542

What worked well:
    -1 layer NN: 0.94508, 0.43958
    -Adding dropout + weight decay: 0.75694, 0.37943
    -Adding same loss function as competition: 0.67512, 0.22144
    -Dynamic stop based on val, 10 fold + saving 2 best of each fold, then averaging for results: 0.58394, 0.24905
    -Submission post processing to work better with loss function (clipping high values basically): 0.5631, 0.25804
    -Switching to 5 fold: 0.55268, 0.26559
    -Increasing model complexity and adding additional data with noise: 0.54741, 0.44774
    -Increasing the noise levels in additional data: 0.50494, 0.22699
    -BEST: Increasing the proportion of the additional data to original data: 0.39372, 0.30131

Thoughts:
    -Avoiding overfitting with a nn for this comp was HARD, needed to use early stop+dropout+weight decay+data modification, and then of top of that results clipping for best results
    -Private set was not well represented by public set or training set
    -Loss function this competition used made overfitting devistating for private score (due to outliers counting for more)
    -5 fold consistently worked the best for me, but could maybe use more folds with better data distribution

