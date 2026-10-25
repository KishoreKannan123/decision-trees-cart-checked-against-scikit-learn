"""
Decision Trees: CART, Checked Against Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - gini
def gini(y):
    # TODO: 1 - sum(p_k^2); 0.0 for an empty array. 
    if(np.size(y) == 0):
        return 0.0    
    p = np.array(np.unique(y, return_counts=True)[1])/np.size(y)
    return 1-np.sum(p**2)

# Step 2 - best_split
def best_split(X, y):
    best_gain = 0.0
    best_threshold = None
    best_feature = None

    gini_parent = gini(y)
    n = len(y)

    for j in range(X.shape[1]):
        vals = np.unique(X[:, j])
        thresholds = (vals[:-1] + vals[1:]) / 2

        for threshold in thresholds:
            left_mask = X[:, j] <= threshold
            right_mask = X[:, j] > threshold

            y_left = y[left_mask]
            y_right = y[right_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            weighted_gini = (
                len(y_left) / n * gini(y_left)
                + len(y_right) / n * gini(y_right)
            )

            gain = gini_parent - weighted_gini

            if gain > best_gain:
                best_gain = float(gain)
                best_feature = j
                best_threshold = float(threshold)

    return best_feature, best_threshold, best_gain

# Step 3 - grow_tree (not yet solved)
# TODO: implement

# Step 4 - predict_tree (not yet solved)
# TODO: implement

# Step 5 - fit_sklearn_tree (not yet solved)
# TODO: implement

# Step 6 - sklearn_splits (not yet solved)
# TODO: implement

# Step 7 - compare_trees (not yet solved)
# TODO: implement

# Step 8 - moons_data (not yet solved)
# TODO: implement

# Step 9 - overfit_vs_regularized (not yet solved)
# TODO: implement

# Step 10 - rotation_sensitivity (not yet solved)
# TODO: implement

# Step 11 - regression_tree (not yet solved)
# TODO: implement

# Step 12 - tree_rules (not yet solved)
# TODO: implement

# Step 13 - save_and_reload_tree (not yet solved)
# TODO: implement

# Step 14 - predict_species (not yet solved)
# TODO: implement

