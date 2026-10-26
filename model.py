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

# Step 3 - grow_tree
import numpy as np


def grow_tree(X, y, max_depth=2, min_samples_leaf=1, depth=0):
    n = len(y)
    
    # Determine majority class with tie-breaking (smallest class index on ties)
    classes, counts = np.unique(y, return_counts=True)
    sorted_indices = np.lexsort((classes, -counts))
    majority_class = int(classes[sorted_indices[0]])
    
    # Stop if node is pure or max_depth is reached
    if len(classes) == 1 or depth >= max_depth:
        return {"leaf": True, "value": majority_class, "n": int(n)}
        
    # Search for the best split with positive gain and min_samples_leaf constraint
    best_gain = 0.0
    best_j = None
    best_t = None
    
    current_gini = gini(y)
    n_features = X.shape[1]
    
    for j in range(n_features):
        vals = np.unique(X[:, j])
        thresholds = (vals[:-1] + vals[1:]) / 2
        for t in thresholds:
            left_mask = X[:, j] <= t
            right_mask = ~left_mask
            
            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)
            
            # Ensure both child nodes satisfy min_samples_leaf
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
                
            weighted_gini = (n_left / n) * gini(y[left_mask]) + (n_right / n) * gini(y[right_mask])
            gain = current_gini - weighted_gini
            
            if gain > best_gain:
                best_gain = gain
                best_j = j
                best_t = t
                
    # Stop if no valid split with positive gain exists
    if best_j is None:
        return {"leaf": True, "value": majority_class, "n": int(n)}
        
    # Recurse for left and right subtrees
    left_mask = X[:, best_j] <= best_t
    right_mask = ~left_mask
    
    left_subtree = grow_tree(X[left_mask], y[left_mask], max_depth, min_samples_leaf, depth + 1)
    right_subtree = grow_tree(X[right_mask], y[right_mask], max_depth, min_samples_leaf, depth + 1)
    
    return {
        "leaf": False,
        "feature": int(best_j),
        "threshold": float(best_t),
        "n": int(n),
        "left": left_subtree,
        "right": right_subtree
    }

# Step 4 - predict_tree
def predict_tree(tree, X):
    # TODO: walk the dict tree per row; return an int array of leaf values.
    result = []
    original_tree = tree
    for i in range(X.shape[0]):
        tree = original_tree
        while(not tree['leaf']):
            if(X[i,tree['feature']]<=tree['threshold']):
                tree = tree['left']
            else:
                tree = tree['right']
            
        result.append(tree['value'])
    
    return np.array(result)

# Step 5 - fit_sklearn_tree
from sklearn.tree import DecisionTreeClassifier

def fit_sklearn_tree(X, y, max_depth=2, min_samples_leaf=1, random_state=42):
    # TODO: fit and return DecisionTreeClassifier(max_depth, min_samples_leaf, random_state).
    clf = DecisionTreeClassifier(max_depth = max_depth,min_samples_leaf = min_samples_leaf,random_state = random_state)
    clf.fit(X,y)

    return clf

# Step 6 - sklearn_splits
def sklearn_splits(clf):
    # TODO: [(feature, threshold)] for internal nodes in node-index order, thresholds rounded to 3 decimals.
    result = []
    for i in range(clf.tree_.node_count):
        if(clf.tree_.children_left[i]!=-1):
            result.append((int(clf.tree_.feature[i]),round(float(clf.tree_.threshold[i]),3)))

    return result

# Step 7 - compare_trees
def compare_trees(X, y, max_depth=2):
    # TODO: {'same_splits': bool, 'agreement': float, 'my_splits': [(feature, threshold), ...]}.
    
    mine = grow_tree(X,y,max_depth = max_depth)
    my_splits = []
    
    def splits(tree):
        if(not tree['leaf']):
            my_splits.append((tree['feature'], round(tree['threshold'], 3)))
            splits(tree['left'])
            splits(tree['right'])

        return     
    
    splits(mine)

    clf = fit_sklearn_tree(X, y, max_depth=max_depth)
    theirs = sklearn_splits(clf)

    agreement = float((predict_tree(mine, X) == clf.predict(X)).mean())

    return {'same_splits':my_splits==theirs,'agreement':agreement,'my_splits':my_splits}

# Step 8 - moons_data
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
def moons_data(n_samples=300, noise=0.25, random_state=42, test_size=0.3):
    # TODO: make_moons + train_test_split; return (X_train, X_test, y_train, y_test).
    X,y = make_moons(n_samples, noise=noise, random_state=random_state)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Step 9 - overfit_vs_regularized
from sklearn.metrics import accuracy_score
def overfit_vs_regularized(X_train, X_test, y_train, y_test, min_samples_leaf=5):
    # TODO: {'free': {...}, 'regularized': {...}} with train_acc, test_acc, leaves for each.
    free = fit_sklearn_tree(X_train,y_train,max_depth = None,min_samples_leaf = 1)
    regularized = fit_sklearn_tree(X_train,y_train,max_depth = None,min_samples_leaf = min_samples_leaf)

    train_pred_free = free.predict(X_train)
    test_pred_free = free.predict(X_test)

    train_pred_reg = regularized.predict(X_train)
    test_pred_reg = regularized.predict(X_test)

    return {'free':{'train_acc':accuracy_score(train_pred_free,y_train),'test_acc':accuracy_score(test_pred_free,y_test),'leaves':free.get_n_leaves()},'regularized':{'train_acc':accuracy_score(train_pred_reg,y_train),'test_acc':accuracy_score(test_pred_reg,y_test),'leaves':regularized.get_n_leaves()}}

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

