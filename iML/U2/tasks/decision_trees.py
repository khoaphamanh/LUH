import sys

sys.path.insert(0, "")

import numpy as np
from sklearn import tree

from tasks.plotting import plot_bar
from utils.dataset import Dataset


def fit_decision_tree(X_train, y_train, random_state=0):
    """
    3.1
    Fits a decision tree on training data.

    Inputs:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Target values.
        random_state (int): Seed of the decision tree.

    Returns:
        model (DecisionTreeClassifier): Fitted decision tree.
    """

    # init model
    model = tree.DecisionTreeClassifier(random_state=random_state)

    # fit the model
    model.fit(X_train, y_train)

    return model


def plot_feature_importance(model, dataset, title=None):
    """
    3.2
    Uses the feature importances of a decision tree and the dataset's input labels to plot a bar.
    Internally, `plot_bar` is called.

    Inputs:
        model (DecisionTreeClassifier): Decision tree.
        dataset (utils.Dataset): Used dataset to train the model. Used to receive the labels.

    Returns:
        x (list): Labels, which are displayed on the x-axis.
        y (list): Values, which are displayed on the y-axis.

    """
    # feature inportance
    y = model.feature_importances_
    x = dataset.get_input_labels()

    plt = plot_bar(x, y, title=title)
    # plt.show()

    return x, y


def compute_feature_importance(model):
    """
    Computes the feature importance of DecisionTreeClassifier from scratch.

    Inputs:
        model (DecisionTreeClassifier): Fitted decision tree.

    Returns:
        feature_importance (np.ndarray): Feature importances with shape (#features,).
    """

    children_left = (
        model.tree_.children_left
    )  # id of the left child of node i or -1 if leaf node

    children_right = (
        model.tree_.children_right
    )  # id of the right child of node i or -1 if leaf node
    feature = model.tree_.feature  # Feature used for splitting node i

    num_features = model.tree_.n_features  # Number of features

    impurity = model.tree_.impurity  # The impurity at node i

    samples = (
        model.tree_.weighted_n_node_samples
    )  # Weighted number of training samples reaching node i

    # calculate the importance node
    importance_node = []
    n_samples = np.max(samples)

    for node_idx in range(len(impurity)):
        gini_parents = impurity[node_idx]
        sample_parents = samples[node_idx]

        node_child_left = children_left[node_idx]
        node_child_right = children_right[node_idx]

        gini_child_left = impurity[node_child_left]
        gini_child_right = impurity[node_child_right]

        samples_child_left = samples[node_child_left]
        samples_child_right = samples[node_child_right]

        i_n = gini_parents * sample_parents - (
            samples_child_left * gini_child_left
            + samples_child_right * gini_child_right
        )
        importance_node.append(i_n / n_samples)

    importance_node = np.array(importance_node)

    # Initialize the feature importance with zeros and shape (num_features,)
    feature_importance = np.zeros((num_features,))
    for fear in range(num_features):
        idx_fear = [i for i, ele in enumerate(feature) if ele == fear]
        feature_importance[fear] = np.sum(importance_node[idx_fear])

    return feature_importance


def normalize_feature_importance(feature_importance):
    """
    Normalizes the given feature importances.

    Inputs:
        feature_importance (np.ndarray): Feature importances with shape (#features,).

    Returns:
        feature_importance (np.ndarray): Normalized feature importances with shape (#features,).
    """
    # calculate normalized feature importance
    feature_importance = feature_importance / np.sum(feature_importance)

    return feature_importance


if __name__ == "__main__":
    dataset = Dataset(
        "wheat_seeds", [0, 1, 2, 3, 4, 5, 6], [7], normalize=True, categorical=True
    )
    (X_train, y_train), (X_test, y_test) = dataset.get_data()

    model = fit_decision_tree(X_train, y_train)

    plot_feature_importance(model, dataset, title="Decision Tree")

    fi = compute_feature_importance(model)
    fi = normalize_feature_importance(fi)
