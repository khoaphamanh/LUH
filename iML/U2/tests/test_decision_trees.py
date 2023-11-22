import sys
sys.path.insert(0, "")

import numpy as np
import pytest

from utils.dataset import Dataset
from tests.config import TEST_DIR


dataset = Dataset("wheat_seeds", [0,1,2,3,4,5,6], [7], normalize=True, categorical=True)
(X_train, y_train), (X_test, y_test) = dataset.get_data()


decision_tree = None
def test_fit_decision_tree():
    global X_train, y_train, decision_tree
    
    dt = __import__(f"{TEST_DIR}.decision_trees", fromlist=['fit_decision_tree'])
    decision_tree = dt.fit_decision_tree(X_train, y_train, random_state=999)
    
    prediction = decision_tree.predict(X_test)

    assert decision_tree is not None
    assert len(prediction) == 80
    assert prediction[0] == 0. or prediction[0] == 0
    assert prediction[1] == 2. or prediction[1] == 2


def test_plot_feature_importance():
    global decision_tree
    
    import warnings
    warnings.filterwarnings("ignore")   

    import matplotlib
    matplotlib.use('Agg')
    
    dt = __import__(f"{TEST_DIR}.decision_trees", fromlist=['plot_feature_importance'])
    x, y = dt.plot_feature_importance(
        decision_tree, dataset, title="Test Figure Decision Tree")
    
    assert isinstance(x, list) or isinstance(x, np.ndarray)
    assert x[1] == 'perimeter'
    
    assert isinstance(y, list) or isinstance(y, np.ndarray)
    assert np.round(y[1], 3) == pytest.approx(0.342, 0.002)
    
    assert len(x) == 7 and len(y) == 7


fi = None
def test_compute_feature_importance():
    global decision_tree, fi
    
    dt = __import__(f"{TEST_DIR}.decision_trees", fromlist=['compute_feature_importance'])
    fi = dt.compute_feature_importance(decision_tree)
    
    assert len(fi) == 7
    assert np.round(fi[1], 3) == pytest.approx(0.228, 0.002)
    assert np.round(fi[5], 3) == pytest.approx(0.060, 0.002)


def test_normalize_feature_importance():
    global fi
    
    dt = __import__(f"{TEST_DIR}.decision_trees", fromlist=['normalize_feature_importance'])
    fi = dt.normalize_feature_importance(fi)
    
    assert len(fi) == 7
    assert np.round(fi[1], 3) == pytest.approx(0.342, 0.002)
    assert np.round(fi[5], 3) == pytest.approx(0.091, 0.002)
    

if __name__ == "__main__":
    test_fit_decision_tree()
    test_plot_feature_importance()
    test_compute_feature_importance()
    test_normalize_feature_importance()
    