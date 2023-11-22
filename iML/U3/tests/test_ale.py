import sys
sys.path.insert(0, "")

import warnings
warnings.filterwarnings("ignore")   

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
from sklearn import ensemble

from utils.dataset import Dataset
from tests.config import TEST_DIR

module = __import__(f"{TEST_DIR}.ale", fromlist=[
    'get_bounds', 'calculate_ale', 'prepare_ale'])

dataset = Dataset("wheat_seeds", [5,6,7], [2], normalize=True, categorical=False)
(X_train, y_train), (X_test, y_test) = dataset.get_data()

model = ensemble.RandomForestRegressor(random_state=0)
model.fit(X_train, y_train)
X = dataset.X
s = 1


def test_get_bounds():
    global module, X_train, s
    
    bounds = module.get_bounds(X_train, s, n_intervals=69)
    
    assert len(bounds) == 70
    assert np.round(bounds[-1], 4) == 0.9744
    assert np.round(bounds[44], 4) == 0.6367


def test_all_elements_included():
    """Tests if all elements are included."""
    global module, model, X, s
    
    _, ale = module.calculate_ale(model, X, s, n_intervals=1, centered=False)
    assert np.round(ale[0], 4) == -0.2260


def test_calculate_ale():
    global module, model, X, s
    
    _, ale = module.calculate_ale(model, X, s, n_intervals=30, centered=False)
    assert len(ale) == 30
    assert np.round(ale[2], 4) == 0.0267
    assert np.round(ale[-1], 2) == -0.45
    

def test_calculate_ale_centered():
    global module, model, X, s
    
    _, ale = module.calculate_ale(model, X, s, n_intervals=30, centered=True)
    assert len(ale) == 30
    assert np.round(ale[0], 3) == 0.143
    assert np.sum(ale) == pytest.approx(0)


def test_prepare_ale():
    global module, model, X, s
    
    centers, _ = module.prepare_ale(model, X, s, n_intervals=30)
    assert len(centers) == 30
    assert np.round(sum(centers), 1) == 15.0

    
if __name__ == "__main__":
    test_get_bounds()
    test_all_elements_included()
    test_calculate_ale()
    test_calculate_ale_centered()
    test_prepare_ale()
