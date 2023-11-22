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

module = __import__(f"{TEST_DIR}.ice", fromlist=[
    'calculate_ice', 'prepare_ice', 'plot_ice', 'prepare_pdp', 'plot_pdp'])

dataset = Dataset("wheat_seeds", [5,6,7], [2], normalize=True, categorical=False)
(X_train, y_train), (X_test, y_test) = dataset.get_data()

model = ensemble.RandomForestRegressor(random_state=0)
model.fit(X_train, y_train)
X = dataset.X
s = 1


def test_calculate_ice():
    global module, model, dataset, X, s
    
    X_ice, y_ice = module.calculate_ice(model, X, s)
    
    assert len(X_ice.shape) == 3
    assert len(y_ice.shape) == 2
    assert X_ice.shape[0] == X_ice.shape[1]
    assert y_ice.shape[0] == y_ice.shape[1]
    assert X_ice.shape[2] == 3
    
    assert np.round(X_ice[49, 51, 1], 4) == pytest.approx(0.304, 0.001)
    assert np.round(y_ice[5, 99], 4) == pytest.approx(0.666, 0.001)


def test_prepare_ice():
    global module, model, dataset, X, s
    
    all_x, all_y = module.prepare_ice(model, X, s, centered=False)
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    assert len(all_x.shape) == 2
    assert len(all_y.shape) == 2
    assert all_x.shape[0] == all_x.shape[1]
    assert all_y.shape[0] == all_y.shape[1]
    
    # Check sorting
    for x1, x2 in zip(all_x[0], all_x[0][1:]):
        assert x1 <= x2
        
    # Check if values are calculated the right way
    assert np.round(all_y[57, 99], 4) == pytest.approx(0.7078, 0.001)
    

def test_prepare_c_ice():
    global module, model, dataset, X, s
    
    all_x, all_y = module.prepare_ice(model, X, s, centered=True)
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    assert len(all_x.shape) == 2
    assert len(all_y.shape) == 2
    assert all_x.shape[0] == all_x.shape[1]
    assert all_y.shape[0] == all_y.shape[1]
    
    # Check sorting
    for x1, x2 in zip(all_x[0], all_x[0][1:]):
        assert x1 <= x2
        
    # Check if values are calculated the right way
    assert np.round(all_y[57, 99], 4) == pytest.approx(0.0246, 0.0001)
    

def test_plot_ice():
    global module, model, dataset, X, s
    
    plt = module.plot_ice(model, dataset, X, s, centered=False)
    ax = plt.gca()
    
    assert ax.get_ylabel() != ""
    assert ax.get_xlabel() != ""
    assert len(ax.lines) == 199
    
    assert isinstance(ax.lines[-1]._alpha, float)
    

def test_prepare_pdp():
    global module, model, dataset, X, s
    
    x, y = module.prepare_pdp(model, X, s)
    
    x = np.array(x)
    y = np.array(y)
    
    
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    
    # Check sorting
    for x1, x2 in zip(x, x[1:]):
        assert x1 <= x2 

    assert np.round(x[48], 4) == pytest.approx(0.2590, 0.001)
    assert np.round(y[48], 4) == pytest.approx(0.6112, 0.001)
    

def test_plot_pdp():
    global module, model, dataset, X, s
    
    plt = module.plot_pdp(model, dataset, X, s)
    ax = plt.gca()
    
    assert ax.get_ylabel() != ""
    assert ax.get_xlabel() != ""
    assert len(ax.lines) == 1
    
    alpha = ax.lines[-1]._alpha
    assert alpha is None or float(alpha) == 1.

    
if __name__ == "__main__":
    test_calculate_ice()
    test_prepare_ice()
    test_prepare_c_ice()
    test_plot_ice()
    test_prepare_pdp()
    test_plot_pdp()