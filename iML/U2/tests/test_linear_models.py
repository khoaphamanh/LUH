import sys
sys.path.insert(0, "")

import numpy as np
import pytest

from utils.dataset import Dataset
from tests.config import TEST_DIR


dataset = Dataset("wheat_seeds", [0,1,2,3,4,5,6], [7], normalize=True, categorical=True)
(X_train, y_train), (X_test, y_test) = dataset.get_data()


linear_regression_model = None
def test_fit_linear_regression():
    global X_train, y_train, linear_regression_model
    
    lm = __import__(f"{TEST_DIR}.linear_models", fromlist=['fit_linear_regression'])
    linear_regression_model = lm.fit_linear_regression(X_train, y_train)
    
    prediction = linear_regression_model.predict(X_test)

    assert linear_regression_model is not None
    assert len(prediction) == 80
    assert np.round(prediction[49], 4) == pytest.approx(-0.0159, 0.002)


my_linear_regression_model = None
def test_fit_my_linear_regression_1():
    """
    Tests if the prediction works.
    """
    
    global X_train, y_train, my_linear_regression_model
    
    lm = __import__(f"{TEST_DIR}.linear_models", fromlist=['fit_my_linear_regression'])
    my_linear_regression_model = lm.fit_my_linear_regression(X_train, y_train)
    
    prediction = linear_regression_model.predict(X_test)
    
    assert isinstance(prediction, list) or isinstance(prediction, np.ndarray)
    assert len(prediction) == 80
    
    
def test_fit_my_linear_regression_2():
    """
    Tests if the `fit` code (coef+bias) is correct.
    """
    
    global X_train, y_train, my_linear_regression_model
    
    coef = my_linear_regression_model.coef_
    bias = my_linear_regression_model.bias_

    assert np.round(coef[0], 3) == pytest.approx(0.668, 0.002)
    assert np.round(coef[1], 3) == pytest.approx(0.697, 0.002)
    assert np.round(coef[2], 3) == pytest.approx(0.301, 0.002)
    assert np.round(bias, 3) == pytest.approx(-0.141, 0.002)

    
def test_plot_linear_regression_weights():
    global linear_regression_model, dataset
    
    import warnings
    warnings.filterwarnings("ignore")   

    import matplotlib
    matplotlib.use('Agg')
    
    lm = __import__(f"{TEST_DIR}.linear_models", fromlist=['plot_linear_regression_weights'])
    x, y = lm.plot_linear_regression_weights(linear_regression_model, dataset, title="Test Figure Linear Regression")
    
    assert isinstance(x, list) or isinstance(x, np.ndarray)
    assert x[2] == 'compactness'
    
    assert isinstance(y, list) or isinstance(y, np.ndarray)
    assert np.round(y[1], 3) == pytest.approx(7.041, 0.001)
    
    assert len(x) == 7 and len(y) == 7
    

generalized_linear_model = None
def test_fit_generalized_linear_model():
    global X_train, y_train, generalized_linear_model
    
    lm = __import__(f"{TEST_DIR}.linear_models", fromlist=['fit_generalized_linear_model'])
    generalized_linear_model = lm.fit_generalized_linear_model(X_train, y_train)
    
    prediction = generalized_linear_model.predict(X_test)

    assert generalized_linear_model is not None
    assert len(prediction) == 80
    
    for p in prediction:
        assert p in [0., 1., 2., 0, 1, 2]
        
    assert prediction[-1] == 0. or prediction[-1] == 0


def test_correlation_analysis():
    global dataset
    
    lm = __import__(f"{TEST_DIR}.linear_models", fromlist=['correlation_analysis'])
    correlations = lm.correlation_analysis(dataset.X)
    
    assert len(correlations.keys()) == 7
    assert len(correlations[0]) == 3
    assert len(correlations[1]) == 3
    assert len(correlations[2]) == 0
    assert len(correlations[3]) == 3
    assert len(correlations[4]) == 2
    assert len(correlations[5]) == 0
    assert len(correlations[6]) == 1

            
if __name__ == "__main__":
    test_fit_linear_regression()
    test_fit_my_linear_regression_1()
    test_fit_my_linear_regression_2()
    test_plot_linear_regression_weights()
    test_fit_generalized_linear_model()
    test_correlation_analysis()
