from testbook import testbook
import numpy as np
import numpy.testing as npt
import pytest


@pytest.fixture
def tb():
    with testbook('exercise.ipynb') as tb:
        yield tb


def test_data_loading(tb):
    tb.execute_cell(range(1,3))

    X = tb.ref('X')
    y = tb.ref('y')

    assert X.shape == (199, 7)
    assert y.shape == (199,)

    # Make sure array is shuffled
    assert X.tolist()[0][0] == pytest.approx(12.72, 0)

def test_sample_drop(tb):
    tb.execute_cell(range(1, 5))

    X = tb.ref('X')
    y = tb.ref('y')

    assert X.shape == (198, 7)
    assert y.shape == (198,)

    assert y.tolist()[150] == 1.0

def test_stratified_split(tb):
    tb.execute_cell(range(1, 7))

    X_train = tb.ref('X_train')
    X_test = tb.ref('X_test')
    y_train = tb.ref('y_train')
    y_test = tb.ref('y_test')

    assert X_train.shape == (138, 7)
    assert X_test.shape == (60, 7)
    assert y_train.shape == (138,)
    assert y_test.shape == (60,)

    label_distribution_y_train = np.unique(np.array(y_train.tolist()), return_counts=True)[1]
    label_distribution_y_test = np.unique(np.array(y_test.tolist()), return_counts=True)[1]

    npt.assert_array_equal(np.asarray([45, 46, 47]), label_distribution_y_train)
    npt.assert_array_equal(np.asarray([19, 20, 21]), label_distribution_y_test)

def test_impurity_importances(tb):
    tb.execute_cell(range(1, 11))

    average_importances = np.array(tb.ref('average_importances').tolist())
    std_importances = np.array(tb.ref('std_importances').tolist())

    npt.assert_array_almost_equal(np.asarray([0.17419392, 0.23442027, 0.04707079, 0.14791778, 0.15531865, 0.06236171, 0.17871688]), average_importances)
    npt.assert_array_almost_equal(np.asarray([0.21261737, 0.24466373, 0.08670116, 0.21020891, 0.18980712, 0.07108178, 0.22750557]), std_importances)


def test_permutation_importances(tb):
    tb.execute_cell(range(1, 18))

    average_importances = np.array(tb.ref('average_permutation_importance').tolist())
    std_importances = np.array(tb.ref('std_permutation_importance').tolist())

    npt.assert_array_almost_equal(
        np.asarray([0.015, 0.03333333, 0., -0.01166667, -0.02, 0.01833333, 0.02166667]),
        average_importances)
    npt.assert_array_almost_equal(
        np.asarray([0.02733537, 0.03073181, 0., 0.02114763, 0.01452966, 0.01166667, 0.01674979]),
        std_importances)

