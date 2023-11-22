from testbook import testbook
import numpy as np
import pytest

@pytest.fixture
def tb():
    with testbook('exercise.ipynb') as tb:
        yield tb


def test_calculate_pfi(tb):
    tb.execute_cell(range(1,9))

    pfi_mean = tb.ref('pfi_mean')

    assert len(pfi_mean) == 4

    assert np.round(pfi_mean[0], 4) == 0.7456
    assert np.round(pfi_mean[2], 4) == 2.0399


def test_plot_scores(tb):
    tb.execute_cell(range(1, 11))

    plot_scores = tb.ref('plot_scores')

    plt = plot_scores(
        feature_names=["A", "B", "C"],
        scores=[0.2, 0.5, 0.4],
        title="Test",
        scores_std=[0.2, 0.05, 0.1]
    )

    ax = plt.gca()

    assert len(ax.containers) == 2
    assert ax.get_title() != ""


def test_pairplot_comparison(tb):
    tb.execute_cell(range(1, 15))

    pairplot_comparison = tb.ref('pairplot_comparison')
    df = tb.ref('df')

    plt = pairplot_comparison(
        data=df,
        fname="x2"
    )

    assert len(plt.figure.axes) == 8


def test_calculate_cfi(tb):
    tb.execute_cell(range(1,26))

    cfi_mean = tb.ref('cfi_mean')

    assert len(cfi_mean) == 4


def test_calculate_loco(tb):
    tb.execute_cell(range(1,32))

    loco_score = tb.ref('loco_score')

    assert len(loco_score) == 4

    assert np.round(loco_score[0], 4) == -0.0001
    assert np.round(loco_score[2], 4) == 1.0641
