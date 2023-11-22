import sys
sys.path.insert(0, "")

from tests.config import TEST_DIR


def test_plotting():
    import warnings
    warnings.filterwarnings("ignore")   

    import matplotlib
    matplotlib.use('Agg')

    p = __import__(f"{TEST_DIR}.plotting", fromlist=['plot_bar'])
    plt = p.plot_bar(
        x=["A", "B", "C"],
        y=[0.2, 0.5, 0.4],
        title="Test Figure",
        ylabel="Label"
    )
    
    ax = plt.gca()
    
    containers = ax.containers
    assert len(containers) == 1
    
    container = containers[0]
    assert len(container.datavalues) > 0
    
    assert ax.get_ylabel() != ""
    assert ax.get_title() != ""

