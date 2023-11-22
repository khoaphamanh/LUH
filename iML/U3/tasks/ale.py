import sys
sys.path.insert(0, "")

import numpy as np
from utils.dataset import Dataset
from utils.styled_plot import plt


def get_bounds(X, s, n_intervals=100):
    """
    Calculates interval bounds given an X's min and max values.
    Only values in the s-th column are considered.
    
    Parameters:
        X (np.ndarray with shape (num_instances, num_features)): Input data.
        s (int): Index of the feature x_s.
        n_intervals (int): Number of intervals.
        
    Returns:
        bounds (list): Values of bounds with `n_intervals`+1 entries.
    """
    #create interval
    feature = X[:,s]
    bounds = np.linspace(feature.min(),feature.max(),n_intervals+1)
    
    return bounds


def calculate_ale(model, X, s, n_intervals=100, centered=False):
    """
    Computes the accumulated local effect of a numeric continuous feature.
    
    Parameters:
        model: Classifier which has a predict method.
        X (np.ndarray with shape (num_instances, num_features)): Input data.
        s (int): Index of the feature x_s.
        n_intervals (int): How many intervals should be used.
        centered (bool): Whether to return uncentered or centered ALE.
        
    Returns:
        bounds (list): See function `get_bounds`.
        ale (list with `n_intervals` entries): Values of ALE.
    """
    
    #init ale
    ale = np.empty(shape=(n_intervals,))
    feature = X[:,s]
    bounds = get_bounds(X=X,s=s,n_intervals=n_intervals)

    for index, (lower, upper) in enumerate(zip(bounds,bounds[1:])):
        index_in_intervals = [i for i in range(len(feature)) 
                        if (index == 0 and lower <= feature[i] <= upper) 
                        or (index > 0 and lower < feature[i] <= upper)]
        
        if len(index_in_intervals) > 0:
            X_tmp = X[index_in_intervals]
            
            X_min = np.copy(X_tmp)
            X_min[:,s] = lower
            y_min = model.predict(X_min)
            
            X_max = np.copy(X_tmp)
            X_max[:,s] = upper
            y_max = model.predict(X_max)
            
            y_diff = y_max - y_min
            y_ale = np.mean(y_diff)

        else:
            y_ale = 0
            
        ale[index] = y_ale 

    #ggregation step
    ale = np.cumsum(ale)
    
    #check centered:
    if centered:
        ale = ale - ale.mean()
    
    return bounds, ale


def prepare_ale(model, X, s, n_intervals=100, centered=False):
    """
    Uses `calculate_ale` to calculate bounds and ale values and use these to prepare x and y data that can be used
    for plotting (the plotting itself is not part of this function).

    Parameters:
        model: Classifier which has a predict method.
        X (np.ndarray with shape (num_instances, num_features)): Input data.
        s (int): Index of the feature x_s.
        n_intervals (int): How many intervals should be used.
        centered (bool): Whether to return uncentered or centered ALE.

    Returns:
        x (list or 1D np.ndarray): Centers of two bounds. `n_intervals` entries.
        y (list or 1D np.ndarray): ALE values. `n_intervals` entries.
    """

    #calculate ale
    bounds, ale = calculate_ale(model=model,X=X,s=s,n_intervals=n_intervals,centered=centered)
    center_bounds = [(bounds[i] + bounds[i+1])/2 for i in range(len(bounds)-1)]
    
    return center_bounds, ale
        
        
if __name__ == "__main__":
    dataset = Dataset("wheat_seeds", [5,6,7], [2], normalize=True, categorical=False)
    (X_train, y_train), (X_test, y_test) = dataset.get_data()
    
    from sklearn import ensemble
    model = ensemble.RandomForestRegressor()
    model.fit(X_train, y_train)
    X = dataset.X
    s = 1
    
    print("Run `get_bounds` ...")
    print(get_bounds(X=X,s=s, n_intervals=4))
    
    num_samples = X.shape[0]
    n_intervals = [int(num_samples/16), int(num_samples/8), int(num_samples/4)]
    num_features = len(dataset.input_ids)
    
    print("Plot ...")
    plt.subplots(cols=len(n_intervals), rows=2, sharey=True)
    for row_id, centered in enumerate([False, True]):
        for col_id, k in enumerate(n_intervals):
                
            plt.subplot(2, len(n_intervals), col_id+1+len(n_intervals)*row_id)
            plt.title(f"{k} bins")
            plt.xlabel(dataset.get_input_labels(s))
            if col_id == 0:
                centered_text = ""
                if not centered:
                    centered_text = "Uncentered"
                        
                plt.ylabel(f"{centered_text} ALE [{dataset.get_output_label()}]")
                    
            x, y = prepare_ale(model, X, s, n_intervals=k, centered=centered)
            plt.plot(x, y, linewidth=0.5)
            
    plt.tight_layout(pad=0.75)
    plt.savefig(f"ale.png")
    plt.show()