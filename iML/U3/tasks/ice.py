import sys

sys.path.insert(0, "")

import numpy as np
from utils.dataset import Dataset
from utils.styled_plot import plt


def calculate_ice(model, X, s):
    """
    Iterates over the observations in X and for each observation i, takes the x_s value of i and replaces the x_s
    values of all other observation with this value. The model is used to make a prediction for this new data.
    The data and prediction of each iteration are added to numpy arrays x_ice and y_ice.
    For the current iteration i and the selected feature index s, the following equation is ensured:
    X_ice[i, :, s] == X[i, s]

    Parameters:
        model: Classifier which can call a predict method.
        X (np.ndarray with shape (num_instances, num_features)): Input data.
        s (int): Index of the feature x_s.

    Returns:
        X_ice (np.ndarray with shape (num_instances, num_instances, num_features)): Changed input data w.r.t. x_s.
        y_ice (np.ndarray with shape (num_instances, num_instances)): Predicted data.
    """
    # init X_ice, y_ice
    num_instance, num_feature = X.shape
    X_ice, y_ice = np.empty(shape=(num_instance, num_instance, num_feature)), np.empty(shape=(num_instance, num_instance))
    seclected_feature = X[:, s]
    X_copy = np.copy(X)

    for index, value in enumerate(seclected_feature):
        tmp = np.full(shape=seclected_feature.shape, fill_value=value)
        X_copy[:, s] = tmp
        X_ice[index] = X_copy

        ouput = model.predict(X_copy)
        y_ice[index] = ouput

    # print(y_ice[57, 99])
    return X_ice, y_ice


def prepare_ice(model, X, s, centered=False):
    """
    Uses `calculate_ice` and iterates over the rows of the returned arrays to obtain as many curves as
    observations.

    Parameters:
        model: Classifier which can call a predict method.
        X (np.ndarray with shape (num_instances, num_features)): Input data.
        s (int): Index of the feature x_s.
        centered (bool): Whether c-ICE should be used.

    Returns:
        all_x (list or 1D np.ndarray): List of lists of the x values.
        all_y (list or 1D np.ndarray): List of lists of the y values.
            Each entry in `all_x` and `all_y` represents one line in the plot.
    """
    # calculate X_ice, y_ice
    X_ice, y_ice = calculate_ice(model=model, X=X, s=s)

    # init all_x, all_y from sorted X_ice, y_ice
    all_x = X_ice[:,:,s]
    sorted_index = np.argsort(all_x[:,0])
    all_x = all_x[sorted_index].transpose()
    all_y = y_ice[sorted_index].transpose()
    
    #check centered:
    if centered:
        X_min = X_ice[sorted_index[0]]
        y_min = model.predict(X_min).reshape(-1,1)
        all_y = all_y - y_min
        
    return all_x, all_y

def plot_ice(model, dataset, X, s, centered=False):
    """
    Creates a plot object and fills it with the content of `prepare_ice`.
    Note: `show` method is not called.

    Parameters:
        model: Classifier which can call a predict method.
        dataset (utils.Dataset): Used dataset to train the model. Required to receive the input and output label.
        s (int): Index of the feature x_s.
        centered (bool): Whether c-ICE should be used.

    Returns:
        plt (matplotlib.pyplot or utils.styled_plot.plt)
    """

    #prepair
    input_label = dataset.get_input_labels(id = s)
    output_label = "predictions"
    all_x, all_y = prepare_ice(model=model,X=X,centered=centered,s=s)
    
    #plot
    plt.figure(cols=2, rows=2)
    
    n_instance = all_x.shape[0]
    for i in range(n_instance):
        x = all_x[i]
        y = all_y[i]
        plt.plot(x,y,alpha = 0.2)
        
    plt.xlabel(input_label)
    plt.ylabel(output_label)
        
    return plt


def prepare_pdp(model, X, s):
    """
    Uses `calculate_ice` and iterates over the rows of the returned arrays, calculating the mean over all the
    corresponding y values for each grid value of x_s.

    Parameters:
        model: Classifier which can call a predict method.
        X (np.ndarray with shape (num_instances, num_features)): Input data.
        s (int): Index of the feature x_s.

    Returns:
        x (list or 1D np.ndarray): x values of the PDP line.
        y (list or 1D np.ndarray): y values of the PDP line.
    """
    #load ice
    all_x, all_y = prepare_ice(model=model,X=X,s=s)
    x = np.mean(all_x,axis=0)
    y = np.mean(all_y,axis=0)
    
    return x, y


def plot_pdp(model, dataset, X, s):
    """
    Creates a plot object and fills it with the content of `prepare_pdp`.
    Note: `show` method is not called.

    Parameters:
        model: Classifier which can call a predict method.
        dataset (utils.Dataset): Used dataset to train the model. Used to receive the labels.
        s (int): Index of the feature x_s.
        centered (bool): Whether c-ICE should be used.
    Returns:
        plt (matplotlib.pyplot or utils.styled_plot.plt)
    """

    #prepair
    input_label = dataset.get_input_labels(id = s)
    output_label = output_label = "predictions"
    x,y = prepare_pdp(model=model,X=X,s=s)
    
    #plot
    plt.figure(cols=2, rows=2)
    plt.plot(x,y)
    plt.xlabel(input_label)
    plt.ylabel(output_label)
        
    return plt


if __name__ == "__main__":
    dataset = Dataset("wheat_seeds", [5, 6, 7], [2], normalize=True, categorical=False)
    (X_train, y_train), (X_test, y_test) = dataset.get_data()

    from sklearn import ensemble

    model = ensemble.RandomForestRegressor()
    model.fit(X_train, y_train)
    X = dataset.X
    s = 1

    print("Run `calculate_ice` ...")
    calculate_ice(model, X, s)

    print("Run `prepare_ice` ...")
    prepare_ice(model, X, s, centered=False)

    print("Run `plot_ice` ...")
    plt = plot_ice(model, dataset, X, s, centered=False)
    plt.savefig(f"ice.png")
    plt.show()

    print("Run `plot_ice` with centered=True ...")
    plt = plot_ice(model, dataset, X, s, centered=True)
    plt.savefig(f"ice_centered.png")
    plt.show()

    print("Run `prepare_pdp` ...")
    prepare_pdp(model, X, s)

    print("Run `plot_pdp` ...")
    plt = plot_pdp(model, dataset, X, s)
    plt.savefig(f"pdp.png")
    plt.show()
