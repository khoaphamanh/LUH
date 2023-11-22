import sys

sys.path.insert(0, "")
from utils.dataset import Dataset
from tasks.plotting import plot_bar
from sklearn import linear_model
import numpy as np
import pandas as pd


def fit_linear_regression(X_train, y_train):
    """
    2.1
    Fits a linear regression model on training data.

    Inputs:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Target values.

    Returns:
        model (LinearRegression): Fitted linear regression model.
    """
    # init a model
    model = linear_model.LinearRegression()

    # fit
    model.fit(X_train, y_train)

    return model


def fit_my_linear_regression(X_train, y_train):
    """
    2.2
    Fits a self-written linear regression model on training data.

    Inputs:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Target values.

    Returns:
        model (MyLinearRegression): Fitted linear regression model.
    """

    class MyLinearRegression:
        def __init__(self):
            self.coef_ = None
            self.bias_ = 0

        def predict(self, X) -> np.ndarray:
            """
            Uses internal coefficients and bias to predict the outcome.

            Returns:
                y (np.ndarray): Predictions of X.
            """
            return X @ self.coef_.T + self.bias_

        def fit(self, X: np.array, y: np.array, learning_rate=1e-1, epochs=1000):
            """
            Adapts the coefficients and bias based on the gradients.
            Coefficients are initialized with zeros.

            Parameters:
                X: Training data.
                y: Target values.
                learning_rate (float): Learning rate decides how much the gradients are updated.
                epochs (int): Iterations of gradient changes.
            """
            # init the weights and bias
            n_features = X.shape[1]  # 7
            self.coef_ = np.zeros(shape=(n_features,))

            # training loop
            for ep in range(epochs):
                # forward pass
                y_predict_train = X @ self.coef_.T + self.bias_

                # calculate the loss:
                mse = self.MSE(y_predict=y_predict_train, y_true=y)
                acc = self.accuracy(y_predict=y_predict_train, y_true=y)

                # calculate the gradient
                d_w = self.dw(X=X, y_predict=y_predict_train, y_true=y)
                d_b = self.db(y_predict=y_predict_train, y_true=y)

                # update the parameters
                self.coef_ = self.coef_ - learning_rate * d_w
                self.bias_ = self.bias_ - learning_rate * d_b

                # print the result:
                if ep % 100 == 0 or ep == epochs - 1:
                    print("Epoch {}, loss {}, accuracy {}".format(ep, mse, acc))

        def MSE(self, y_predict: np.array, y_true: np.array):
            """
            Discription:
                Loss function of model. In this case MSE = 1/2n * (y_predict - y_true)^2

            Parameters:
                y_predict: Output of the models. Same shape as X and y_true. y_predict = X*w.T + b
                y_true: Labels.
            Returns:
                MSE Loss: Scala.
            """
            # number of features
            n = len(y_predict)
            # flatt the y_predict
            y_predict = y_predict.flatten()

            return 0.5 / n * np.sum((y_predict - y_true) ** 2)

        def dw(self, X: np.array, y_predict: np.array, y_true: np.array):
            """
            Discription:
                Gradient of weights base on MSE.
            Parameters:
                y_predict: Output of the models. Same shape as X and y_true. y_predict = X*w + b
                y_true: Labels.
            Returns:
                dw: Same shape as w: (1,n_features)
            """
            # number of features
            n = len(y_predict)
            # flatt the y_predict
            y_predict = y_predict.flatten()

            return 1 / n * (y_predict - y_true) @ X

        def db(self, y_predict: np.array, y_true: np.array):
            """
            Discription:
                Gradient of bias base on MSE.
            Parameters:
                y_predict: Output of the models. Same shape as X and y_true. y_predict = X*w + b
                y_true: Labels.
            Returns:
                dw: Same shape as b: Scala
            """
            # number of features
            n = len(y_predict)
            # flatt the y_predict
            y_predict = y_predict.flatten()

            return 1 / n * np.sum((y_predict - y_true))

        def accuracy(self, y_predict: np.array, y_true: np.array):
            """
            Discription:
                Accuracy of y_predict compare to y_true.
            Parameters:
                y_predict: Output of the models. Same shape as X and y_true. y_predict = X*w + b
                y_true: Labels.
            Returns:
                acc: value in range (0,1)
            """
            # flatt the y_predict
            y_predict = y_predict.flatten()

            # calculate the nearest label:
            classes = np.unique(y_true)
            diff = np.abs(classes[:, None] - y_predict)
            indices = np.argmin(diff, axis=0)
            y_predict_label = classes[indices]

            # calculate the accuracy
            check = np.equal(y_predict_label, y_true)
            acc = sum(check) / len(y_true)

            return acc

    model = MyLinearRegression()
    model.fit(X_train, y_train)

    return model


def plot_linear_regression_weights(model, dataset: Dataset, title=None):
    """
    2.3
    Uses the coefficients of a linear regression model and the dataset's input labels to plot a bar.
    Internally, `plot_bar` is called.

    Inputs:
        model (LinearRegression or MyLinearRegression): Linear regression model.
        dataset (utils.Dataset): Used dataset to train the model. Used to receive the labels.

    Returns:
        x (list): Labels, which are displayed on the x-axis.
        y (list): Values, which are displayed on the y-axis.

    """
    # get the name of feature
    x = dataset.get_input_labels()

    # get the weight for each feature
    y = model.coef_

    # plot
    plt = plot_bar(x, y, title=title)
    # plt.show()

    return x, y


def fit_generalized_linear_model(X_train, y_train):
    """
    2.4
    Fits a GLM on training data, solving a multi-classification problem.

    Inputs:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Target values.

    Returns:
        model: Fitted GLM.
    """
    # init the model
    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)

    return model


def correlation_analysis(X):
    """
    2.5
    Performs a correlation analysis using X.
    Two features are correlated if the correlation value is higher than 0.9.

    Inputs:
        X (np.ndarray): Data to perform correlation analysis on.

    Returns:
        correlations (dict):
            Holds the correlated feature ids of a feature id.
            Key: feature id (e.g. X has 8 columns/features so the dict
                 should have 8 keys with 0..7)
            Values: list of correlated feature ids. Exclude the id from the key.
    """
    # calculate the pairwise correlation
    data = pd.DataFrame(X)
    corr = data.corr().to_numpy()

    # create the correlation dict
    threshold = 0.9
    correlations = {}
    for i in range(len(corr)):
        corr_fearture = list(corr[i])
        correlations[i] = [
            j
            for j in range(len(corr_fearture))
            if corr_fearture[j] > threshold and j != i
        ]

    return correlations


if __name__ == "__main__":
    dataset = Dataset(
        "wheat_seeds", [0, 1, 2, 3, 4, 5, 6], [7], normalize=True, categorical=True
    )
    (X_train, y_train), (X_test, y_test) = dataset.get_data()

    model1 = fit_linear_regression(X_train, y_train)
    model2 = fit_my_linear_regression(X_train, y_train)

    plot_linear_regression_weights(model1, dataset, title="Linear Regression")
    plot_linear_regression_weights(model2, dataset, title="My Linear Regression")

    model3 = fit_generalized_linear_model(X_train, y_train)

    correlations = correlation_analysis(dataset.X)
    print(correlations)
