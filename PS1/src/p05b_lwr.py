import matplotlib.pyplot as plt
import numpy as np
import util
from sklearn.metrics import mean_squared_error
from linear_model import LinearModel
import warnings

warnings.filterwarnings("ignore")


def plot(x, y_label, y_pred, title, save_path):
    plt.figure()
    plt.plot(x[:, -1], y_label, 'bx', label='label')
    plt.plot(x[:, -1], y_pred, 'ro', label='prediction')
    plt.suptitle(title, fontsize=12)
    plt.legend(loc='upper left')
    plt.savefig(save_path)


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)
    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)
    # Get MSE value on the validation set
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)
    MSE = mean_squared_error(y_val_pred, y_val)
    print(f"MSE: {MSE}")
    # Plot validation predictions on top of training set
    plot(x_train, y_train, y_train_pred, 'Training Set',
         save_path='/media/kiennguyen/New Volume/VinAI_ROADMAP/ML/cs229-2018-autumn/problem-sets/PS1/figure/p05b_train.jpg')
    plot(x_val, y_val, y_val_pred, "Validation Set",
         save_path='/media/kiennguyen/New Volume/VinAI_ROADMAP/ML/cs229-2018-autumn/problem-sets/PS1/figure/p05b_val.jpg')
    # No need to save predictions
    # Plot data

    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        self.x = x  # (m,n)
        self.y = y  # (m,)
        # *** START CODE HERE ***
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (l, n).

        Returns:
            Outputs of shape (l,).
        """
        m, n = x.shape
        w_vector = np.exp(-np.linalg.norm(self.x - x.reshape(m, -1, n), axis=-1) ** 2 / (2 * self.tau ** 2))  # (l,m)

        # *** START CODE HERE ***
        w_matrix = np.apply_along_axis(np.diag, axis=1, arr=w_vector)  # (l,m,m)
        theta = np.linalg.inv(self.x.T @ w_matrix @ self.x) @ self.x.T @ w_matrix @ self.y
        return np.einsum("ij,ij->i", x, theta)
        # *** END CODE HERE ***


if __name__ == '__main__':
    main(tau=5e-1,
         train_path='../data/ds5_train.csv',
         eval_path='../data/ds5_valid.csv')
