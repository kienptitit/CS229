import numpy as np
import util
from scipy.stats import multivariate_normal
from linear_model import LinearModel
from sklearn.metrics import accuracy_score
import pandas as pd


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # *** END CODE HERE ***
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    # *** START CODE HERE ***
    # *** END CODE HERE ***
    model = GDA()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    y_pred = [1 if y > 0.5 else 0 for y in y_pred]
    data = {
        'pred': y_pred,
        'gt': y_val.tolist()
    }
    print(f"ACC : {accuracy_score(y_val, y_pred):.4f}")
    df = pd.DataFrame(data)
    df.to_csv(pred_path, index=False)
    util.plot(x_val, y_val, model.theta, save_path='../figure/GDA_data2.jpg')


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***

        m, n = x.shape
        self.theta = np.zeros(n + 1)
        # TODO : Compute phi
        phi = (y == 1).sum() / len(y)
        # TODO : Compute u0
        u0 = x[y == 0].mean(axis=0)
        # TODO : Compute u1
        u1 = x[y == 1].mean(axis=0)
        # TODO : Compute Sigma
        x0 = x[y == 0] - u0
        sigma = x0.T @ x0
        sigma_inv = np.linalg.inv(sigma)
        self.theta[0] = 0.5 * (u0 @ sigma_inv @ u0 - u1 @ sigma_inv @ u1) + np.log(1 / phi - 1)
        self.theta[1:] = sigma_inv @ (u1 - u0)

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE

        return 1 / (1 + np.exp(-x.dot(self.theta)))


if __name__ == '__main__':
    # main(train_path='../data/ds1_train.csv',
    #      eval_path='../data/ds1_valid.csv',
    #      pred_path='../output/p01e_pred_1.csv')
    main(train_path='../data/ds2_train.csv',
         eval_path='../data/ds2_valid.csv',
         pred_path='../output/p01e_pred_2.csv')
