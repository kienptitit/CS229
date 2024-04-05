import numpy as np
import util
import pandas as pd
from linear_model import LinearModel
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    # *** START CODE HERE ***
    # *** END CODE HERE ***
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)

    util.plot(x_val, y_val, model.theta, save_path='../figure/logreg_data2.jpg')
    data = {
        'pred': y_pred,
        'gt': y_val.tolist()
    }
    print(f"ACC : {accuracy_score(y_val, y_pred):.4f}")
    df = pd.DataFrame(data)
    df.to_csv(pred_path, index=False)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***
        # TODO: Construct Theta

        m, n = x.shape
        iter = 0
        self.theta = np.zeros((n,))
        while True:
            # TODO: Construct First Derivative
            g = sigmoid(np.dot(x, self.theta))
            first_derivative = (x.T @ (g - y)) / m  # (n,)

            # TODO: Construct Hessian Matrix

            H = (x * g[:, None] * (1 - g[:, None])).T.dot(x) / m  # (n,n)
            H_tmp = (x.T * g * (1 - g)).dot(x) / m
            # TODO: Compute Newton's direction
            d = -np.linalg.inv(H) @ first_derivative  # (n,1)
            # TODO: Update theta
            new_theta = self.theta + self.step_size * d
            # TODO: Check converege condition
            epsilon = np.linalg.norm(new_theta - self.theta)
            if epsilon < self.eps or iter > self.max_iter:
                break
            iter += 1
            self.theta = new_theta

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***
        pred = sigmoid(np.matmul(x, self.theta))
        return pred


if __name__ == '__main__':
    # main(train_path='../data/ds1_train.csv',
    #      eval_path='../data/ds1_valid.csv',
    #      pred_path='../output/p01b_pred_1.csv')
    main(train_path='../data/ds2_train.csv',
         eval_path='../data/ds2_valid.csv',
         pred_path='../output/p01b_pred_2.csv')
