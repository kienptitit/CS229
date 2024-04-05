import numpy as np
import util

from p01b_logreg import LogisticRegression
from sklearn.metrics import accuracy_score

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')
    # *** START CODE HERE ***
    # *** END CODE HERE ***
    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    x_train, y_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_val, y_val = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_val)
    y_pred = [1 if y > 0.5 else 0 for y in y_pred]
    print("ACC : ", accuracy_score(y_val, y_pred))
    # Make sure to save outputs to pred_path_c
    # Part (d): Train on y-labels and test on true labels
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_val, y_val = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    logreg2 = LogisticRegression()
    logreg2.fit(x_train, y_train)
    y_pred = logreg2.predict(x_val)
    output = [1 if y > 0.5 else 0 for y in y_pred]
    print("ACC : ", accuracy_score(y_val, output))
    # Make sure to save outputs to pred_path_d
    # Part (e): Apply correction factor using validation set and test on true labels
    _, y_tmp = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    alpha = y_pred[y_tmp == 1].mean()

    y_pred = y_pred / alpha
    output = [1 if y > 0.5 else 0 for y in y_pred]
    print("ACC : ", accuracy_score(y_val, output))
    # Plot and use np.savetxt to save outputs to pred_path_e
    theta_prime = logreg2.theta + np.log(2 / alpha - 1) * np.array([1, 0, 0])
    util.plot(x_val, y_val, logreg.theta, save_path='../figure/prob2_figure1.jpg')
    util.plot(x_val, y_val, logreg2.theta, save_path='../figure/prob2_figure2.jpg')
    util.plot(x_val, y_val, theta_prime, save_path='../figure/prob2_figure3.jpg')
    # *** END CODER HERE


if __name__ == '__main__':
    main(train_path='../data/ds3_train.csv',
         valid_path='../data/ds3_valid.csv',
         test_path='../data/ds3_test.csv',
         pred_path='output/p02X_pred.txt')
