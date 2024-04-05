import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4  # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = r"F:\VinAI_ROADMAP\ML\cs229-2018-autumn\problem-sets\PS3\data\ds3_train.csv"
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        # Split into labeled and unlabeled examples
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]  # Labeled examples
        z = z[labeled_idxs, :]  # Corresponding labels
        x = x[~labeled_idxs, :]  # Unlabeled examples
    m = x.shape[0]
    idx = np.random.permutation(m)
    group_member = int(m / K)
    mu = []
    sigma = []

    for i in range(K):
        if i != K - 1:
            x_temp = x[idx[i * group_member: (i + 1) * group_member], :]
        else:
            x_temp = x[idx[i * group_member: m], :]

        mu_temp = np.mean(x_temp, axis=0)
        mu.append(mu_temp)
        sigma.append((x_temp - mu_temp).T.dot(x_temp - mu_temp) / x_temp.shape[0])
    phi = np.ones(K) / K
    w = np.ones((m, K)) / K
    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """

    def gaussian_ll(x, u, sigma):
        """
        Args:
            x: [m,d]
            u: [k,d]
            sigma:[k,d,d]
        Returns: [m,k]

        """
        d = x.shape[1]
        k = len(sigma)
        det = np.zeros(k)
        inv_matrix = np.zeros((k, d, d))
        for i in range(k):
            det[i] = np.linalg.det(sigma[i])
            inv_matrix[i] = np.linalg.inv(sigma[i])

        first_term = 1 / ((2 * np.pi) ** (d // 2) * det ** 0.5)
        x_ = (x[:, None, :] - u[None, :, :]).transpose(1, 0, 2)  # m,k,d -> k,m,d
        second_term = np.exp(
            -0.5 * np.sum((x_ @ inv_matrix) * x_, axis=-1))

        return (first_term[:, None] * second_term).transpose(1, 0)

    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    tmp = mu
    mu = np.stack(mu)
    m, n = x.shape
    k = w.shape[1]
    w2 = np.zeros_like(w)

    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        phi = np.repeat(phi[None, :], m, axis=0)
        prior = gaussian_ll(x, mu, sigma)
        denominator = np.sum(prior * phi, axis=-1)
        w = phi * prior / denominator[:, None]  # [m,k]
        # (2) M-step: Update the model parameters phi, mu, and sigma

        mu = np.sum(w[:, :, None] @ x[:, None, :], axis=0) / np.sum(w, axis=0)[:, None]  # [k,n]
        phi = np.mean(w, axis=0)  # [k]
        for i in range(k):
            sigma[i] = (w[:, i][:, None] * (x - mu[i])).T.dot(x - mu[i]) / sum(w[:, i])

        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        prev_ll = ll
        ll = np.sum(np.log(np.sum(prior * phi, axis=-1)))  # m
        it += 1
    print(f'Number of iterations:{it}')

    # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
    # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
    # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """

    # No need to change any of these parameters
    def gaussian_ll(x, u, sigma):
        """
        Args:
            x: [m,d]
            u: [k,d]
            sigma:[k,d,d]
        Returns: [m,k]

        """
        d = x.shape[1]
        k = len(sigma)
        det = np.zeros(k)
        inv_matrix = np.zeros((k, d, d))
        for i in range(k):
            det[i] = np.linalg.det(sigma[i])
            inv_matrix[i] = np.linalg.inv(sigma[i])

        first_term = 1 / ((2 * np.pi) ** (d // 2) * det ** 0.5)
        x_ = (x[:, None, :] - u[None, :, :]).transpose(1, 0, 2)  # m,k,d -> k,m,d
        second_term = np.exp(
            -0.5 * np.sum((x_ @ inv_matrix) * x_, axis=-1))

        return (first_term[:, None] * second_term).transpose(1, 0)

    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    m, n = x.shape
    k = len(mu)
    m_tilde = x_tilde.shape[0]
    mu = np.stack(mu)
    z = z[:, 0]
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        prior = gaussian_ll(x, mu, sigma)
        tmp_phi = np.repeat(phi[None, :], m, axis=0)
        denominator = np.sum(prior * tmp_phi, axis=-1)
        w = prior * tmp_phi / denominator[:, None]
        # (2) M-step: Update the model parameters phi, mu, and sigma
        for j in range(k):
            sigma[j] = (np.sum(w[:, j][:, None, None] * np.matmul((x - mu[j])[:, :, None], (x - mu[j])[:, None, :]),
                               axis=0) + alpha * np.sum(np.matmul((x_tilde[z == j] - mu[j])[:, :, None],
                                                                  (x_tilde[z == j] - mu[j])[:, None, :]), axis=0)) / (
                               np.sum(w[:, j]) + alpha * np.sum(z == j))
            mu[j] = (np.sum(w[:, j][:, None] * x, axis=0) + alpha * np.sum(x_tilde[z == j], axis=0)) / (
                    np.sum(w[:, j]) + alpha * np.sum(z == j))
            phi[j] = (np.sum(w[:, j]) + alpha * np.sum(z == j)) / (m + alpha * m_tilde)

            # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        it += 1
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join(r"F:\VinAI_ROADMAP\ML\cs229-2018-autumn\problem-sets\PS3\output", file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model (problem 3).

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (m, n)
        z: NumPy array shape (m, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        # main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
