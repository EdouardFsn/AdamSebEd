"""Implementations of the six basic methods of the project"""

import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the features.
        initial_w: numpy array of shape (D,), the initial weights.
        max_iters: int, number of gradient descent steps.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the final weights.
        loss: float, the MSE loss corresponding to w.
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_mse_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = compute_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1).

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the features.
        initial_w: numpy array of shape (D,), the initial weights.
        max_iters: int, number of SGD steps.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the final weights.
        loss: float, the MSE loss corresponding to w.
    """
    raise NotImplementedError


def least_squares(y, tx):
    """Least squares regression using normal equations.

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the features.

    Returns:
        w: numpy array of shape (D,), the optimal weights.
        loss: float, the MSE loss corresponding to w.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the features.
        lambda_: float, the regularization parameter.

    Returns:
        w: numpy array of shape (D,), the optimal weights.
        loss: float, the MSE loss without the penalty term.
    """
    n, d = tx.shape
    a = tx.T.dot(tx) + 2 * n * lambda_ * np.eye(d)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent, with y in {0, 1}.

    Args:
        y: numpy array of shape (N,), the labels in {0, 1}.
        tx: numpy array of shape (N, D), the features.
        initial_w: numpy array of shape (D,), the initial weights.
        max_iters: int, number of gradient descent steps.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the final weights.
        loss: float, the negative log likelihood corresponding to w.
    """
    raise NotImplementedError


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent, y in {0, 1}.

    Args:
        y: numpy array of shape (N,), the labels in {0, 1}.
        tx: numpy array of shape (N, D), the features.
        lambda_: float, the regularization parameter.
        initial_w: numpy array of shape (D,), the initial weights.
        max_iters: int, number of gradient descent steps.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the final weights.
        loss: float, the negative log likelihood without the penalty term.
    """
    raise NotImplementedError


# Helper functions


def compute_mse(y, tx, w):
    """Compute the mean squared error loss.

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the features.
        w: numpy array of shape (D,), the weights.

    Returns:
        loss: float, the MSE loss.
    """
    e = y - tx.dot(w)
    loss = np.mean(e**2) / 2
    return loss


def compute_mse_gradient(y, tx, w):
    """Compute the gradient of the mean squared error loss.

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the features.
        w: numpy array of shape (D,), the weights.

    Returns:
        gradient: numpy array of shape (D,), the gradient of the MSE loss.
    """
    e = y - tx.dot(w)
    gradient = -tx.T.dot(e) / len(y)
    return gradient


def sigmoid(t):
    """Apply the sigmoid function on t.

    Args:
        t: A scalar or numpy array.

    Returns:
        The sigmoid of t.
    """
    return 1 / (1 + np.exp(-t))


def compute_logistic_loss(y, tx, w):
    """Compute the negative log likelihood loss for logistic regression.

    Args:
        y: numpy array of shape (N,), the labels in {0, 1}.
        tx: numpy array of shape (N, D), the features.
        w: numpy array of shape (D,), the weights.

    Returns:
        loss: float, the negative log likelihood loss.
    """
    pred = sigmoid(tx.dot(w))
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    return loss


def compute_logistic_gradient(y, tx, w):
    """Compute the gradient of the negative log likelihood loss for logistic regression.

    Args:
        y: numpy array of shape (N,), the labels in {0, 1}.
        tx: numpy array of shape (N, D), the features.
        w: numpy array of shape (D,), the weights.

    Returns:
        gradient: numpy array of shape (D,), the gradient of the negative log likelihood loss.
    """
    pred = sigmoid(tx.dot(w))
    gradient = tx.T.dot(pred - y) / len(y)
    return gradient
