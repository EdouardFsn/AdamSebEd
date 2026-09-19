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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
