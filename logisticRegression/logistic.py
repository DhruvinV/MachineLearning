
""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid

def augment_data(data):
    (n, m) = data.shape
    x0 = np.ones((n,1))
    return np.concatenate((data, x0), axis=1)

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """

    # TODO: Finish this function
    # adding columns of 1
    ones_columns = np.ones((data.shape[0],1))
    weight = np.c_[data,ones_columns]
    z = np.matmul(weight,weights)
    return sigmoid(z)

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    nm_tar = 1.0*targets
    log_y = np.sum((nm_tar*np.log(y)),0)
    n_tar = (1-targets)
    log_1_y = np.sum((n_tar*np.log(1-y)),0) 
    ce = -1*(log_y+log_1_y)
    targets = targets>0.5
    y = y>0.5
    n = y.shape[0]
    frac_correct = 0
    for i in range(targets.shape[0]):
        if(targets[i][0]== y[i][0]):
            frac_correct += 1
    frac_correct = frac_correct/targets.shape[0]
    return ce[0], frac_correct

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """

    # TODO: Finish this function
    y = logistic_predict(weights, data)
    f = evaluate(targets,y)[0]
    y_t = y - targets
    aug_data = augment_data(data)
    ones_columns = np.ones((data.shape[0],1))
    x1 = np.c_[data,ones_columns]
    df = np.matmul(np.transpose(x1),y_t)
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
    """

    # TODO: Finish this function
    y = logistic_predict(weights,data)
    lambdaa = hyperparameters['weight_regularization'] 
    w = weights
    w[data.shape[1]]=0
    f = evaluate(targets,y)[0] + 0.5*lambdaa*np.dot(np.transpose(w),w)
    ones_columns = np.ones((data.shape[0],1))
    x1 = np.c_[data,ones_columns]
    df = np.matmul(np.transpose(x1),(y-1)) + lambdaa*w
    return f, df, y
