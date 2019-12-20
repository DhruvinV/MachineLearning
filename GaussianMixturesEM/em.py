import scipy
import numpy as np
import itertools
import matplotlib.pyplot as plt
# generate some data
num_samples = 400
cov = np.array([[1., .7], [.7, 1.]]) * 10
mean_1 = [.1, .1]
mean_2 = [6., .1]
x_class1 = np.random.multivariate_normal(mean_1, cov, num_samples // 2)
x_class2 = np.random.multivariate_normal(mean_2, cov, num_samples // 2)
xy_class1 = np.column_stack((x_class1, np.zeros(num_samples // 2)))
xy_class2 = np.column_stack((x_class2, np.ones(num_samples // 2)))
data_full = np.row_stack([xy_class1, xy_class2])
np.random.shuffle(data_full)
data = data_full[:, :2]
labels = data_full[:, 2]

def normal_density(x, mu, Sigma):
return np.exp(-.5 * np.dot(x - mu, np.linalg.solve(Sigma, x - mu))) \
    / np.sqrt(np.linalg.det(2 * np.pi * Sigma))
    
    
# log-likelihood
def log_likelihood(data, Mu, Sigma, Pi):
""" Compute log likelihood on the data given the Gaussian Mixture Parameters.

Args:
    data: a NxD matrix for the data points
    Mu: a DxK matrix for the means of the K Gaussian Mixtures
    Sigma: a list of size K with each element being DxD covariance matrix
    Pi: a vector of size K for the mixing coefficients

Returns:
    L: a scalar denoting the log likelihood of the data given the Gaussian Mixture
"""
# Fill this in:
N, D = data.shape[0],data.shape[1]  # Number of datapoints and dimension of datapoint
K = Mu.shape[1] # number of mixtures
L, T = 0,0
for n in range(N):
    for k in range(K):
        T += (Pi[k] * (normal_density(data[n],Mu[:,k],Sigma[k]))) # Compute the likelihood from the k-th Gaussian weighted by the mixing coefficients
    L += np.log(T)
return L
# Gaussian Mixture Expectation Step
def gm_e_step(data, Mu, Sigma, Pi):
    """ Gaussian Mixture Expectation Step.

    Args:
        data: a NxD matrix for the data points
        Mu: a DxK matrix for the means of the K Gaussian Mixtures
        Sigma: a list of size K with each element being DxD covariance matrix
        Pi: a vector of size K for the mixing coefficients
    
    Returns:
        Gamma: a NxK matrix of responsibilities
    """
    # Fill this in:
    N, D = data.shape[0],data.shape[1] # Number of datapoints and dimension of datapoint
    K = Mu.shape[1] # number of mixtures
    Gamma = np.zeros((N,K)) # zeros of shape (N,K), matrix of responsibilities
    # print(Gamma[n]shape)
    for n in range(N):
        for k in range(K):
            Gamma[n, k] = (Pi[k] * (normal_density(data[n],Mu[:,k],Sigma[k])))
        Gamma[n, :] /= np.sum(Gamma[n,:]) # Normalize by sum across second dimension (mixtures)
    return Gamma
# Gaussian Mixture Maximization Step
def gm_m_step(data, Gamma):
    """ Gaussian Mixture Maximization Step.

    Args:
        data: a NxD matrix for the data points
        Gamma: a NxK matrix of responsibilities
    
    Returns:
        Mu: a DxK matrix for the means of the K Gaussian Mixtures
        Sigma: a list of size K with each element being DxD covariance matrix
        Pi: a vector of size K for the mixing coefficients
    """
    # Fill this in:
    N, D = data.shape[0],data.shape[1]  # Number of datapoints and dimension of datapoint
    K = Gamma.shape[1]  # number of mixtures
    Nk = np.sum(Gamma,axis=0) # Sum along first axis
    # print(Nk.shape)
    Sigma = [np.identity(D) for k in range(K)]
    Mu = 1/Nk * (np.matmul(data.T,Gamma))
    # print(Sigma[0].shape)
    for k in range(K):
        x1 = data - Mu[:,k]
        dig = np.diagflat(Gamma[:,k])
        x4 = np.matmul(x1.T,dig)
        x5 = np.matmul(x4,x1)
        x6 = (1/Nk[k])*((x5))
        Sigma[k] = x6
    Pi = Nk/N
    return Mu, Sigma, Pi
if __name__ = "__main__":
    N, D = data.shape
    K = 2
    Mu = np.zeros([D, K])
    Mu[:, 1] = 1.
    Sigma = [np.eye(2), np.eye(2)]
    Pi = np.ones(K) / K
    Gamma = np.zeros([N, K]) # Gamma is the matrix of responsibilities

    max_iter  = 200
    itera,liklehood = [],[]
    for it in range(max_iter):
        Gamma = gm_e_step(data, Mu, Sigma, Pi)
        Mu, Sigma, Pi = gm_m_step(data, Gamma)
        itera.append(it)
        liklehood.append(log_likelihood(data, Mu, Sigma, Pi))
    class_1 = np.where(Gamma[:, 0] >= .5)
    class_2 = np.where(Gamma[:, 1] >= .5)
