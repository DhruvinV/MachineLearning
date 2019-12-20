import scipy
import numpy as np
import itertools
import matplotlib.pyplot as plt


# Gaussian Data  for testing purposes.
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

# cost function to test for convergence
def cost(data, R, Mu):
N, D = data.shape
K = Mu.shape[1]
J = 0
for k in range(K):
    J += np.sum(np.dot(np.linalg.norm(data - np.array([Mu[:, k], ] * N), axis=1)**2, R))
return J

#assignmnet Step
def km_assignment_step(data, Mu):
""" Compute K-Means assignment step

Args:
    data: a NxD matrix for the data points
    Mu: a DxK matrix for the cluster means locations

Returns:
    R_new: a NxK matrix of responsibilities
"""

# Fill this in:
N, D = 400,2 # Number of datapoints and dimension of datapoint
K = 2 # number of clusters
# r = np.zeros((N,K))
r = np.zeros((N,K))
for k in range(K):
    r[:, k] = np.linalg.norm(data-Mu[:,k],axis=1)
# print(r.shape)
arg_min = np.argmin(r,axis=1) # argmax/argmin along dimension 1
# print(arg_min)
R_new = np.zeros((N,K)) # Set to zeros/ones with shape (N, K)
# print(R_new)
R_new[range(N),arg_min] = 1 # Assign to 1
# print(R_new)
return R_new
# refitting step
def km_refitting_step(data, R, Mu):
    """ Compute K-Means refitting step.
    
    Args:
        data: a NxD matrix for the data points
        R: a NxK matrix of responsibilities
        Mu: a DxK matrix for the cluster means locations
    
    Returns:
        Mu_new: a DxK matrix for the new cluster means locations
    """
    N, D = data.shape[0],data.shape[1] # Number of datapoints and dimension of datapoint
    K = R.shape[1]  # number of clusters
    Mu_new = np.zeros((D,K))
#     for k in range(K):
#       Mu_new[:,k] = np.mean(data[R==k],0)
    Mu_new = (np.transpose(np.matmul(np.transpose(R),data)))/( np.sum(R,axis=0))
    # Mu_new[:,range(K)] = np.mean(data[R==range(K)],0)
    # Mu_new = ...
    return Mu_new
if __name__ == '__main__':
    N, D = data.shape
    K = 2
    max_iter = 100
    class_init = np.random.binomial(1., .5, size=N)
    R = np.vstack([class_init, 1 - class_init]).T

    Mu = np.zeros([D, K])
    Mu[:, 1] = 1.
    R.T.dot(data), np.sum(R, axis=0)
    cost_list,iteras = [],[]

    for it in range(max_iter):
        R = km_assignment_step(data, Mu)
        Mu = km_refitting_step(data, R, Mu)
    #     print(it, cost(data, R, Mu))
        cost_list.append(cost(data,R,Mu))
        iteras.append(it)
        # it, cost(data, R, Mu)

    class_1 = np.where(R[:, 0])
    class_2 = np.where(R[:, 1])
    

