import numpy as np
import scipy
from numpy.linalg import norm
from numpy import linspace
import matplotlib.pyplot as plt

data_train = {'X': np.genfromtxt('data_train_X.csv', delimiter=","),
            'y': np.genfromtxt('data_train_y.csv', delimiter=",")}
data_test = {'X': np.genfromtxt('data_test_X.csv', delimiter=","),
            'y': np.genfromtxt('data_test_y.csv', delimiter=",")}

def shuffle_data(data):
    new_array = np.c_[data[1], data[0]]
    np.random.permutation(new_array)
    column = new_array[:, new_array.shape[1]-1]
    shuffle_data = np.delete(new_array, new_array.shape[1]-1, axis=1)
    return (column,shuffle_data)
def split_columns(new_array):
    column = new_array[:, new_array.shape[1]-1]
    shuffle_data = np.delete(new_array, new_array.shape[1]-1, axis=1)
    return (column,shuffle_data)
def split_data(data, num_folds, fold):
    new_array = np.c_[data[1], data[0]]
    split_array = np.array_split(new_array, num_folds)
    data_fold = tuple()
    data_rest = None
    data_foldc = None
    data_foldm = None
    for i in range(len(split_array)):
        if(i == fold-1):
            (data_foldc,data_foldm) = split_columns(split_array[i])
        else:
            if(type(data_test) == None):
                data_rest = split_array[i]
            else:
                data_rest = np.vstack(split_array[i])
    (data_restc,data_foldm) = split_columns(data_rest)
    return ((data_foldc,data_foldm) ,(data_restc,data_foldm))

def train_model(data,lambd):
    w = (np.matmul(np.transpose(data[1]),data[1])) + (np.eye(data[1].shape[1]) * lambd)
    w_inv =  np.linalg.inv(w)
    final_w = np.matmul(w_inv,(np.transpose(data[1]).dot(data[0])))
    return final_w
def predict(data, model):
    pred = np.matmul(data[1],model)
    return pred
def loss(data,model):
    prediction = predict(data,model)
    final_array = data[0]-prediction
    norm1 = norm(final_array)
    loss = (norm1**2)/(data[0].shape[0])
    return loss
def cross_validation(data,num_folds,lambda_seq): 
    data = shuffle_data(data)
    cv_error = [None]*50
    for i in range(len(lambd_seq)):
            lambd = lambd_seq[i]
            cv_loss_lmd = 0.
            for j in range(1,num_folds):
                val_cv, train_cv = split_data(data, num_folds, j)
                model = train_model(train_cv, lambd)
                cv_loss_lmd += loss(val_cv, model)
            cv_error[i] = cv_loss_lmd / num_folds
    return cv_error

lambd_seq = linspace(0.02, 1.5,num=50)
lamtrain = []
lamtesting= []
fivefold = []
tenfold = []
five = cross_validation((data_train['y'],data_train['X']),5,lambd_seq)
# print(len(five))
ten = cross_validation((data_train['y'],data_train['X']),10,lambd_seq)
for i in range(len(lambd_seq)):
    model = train_model((data_train['y'],data_train['X']),lambd_seq[i])
    train = predict((data_train['y'],data_train['X']),model)
    trainz = train-data_train['y']
    result = 1/2*(norm(trainz))
    lamtrain.append((lambd_seq[i],result))
    test = predict((data_test['y'],data_test['X']),model)
    tests = test-data_test['y']
    result = 1/2*(norm(tests))
    lamtesting.append((lambd_seq[i],result))
    fivefold.append((lambd_seq[i],five[i]))
    tenfold.append((lambd_seq[i],ten[i]))



x_val = [x[0] for x in lamtrain]
y_val = [x[1] for x in lamtrain]
x_v2 = [x[0] for x in lamtesting]
y_v2 = [x[1] for x in lamtesting]
fx = [x[0] for x in fivefold]
fy = [x[1] for x in fivefold]
tx = [x[0] for x in tenfold]
ty = [x[1] for x in tenfold]
plt.plot(x_val, y_val,x_v2, y_v2,fx,fy,tx,ty)
plt.gca().legend(('training','testing',"5-fold","10-fold"))
plt.xlim(1.5, 0.02) 
plt.xlabel("Lambda Value")
plt.ylabel("Error rate")
plt.show()
