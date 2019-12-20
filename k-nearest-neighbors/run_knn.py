import numpy as np
from l2_distance import l2_distance
import utils
import matplotlib.pyplot as plt

def run_knn(k, train_data, train_labels, valid_data):
    """Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples, 
          and M is the number of features per example.

    Inputs:
        k:            The number of neighbours to use for classification 
                      of a validation example.
        train_data:   The N_TRAIN x M array of training
                      data.
        train_labels: The N_TRAIN x 1 vector of training labels
                      corresponding to the examples in train_data 
                      (must be binary).
        valid_data:   The N_VALID x M array of data to
                      predict classes for.

    Outputs:
        valid_labels: The N_VALID x 1 vector of predicted labels 
                      for the validation data.
    """

    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:,:k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # note this only works for binary labels
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1,1)

    return valid_labels


if(__name__=="__main__"):
    train_x, train_y = utils.load_train()
    valid_x, valid_y = utils.load_valid()
    k = [1,3,5,7,9]
    y1 = []
    for i in k:
        knn = run_knn(i,train_x,train_y,valid_x)
        # print(knn)
        # print(valid_y)
        # print(valid_y == knn)
        result = (np.sum(np.bitwise_xor(knn, valid_y))/float(valid_x.shape[0]))
        # result = np.mean(valid_y == knn)
        y1.append(result)
        print("Error Rate "+ str(result))
    test_x,test_y = utils.load_test()
    y2 = []
    for j in k:
        knn = run_knn(j,train_x,train_y,test_x)
        result = (np.sum(np.bitwise_xor(knn, test_y))/float(test_x.shape[0]))
        # result = np.mean(test_y == knn)
        y2.append(result)
        print("Error Rate "+ str(result))

    plt.plot(k,y1,k,y2)
    plt.gca().legend(('Validation','Testing'))
    plt.xlabel("K-value")
    plt.ylabel("Classification Rate")
    plt.show()