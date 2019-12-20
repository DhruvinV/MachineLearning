import numpy as np
from check_grad import check_grad
from utils import *
from logistic import *
import matplotlib.pyplot as plt
def run_logistic_regression(i):
    # train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()
    N, M = train_inputs.shape
    x = []
    y1 = []
    y2 = []
    y3 = []
    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate':0.04,
                    'weight_regularization':j,
                    'num_iterations':400
                }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    # weights = [np.random.randn() for l in range(M+1)]
    weights = weights = np.random.randn(M + 1, 1)

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)
    cros_train = []
    cros_valid = []
    train_acc = []
    train_valid = []
    # Begin learning with gradient descent
    for t in range(hyperparameters['num_iterations']):

        # TODO: you may need to modify this loop to create plots, etc.

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
        
        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters
        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        
        # print some stats
        # print (("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
        #     "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
        #         t+1, f / N, cross_entropy_train, frac_correct_train*100,
        #         cross_entropy_valid, frac_correct_valid*100))
        cros_train.append(cross_entropy_train)
        cros_valid.append(cross_entropy_valid)
        train_acc.append(frac_correct_train*100)
        train_valid.append(frac_correct_valid*100)

    return i,cros_train,cros_valid,train_acc,train_valid
    # prediction_valid = logistic_predict(weights, test_inputs)
    # cross_entropy_valid, frac_correct_valid = evaluate(test_targets, predictions_valid)
    # print("CE_TEST", cross_entropy_valid, "ACC_TEST",frac_correct_valid)
        # x.append(t)
    # y1.append(cros_train[-1])
    # y2.append(cros_valid[-1])
    # y3.append(train_acc[-1])
    # y4.append(train_valid[-1])
    # plt.plot(x,cros_train,x,cros_valid)
    # plt.gca().legend(('Training','Validation'))
    # plt.xlabel("Iterations")
    # plt.ylabel("Cross Entropy")
    # plt.show()

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and 
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print ("diff =", diff)

if __name__ == '__main__':
    train_inputs, train_targets = load_train_small()
    M = train_inputs.shape[1]
    x = [0,0.001,0.01,0.1,1.0]
    y1 = []
    y2 = []
    for j in [0,0.001,0.01,0.1,1.0]:
        cros_train1 = 0
        train_acc1 = 0
        cros_valid1 = 0
        train_valid1 = 0
        for k in range(5):
            i,cros_train,cros_valid,train_acc,train_valid = run_logistic_regression(j)
            cros_train1 += cros_train[-1]
            cros_valid1 += cros_valid[-1]
            train_acc1  += train_acc[-1]
            train_valid1 +=train_valid[-1]
        y1.append(train_acc1/5)
        y2.append(train_valid1/5)
    plt.plot(x,y1,x,y2)
    plt.gca().legend(('Training','Validation'))
    plt.xlabel("Lambda")
    plt.ylabel("Cross Entropy")
    plt.show()
    
        

