import sklearn
import scipy
import numpy
import pandas
import random
import sklearn.feature_extraction
import sklearn.model_selection
from sklearn import tree
import graphviz
import pydotplus
import collections
vectorizer = sklearn.feature_extraction.text.CountVectorizer()

def load_data():
    Y = []
    # load data in pandas dataframe
    data = pandas.read_csv('clean_fake.txt', sep ="\t",header=None)
    # add label 0 for y which implies false
    r,c = data.shape
    for i  in range(0,r):
        Y.append(0)
    data2 = data.append(pandas.read_csv('clean_real.txt', sep = "\t", header=None))
    # add label 0 for y which implies false
    newr , newc =  data2.shape
    r2  =  newr-r
    for i in range(0,r2):
        Y.append(1)
    array = pandas.DataFrame(data2).to_numpy()
    l = array.tolist()
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)
    X = vectorizer.fit_transform(flat_list)
    # print(Y)
    X_train, X_validate_test, Y_train, Y_validate_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.3)
    X_validate, X_test, Y_validate, Y_test = sklearn.model_selection.train_test_split(X_validate_test, Y_validate_test, test_size=0.3)
    return X_train, X_validate, X_test, Y_train, Y_validate, Y_test
def check_accuracy(model,x,y):
    pred = model.predict(x)
    #  True positives+ false negative / total number of samples
    sum = 0
    for i in range(0,len(y)):
        if(y[i]==pred[i]):
            sum += 1
    return float(sum/len(y))
def select_model(trainx,trainy,validx,validy):
    trees = {}
    criterion = ["gini","entropy"]
    depth = 7
    for i in range(1,5):
        for j in criterion:
            tree = sklearn.tree.DecisionTreeClassifier(criterion=j, max_depth=i)
            tree.fit(trainx,trainy)
            accuracy = check_accuracy(tree,validx,validy)
            print("Model(depth= " +  str(i) + "criteria= "  +  str(j)+") with accuracy = " + str(accuracy))
            trees[accuracy] = tree
    keys =  trees.keys()
    max_key =  max(keys)
    solution = trees[max_key]
    return solution

def compute_information_gain

(X_train, X_validate, X_test, Y_train, Y_validate, Y_test) = load_data()
best_model = select_model(X_train,Y_train,X_validate,Y_validate)
accuracy =  check_accuracy(best_model,X_test, Y_test )
print("Best_Model Testing Accuracy =  " +  str(accuracy))
# Draw graph code over here
