########################################
# CS/CNS/EE 155 2018
# Project 1
#
# Author:       Andrew Taylor
# Description:  Project 1 SVM Helper
########################################

import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl

# These two functions can be used to train a linear classififer with a variety of different loss functions and regularization, and evaluate the accuracy on a classification test data set. 

def svm_general(X_train,y_train,X_test,y_test,loss,penalty,alpha,max_iter,tol):
    clf = skl.SGDClassifier(loss = loss, penalty = penalty, alpha = alpha, max_iter = max_iter, tol = tol)
    clf.fit(X_train,y_train)
    accuracy = loss_eval(clf.coef_,X_test,y_test)
    
    return clf.coef_, accuracy

def loss_eval(coefs,X_test,y_test):
    
    correct = 0
    for i in range(y_test.size):
        
        val = np.dot(coefs,X_test[i,:])
        if(val>0):
            val = 1
        elif(val<0):
            val = 0
        if(val==y_test[i]):
                correct = correct + 1
    
    accuracy = correct/y_test.size
    
    return accuracy