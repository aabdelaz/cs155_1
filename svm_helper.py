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
import sklearn.model_selection as skl_model_selection
import process_data_helper as pdh

# These two functions can be used to train a linear classififer with a variety of different loss functions and regularization, and evaluate the accuracy on a classification test data set. 

def svm_general(X_train,y_train,X_test,y_test,loss,penalty,alpha,max_iter,tol):
    clf = skl.SGDClassifier(loss = loss, penalty = penalty, alpha = alpha, max_iter = max_iter, tol = tol)
    clf.fit(X_train,y_train)
     
    train_acc,test_acc = loss_eval(clf,X_train,y_train,X_test,y_test)
    
    return clf, train_acc, test_acc

def loss_eval(clf,X_train,y_train,X_test,y_test):
    '''
    Evaluate the training and testing error.
    
    Inputs:
        clf: A trained classifier model
        X_train: A (N1,D) array containing all of the training data.
        y_train: A (N1,1) array containing all of the test data.
        X_test: A (N2,D) array containing all of the test data.
        y_test: A (N2,1) array containing all of the test labels.
        
    Outputs:
        train_acc: A scalar value with percentage of correct classifications on training set.
        test_acc: A scalar value with percentage of correct classifications on training set.
    '''
    # Initialize a correct counter
    correct_train = 0
    correct_test = 0
    
    # Get a vector of predictions
    train_predictions = clf.predict(X_train)
    test_predictions = clf.predict(X_test)
    
    # Iterave over all of the predictions
    for i in range(y_train.size):
        
        if(train_predictions[i] == y_train[i]):
            correct_train = correct_train+1
    
    for j in range(y_test.size):
        if(test_predictions[j] == y_test[j]):
            correct_test = correct_test+1
    
    # Compute the accuracy as number of correct classifications over total number of test data points.
    train_acc = correct_train/y_train.size
    test_acc = correct_test/y_test.size
    
    return train_acc, test_acc

def svm_kfold_eval(data,labels,folds,loss,penalty,alpha,max_iter,tol):
    '''
    Evaluate training and testing accuracy over a number of data folds created using the Kfolds method.
    
    Inputs:
        data: An (N,D) array of feature data
        labels: An (N,1) array of label data
        folds: An integer value for the number of folds to be used.
        loss: A string with the type of loss to be used in the SVM.
        penalty: A string with the type of regression to be used with the SVM.
        alpha: A scalar value with the regression penalty coefficient.
        max_iter: An integer number containing the maximum number of iterations.
        tol: A float containing the tolerance on the optimization method. 
        
    Outputs:
        train_acc: A (folds,1) array of training accuracies from each possible KFold.
        test_acc: A (folds,1) array of testing accuracies from each possible KFold.
    '''
    # Produce sets of indices with KFold.
    kf = skl_model_selection.KFold(folds)
    inds = [ind for ind in kf.split(data,labels)]
    
    # Set up matrices to store training and testing errors.
    train_acc_mat = np.empty((folds,))
    test_acc_mat = np.empty((folds,))
    
    # Iterate through the KFolds
    
    for k in range(folds):
    
        # Get set of indices, then extract data
        train_ind, test_ind = inds[k]
        X_train = data[train_ind]
        y_train = labels[train_ind]
        X_test = data[test_ind]
        y_test = labels[test_ind]
        
        # Fit a classifier and evaluate its performance
        svm, train_acc, test_acc = svm_general(X_train,y_train,X_test,y_test,loss,penalty,alpha,max_iter,tol)
    
        # Store errors
        train_acc_mat[k] = train_acc
        test_acc_mat[k] = test_acc

    return train_acc_mat, test_acc_mat, svm
