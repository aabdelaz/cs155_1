########################################
# CS/CNS/EE 155 2018
# Project 1
#
# Author:       Ameera Abdelaziz
# Description:  Project 1 Decision Tree/Forest Helper
########################################

import numpy as np
import sklearn as skl
import sklearn.ensemble as skl_ensemble
from sklearn.model_selection import KFold
import process_data_helper as pdh

# These two functions can be used to train a linear classififer with a variety of different loss functions and regularization, and evaluate the accuracy on a classification test data set. 

def adaboost_eval(X_train,y_train,X_test,y_test, n_estimators, split_criterion, learning_rate, max_depth, min_samples_leaf):
    base_clf = skl.tree.DecisionTreeClassifier(criterion = split_criterion,
    max_depth = max_depth, min_samples_leaf = min_samples_leaf)
    clf = skl.ensemble.AdaBoostClassifier(base_estimator = base_clf, n_estimators = n_estimators, learning_rate = learning_rate)
    clf.fit(X_train,y_train)
     
    train_acc,test_acc = loss_eval(clf,X_train,y_train,X_test,y_test)
    
    return train_acc, test_acc

def forest_eval(X_train,y_train,X_test,y_test, n_estimators, split_criterion, max_depth, min_samples_leaf):
    clf = skl.ensemble.RandomForestClassifier(n_estimators = n_estimators, criterion = split_criterion,
    max_depth = max_depth, min_samples_leaf = min_samples_leaf)
    clf.fit(X_train,y_train)
     
    train_acc,test_acc = loss_eval(clf,X_train,y_train,X_test,y_test)
    
    return train_acc, test_acc

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

def adaboost_kfold_eval(data,labels,folds,n_estimators,split_criterion, learning_rate=1,max_depth=None, min_samples_leaf=1):
    '''
    Evaluate training and testing accuracy over a number of data folds created using the Kfolds method.
    
    Inputs:
        data: An (N,D) array of feature data
        labels: An (N,1) array of label data
        folds: An integer value for the number of folds to be used.
        n_estimators, split_criterion, max_depth, min_samples_leaf: parameters for decision forest classifiers
        
    Outputs:
        train_acc: A (folds,1) array of training accuracies from each possible KFold.
        test_acc: A (folds,1) array of testing accuracies from each possible KFold.
    '''
    # Produce sets of indices with KFold.
    kf = KFold(folds)
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
        train_acc, test_acc = adaboost_eval(X_train,y_train,X_test,y_test,n_estimators,split_criterion, learning_rate,
        max_depth, min_samples_leaf)
    
        # Store errors
        train_acc_mat[k] = train_acc
        test_acc_mat[k] = test_acc

    return train_acc_mat, test_acc_mat                                          
                                          
def forest_kfold_eval(data,labels,folds,n_estimators,split_criterion,max_depth=None, min_samples_leaf=1):
    '''
    Evaluate training and testing accuracy over a number of data folds created using the Kfolds method.
    
    Inputs:
        data: An (N,D) array of feature data
        labels: An (N,1) array of label data
        folds: An integer value for the number of folds to be used.
        n_estimators, split_criterion, max_depth, min_samples_leaf: parameters for decision forest classifiers
        
    Outputs:
        train_acc: A (folds,1) array of training accuracies from each possible KFold.
        test_acc: A (folds,1) array of testing accuracies from each possible KFold.
    '''
    # Produce sets of indices with KFold.
    kf = KFold(folds)
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
        train_acc, test_acc = forest_eval(X_train,y_train,X_test,y_test,n_estimators,split_criterion,
        max_depth, min_samples_leaf)
    
        # Store errors
        train_acc_mat[k] = train_acc
        test_acc_mat[k] = test_acc

    return train_acc_mat, test_acc_mat

def adaboost_train(data,labels,n_estimators, split_criterion,learning_rate=1, max_depth=None, min_samples_leaf=1):
    '''
    Evaluate training and testing accuracy over a full data set, and return the classifier.
    
    Inputs:
        data: An (N,D) array of feature data
        labels: An (N,1) array of label data
        n_estimators,split_criterion,max_depth, min_samples_leaf: Parameters for decision forest.
        
    Outputs:
        train_acc: A scalar value of training accuracy
        clf: The trained adaboost classifer.
    '''
    # Create a classifier
    base_clf = skl.tree.DecisionTreeClassifier(criterion = split_criterion,
    max_depth = max_depth, min_samples_leaf = min_samples_leaf)
    clf = skl.ensemble.AdaBoostClassifier(base_estimator = base_clf, n_estimators = n_estimators, learning_rate = learning_rate)
    
    # Train it
    clf.fit(data,labels)
    
    # Compute training error
    correct_train = 0
    
    # Get a vector of predictions
    train_predictions = clf.predict(data)
    
    # Iterave over all of the predictions
    for i in range(labels.size):
        if(train_predictions[i] == labels[i]):
            correct_train = correct_train+1
    
    # Compute the accuracy as number of correct classifications over total number of test data points.
    train_acc = correct_train/labels.size

    return train_acc, clf                                          
                                          
def forest_train(data,labels,n_estimators,split_criterion,max_depth=None, min_samples_leaf=1):
    '''
    Evaluate training and testing accuracy over a full data set, and return the classifier.
    
    Inputs:
        data: An (N,D) array of feature data
        labels: An (N,1) array of label data
        n_estimators,split_criterion,max_depth, min_samples_leaf: Parameters for decision forest.
        
    Outputs:
        train_acc: A scalar value of training accuracy
        clf: The trained decision forest classifer.
    '''
    # Create a classifier
    clf = skl_ensemble.RandomForestClassifier(n_estimators=n_estimators,criterion=split_criterion,
                                                max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    
    # Train it
    clf.fit(data,labels)
    
    # Compute training error
    correct_train = 0
    
    # Get a vector of predictions
    train_predictions = clf.predict(data)
    
    # Iterave over all of the predictions
    for i in range(labels.size):
        if(train_predictions[i] == labels[i]):
            correct_train = correct_train+1
    
    # Compute the accuracy as number of correct classifications over total number of test data points.
    train_acc = correct_train/labels.size

    return train_acc, clf