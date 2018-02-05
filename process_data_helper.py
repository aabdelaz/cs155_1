########################################
# CS/CNS/EE 155 2018
# Project 1
#
# Author:       Ameera Abdelaziz, Andrew Taylor, Jiexin Chen
# Description:  Project 1 Data Processing Helper
########################################


import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

def load_data(filename):
    """
    Function loads data stored in the file filename and returns it as a numpy ndarray.
    
    Inputs:
        filename: given as a string.
        
    Outputs:
        Data contained in the file, returned as a numpy ndarray
    """
    return np.loadtxt(filename, skiprows=1, delimiter=' ')

def process_training_data(data, normalize='None', add_bias=True):
    '''
    Process the data by:
        1. Splitting the first column (Y) from the other columns (X)
        2. Normalizing the columns of X.
        3. Appending a column of ones to X.
    
    Inputs:
        data: A (N, D) array containing Y as its first column and X as
        its last columns.
        normalize: 'stats', or 'l1', denoting the type of normalization. Defaults to none.
        add_bias: Whether to add a column of 1's to the data.
        
    Outputs:
        X: A (N,D) array containing the normalized data
        with a column of ones appended.
        Y: A (N, ) vector containing the labels for X
    '''
    N, D = data.shape
    
    # Split X and Y.
    Y, X = np.split(data, [1], axis=1);
    
    if (normalize == 'stats'):
        # Standard deviation of each column of X.
        sigma = np.std(X, axis=0)
    
        # Mean of each column of X
        mean = np.mean(X, axis=0)
    
        # Normalize X
        X = (X - mean)/sigma
    elif (normalize == 'l1'):
        row_sum = np.sum(X, axis=1).T
        row_sum[row_sum == 0] = 1
        X = (X.T/row_sum).T
    elif (normalize == 'tfidf'):
        transformer = TfidfTransformer(smooth_idf=False)
        X = transformer.fit_transform(counts)
        X.toarray()
    elif (normalize != 'None'):
        print('Invalid normalization keyword. Defaulting to no normalization')
    if (add_bias == True):
        # Append column of ones to original data
        X_padded = np.ones((N,D))
        X_padded[:,:-1] = X
        X = X_padded
    
    return X, Y

def process_testing_data(data, normalize='None', add_bias=True):
    '''
    Process the data by:
        1. Normalizing the columns of X.
        2. Appending a column of ones to X.
    
    Inputs:
        data: A (N, D) array containing Y as its first column and X as
        its last columns.
        normalize: 'stats', or 'l1', denoting the type of normalization. Defaults to none.
        add_bias: Whether to add a column of 1's to the data.
        
    Outputs:
        X: A (N,D) array containing the normalized data
        with a column of ones appended.
    '''
    N, D = data.shape
    
    if (normalize == 'stats'):
        # Standard deviation of each column of X.
        sigma = np.std(data, axis=0)
    
        # Mean of each column of X
        mean = np.mean(data, axis=0)
    
        # Normalize X
        data = (data - mean)/sigma
    elif (normalize == 'l1'):
        row_sum = np.sum(data, axis=1).T
        data = (data.T/row_sum).T
    elif (normalize == 'tfidf'):
        transformer = TfidfTransformer(smooth_idf=False)
        X = transformer.fit_transform(counts)
        X.toarray() 
    elif (normalize != 'None'):
        print('Invalid normalization keyword. Defaulting to no normalization')
    
    if (add_bias == True):
        # Append column of ones to original data
        X = np.ones((N,D+1))
        X[:,:-1] = data
        data = X
    
    return data

def write_predictions(clf, X, filename):
    '''
    Predict labels of X using clf, and write to filename.
    
    Inputs:
        clf: Classifier used to predict.
        X: A (N, D) data array.
        filename: Name of output file.
        
    Outputs:
        X: A (N,D) array containing the normalized data
        with a column of ones appended.
    '''
    Y = clf.predict(X)

    
    lines = ["Id,Prediction\n"]
    for i in range(Y.shape[0]):
        lines.append(str(i+1) + ',' + str(int(Y[i])) + '\n')

    fh = open(filename, "w")
    fh.writelines(lines)
    fh.close()

