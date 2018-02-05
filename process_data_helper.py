########################################
# CS/CNS/EE 155 2018
# Project 1
#
# Author:       Ameera Abdelaziz, Andrew Taylor
# Description:  Project 1 Data Processing Helper
########################################


import numpy as np

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
    Y, Xorig = np.split(data, [1], axis=1);
    
    if (normalize == 'stats'):
        # Standard deviation of each column of X.
        sigma = np.std(Xorig, axis=0)
    
        # Mean of each column of X
        mean = np.mean(Xorig, axis=0)
    
        # Normalize X
        Xorig = (Xorig - mean)/sigma
    elif (normalize == 'l1'):
        row_sum = np.sum(Xorig, axis=1).T
        row_sum[row_sum == 0] = 1
        Xorig = (Xorig.T/row_sum).T
    elif (normalize != 'None'):
        print('Invalid normalization keyword. Defaulting to no normalization')
    
    if (add_bias == True):
        # Append column of ones to original data
        X = np.ones((N,D))
        X[:,:-1] = Xorig
    
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
    elif (normalize != 'None'):
        print('Invalid normalization keyword. Defaulting to no normalization')
    
    if (add_bias == True):
        # Append column of ones to original data
        X = np.ones((N,D+1))
        X[:,:-1] = data
        data = X
    
    return data