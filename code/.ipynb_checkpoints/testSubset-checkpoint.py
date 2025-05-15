#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from module import *  # This should import the is_diagonally_dominant() function

def generate_cov_matrix(csv_filepath):
    """
    Reads a CSV file containing test data, extracts the numeric portion (rows from 1 onward),
    converts the values to numeric (coercing errors to NaN), drops rows with all missing values,
    and returns the covariance matrix of the numeric data as a numpy array.
    
    Parameters:
        csv_filepath (str): Path to the CSV file.
    
    Returns:
        cov_matrix (np.ndarray): Covariance matrix computed from the numeric data.
    """
    # Read CSV file
    normalTests = pd.read_csv(csv_filepath)
    
    # The 0-th row is assumed to be non-numeric (e.g., questions or labels)
    numeric_data = normalTests.iloc[1:]
    
    # Convert all entries to numeric values (non-convertible values become NaN)
    numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')
    
    # Optionally, drop rows that are entirely NaN
    numeric_data = numeric_data.dropna(how='all')
    
    # Compute the covariance matrix and convert it to a numpy array.
    cov_matrix = numeric_data.cov().to_numpy()
    return cov_matrix

def generate_random_indices(max_index, min_length=3, max_length=12):
    """
    Generates a random sorted sequence of unique indices.
    
    Parameters:
        max_index (int): The maximum (exclusive) index available.
        min_length (int): The minimum length of the generated sequence.
        max_length (int): The maximum length of the generated sequence.
        
    Returns:
        indices (list): Sorted list of randomly selected unique indices.
    """
    # Random length between min_length and max_length (inclusive)
    length = np.random.randint(min_length, max_length + 1)
    
    # Choose 'length' unique indices from 0 to max_index-1 without replacement.
    indices = np.random.choice(range(max_index), size=length, replace=False)
    
    # Return the sorted indices (for clarity when viewing the submatrix)
    return sorted(indices)

if __name__ == '__main__':
    # Path to the CSV file containing the normalTests data.
    csv_filepath = '../normalTests.csv'
    
    # Generate the covariance matrix (as a NumPy array).
    cov_matrix = generate_cov_matrix(csv_filepath)
    print("Covariance Matrix (NumPy array):")
    print(cov_matrix)
    
    # Determine the dimension (number of tests) available.
    # (Assumes that the covariance matrix is square.)
    max_index = cov_matrix.shape[0]
    
    # Generate a few random sequences (e.g., 5 sequences)
    num_sequences = 5
    for i in range(num_sequences):
        # Generate a random sequence of indices (each sequence length between 3 and 12)
        indices = generate_random_indices(max_index, min_length=3, max_length=12)
        print("\nRandom Sequence {}:".format(i+1), indices)
        
        # Extract the submatrix using numpy indexing with np.ix_
        submatrix = cov_matrix[np.ix_(indices, indices)]
        print("Submatrix:")
        print(submatrix)
        
        # Check if the submatrix is diagonally dominant using the imported function.
        if is_diagonally_dominant(submatrix):
            print("This submatrix is diagonally dominant.")
        else:
            print("This submatrix is NOT diagonally dominant.")
    indices = [64, 37, 40]
    submatrix = cov_matrix[np.ix_(indices, indices)]
    print("Submatrix:")
    print(submatrix)
    if is_diagonally_dominant(submatrix):
        print("This submatrix is diagonally dominant.")
    else:
        print("This submatrix is NOT diagonally dominant.")

# Random Sequence 1: [37, 40, 64]
# Submatrix:
# [[ 1.4924669  -0.92948321  0.0319181 ]
#  [-0.92948321  4.04609993 -0.75230051]
#  [ 0.0319181  -0.75230051 14.09275362]]
# This submatrix is diagonally dominant.

# Random Sequence 2: [1, 5, 9, 10, 30, 40, 47, 53]
# Submatrix:
# [[24.71234895 13.3868066   9.20518688  9.97139588  6.09016018  4.44539093
#   -1.91361004 -0.82824561]
#  [13.3868066  28.89714195 12.12869508 18.05095541  5.80602593  7.06404584
#   -1.98181818 -3.60486486]
#  [ 9.20518688 12.12869508 28.91082803 33.52098644  5.64271548  4.59459924
#   -3.64545455 -5.82774775]
#  [ 9.97139588 18.05095541 33.52098644 67.88771844  8.05644546  6.30267243
#   -7.2         1.6227027 ]
#  [ 6.09016018  5.80602593  5.64271548  8.05644546 11.46356822  2.84061293
#   -4.2208816  -1.15982456]
#  [ 4.44539093  7.06404584  4.59459924  6.30267243  2.84061293  4.04609993
#   -1.72136364 -0.78268468]
#  [-1.91361004 -1.98181818 -3.64545455 -7.2        -4.2208816  -1.72136364
#   11.90017696 -5.81508772]
#  [-0.82824561 -3.60486486 -5.82774775  1.6227027  -1.15982456 -0.78268468
#   -5.81508772 30.97877193]]
# This submatrix is NOT diagonally dominant.