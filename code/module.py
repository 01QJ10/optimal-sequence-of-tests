from itertools import combinations
import numpy as np
from scipy.stats import kstest, shapiro
import pandas as pd

def is_diagonally_dominant(cov_matrix):

    """
    Diagonal dominance here is defined as the condition where the sum of the elements of the inverse
    of the covariance submatrix formed by the top k variances is greater than or equal to the sum of the
    elements of the inverse of any other submatrix of the same size.
    """
    num_tests = len(cov_matrix)
    variances = np.diag(cov_matrix)
    ordered_indices = np.argsort(variances)  # Indices sorted by descending variance

    # Loop through all possible submatrix sizes
    for k in range(1, num_tests + 1):
        # Generate the submatrix from the top k ordered indices
        main_indices = ordered_indices[:k]
        main_submatrix = cov_matrix[np.ix_(main_indices, main_indices)]
        #print(main_submatrix)
        # Check invertibility and compute the sum of the inverse matrix
        if np.linalg.det(main_submatrix) == 0:
            raise ValueError("The main submatrix is not invertible.")

        
            
        # Compute the sum of the elements of the inverse of the main submatrix   
        main_sum = np.sum(np.linalg.inv(main_submatrix))

        # Compare with all other combinations of k indices
        for sub_indices in combinations(range(num_tests), k):
            if set(sub_indices) == set(main_indices):
                continue
            submatrix = cov_matrix[np.ix_(sub_indices, sub_indices)]
            #print(submatrix)
            # Ensure submatrix is invertible
            if np.linalg.det(submatrix) == 0:
                continue  # Skip non-invertible submatrices

            # Calculate the sum of the elements of the inverse of this submatrix
            sub_sum = np.sum(np.linalg.inv(submatrix))
            if main_sum < sub_sum:
                return False
    return True

def find_optimal_tests(cov_matrix):

    """
    Diagonal dominance here is defined as the condition where the sum of the elements of the inverse
    of the covariance submatrix formed by the top k variances is greater than or equal to the sum of the
    elements of the inverse of any other submatrix of the same size.
    """
    num_tests = len(cov_matrix)
    variances = np.diag(cov_matrix)
    ordered_indices = np.argsort(variances)  # Indices sorted by descending variance

    # Loop through all possible submatrix sizes
    for k in range(1, num_tests + 1):
        # Generate the submatrix from the top k ordered indices
        main_indices = ordered_indices[:k]
        main_submatrix = cov_matrix[np.ix_(main_indices, main_indices)]
        #print(main_submatrix)
        # Check invertibility and compute the sum of the inverse matrix
        if np.linalg.det(main_submatrix) == 0:
            raise ValueError("The main submatrix is not invertible.")

        
            
        # Compute the sum of the elements of the inverse of the main submatrix   
        main_sum = np.sum(np.linalg.inv(main_submatrix))

        # Compare with all other combinations of k indices
        for sub_indices in combinations(range(num_tests), k):
            if set(sub_indices) == set(main_indices):
                continue
            submatrix = cov_matrix[np.ix_(sub_indices, sub_indices)]
            #print(submatrix)
            # Ensure submatrix is invertible
            if np.linalg.det(submatrix) == 0:
                continue  # Skip non-invertible submatrices

            # Calculate the sum of the elements of the inverse of this submatrix
            sub_sum = np.sum(np.linalg.inv(submatrix))
            if main_sum < sub_sum:
                return False
        
    return True

def find_optimal_indices(cov_matrix):
    variances = np.diag(cov_matrix)
    sorted_indices = np.argsort(-variances)

    return sorted_indices

def optimal_test_selection(cov_matrix, T):
    """
    Implements the optimal policy based on Lemma 9 and Theorem 1. It computes the optimal subset 
    of indices from a covariance matrix that maximizes the sum of elements in the inverse of the 
    submatrix formed by these indices, minimizing 1/inverse sum, and thus minimizing the effective variance.


    Returns:
    list: Sequence of indices representing the optimal selection of tests over time.
    """
    num_tests = len(cov_matrix)  # Total number of available tests
    best_indices = None  # To store the indices of the optimal subset of tests
    max_sum_inverse = -np.inf  # Initialize with a very small number to ensure any sum will be larger

    # Evaluate all combinations of test indices to find the one that maximizes the sum of the inverse submatrix
    for indices in combinations(range(num_tests), T):
        submatrix = cov_matrix[np.ix_(indices, indices)]  # Extract the submatrix for the current combination of indices
        try:
            # Calculate the inverse of the submatrix
            inverse_submatrix = np.linalg.inv(submatrix)
            # Sum of all elements in the inverse matrix, equivalent to 1' * inv(Sigma) * 1 for a vector of ones
            sum_inverse = np.sum(inverse_submatrix)

            # Check if the current sum is greater than the previously found maximum
            if sum_inverse > max_sum_inverse:
                max_sum_inverse = sum_inverse  # Update the maximum sum found
                best_indices = indices  # Update the best indices corresponding to this maximum
        except np.linalg.LinAlgError:
            continue  # Skip this combination if the matrix is non-invertible

    return list(best_indices)  # Return the list of indices for the optimal subset]

def convert_percentages(value):
    if isinstance(value, str) and '%' in value:
        return float(value.strip('%')) / 100
    return value

def test_normality(df: pd.DataFrame, output_csv: str = "../data/normality_results.csv") -> pd.DataFrame:
    """
    Takes in a DataFrame with the first row as question labels and the rest as numerical data,
    standardizes each column, and performs KS and Shapiro normality tests.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        output_csv (str): Path to save the resulting CSV. Default is "normality_results.csv".

    Returns:
        pd.DataFrame: A DataFrame containing test statistics and p-values.
    """
    # Extract first row as question labels
    questions = df.iloc[0]
    df_data = df.iloc[1:].copy()

    # Convert percentages and other strings to numeric
    df_data = df_data.applymap(convert_percentages)
    df_data = df_data.apply(pd.to_numeric, errors='coerce')

    results = []
    for idx in range(df_data.shape[1]):
        col_data = df_data.iloc[:, idx].dropna()

        # Standardize the data
        mean = col_data.mean()
        std_dev = col_data.std()
        if std_dev == 0 or len(col_data) < 3:
            continue  # Skip if std is zero or not enough data for Shapiro

        standardized_data = (col_data - mean) / std_dev

        # Perform normality tests
        ks_stat, ks_p = kstest(standardized_data, 'norm')
        shapiro_stat, shapiro_p = shapiro(standardized_data)

        results.append({
            'Test Name': df.columns[idx],
            'Question': questions[idx],
            'KS Statistic': ks_stat,
            'KS P-value': ks_p,
            'Shapiro Statistic': shapiro_stat,
            'Shapiro P-value': shapiro_p
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    return results_df