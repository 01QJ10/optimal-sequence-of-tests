#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from itertools import combinations
from module import is_diagonally_dominant, optimal_test_selection


def generate_cov_matrix(csv_filepath):
    """
    Reads CSV, extracts numeric rows, computes covariance matrix and returns it along with column names.
    """
    df = pd.read_csv(csv_filepath)
    numeric_data = df.iloc[1:]
    numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')
    numeric_data = numeric_data.dropna(how='all')
    cov_matrix = numeric_data.cov().to_numpy()
    column_names = numeric_data.columns.tolist()
    return cov_matrix, column_names


def main():
    # Prompt user for sequence length and CSV file
    length = int(input("Enter desired sequence length: "))
    csv_filepath = input("Enter input CSV filepath: ")

    # Generate covariance matrix and get column names
    cov_matrix, col_names = generate_cov_matrix(csv_filepath)
    print("Covariance matrix:")
    print(cov_matrix)

    # Choose method
    choice = input("Use fast method (Definition 1 check on each subset)? [Y/n]: ").strip().lower()
    use_fast = (choice != 'n' and choice != 'no')

    # Prepare output directory and filename
    output_dir = os.path.join(os.getcwd(), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_name = input("Enter output txt filename (without extension): ").strip()
    output_path = os.path.join(output_dir, f"{output_name}.txt")

    result_indices = None

    if use_fast:
        if is_diagonally_dominant(cov_matrix):
            sorted_idx = np.argsort(np.diag(cov_matrix))
            result_indices = sorted_idx.tolist()
        else:
            print("Full covariance is not diagonally-dominant—fast method not applicable.")
    else:
        # Brute-force via Lemma 9
        best = optimal_test_selection(cov_matrix, length)
        if best:
            result_indices = best
        else:
            print(f"No optimal sequence found of length {length}.")

    # Write results if found
    if result_indices:
        with open(output_path, 'w') as f:
            f.write(f"Optimal sequence of length {length}:\n")
            for idx in result_indices:
                name = col_names[idx] if idx < len(col_names) else ""
                f.write(f"{idx}\t{name}\n")
        print(f"Written optimal sequence to {output_path}")
    else:
        print("No sequence written to file.")

if __name__ == '__main__':
    main()
