import numpy as np
import os
from module import *
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def generate_random_covariance_matrix(size, sample_size=1000):
    df = sample_size - 1  # degrees of freedom
    variances = np.random.chisquare(df, size=size)
    covariance_matrix = np.diag(variances)
    return covariance_matrix

def generate_matrices(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs('diagonalMatrices')

    # Generate and save the matrices
    for size in range(1, 101):  # from 1x1 to 100x100 matrices
        for i in range(10):  # generate 10 matrices for each size
            matrix = generate_random_covariance_matrix(size)
            filename = f"{directory}/matrix_{size}_{i+1}.txt"
            np.savetxt(filename, matrix, fmt='%.5f')

# def read_matrices(folder_path):
#     matrices = {}
#     for size in range(1, 2):
#         matrices[size] = []
#         for i in range(10):
#             filename = f"{folder_path}/matrix_{size}_{i+1}.txt"
#             matrix = np.loadtxt(filename)
#             matrices[size].append(matrix)
#     return matrices


def check_matrices(folder_path):
    for size in range(10, 13):
        for i in range(10):
            filename = f"{folder_path}/matrix_{size}_{i+1}.txt"
            start_time = time.time()  # Start timing

            matrix = np.loadtxt(filename)
            if size == 1:
                matrix = np.array([[matrix]])
            # print(matrix)
            # print(type(matrix))
            # print(len(matrix))
            if not is_diagonally_dominant(matrix):
                return False
            end_time = time.time()  # End timing
            # Print time taken for this matrix
            print(f"Checked matrix {filename} in {end_time - start_time:.4f} seconds.")
    
    return True



def check_matrix(file_path):
    start_time = time.time()  # Start timing

    matrix = np.loadtxt(file_path)
    if matrix.size == 1:
        matrix = np.array([[matrix]])  # Ensure the matrix is 2D

    dominant = is_diagonally_dominant(matrix)
    end_time = time.time()  # End timing

    # Print time taken for this matrix
    print(f"Checked matrix {file_path} in {end_time - start_time:.4f} seconds.")
    return dominant

def check_matrices_concurrently(folder_path):
    tasks = []
    with ThreadPoolExecutor(max_workers=8) as executor:  # You can adjust max_workers based on your system
        for size in range(1, 101):
            for i in range(0, 10):
                file_path = f"{folder_path}/matrix_{size}_{i+1}.txt"
                tasks.append(executor.submit(check_matrix, file_path))

        # Collect results
        for future in as_completed(tasks):
            if not future.result():
                print("A non-diagonally dominant matrix found.")
                return False

    return True
if __name__ == "__main__":
    directory = '../tests/diagonalMatrices'
    # generate_matrices(directory)
    # print("Matrices generated and saved in 'diagonalMatrices' folder.")
    # results = check_matrices(directory)
    results = check_matrices_concurrently(directory)
    print(results)