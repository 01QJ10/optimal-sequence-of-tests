import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from module import is_diagonally_dominant

# Dynamically determine number of workers (default to CPU cores)
MAX_WORKERS = int(os.getenv("MAX_WORKERS", os.cpu_count() or 1))


def generate_random_covariance_matrix(size, sample_size=1000):
    df = sample_size - 1
    variances = np.random.chisquare(df, size=size)
    return np.diag(variances)


def generate_matrices(folder_path, generator, min, max, n):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for size in range(min, max):
        for i in range(n):
            matrix = generator(size)
            filename = f"{folder_path}/matrix_{size}_{i+1}.txt"
            np.savetxt(filename, matrix, fmt='%.5f')

def generate_non_diagonally_dominant(n: int) -> np.ndarray:
    """
    Return an n×n symmetric positive-definite matrix Σ
    that violates Definition 1 (i.e. is NOT diagonally 
    dominant in the paper's precision-based sense).
    
    Requires n >= 3 (smallest size admitting a k=2 violation).
    """
    if n < 3:
        raise ValueError("Need n >= 3 to violate Definition 1 (k=2 check).")

    # --- 1) A 3×3 "bad" block B that fails at k=2 ---
    # variances = [1, 4, 9], covariances chosen so that
    # the prefix {0,1} loses to {0,2} under the inverse-sum test.
    B = np.array([
        [1.0,  0.1,  2.5],   # var0=1, cov01=0.1, cov02=2.5
        [0.1,  4.0,  0.0],   # var1=4
        [2.5,  0.0,  9.0],   # var2=9
    ])

    # Quick sanity check: B must be PD
    if np.any(np.linalg.eigvalsh(B) <= 0):
        raise RuntimeError("Block B is not positive-definite!")

    # --- 2) Build Σ by embedding B and padding the rest ---
    res = np.zeros((n, n))
    res[:3, :3] = B

    # For the remaining indices 3..n-1, just use large diagonal entries
    for i in range(3, n):
        res[i, i] = (i + 1)**2  # e.g. variances 16,25,36,...

    return res

def check_matrix(file_path):
    start_time = time.time()
    matrix = np.loadtxt(file_path)
    if matrix.size == 1:
        matrix = np.array([[matrix]])

    try:
        dominant = is_diagonally_dominant(matrix)
    except Exception as e:
        print(f"Error checking {file_path}: {e}")
        return False

    elapsed = time.time() - start_time
    status = "dominant" if dominant else "NOT dominant"
    print(f"Checked {file_path}: {status} in {elapsed:.4f}s")

    if not dominant:
        print("Matrix contents (non-dominant):")
        print(matrix)

    return dominant


def check_matrices_concurrently(folder_path, min, max, n):
    failures = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(check_matrix, f"{folder_path}/matrix_{size}_{i+1}.txt"): (size, i+1)
            for size in range(min, max)
            for i in range(n)
        }

        for future in as_completed(futures):
            if not future.result():
                failures += 1

    if failures:
        print(f"Completed checks with {failures} non-dominant matrices.")
        return False
    else:
        print("All matrices are diagonally dominant.")
        return True


if __name__ == "__main__":
    # directory = '../tests/diagonalMatrices'
    # print(f"Using up to {MAX_WORKERS} worker threads.")
    # results = check_matrices_concurrently(directory, 1, 101, 10)
    # print("Overall result:", results)

    directory = '../tests/nonDiagonalMatrices'
    print(f"Using up to {MAX_WORKERS} worker threads.")
    results = check_matrices_concurrently(directory, 3, 21, 1)
    print("Overall result:", results)
