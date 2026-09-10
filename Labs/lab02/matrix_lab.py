"""AIC 380 - Artificial Neural Networks
Lab 02, Lab Task 1: declaring, indexing and slicing matrices with NumPy.
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Matrix Declaration
# ---------------------------------------------------------------------------

# Create a 3x3 matrix of integers
int_matrix = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

# Create a 4x4 matrix of floats
float_matrix = np.array([[1.5, 2.5, 3.5, 4.5],
                         [5.5, 6.5, 7.5, 8.5],
                         [9.5, 10.5, 11.5, 12.5],
                         [13.5, 14.5, 15.5, 16.5]])

print("3x3 integer matrix:")
print(int_matrix)
print("dtype:", int_matrix.dtype, "| shape:", int_matrix.shape)

print("\n4x4 float matrix:")
print(float_matrix)
print("dtype:", float_matrix.dtype, "| shape:", float_matrix.shape)

# ---------------------------------------------------------------------------
# 2. Matrix Indexing
# ---------------------------------------------------------------------------
# Indices start at 0, so the 2nd row is index 1 and the 3rd column is index 2.

# Access the element at the 2nd row and 3rd column of the integer matrix
element = int_matrix[1, 2]
print("\nElement at 2nd row, 3rd column of the integer matrix:", element)

# Change the element at the 1st row and 4th column of the float matrix to 20.5
print("1st row of the float matrix before the change:", float_matrix[0])
float_matrix[0, 3] = 20.5
print("1st row of the float matrix after  the change:", float_matrix[0])

print("\nFloat matrix after the update:")
print(float_matrix)

# ---------------------------------------------------------------------------
# 3. Matrix Slicing
# ---------------------------------------------------------------------------
# Slicing is inclusive of the start index and exclusive of the stop index.

# Extract the first two rows and columns of the integer matrix
first_two = int_matrix[:2, :2]
print("\nFirst two rows and columns of the integer matrix:")
print(first_two)

# Extract the last row of the float matrix
last_row = float_matrix[-1, :]
print("\nLast row of the float matrix:", last_row)
