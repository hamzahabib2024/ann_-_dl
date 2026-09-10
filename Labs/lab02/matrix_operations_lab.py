"""AIC 380 - Artificial Neural Networks
Lab 02, Lab Task 2: basic and advanced matrix operations with NumPy.
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Matrix Declaration
# ---------------------------------------------------------------------------

# A 3x3 integer matrix.
# The last element is 10 rather than 9 on purpose: [[1,2,3],[4,5,6],[7,8,9]]
# has a determinant of 0, which would make the determinant step uninteresting.
matrix_int = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 10]])

# A 3x3 float matrix
matrix_float = np.array([[1.5, 2.5, 3.5],
                         [4.5, 5.5, 6.5],
                         [7.5, 8.5, 10.5]])

print("matrix_int:")
print(matrix_int)
print("\nmatrix_float:")
print(matrix_float)

# ---------------------------------------------------------------------------
# 2. Basic Matrix Operations
# ---------------------------------------------------------------------------
# These work element by element: the result at [i, j] depends only on the two
# values at [i, j]. The integer matrix is promoted to float automatically.

# Element-wise addition
addition = matrix_int + matrix_float
print("\nElement-wise addition (matrix_int + matrix_float):")
print(addition)

# Element-wise multiplication - note the * operator, NOT matrix multiplication
elementwise_product = matrix_int * matrix_float
print("\nElement-wise multiplication (matrix_int * matrix_float):")
print(elementwise_product)

# Transpose of matrix_int - rows become columns
print("\nTranspose of matrix_int:")
print(matrix_int.T)

# ---------------------------------------------------------------------------
# 3. Advanced Matrix Operations
# ---------------------------------------------------------------------------

# Matrix multiplication (dot product) - the @ operator, or np.dot().
# Here every element of the result is a row of the left matrix combined with a
# column of the right one, so it is not the same as the * above.
matrix_product = matrix_int @ matrix_float
print("\nMatrix multiplication (matrix_int @ matrix_float):")
print(matrix_product)
print("Same result with np.dot():", np.allclose(matrix_product, np.dot(matrix_int, matrix_float)))

# Determinant of matrix_int
determinant = np.linalg.det(matrix_int)
print("\nDeterminant of matrix_int:", determinant)
print("Rounded to 2 decimals    :", round(determinant, 2))

# Inverse of matrix_float
inverse_float = np.linalg.inv(matrix_float)
print("\nInverse of matrix_float:")
print(inverse_float)

# A matrix multiplied by its inverse gives the identity matrix.
# np.allclose() is used instead of == because of floating point rounding.
identity_check = matrix_float @ inverse_float
print("\nmatrix_float @ inverse_float (should be the identity matrix):")
print(np.round(identity_check, 10))
print("Is it the identity matrix?", np.allclose(identity_check, np.eye(3)))
