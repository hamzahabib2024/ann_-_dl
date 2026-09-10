# Lab 02 — NumPy

**Course:** AIC 380 – Artificial Neural Networks
**Topic:** NumPy, the Python library used to represent the vectors and matrices
that neural networks are built from.

## Files in this folder

| File | What it is |
|---|---|
| `Lab 02.pdf` | The lab manual: concepts, solved activities, and the graded tasks. |
| `activities_ANN_lab_02.ipynb` | The 9 solved activities from the manual (done in class). |
| `tasks_ANN_lab_02.ipynb` | **The graded work** — Lab Tasks 1, 2 and 3, all cells executed with outputs. |

The manual asks for each task to be submitted as a `.py` script
(`matrix_lab.py`, `matrix_operations_lab.py`, `data_analysis_lab.py`). This
repository keeps every lab in notebook form instead, so the three tasks are the
three sections of `tasks_ANN_lab_02.ipynb`. Any section can be exported to the
script the manual names with **File → Download as → Python (.py)**.

## How to run

Open `tasks_ANN_lab_02.ipynb` in Jupyter, VS Code, or Google Colab and run the
cells from top to bottom (`Run All`).

Requirements: `numpy` for all three tasks and `matplotlib` for Task 3 only.

```
pip install numpy matplotlib
```

Run the cells **in order** — each task builds on the arrays created in its own
first cell.

## NumPy concepts covered in this lab

NumPy (Numerical Python) provides the `ndarray`, a multi-dimensional array of
values that all share one type. Its operations are implemented in C, so they run
far faster than the equivalent Python loops, which is why every neural network
library is built on top of it.

```python
import numpy as np
```

### Vectors and matrices

A **vector** is a one-dimensional array, a **matrix** is a two-dimensional one —
a grid of rows and columns.

```python
vector = np.array([1, 2, 3, 4, 5, 6])          # 1-D
matrix = np.array([[1, 2, 3], [4, 5, 6]])      # 2-D, 2 rows x 3 columns
```

### Describing an array

```python
matrix.shape   # (2, 3) — rows and columns
matrix.size    # 6      — total number of elements
matrix.ndim    # 2      — number of dimensions
matrix.dtype   # int64  — the type of the elements
```

### Indexing and slicing

Indexing starts at `0`; negative indices count from the end. A 2-D array is
indexed as `[row, column]`, and slicing is *inclusive–exclusive* — the `start`
item is included, the `stop` item is not.

```python
vector[2]          # 3rd element
matrix[1, 1]       # 2nd row, 2nd column
vector[:3]         # up to and including the 3rd element
vector[3:]         # everything after the 3rd element
vector[-1]         # last element
matrix[:2, :]      # first 2 rows, all columns
matrix[:, 1:2]     # all rows, 2nd column
matrix[matrix > 4] # boolean indexing — every element matching a condition
```

### Element-wise operations vs matrix multiplication

This is the distinction that matters most in this lab.

| | Operator | What it does |
|---|---|---|
| Element-wise | `+` `-` `*` `/` | Combines the two values at the same position |
| Matrix multiplication | `@` or `np.dot()` | Combines a whole row of the left matrix with a whole column of the right |

```python
a * b        # element-wise product — result[i, j] = a[i, j] * b[i, j]
a @ b        # matrix product (dot product) — a different result entirely
np.dot(a, b) # the same as a @ b
```

Matrix multiplication is what happens when a layer of a neural network is
evaluated, so `*` and `@` are not interchangeable.

### Reshaping and transposing

```python
matrix.reshape(9, 1)   # same elements, new shape
matrix.reshape(1, -1)  # -1 means "as many columns as needed"
matrix.flatten()       # collapse to a 1-D array
matrix.T               # transpose — rows become columns
```

### Descriptive statistics

`axis=0` works down each column, `axis=1` works across each row, and leaving
`axis` out uses the whole array.

```python
np.max(matrix), np.min(matrix)
np.max(matrix, axis=0)   # max of each column
np.max(matrix, axis=1)   # max of each row
matrix.mean(), np.var(matrix), np.std(matrix)
```

### Linear algebra

```python
np.linalg.det(matrix)          # determinant
np.linalg.matrix_rank(matrix)  # rank
np.linalg.inv(matrix)          # inverse
matrix.diagonal()              # the diagonal elements
matrix.trace()                 # their sum
```

A matrix only has an inverse when its determinant is not `0`.

### Random values

Setting a **seed** makes the "random" numbers repeatable, which is what makes a
training run reproducible.

```python
np.random.seed(1)
np.random.randint(0, 11, 3)     # 3 integers between 0 and 10
np.random.normal(1.0, 2.0, 3)   # 3 values from a normal distribution
```

### Sparse matrices and vectorize

Two concepts the manual covers that the activities do not use directly:

```python
from scipy import sparse
sparse.csr_matrix(matrix)   # stores only the non-zero values, for mostly-zero data

add_100 = np.vectorize(lambda i: i + 100)
add_100(matrix)             # apply a function to every element
```

## What each task does

### Task 1 — Matrix declaration, indexing and slicing
- **Declaration:** a 3x3 integer matrix and a 4x4 float matrix.
- **Indexing:** reading the element at the 2nd row and 3rd column, and changing
  the element at the 1st row and 4th column to `20.5`. Both are index `[1, 2]`
  and `[0, 3]`, because indices start at `0`.
- **Slicing:** the first two rows and columns of the integer matrix, and the
  last row of the float matrix.

### Task 2 — Matrix operations
- **Declaration:** `matrix_int` and `matrix_float`, both 3x3.
- **Basic operations:** element-wise addition, element-wise multiplication, and
  the transpose of `matrix_int`.
- **Advanced operations:** matrix multiplication with `@`, the determinant of
  `matrix_int`, and the inverse of `matrix_float`. The result is checked by
  multiplying the matrix by its inverse, which gives the identity matrix.

`matrix_int` is `[[1, 2, 3], [4, 5, 6], [7, 8, 10]]` rather than the manual's
`[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`, because that matrix has a determinant of
`0` and would make the determinant step uninteresting.

Floating point results are compared with `np.allclose()` instead of `==`,
because rounding makes exact equality unreliable — the determinant prints as
`-3.000000000000001` rather than `-3`.

### Task 3 — Data analysis on a grade dataset
The dataset is 5 students, each stored as `[ID, math, science, english]`.

- The list of lists is converted to a NumPy array and displayed.
- The math, science and English grades are extracted as separate columns and
  each subject average is calculated.
- The overall average is calculated per student, and the highest and lowest are
  found with `np.argmax()` and `np.argmin()`.
- 5 bonus points are added to every math grade, and the student averages are
  recalculated and compared with the originals.
- The subject averages are plotted as a bar chart with `matplotlib`.

**The trap in this task:** column `0` is the student **ID**, not a grade. It has
to be excluded from every average — the grade columns are addressed as
`grades[:, 1:]` — otherwise the ID is treated as a fourth subject and every
result is wrong.

The bonus points are applied to a copy of the array (`grades.copy()`) so that
the original grades survive for the before/after comparison.

## Note on the activities notebook

The activity code is reproduced as it appears in the manual, with three typos
corrected so that every cell runs:

| Activity | In the manual | Corrected to |
|---|---|---|
| 2 | `print(vector[-1])` | `print(vector_row[-1])` — the vector is called `vector_row` |
| 5 | `print(marix.reshape(9))` | `print(matrix.reshape(9))` |
| 7 | `vector_1 = np.array([4,5,6])` | `vector_2 = np.array([4,5,6])` — otherwise vector-1 is overwritten and `vector_2` never exists |

The manual also repeats Activity 7 as Activity 8. Activity 8 keeps the same dot
product and adds the `vector_1.dot(vector_2)` form, so all three ways of writing
a dot product appear.
