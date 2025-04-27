---
title: 'CE7453: Linear Systems'
date: 2024-06-22
permalink: /posts/2024/06/ce7453-linear-systems/
tags:
  - CE7453
  - linear systems
  - numerical methods
  - exam preparation
---

This post summarizes key concepts and methods for solving linear systems covered in CE7453, based on the lecture notes "02-LinearSystems".

## Key Concepts

Linear systems of the form Ax = b arise in numerous engineering applications. Solving these systems efficiently and accurately is essential for many computational tasks from finite element analysis to optimization problems.

### Real-world Applications

- **Structural Analysis**: Determining displacements and forces in structures
- **Circuit Analysis**: Finding currents and voltages in electrical networks
- **Image Processing**: Image reconstruction and filtering operations
- **Economics**: Input-output models of economic systems
- **Machine Learning**: Solving normal equations in linear regression

## Direct Methods

### 1. Gaussian Elimination

- **Principle**: Transform the augmented matrix [A|b] to upper triangular form through row operations
- **Complexity**: O(n³) operations
- **Key Features**: 
  - Forward elimination to achieve upper triangular form
  - Back substitution to find solution vector
- **Variants**: With partial pivoting for numerical stability

#### Step-by-Step Procedure

1. Form the augmented matrix [A|b]
2. **Forward Elimination**: 
   - For each pivot row i = 1 to n-1
   - For each row j = i+1 to n
     - Compute multiplier m<sub>ji</sub> = a<sub>ji</sub>/a<sub>ii</sub>
     - Update row j: row<sub>j</sub> = row<sub>j</sub> - m<sub>ji</sub> × row<sub>i</sub>
3. **Back Substitution**:
   - x<sub>n</sub> = b<sub>n</sub>/a<sub>nn</sub>
   - For i = n-1 down to 1
     - x<sub>i</sub> = (b<sub>i</sub> - Σ<sub>j=i+1</sub><sup>n</sup> a<sub>ij</sub>x<sub>j</sub>)/a<sub>ii</sub>

#### Worked Example: 3×3 System

Solve the system:
```
2x + y - z = 8
-3x - y + 2z = -11
-2x + y + 2z = -3
```

**Step 1**: Form the augmented matrix
```
[ 2  1 -1 |  8 ]
[-3 -1  2 | -11]
[-2  1  2 | -3 ]
```

**Step 2**: Forward Elimination
- Eliminate x from row 2:
  - Multiplier = -3/2
  - Row 2 = Row 2 - (-3/2)×Row 1
```
[ 2  1 -1 |  8 ]
[ 0  0.5 0.5 | 1 ]
[-2  1  2 | -3 ]
```

- Eliminate x from row 3:
  - Multiplier = -2/2 = -1
  - Row 3 = Row 3 - (-1)×Row 1
```
[ 2  1 -1 |  8 ]
[ 0  0.5 0.5 | 1 ]
[ 0  2  1 | 5 ]
```

- Eliminate y from row 3:
  - Multiplier = 2/0.5 = 4
  - Row 3 = Row 3 - 4×Row 2
```
[ 2  1 -1 |  8 ]
[ 0  0.5 0.5 | 1 ]
[ 0  0 -1 | 1 ]
```

**Step 3**: Back Substitution
- z = -1 (from row 3)
- y = (1 - 0.5×(-1))/0.5 = 3 (from row 2)
- x = (8 - 1×3 - (-1)×(-1))/2 = 2 (from row 1)

Solution: x = 2, y = 3, z = -1

#### Partial Pivoting

To improve numerical stability, we should use partial pivoting:
1. Before eliminating with pivot a<sub>ii</sub>, find the largest element in column i from rows i to n
2. Interchange rows to make this element the pivot
3. Then proceed with elimination

#### Python Implementation
```python
import numpy as np

def gaussian_elimination(A, b):
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.
    
    Parameters:
    - A: Coefficient matrix (n×n)
    - b: Right-hand side vector (n×1)
    
    Returns:
    - x: Solution vector (n×1)
    """
    n = len(b)
    # Create augmented matrix
    aug = np.column_stack((A, b))
    
    # Forward elimination with partial pivoting
    for i in range(n):
        # Partial pivoting
        max_row = i + np.argmax(abs(aug[i:, i]))
        if max_row != i:
            aug[[i, max_row]] = aug[[max_row, i]]
        
        # Eliminate entries below pivot
        for j in range(i+1, n):
            factor = aug[j, i] / aug[i, i]
            aug[j, i:] = aug[j, i:] - factor * aug[i, i:]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (aug[i, n] - np.sum(aug[i, i+1:n] * x[i+1:])) / aug[i, i]
    
    return x
```

### 2. LU Decomposition

- **Principle**: Factorize A = LU where L is lower triangular and U is upper triangular
- **Advantages**: 
  - Can solve multiple right-hand sides efficiently
  - Useful for calculating determinants and inverses
- **Implementation**: 
  - Doolittle algorithm
  - Crout algorithm
  - Cholesky decomposition (for symmetric positive definite matrices)

#### Algorithms for LU Decomposition

**Doolittle Algorithm** (u<sub>ii</sub> = 1):
- For i = 1 to n:
  - For j = i to n: u<sub>ij</sub> = a<sub>ij</sub> - Σ<sub>k=1</sub><sup>i-1</sup> l<sub>ik</sub>u<sub>kj</sub>
  - For j = i+1 to n: l<sub>ji</sub> = (a<sub>ji</sub> - Σ<sub>k=1</sub><sup>i-1</sup> l<sub>jk</sub>u<sub>ki</sub>)/u<sub>ii</sub>

**Crout Algorithm** (l<sub>ii</sub> = 1):
- For j = 1 to n:
  - For i = j to n: l<sub>ij</sub> = a<sub>ij</sub> - Σ<sub>k=1</sub><sup>j-1</sup> l<sub>ik</sub>u<sub>kj</sub>
  - For i = j+1 to n: u<sub>ji</sub> = (a<sub>ji</sub> - Σ<sub>k=1</sub><sup>j-1</sup> l<sub>jk</sub>u<sub>ki</sub>)/l<sub>jj</sub>

#### Worked Example: LU Decomposition

For the matrix:
```
A = [ 2  1 -1 ]
    [-3 -1  2 ]
    [-2  1  2 ]
```

We find:
```
L = [ 1  0  0 ]    U = [ 2  1 -1 ]
    [-1.5 1  0 ]       [ 0  0.5 0.5]
    [-1  4  1 ]        [ 0  0 -1 ]
```

To solve Ax = b using LU decomposition:
1. Solve Ly = b for y (forward substitution)
2. Solve Ux = y for x (back substitution)

For b = [8, -11, -3]<sup>T</sup>:
- y<sub>1</sub> = 8
- y<sub>2</sub> = -11 - (-1.5)(8) = 1
- y<sub>3</sub> = -3 - (-1)(8) - 4(1) = 1

- x<sub>3</sub> = 1/(-1) = -1
- x<sub>2</sub> = (1 - 0.5(-1))/0.5 = 3
- x<sub>1</sub> = (8 - 1(3) - (-1)(-1))/2 = 2

Solution: x = 2, y = 3, z = -1 (same as with Gaussian elimination)

#### Python Implementation
```python
def lu_decomposition(A):
    """
    Perform LU decomposition using Doolittle algorithm.
    
    Parameters:
    - A: Square matrix
    
    Returns:
    - L: Lower triangular matrix
    - U: Upper triangular matrix
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        # Upper triangular matrix
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        
        # Lower triangular matrix
        L[i, i] = 1  # Diagonal of L is 1
        for j in range(i+1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]
    
    return L, U

def solve_lu(L, U, b):
    """
    Solve LUx = b using forward and back substitution.
    """
    n = L.shape[0]
    # Forward substitution for Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i, j] * y[j] for j in range(i))
    
    # Back substitution for Ux = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i+1, n))) / U[i, i]
    
    return x
```

### 3. QR Decomposition

- **Principle**: Factorize A = QR where Q is orthogonal and R is upper triangular
- **Methods**:
  - Gram-Schmidt process
  - Householder reflections
  - Givens rotations
- **Applications**: Least squares problems, eigenvalue calculations

#### QR Decomposition Methods

**1. Gram-Schmidt Process**:
- Convert columns of A into orthonormal basis
- For j = 1 to n:
  - v<sub>j</sub> = a<sub>j</sub>
  - For i = 1 to j-1:
    - r<sub>ij</sub> = q<sub>i</sub><sup>T</sup>v<sub>j</sub>
    - v<sub>j</sub> = v<sub>j</sub> - r<sub>ij</sub>q<sub>i</sub>
  - r<sub>jj</sub> = ||v<sub>j</sub>||
  - q<sub>j</sub> = v<sub>j</sub>/r<sub>jj</sub>

**2. Householder Reflections**:
- More numerically stable than Gram-Schmidt
- Uses reflections to zero out elements below diagonal
- For k = 1 to n:
  - Apply Householder transformation to create zeros in column k below diagonal

#### Solving with QR Decomposition

To solve Ax = b using QR decomposition:
1. Compute A = QR
2. Compute y = Q<sup>T</sup>b
3. Solve Rx = y using back substitution

#### Numerical Stability

QR decomposition is more numerically stable than Gaussian elimination for many problems, especially those that are ill-conditioned.

## Iterative Methods

### 1. Jacobi Method

- **Principle**: Solve for each variable using values from previous iteration
- **Convergence**: Requires diagonal dominance
- **Iteration Formula**: x<sub>i</sub><sup>(k+1)</sup> = (b<sub>i</sub> - Σ<sub>j≠i</sub> a<sub>ij</sub>x<sub>j</sub><sup>(k)</sup>)/a<sub>ii</sub>

#### Matrix Formulation

Decompose A = D + L + U where:
- D is the diagonal
- L is the strictly lower triangular part
- U is the strictly upper triangular part

Jacobi iteration: x<sup>(k+1)</sup> = D<sup>-1</sup>(b - (L+U)x<sup>(k)</sup>)

#### Worked Example: Jacobi Method

Solve:
```
4x - y + z = 7
4x - 8y + z = -21
-2x + y + 5z = 15
```

**Step 1**: Rearrange to isolate diagonal terms
```
x = (7 + y - z)/4
y = (4x + z + 21)/8
z = (15 + 2x - y)/5
```

**Step 2**: Iterate starting with initial guess x<sup>(0)</sup> = [0, 0, 0]<sup>T</sup>

| Iteration | x<sup>(k)</sup> | y<sup>(k)</sup> | z<sup>(k)</sup> |
|-----------|-----------------|-----------------|-----------------|
| 0         | 0               | 0               | 0               |
| 1         | 1.75            | 2.625           | 3               |
| 2         | 1.46875         | 2.28125         | 2.325           |
| 3         | 1.64             | 2.01            | 2.63            |
| 4         | 1.595           | 2.10            | 2.544           |
| ...       | ...             | ...             | ...             |
| 10        | 1.60            | 2.05            | 2.55            |

The solution converges to x = 1.6, y = 2.05, z = 2.55

#### Python Implementation
```python
def jacobi_method(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Solve Ax = b using Jacobi iterative method.
    
    Parameters:
    - A: Coefficient matrix
    - b: Right-hand side vector
    - x0: Initial guess (zeros if None)
    - tol: Tolerance for convergence
    - max_iter: Maximum iterations
    
    Returns:
    - x: Solution vector
    - iterations: Number of iterations performed
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    
    x = x0.copy()
    x_new = np.zeros(n)
    iterations = 0
    
    while iterations < max_iter:
        for i in range(n):
            sum_ax = np.sum(A[i, :] * x) - A[i, i] * x[i]
            x_new[i] = (b[i] - sum_ax) / A[i, i]
        
        if np.linalg.norm(x_new - x) < tol:
            return x_new, iterations
        
        x = x_new.copy()
        iterations += 1
    
    return x, iterations
```

### 2. Gauss-Seidel Method

- **Principle**: Use most recently computed values during iteration
- **Convergence**: Faster than Jacobi when it converges
- **Iteration Formula**: x<sub>i</sub><sup>(k+1)</sup> = (b<sub>i</sub> - Σ<sub>j<i</sub> a<sub>ij</sub>x<sub>j</sub><sup>(k+1)</sup> - Σ<sub>j>i</sub> a<sub>ij</sub>x<sub>j</sub><sup>(k)</sup>)/a<sub>ii</sub>

#### Matrix Formulation

Gauss-Seidel iteration: x<sup>(k+1)</sup> = (D+L)<sup>-1</sup>(b - Ux<sup>(k)</sup>)

#### Convergence Comparison with Jacobi

Gauss-Seidel usually converges faster than Jacobi for the same problem because it uses the most recent values of the variables. For our previous example, Gauss-Seidel might take 6-7 iterations versus Jacobi's 10.

#### Python Implementation
```python
def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Solve Ax = b using Gauss-Seidel iterative method.
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    
    x = x0.copy()
    iterations = 0
    
    while iterations < max_iter:
        x_old = x.copy()
        
        for i in range(n):
            # Sum for j < i (already updated values)
            sum1 = np.sum(A[i, :i] * x[:i])
            # Sum for j > i (old values)
            sum2 = np.sum(A[i, i+1:] * x_old[i+1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]
        
        if np.linalg.norm(x - x_old) < tol:
            return x, iterations
        
        iterations += 1
    
    return x, iterations
```

### 3. Successive Over-Relaxation (SOR)

- **Principle**: Accelerate Gauss-Seidel using relaxation parameter ω
- **Optimization**: Finding optimal ω can significantly improve convergence
- **Iteration Formula**: x<sub>i</sub><sup>(k+1)</sup> = (1-ω)x<sub>i</sub><sup>(k)</sup> + ω(b<sub>i</sub> - Σ<sub>j≠i</sub> a<sub>ij</sub>x<sub>j</sub><sup>(k+1/k)</sup>)/a<sub>ii</sub>

#### Optimal Relaxation Parameter

For symmetric positive definite matrices, the optimal ω is:
ω<sub>opt</sub> = 2/(1 + √(1-ρ²))

where ρ is the spectral radius of the Jacobi iteration matrix.

For 1 < ω < 2, SOR can converge much faster than Gauss-Seidel. When ω = 1, SOR reduces to Gauss-Seidel.

#### Python Implementation
```python
def sor_method(A, b, omega, x0=None, tol=1e-6, max_iter=100):
    """
    Solve Ax = b using Successive Over-Relaxation (SOR) method.
    
    Parameters:
    - A: Coefficient matrix
    - b: Right-hand side vector
    - omega: Relaxation parameter (1 < omega < 2)
    - x0: Initial guess
    - tol: Tolerance for convergence
    - max_iter: Maximum iterations
    
    Returns:
    - x: Solution vector
    - iterations: Number of iterations performed
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    
    x = x0.copy()
    iterations = 0
    
    while iterations < max_iter:
        x_old = x.copy()
        
        for i in range(n):
            # Sum for j < i (already updated values)
            sum1 = np.sum(A[i, :i] * x[:i])
            # Sum for j > i (old values)
            sum2 = np.sum(A[i, i+1:] * x_old[i+1:])
            
            # SOR update
            x[i] = (1 - omega) * x_old[i] + omega * (b[i] - sum1 - sum2) / A[i, i]
        
        if np.linalg.norm(x - x_old) < tol:
            return x, iterations
        
        iterations += 1
    
    return x, iterations
```

## Special Matrices and Optimizations

- **Sparse Matrices**: Special storage schemes and algorithms
- **Banded Matrices**: Efficient algorithms for matrices with limited bandwidth
- **Symmetric Positive Definite**: Cholesky factorization
- **Condition Number**: Measure of how sensitive the solution is to changes in the input

### Sparse Matrix Storage

For matrices with many zeros, we can use specialized storage formats:
1. **Compressed Sparse Row (CSR)**: Stores non-zero values, column indices, and row pointers
2. **Compressed Sparse Column (CSC)**: Similar to CSR but column-oriented
3. **Coordinate format (COO)**: Stores (row, column, value) triplets

### Banded Matrix Operations

For a matrix with bandwidth m (m non-zero diagonals), the complexity of operations can be reduced:
- Gaussian elimination: O(nm²) instead of O(n³)
- LU decomposition: O(nm²) instead of O(n³)

### Cholesky Decomposition

For symmetric positive definite matrices, we can use the Cholesky decomposition:
- A = LL<sup>T</sup> where L is lower triangular
- More efficient than general LU decomposition (roughly half the operations)

```python
def cholesky(A):
    """
    Compute Cholesky decomposition A = LL^T.
    A must be symmetric positive definite.
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1):
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - sum(L[i, k]**2 for k in range(j)))
            else:
                L[i, j] = (A[i, j] - sum(L[i, k] * L[j, k] for k in range(j))) / L[j, j]
    
    return L
```

## Error Analysis

- **Forward Error**: ||x - x̂|| / ||x||
- **Backward Error**: ||Ax̂ - b|| / ||b||
- **Residual**: r = b - Ax̂

### Condition Number

The condition number κ(A) = ||A|| · ||A<sup>-1</sup>|| affects the error propagation:
- Forward error ≤ κ(A) × Backward error
- Large condition number indicates an ill-conditioned system
- For κ(A) = 10<sup>d</sup>, you may lose up to d digits of accuracy

### Example: Effect of Condition Number

For a well-conditioned system (κ(A) ≈ 10):
- Small changes in b result in small changes in x

For an ill-conditioned system (κ(A) ≈ 10⁶):
- Small changes in b can result in large changes in x
- You might lose 6 digits of accuracy in the solution

## Method Comparison: A Case Study

Consider the following system:
```
10x + y = 11
x + 10y = 11
```

| Method        | Initial Guess | Iterations | Solution               |
|---------------|---------------|------------|------------------------|
| Direct (LU)   | -             | -          | x = 1, y = 0.1         |
| Jacobi        | [0, 0]        | 24         | x = 1, y = 0.1         |
| Gauss-Seidel  | [0, 0]        | 13         | x = 1, y = 0.1         |
| SOR (ω=1.5)   | [0, 0]        | 6          | x = 1, y = 0.1         |

## Exam Focus Areas

1. **Algorithm Selection**: Know when to use direct vs. iterative methods
2. **Implementation Details**: Steps for the main algorithms
3. **Convergence Analysis**: When and how fast iterative methods converge
4. **Computational Complexity**: Understanding operation counts
5. **Error Estimation**: Calculating and interpreting errors and residuals

## Practice Problems

1. Solve a linear system using Gaussian elimination with partial pivoting
2. Implement LU decomposition for a given matrix
3. Compare convergence rates of Jacobi and Gauss-Seidel methods
4. Analyze condition number and its effect on solution accuracy

### Challenge Problem

Consider the system Ax = b where:
```
A = [ 10  7  0  1 ]    b = [ 32 ]
    [  3 10  0  2 ]        [ 24 ]
    [  0  4  8  5 ]        [ 28 ]
    [  1  6  2 10 ]        [ 16 ]
```

1. Solve this system using LU decomposition
2. Calculate the condition number of A
3. Solve using Jacobi and Gauss-Seidel methods
4. Determine which method converges faster and why

Original lecture notes are available at: `/files/CE7453/CE7453/02-LinearSystems-4slides1page(1).pdf` 