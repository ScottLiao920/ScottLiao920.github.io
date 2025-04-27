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

## Direct Methods

### 1. Gaussian Elimination

- **Principle**: Transform the augmented matrix [A|b] to upper triangular form through row operations
- **Complexity**: O(n³) operations
- **Key Features**: 
  - Forward elimination to achieve upper triangular form
  - Back substitution to find solution vector
- **Variants**: With partial pivoting for numerical stability

### 2. LU Decomposition

- **Principle**: Factorize A = LU where L is lower triangular and U is upper triangular
- **Advantages**: 
  - Can solve multiple right-hand sides efficiently
  - Useful for calculating determinants and inverses
- **Implementation**: 
  - Doolittle algorithm
  - Crout algorithm
  - Cholesky decomposition (for symmetric positive definite matrices)

### 3. QR Decomposition

- **Principle**: Factorize A = QR where Q is orthogonal and R is upper triangular
- **Methods**:
  - Gram-Schmidt process
  - Householder reflections
  - Givens rotations
- **Applications**: Least squares problems, eigenvalue calculations

## Iterative Methods

### 1. Jacobi Method

- **Principle**: Solve for each variable using values from previous iteration
- **Convergence**: Requires diagonal dominance
- **Iteration Formula**: x<sub>i</sub><sup>(k+1)</sup> = (b<sub>i</sub> - Σ<sub>j≠i</sub> a<sub>ij</sub>x<sub>j</sub><sup>(k)</sup>)/a<sub>ii</sub>

### 2. Gauss-Seidel Method

- **Principle**: Use most recently computed values during iteration
- **Convergence**: Faster than Jacobi when it converges
- **Iteration Formula**: x<sub>i</sub><sup>(k+1)</sup> = (b<sub>i</sub> - Σ<sub>j<i</sub> a<sub>ij</sub>x<sub>j</sub><sup>(k+1)</sup> - Σ<sub>j>i</sub> a<sub>ij</sub>x<sub>j</sub><sup>(k)</sup>)/a<sub>ii</sub>

### 3. Successive Over-Relaxation (SOR)

- **Principle**: Accelerate Gauss-Seidel using relaxation parameter ω
- **Optimization**: Finding optimal ω can significantly improve convergence
- **Iteration Formula**: x<sub>i</sub><sup>(k+1)</sup> = (1-ω)x<sub>i</sub><sup>(k)</sup> + ω(b<sub>i</sub> - Σ<sub>j≠i</sub> a<sub>ij</sub>x<sub>j</sub><sup>(k+1/k)</sup>)/a<sub>ii</sub>

## Special Matrices and Optimizations

- **Sparse Matrices**: Special storage schemes and algorithms
- **Banded Matrices**: Efficient algorithms for matrices with limited bandwidth
- **Symmetric Positive Definite**: Cholesky factorization
- **Condition Number**: Measure of how sensitive the solution is to changes in the input

## Error Analysis

- **Forward Error**: ||x - x̂|| / ||x||
- **Backward Error**: ||Ax̂ - b|| / ||b||
- **Residual**: r = b - Ax̂

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

Original lecture notes are available at: `/files/CE7453/CE7453/02-LinearSystems-4slides1page(1).pdf` 