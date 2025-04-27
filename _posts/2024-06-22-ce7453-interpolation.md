---
title: 'CE7453: Interpolation'
date: 2024-06-22
permalink: /posts/2024/06/ce7453-interpolation/
tags:
  - CE7453
  - interpolation
  - numerical methods
  - exam preparation
---

This post summarizes key concepts of interpolation methods covered in CE7453, based on the lecture notes "05-interpolation".

## Key Concepts

Interpolation is the process of estimating unknown values between known data points. It's a fundamental technique in numerical methods with applications in data analysis, computer graphics, physical simulation, and many other fields.

## Polynomial Interpolation

### 1. Lagrange Interpolation

- **Form**: P(x) = Σ<sub>i=0</sub><sup>n</sup> y<sub>i</sub> L<sub>i</sub>(x)
- **Basis Functions**: L<sub>i</sub>(x) = Π<sub>j≠i</sub> (x-x<sub>j</sub>)/(x<sub>i</sub>-x<sub>j</sub>)
- **Properties**:
  - Exact fit through all data points
  - Degree n polynomial for n+1 points
- **Limitations**:
  - High oscillations (Runge phenomenon)
  - Computationally intensive for large datasets
  - Sensitive to changes in data points

### 2. Newton's Divided Differences

- **Form**: P(x) = a<sub>0</sub> + a<sub>1</sub>(x-x<sub>0</sub>) + a<sub>2</sub>(x-x<sub>0</sub>)(x-x<sub>1</sub>) + ...
- **Coefficients**: Calculated using divided differences table
- **Advantages**:
  - Easy to add new points without recalculating
  - More numerically stable than Lagrange
  - More efficient evaluation

### 3. Hermite Interpolation

- **Concept**: Interpolates both function values and derivatives
- **Order**: Higher continuity at interpolation points
- **Applications**: Smooth curve generation, physical simulations

## Piecewise Interpolation

### 1. Linear Interpolation

- **Form**: P(x) = y<sub>i</sub> + (y<sub>i+1</sub>-y<sub>i</sub>)/(x<sub>i+1</sub>-x<sub>i</sub>) × (x-x<sub>i</sub>)
- **Properties**:
  - C<sup>0</sup> continuity at knots
  - Simple and efficient
  - Limited accuracy

### 2. Cubic Spline Interpolation

- **Types**:
  - Natural Splines: Second derivatives are zero at endpoints
  - Clamped Splines: First derivatives specified at endpoints
  - Not-a-knot: Third derivatives match at first and last interior points
- **Properties**:
  - C<sup>2</sup> continuity across all points
  - Minimizes oscillation (minimum curvature)
  - Efficiently representable as piecewise cubics
- **Matrix Form**: Tridiagonal system for coefficient calculation

### 3. Cubic Hermite Splines

- **Form**: Local hermite interpolation between adjacent points
- **Parametrization**: Often uses normalized parameter t ∈ [0,1]
- **Basis Functions**: Hermite basis functions H<sub>i</sub>(t)
- **Applications**: Path interpolation, animation

## Multivariate Interpolation

### 1. Bilinear Interpolation

- **Concept**: Extension of linear interpolation to 2D grid
- **Process**: Interpolate along rows, then along resulting column
- **Applications**: Image scaling, texture mapping

### 2. Bicubic Interpolation

- **Concept**: Extension of cubic interpolation to 2D
- **Data Required**: Function values, partial derivatives, and cross derivatives
- **Applications**: Image processing, terrain modeling

### 3. Radial Basis Functions

- **Form**: s(x) = Σ<sub>i=1</sub><sup>n</sup> λ<sub>i</sub> φ(||x-x<sub>i</sub>||)
- **Common RBFs**: Gaussian, Multiquadric, Thin-plate spline
- **Applications**: Scattered data interpolation, mesh deformation

## Error Analysis

- **Error Bound**: |f(x) - P(x)| ≤ (M/(n+1)!) × Π<sub>i=0</sub><sup>n</sup> |x-x<sub>i</sub>|
  where M is bound on (n+1)th derivative
- **Runge Phenomenon**: High oscillations near edges with equidistant points
- **Chebyshev Nodes**: Optimal node placement to minimize error

## Exam Focus Areas

1. **Basis Construction**: Deriving basis functions for different methods
2. **Error Analysis**: Understanding and calculating interpolation error
3. **Algorithm Implementation**: Steps for constructing and evaluating interpolants
4. **Method Selection**: Choosing appropriate interpolation methods
5. **Continuity Analysis**: Determining and ensuring continuity properties

## Practice Problems

1. Construct a Lagrange polynomial for a given dataset
2. Set up and solve the tridiagonal system for natural cubic splines
3. Compare errors between different interpolation methods
4. Implement bilinear interpolation for a 2D grid of values

Original lecture notes are available at: `/files/CE7453/CE7453/05-interpolation-4slides1page(1).pdf` 