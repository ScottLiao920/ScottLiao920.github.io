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

### Mathematical Definition

Given a set of n+1 distinct points (x₀, y₀), (x₁, y₁), ..., (xₙ, yₙ), interpolation seeks to find a function f(x) that passes through all these points, i.e., f(xᵢ) = yᵢ for all i = 0, 1, ..., n.

### Real-world Applications

- **Digital Signal Processing**: Reconstructing continuous signals from discrete samples
- **Computer Graphics**: Smooth animation transitions, curve generation
- **Medical Imaging**: Reconstruction of high-resolution images from lower-resolution scans
- **Geospatial Analysis**: Creating elevation models from sampled points
- **Engineering Design**: Creating smooth curves through design constraints

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

#### Derivation of Lagrange Basis Functions

For each point (xᵢ, yᵢ), we need a basis function Lᵢ(x) that equals 1 at x = xᵢ and 0 at all other data points xⱼ where j ≠ i.

The formula:
L<sub>i</sub>(x) = Π<sub>j≠i</sub> (x-x<sub>j</sub>)/(x<sub>i</sub>-x<sub>j</sub>)

satisfies these properties:
- When x = xᵢ, each term in the product becomes (xᵢ-xⱼ)/(xᵢ-xⱼ) = 1, so Lᵢ(xᵢ) = 1
- When x = xₖ where k ≠ i, one term becomes (xₖ-xₖ)/(xᵢ-xₖ) = 0, making Lᵢ(xₖ) = 0

#### Worked Example: Lagrange Interpolation

Consider the data points: (0, 1), (1, 3), (2, 2).

**Step 1**: Calculate Lagrange basis functions
- L₀(x) = ((x-1)(x-2))/((0-1)(0-2)) = (x-1)(x-2)/2
- L₁(x) = ((x-0)(x-2))/((1-0)(1-2)) = (x)(x-2)/(-1) = -x(x-2)
- L₂(x) = ((x-0)(x-1))/((2-0)(2-1)) = (x)(x-1)/2

**Step 2**: Form the interpolating polynomial
P(x) = 1·L₀(x) + 3·L₁(x) + 2·L₂(x)
     = 1·((x-1)(x-2))/2 + 3·(-x(x-2)) + 2·(x(x-1))/2
     = (x²-3x+2)/2 - 3x² + 6x + x²/2 - x/2
     = -x² + 4.5x + 1

**Step 3**: Verify the result
- P(0) = 1 ✓
- P(1) = -1 + 4.5 + 1 = 3 ✓
- P(2) = -4 + 9 + 1 = 2 ✓

#### Python Implementation
```python
import numpy as np

def lagrange_interpolation(x_points, y_points, x):
    """
    Compute the value of the Lagrange interpolation polynomial at point x.
    
    Parameters:
    - x_points: Array of x coordinates of data points
    - y_points: Array of y coordinates of data points
    - x: Point at which to evaluate the polynomial
    
    Returns:
    - y: Interpolated value at x
    """
    n = len(x_points)
    y = 0.0
    
    for i in range(n):
        # Calculate Lagrange basis polynomial L_i(x)
        L_i = 1.0
        for j in range(n):
            if j != i:
                L_i *= (x - x_points[j]) / (x_points[i] - x_points[j])
        
        y += y_points[i] * L_i
    
    return y
```

### 2. Newton's Divided Differences

- **Form**: P(x) = a<sub>0</sub> + a<sub>1</sub>(x-x<sub>0</sub>) + a<sub>2</sub>(x-x<sub>0</sub>)(x-x<sub>1</sub>) + ...
- **Coefficients**: Calculated using divided differences table
- **Advantages**:
  - Easy to add new points without recalculating
  - More numerically stable than Lagrange
  - More efficient evaluation

#### Divided Differences Formula

The first divided difference is:
f[x<sub>i</sub>, x<sub>i+1</sub>] = (f(x<sub>i+1</sub>) - f(x<sub>i</sub>))/(x<sub>i+1</sub> - x<sub>i</sub>)

Higher-order divided differences are calculated recursively:
f[x<sub>i</sub>, ..., x<sub>i+k</sub>] = (f[x<sub>i+1</sub>, ..., x<sub>i+k</sub>] - f[x<sub>i</sub>, ..., x<sub>i+k-1</sub>])/(x<sub>i+k</sub> - x<sub>i</sub>)

#### Worked Example: Newton's Divided Differences

Using the same data points: (0, 1), (1, 3), (2, 2)

**Step 1**: Build the divided differences table

| x | f(x) | 1st differences | 2nd differences |
|---|------|----------------|-----------------|
| 0 | 1    |                |                 |
|   |      | (3-1)/(1-0) = 2|                 |
| 1 | 3    |                | ((2-3)/(2-1) - (3-1)/(1-0))/(2-0) = -1.5 |
|   |      | (2-3)/(2-1) = -1|                |
| 2 | 2    |                |                 |

**Step 2**: Form the interpolating polynomial using Newton's form
P(x) = f(x₀) + f[x₀,x₁](x-x₀) + f[x₀,x₁,x₂](x-x₀)(x-x₁)
     = 1 + 2(x-0) + (-1.5)(x-0)(x-1)
     = 1 + 2x - 1.5x(x-1)
     = 1 + 2x - 1.5x² + 1.5x
     = 1 + 3.5x - 1.5x²
     = -1.5x² + 3.5x + 1

This matches our result from Lagrange interpolation.

#### Python Implementation
```python
def newton_divided_differences(x_points, y_points):
    """
    Compute the coefficients of Newton's divided differences form.
    
    Parameters:
    - x_points: Array of x coordinates of data points
    - y_points: Array of y coordinates of data points
    
    Returns:
    - coef: Array of coefficients for Newton's interpolation formula
    """
    n = len(x_points)
    coef = np.copy(y_points)
    
    # Create the divided differences table
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coef[i] = (coef[i] - coef[i-1]) / (x_points[i] - x_points[i-j])
    
    return coef

def newton_interpolation(x_points, y_points, x):
    """
    Evaluate Newton's interpolation polynomial at point x.
    
    Parameters:
    - x_points: Array of x coordinates of data points
    - y_points: Array of y coordinates of data points
    - x: Point at which to evaluate the polynomial
    
    Returns:
    - y: Interpolated value at x
    """
    n = len(x_points)
    coef = newton_divided_differences(x_points, y_points)
    
    # Evaluate using Horner's rule
    result = coef[n-1]
    for i in range(n-2, -1, -1):
        result = result * (x - x_points[i]) + coef[i]
    
    return result
```

### 3. Hermite Interpolation

- **Concept**: Interpolates both function values and derivatives
- **Order**: Higher continuity at interpolation points
- **Applications**: Smooth curve generation, physical simulations

#### Hermite Interpolation Formula

For each point xᵢ, we need to match both f(xᵢ) and f'(xᵢ). For two points x₀ and x₁ with values f(x₀), f'(x₀), f(x₁), f'(x₁), the cubic Hermite interpolant is:

P(x) = h₀₀(t)f(x₀) + h₁₀(t)(x₁-x₀)f'(x₀) + h₀₁(t)f(x₁) + h₁₁(t)(x₁-x₀)f'(x₁)

where t = (x-x₀)/(x₁-x₀) and the basis functions are:
- h₀₀(t) = 2t³ - 3t² + 1
- h₁₀(t) = t³ - 2t² + t
- h₀₁(t) = -2t³ + 3t²
- h₁₁(t) = t³ - t²

#### Worked Example: Cubic Hermite Interpolation

Given two points:
- (x₀, f(x₀), f'(x₀)) = (0, 1, 2)
- (x₁, f(x₁), f'(x₁)) = (1, 3, -1)

For x = 0.5, t = (0.5-0)/(1-0) = 0.5

Calculate basis functions:
- h₀₀(0.5) = 2(0.5)³ - 3(0.5)² + 1 = 0.25 - 0.75 + 1 = 0.5
- h₁₀(0.5) = (0.5)³ - 2(0.5)² + 0.5 = 0.125 - 0.5 + 0.5 = 0.125
- h₀₁(0.5) = -2(0.5)³ + 3(0.5)² = -0.25 + 0.75 = 0.5
- h₁₁(0.5) = (0.5)³ - (0.5)² = 0.125 - 0.25 = -0.125

The interpolated value is:
P(0.5) = 0.5(1) + 0.125(1)(2) + 0.5(3) + (-0.125)(1)(-1)
       = 0.5 + 0.25 + 1.5 - 0.125
       = 2.125

## Piecewise Interpolation

### 1. Linear Interpolation

- **Form**: P(x) = y<sub>i</sub> + (y<sub>i+1</sub>-y<sub>i</sub>)/(x<sub>i+1</sub>-x<sub>i</sub>) × (x-x<sub>i</sub>)
- **Properties**:
  - C<sup>0</sup> continuity at knots
  - Simple and efficient
  - Limited accuracy

#### Linear Interpolation Formula

For x ∈ [x<sub>i</sub>, x<sub>i+1</sub>]:
P(x) = y<sub>i</sub> + (y<sub>i+1</sub> - y<sub>i</sub>)/(x<sub>i+1</sub> - x<sub>i</sub>) × (x - x<sub>i</sub>)

This can be rewritten as:
P(x) = (1-t)y<sub>i</sub> + ty<sub>i+1</sub>

where t = (x-x<sub>i</sub>)/(x<sub>i+1</sub>-x<sub>i</sub>)

#### Python Implementation
```python
def linear_interpolation(x_points, y_points, x):
    """
    Perform linear interpolation at point x.
    
    Parameters:
    - x_points: Array of x coordinates of data points
    - y_points: Array of y coordinates of data points
    - x: Point at which to evaluate
    
    Returns:
    - y: Interpolated value at x
    """
    # Find the interval containing x
    i = 0
    while i < len(x_points) - 1 and x > x_points[i+1]:
        i += 1
    
    # Check if x is within range
    if i >= len(x_points) - 1:
        return y_points[-1]  # Return last point if beyond range
    
    # Linear interpolation formula
    t = (x - x_points[i]) / (x_points[i+1] - x_points[i])
    return (1 - t) * y_points[i] + t * y_points[i+1]
```

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

#### Cubic Spline Conditions

For n+1 data points, we construct n cubic polynomials S₁(x), S₂(x), ..., Sₙ(x) where Sᵢ(x) interpolates the interval [x<sub>i-1</sub>, x<sub>i</sub>].

Each cubic polynomial has the form:
Sᵢ(x) = aᵢ + bᵢ(x-x<sub>i-1</sub>) + cᵢ(x-x<sub>i-1</sub>)² + dᵢ(x-x<sub>i-1</sub>)³

The conditions imposed are:
1. Sᵢ(x<sub>i-1</sub>) = y<sub>i-1</sub> and Sᵢ(x<sub>i</sub>) = y<sub>i</sub> (interpolation)
2. S'ᵢ(x<sub>i</sub>) = S'<sub>i+1</sub>(x<sub>i</sub>) (C¹ continuity)
3. S''ᵢ(x<sub>i</sub>) = S''<sub>i+1</sub>(x<sub>i</sub>) (C² continuity)
4. Plus two additional conditions (typically natural or clamped)

#### Worked Example: Natural Cubic Spline

Consider the data points: (0, 1), (1, 3), (2, 2)

With the natural spline condition: S''(x₀) = S''(xₙ) = 0

**Step 1**: Set up the tridiagonal system to solve for the second derivatives M₀, M₁, M₂

For n = 2, we have the tridiagonal system:
[h₀    | h₀/3   | 0     ] [M₀] = [d₁ - d₀]
[h₀/3  | 2(h₀+h₁)/3 | h₁/3 ] [M₁] = [d₂ - d₁]
[0     | h₁/3   | h₁    ] [M₂] = [0     ]

where:
- hᵢ = x<sub>i+1</sub> - x<sub>i</sub>
- dᵢ = (y<sub>i+1</sub> - y<sub>i</sub>)/hᵢ

Substituting our values:
- h₀ = 1 - 0 = 1, h₁ = 2 - 1 = 1
- d₀ = (3 - 1)/1 = 2, d₁ = (2 - 3)/1 = -1

For natural splines, M₀ = M₂ = 0, so we only need to solve for M₁:
(2(h₀+h₁)/3)M₁ = d₁ - d₀
(2(1+1)/3)M₁ = -1 - 2
(4/3)M₁ = -3
M₁ = -9/4 = -2.25

**Step 2**: Calculate the coefficients for each spline segment

For segment S₁ (between x₀ and x₁):
a₁ = y₀ = 1
b₁ = (y₁ - y₀)/h₀ - h₀(2M₀ + M₁)/6 = (3 - 1)/1 - 1(2(0) + (-2.25))/6 = 2 + 0.375 = 2.375
c₁ = M₀/2 = 0
d₁ = (M₁ - M₀)/(6h₀) = (-2.25 - 0)/(6(1)) = -0.375

For segment S₂ (between x₁ and x₂):
a₂ = y₁ = 3
b₂ = (y₂ - y₁)/h₁ - h₁(2M₁ + M₂)/6 = (2 - 3)/1 - 1(2(-2.25) + 0)/6 = -1 + 0.75 = -0.25
c₂ = M₁/2 = -2.25/2 = -1.125
d₂ = (M₂ - M₁)/(6h₁) = (0 - (-2.25))/(6(1)) = 0.375

**Step 3**: The cubic spline is:
S₁(x) = 1 + 2.375(x - 0) + 0(x - 0)² - 0.375(x - 0)³ for x ∈ [0, 1]
S₂(x) = 3 - 0.25(x - 1) - 1.125(x - 1)² + 0.375(x - 1)³ for x ∈ [1, 2]

#### Python Implementation
```python
def cubic_spline(x_points, y_points, x, boundary_type='natural'):
    """
    Compute cubic spline interpolation at point x.
    
    Parameters:
    - x_points: Array of x coordinates of data points
    - y_points: Array of y coordinates of data points
    - x: Point at which to evaluate
    - boundary_type: 'natural', 'clamped', or 'not-a-knot'
    
    Returns:
    - y: Interpolated value at x
    """
    n = len(x_points) - 1  # Number of intervals
    
    # Step 1: Compute interval lengths and divided differences
    h = [x_points[i+1] - x_points[i] for i in range(n)]
    d = [(y_points[i+1] - y_points[i]) / h[i] for i in range(n)]
    
    # Step 2: Set up tridiagonal system for second derivatives
    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)
    
    # Set up equations for interior points
    for i in range(1, n):
        A[i, i-1] = h[i-1] / 6
        A[i, i] = (h[i-1] + h[i]) / 3
        A[i, i+1] = h[i] / 6
        b[i] = d[i] - d[i-1]
    
    # Set up boundary conditions
    if boundary_type == 'natural':
        A[0, 0] = 1
        A[n, n] = 1
    elif boundary_type == 'clamped':
        # Would need first derivative values at endpoints
        pass
    
    # Step 3: Solve for second derivatives
    M = np.linalg.solve(A, b)
    
    # Step 4: Find interval containing x
    i = 0
    while i < n and x > x_points[i+1]:
        i += 1
    
    # Step 5: Compute spline value
    t = (x - x_points[i]) / h[i]
    a = y_points[i]
    b = d[i] - h[i] * (2 * M[i] + M[i+1]) / 6
    c = M[i] / 2
    d_coef = (M[i+1] - M[i]) / (6 * h[i])
    
    return a + b * (x - x_points[i]) + c * (x - x_points[i])**2 + d_coef * (x - x_points[i])**3
```

### 3. Cubic Hermite Splines

- **Form**: Local hermite interpolation between adjacent points
- **Parametrization**: Often uses normalized parameter t ∈ [0,1]
- **Basis Functions**: Hermite basis functions H<sub>i</sub>(t)
- **Applications**: Path interpolation, animation

#### Piecewise Cubic Hermite Interpolation

For data points (x₀, y₀, m₀), (x₁, y₁, m₁), ..., (xₙ, yₙ, mₙ) where m<sub>i</sub> are the specified derivatives, the cubic Hermite spline over [x<sub>i</sub>, x<sub>i+1</sub>] is:

P(x) = H₀₀(t)y<sub>i</sub> + H₁₀(t)h<sub>i</sub>m<sub>i</sub> + H₀₁(t)y<sub>i+1</sub> + H₁₁(t)h<sub>i</sub>m<sub>i+1</sub>

where t = (x-x<sub>i</sub>)/h<sub>i</sub> and h<sub>i</sub> = x<sub>i+1</sub> - x<sub>i</sub>

#### Estimating Derivatives for Cubic Hermite Splines

If derivatives are not provided, we can estimate them using the centered finite difference:

m<sub>i</sub> = (y<sub>i+1</sub> - y<sub>i-1</sub>)/(x<sub>i+1</sub> - x<sub>i-1</sub>)

For endpoints, we use one-sided differences:
m₀ = (y₁ - y₀)/(x₁ - x₀)
mₙ = (yₙ - y<sub>n-1</sub>)/(xₙ - x<sub>n-1</sub>)

This creates what's known as a "Cardinal spline".

## Multivariate Interpolation

### 1. Bilinear Interpolation

- **Concept**: Extension of linear interpolation to 2D grid
- **Process**: Interpolate along rows, then along resulting column
- **Applications**: Image scaling, texture mapping

#### Bilinear Interpolation Formula

For a point (x, y) in the rectangle with corners (x₁, y₁), (x₂, y₁), (x₁, y₂), (x₂, y₂) and function values f₁₁, f₂₁, f₁₂, f₂₂:

1. Interpolate along the top edge: f(x, y₁) = f₁₁ + (x - x₁)/(x₂ - x₁) × (f₂₁ - f₁₁)
2. Interpolate along the bottom edge: f(x, y₂) = f₁₂ + (x - x₁)/(x₂ - x₁) × (f₂₂ - f₁₂)
3. Interpolate between these values: f(x, y) = f(x, y₁) + (y - y₁)/(y₂ - y₁) × (f(x, y₂) - f(x, y₁))

#### Worked Example: Bilinear Interpolation

Given values at four corners of a unit square:
- f(0, 0) = 10
- f(1, 0) = 20
- f(0, 1) = 15
- f(1, 1) = 25

Find the interpolated value at (0.6, 0.4):

Step 1: Interpolate along x at y = 0
f(0.6, 0) = 10 + 0.6 × (20 - 10) = 10 + 6 = 16

Step 2: Interpolate along x at y = 1
f(0.6, 1) = 15 + 0.6 × (25 - 15) = 15 + 6 = 21

Step 3: Interpolate along y at x = 0.6
f(0.6, 0.4) = 16 + 0.4 × (21 - 16) = 16 + 2 = 18

#### Python Implementation
```python
def bilinear_interpolation(x, y, points):
    """
    Perform bilinear interpolation at point (x, y).
    
    Parameters:
    - x, y: Coordinates at which to interpolate
    - points: Dictionary with (x,y) tuples as keys and function values as values
    
    Returns:
    - Interpolated value
    """
    # Extract corners of rectangle containing (x, y)
    x1, y1 = math.floor(x), math.floor(y)
    x2, y2 = math.ceil(x), math.ceil(y)
    
    # Handle case where point is exactly on a grid point
    if x == x1 and y == y1:
        return points[(x1, y1)]
    
    # Extract function values at corners
    f11 = points[(x1, y1)]
    f21 = points[(x2, y1)]
    f12 = points[(x1, y2)]
    f22 = points[(x2, y2)]
    
    # Normalize coordinates to [0,1]
    x_ratio = (x - x1) / (x2 - x1) if x2 != x1 else 0
    y_ratio = (y - y1) / (y2 - y1) if y2 != y1 else 0
    
    # Bilinear interpolation formula
    result = (1 - x_ratio) * (1 - y_ratio) * f11 + \
             x_ratio * (1 - y_ratio) * f21 + \
             (1 - x_ratio) * y_ratio * f12 + \
             x_ratio * y_ratio * f22
    
    return result
```

### 2. Bicubic Interpolation

- **Concept**: Extension of cubic interpolation to 2D
- **Data Required**: Function values, partial derivatives, and cross derivatives
- **Applications**: Image processing, terrain modeling

#### Bicubic Interpolation Formula

Bicubic interpolation requires 16 coefficients, determined by:
- Function values f(x, y) at the four corners
- Partial derivatives ∂f/∂x and ∂f/∂y at the four corners
- Cross derivatives ∂²f/∂x∂y at the four corners

The interpolating function has the form:
f(x, y) = Σ<sub>i=0</sub><sup>3</sup> Σ<sub>j=0</sub><sup>3</sup> a<sub>ij</sub> x<sup>i</sup> y<sup>j</sup>

### 3. Radial Basis Functions

- **Form**: s(x) = Σ<sub>i=1</sub><sup>n</sup> λ<sub>i</sub> φ(||x-x<sub>i</sub>||)
- **Common RBFs**: Gaussian, Multiquadric, Thin-plate spline
- **Applications**: Scattered data interpolation, mesh deformation

#### Common Radial Basis Functions

1. **Gaussian**: φ(r) = exp(-r²/σ²)
2. **Multiquadric**: φ(r) = √(r² + σ²)
3. **Inverse Multiquadric**: φ(r) = 1/√(r² + σ²)
4. **Thin-plate spline**: φ(r) = r² log(r)

Where r = ||x-x<sub>i</sub>|| is the distance from the evaluation point to the data point, and σ is a shape parameter.

#### Example: RBF Interpolation Process

For data points (x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ) with values f₁, f₂, ..., fₙ:

1. Choose a radial basis function φ(r)
2. Set up the system of equations: Σ<sub>j=1</sub><sup>n</sup> λ<sub>j</sub> φ(||x<sub>i</sub>-x<sub>j</sub>||) = f<sub>i</sub> for i = 1, 2, ..., n
3. Solve for the weights λ<sub>j</sub>
4. Use the weights to evaluate the interpolant at any point: s(x) = Σ<sub>j=1</sub><sup>n</sup> λ<sub>j</sub> φ(||x-x<sub>j</sub>||)

## Error Analysis

- **Error Bound**: |f(x) - P(x)| ≤ (M/(n+1)!) × Π<sub>i=0</sub><sup>n</sup> |x-x<sub>i</sub>|
  where M is bound on (n+1)th derivative
- **Runge Phenomenon**: High oscillations near edges with equidistant points
- **Chebyshev Nodes**: Optimal node placement to minimize error

### Error Formula Derivation

For an interpolating polynomial P(x) of degree n, the error at point x is:

E(x) = f(x) - P(x) = (f<sup>(n+1)</sup>(ξ)/(n+1)!) × Π<sub>i=0</sub><sup>n</sup> (x-x<sub>i</sub>)

for some ξ in the range of the data points. If |f<sup>(n+1)</sup>(x)| ≤ M, then:

|E(x)| ≤ (M/(n+1)!) × Π<sub>i=0</sub><sup>n</sup> |x-x<sub>i</sub>|

### Runge Phenomenon

When using high-degree polynomials with equally spaced points, severe oscillations can occur near the endpoints of the interval. This is known as the Runge phenomenon.

For example, interpolating f(x) = 1/(1+25x²) over [-1, 1] with equally spaced points can lead to errors that grow exponentially with the number of points.

### Chebyshev Nodes

To mitigate the Runge phenomenon, we can use Chebyshev nodes:

x<sub>i</sub> = cos((2i+1)π/(2n+2)) for i = 0, 1, ..., n

These nodes are concentrated at the endpoints and minimize the maximum interpolation error.

#### Error Comparison: Equally Spaced vs. Chebyshev Nodes

When interpolating f(x) = 1/(1+25x²) over [-1, 1] with 10 points:
- Maximum error with equally spaced points: ~10²
- Maximum error with Chebyshev nodes: ~10⁻¹

## Exam Focus Areas

1. **Basis Construction**: Deriving basis functions for different methods
2. **Error Analysis**: Understanding and calculating interpolation error
3. **Algorithm Implementation**: Steps for constructing and evaluating interpolants
4. **Method Selection**: Choosing appropriate interpolation methods
5. **Continuity Analysis**: Determining and ensuring continuity properties

### Sample Exam Questions

1. Derive the Lagrange basis function for a specific data point.
2. Calculate the divided differences table for a set of data points.
3. Determine the error bound for a polynomial interpolation.
4. Construct a cubic spline with specific boundary conditions.
5. Compare the accuracy of different interpolation methods for a given function.

## Practice Problems

1. Construct a Lagrange polynomial for a given dataset
2. Set up and solve the tridiagonal system for natural cubic splines
3. Compare errors between different interpolation methods
4. Implement bilinear interpolation for a 2D grid of values

### Challenge Problem

A rocket's position (in meters) is measured at three time points:
- t = 0s: x = 0m
- t = 1s: x = 10m
- t = 3s: x = 90m

1. Find the Lagrange polynomial that interpolates this data
2. Estimate the position at t = 2s
3. Calculate the velocity (first derivative) at t = 1.5s
4. Discuss whether this interpolation is physically realistic and why

**Solution**:
1. The Lagrange polynomial is P(t) = 10t² + 0t + 0 = 10t²
2. At t = 2s: P(2) = 10(2)² = 40m
3. Velocity at t = 1.5s: P'(t) = 20t, so P'(1.5) = 20(1.5) = 30m/s
4. This interpolation suggests constant acceleration, which is physically plausible for a rocket during a phase of constant thrust.

Original lecture notes are available at: `/files/CE7453/CE7453/05-interpolation-4slides1page(1).pdf` 