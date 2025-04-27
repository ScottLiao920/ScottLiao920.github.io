---
title: 'CE7453: Numerical Integration'
date: 2024-06-22
permalink: /posts/2024/06/ce7453-numerical-integration/
tags:
  - CE7453
  - numerical integration
  - quadrature
  - exam preparation
---

This post summarizes key concepts of numerical integration covered in CE7453, based on the lecture notes "06-numerical integration".

## Introduction to Numerical Integration

Numerical integration, also known as quadrature, is the process of approximating the definite integral of a function when analytical integration is difficult or impossible.

### Mathematical Definition

Given a function f(x), numerical integration approximates:
$\int_{a}^{b} f(x) dx \approx \sum_{i=0}^{n} w_i f(x_i)$

where $x_i$ are evaluation points (nodes), and $w_i$ are weights.

### Real-world Applications

- **Physics**: Calculating work, energy, momentum
- **Engineering**: Stress analysis, fluid dynamics
- **Finance**: Option pricing, risk assessment
- **Statistics**: Probability calculations, expected values
- **Computer Graphics**: Rendering, global illumination

## Newton-Cotes Formulas

Newton-Cotes formulas approximate integrals using polynomial interpolation over equally spaced points.

### 1. Trapezoidal Rule

- **Formula**: $\int_{a}^{b} f(x) dx \approx \frac{b-a}{2} [f(a) + f(b)]$
- **Order of Accuracy**: O(h²)
- **Error Term**: $-\frac{(b-a)^3}{12} f''(\xi)$, for some $\xi \in [a,b]$

#### Derivation of the Trapezoidal Rule

The trapezoidal rule approximates the function f(x) with a linear function between x = a and x = b. The integral is then the area of the trapezoid:

$\int_{a}^{b} f(x) dx \approx \int_{a}^{b} \left( f(a) + \frac{f(b) - f(a)}{b-a}(x-a) \right) dx$

Evaluating this integral:
$\int_{a}^{b} \left( f(a) + \frac{f(b) - f(a)}{b-a}(x-a) \right) dx = f(a)(b-a) + \frac{f(b) - f(a)}{b-a} \frac{(b-a)^2}{2} = \frac{b-a}{2}[f(a) + f(b)]$

#### Worked Example: Trapezoidal Rule

Calculate $\int_{0}^{1} x^2 dx$ using the trapezoidal rule.

**Solution**:
$\int_{0}^{1} x^2 dx \approx \frac{1-0}{2} [f(0) + f(1)] = \frac{1}{2} [0 + 1] = 0.5$

The exact value is $\int_{0}^{1} x^2 dx = \left[ \frac{x^3}{3} \right]_{0}^{1} = \frac{1}{3}$

Error = $0.5 - \frac{1}{3} = \frac{1}{6} \approx 0.167$

#### Composite Trapezoidal Rule

For better accuracy, divide [a,b] into n subintervals of width h = (b-a)/n:

$\int_{a}^{b} f(x) dx \approx \frac{h}{2} \left[ f(a) + f(b) + 2\sum_{i=1}^{n-1} f(a+ih) \right]$

#### Python Implementation
```python
import numpy as np

def trapezoidal_rule(f, a, b, n):
    """
    Approximate the integral of f(x) from a to b using the composite trapezoidal rule.
    
    Parameters:
    - f: Function to integrate
    - a, b: Integration limits
    - n: Number of subintervals
    
    Returns:
    - Approximate value of the integral
    """
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    
    return h * (0.5 * y[0] + 0.5 * y[n] + np.sum(y[1:n]))
```

### 2. Simpson's Rules

#### Simpson's 1/3 Rule

- **Formula**: $\int_{a}^{b} f(x) dx \approx \frac{b-a}{6} [f(a) + 4f(\frac{a+b}{2}) + f(b)]$
- **Order of Accuracy**: O(h⁴)
- **Error Term**: $-\frac{(b-a)^5}{2880} f^{(4)}(\xi)$, for some $\xi \in [a,b]$

#### Derivation of Simpson's 1/3 Rule

Simpson's 1/3 rule approximates the function with a quadratic polynomial. If we place nodes at a, (a+b)/2, and b, the Lagrange interpolating polynomial is:

$P(x) = \frac{(x-m)(x-b)}{(a-m)(a-b)}f(a) + \frac{(x-a)(x-b)}{(m-a)(m-b)}f(m) + \frac{(x-a)(x-m)}{(b-a)(b-m)}f(b)$

where m = (a+b)/2. Integrating this polynomial and simplifying leads to Simpson's 1/3 rule.

#### Worked Example: Simpson's 1/3 Rule

Calculate $\int_{0}^{1} x^2 dx$ using Simpson's 1/3 rule.

**Solution**:
With a = 0, b = 1, and m = 0.5:
$\int_{0}^{1} x^2 dx \approx \frac{1-0}{6} [f(0) + 4f(0.5) + f(1)] = \frac{1}{6} [0 + 4(0.25) + 1] = \frac{1}{6} [0 + 1 + 1] = \frac{1}{3}$

The exact value is $\int_{0}^{1} x^2 dx = \frac{1}{3}$, so in this case Simpson's rule gives the exact answer because our integrand is a quadratic polynomial.

#### Composite Simpson's 1/3 Rule

For n subintervals (n must be even):
$\int_{a}^{b} f(x) dx \approx \frac{h}{3} \left[ f(a) + f(b) + 4\sum_{i=1,3,5,...}^{n-1} f(a+ih) + 2\sum_{i=2,4,6,...}^{n-2} f(a+ih) \right]$

where h = (b-a)/n.

#### Python Implementation
```python
def simpsons_rule(f, a, b, n):
    """
    Approximate the integral of f(x) from a to b using the composite Simpson's 1/3 rule.
    
    Parameters:
    - f: Function to integrate
    - a, b: Integration limits
    - n: Number of subintervals (must be even)
    
    Returns:
    - Approximate value of the integral
    """
    if n % 2 != 0:
        raise ValueError("Number of subintervals must be even")
    
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    
    result = y[0] + y[n]
    result += 4 * np.sum(y[1:n:2])  # Sum of odd indices
    result += 2 * np.sum(y[2:n-1:2])  # Sum of even indices
    
    return h * result / 3
```

#### Simpson's 3/8 Rule

- **Formula**: $\int_{a}^{b} f(x) dx \approx \frac{b-a}{8} [f(a) + 3f(a + \frac{b-a}{3}) + 3f(a + \frac{2(b-a)}{3}) + f(b)]$
- **Order of Accuracy**: O(h⁴)
- **Applications**: Often used when Simpson's 1/3 rule doesn't apply well

### 3. Higher Order Newton-Cotes Formulas

- **Definition**: Extensions with more points for higher accuracy
- **Limitation**: Can be unstable for large numbers of points
- **Typical Use**: Usually limited to n ≤ 8 due to Runge phenomenon

### 4. Error Analysis of Newton-Cotes Formulas

- **Trapezoidal Rule**: Error ~ O(h²)
- **Simpson's Rule**: Error ~ O(h⁴)
- **General Case**: For a Newton-Cotes formula with n+1 points, if n is odd, the error is O(h^(n+3)), and if n is even, the error is O(h^(n+2))

## Gaussian Quadrature

Gaussian quadrature allows us to choose both the weights and evaluation points optimally.

### 1. Legendre-Gauss Quadrature

- **Formula**: $\int_{-1}^{1} f(x) dx \approx \sum_{i=1}^{n} w_i f(x_i)$
- **Nodes**: Roots of Legendre polynomials P_n(x)
- **Weights**: $w_i = \frac{2}{(1-x_i^2)[P_n'(x_i)]^2}$
- **Accuracy**: Exact for polynomials of degree ≤ 2n-1

#### Legendre Polynomials

Legendre polynomials P_n(x) are defined by the recurrence relation:
- P₀(x) = 1
- P₁(x) = x
- (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)

The first few Legendre polynomials are:
- P₀(x) = 1
- P₁(x) = x
- P₂(x) = (3x² - 1)/2
- P₃(x) = (5x³ - 3x)/2

#### Nodes and Weights for n = 2, 3

For n = 2:
- Nodes: x₁ = -1/√3 ≈ -0.577, x₂ = 1/√3 ≈ 0.577
- Weights: w₁ = w₂ = 1

For n = 3:
- Nodes: x₁ = -√(3/5) ≈ -0.775, x₂ = 0, x₃ = √(3/5) ≈ 0.775
- Weights: w₁ = w₃ = 5/9, w₂ = 8/9

#### Worked Example: Gauss Quadrature with n = 2

Calculate $\int_{-1}^{1} x^3 dx$ using Gauss quadrature with n = 2.

**Solution**:
For n = 2, the nodes are x₁ = -1/√3 and x₂ = 1/√3, with weights w₁ = w₂ = 1.

$\int_{-1}^{1} x^3 dx \approx w_1 f(x_1) + w_2 f(x_2) = 1 \cdot (-1/\sqrt{3})^3 + 1 \cdot (1/\sqrt{3})^3 = -1/3\sqrt{3} + 1/3\sqrt{3} = 0$

The exact value is $\int_{-1}^{1} x^3 dx = \left[ \frac{x^4}{4} \right]_{-1}^{1} = \frac{1}{4} - \frac{1}{4} = 0$, so Gauss quadrature gives the exact result.

#### Python Implementation
```python
def gauss_legendre(f, a, b, n):
    """
    Approximate the integral of f(x) from a to b using Gauss-Legendre quadrature with n points.
    
    Parameters:
    - f: Function to integrate
    - a, b: Integration limits
    - n: Number of quadrature points
    
    Returns:
    - Approximate value of the integral
    """
    # Get Gauss-Legendre nodes and weights for [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n)
    
    # Transform from [-1, 1] to [a, b]
    x_transformed = 0.5 * (b - a) * x + 0.5 * (b + a)
    
    # Compute the integral
    result = 0.5 * (b - a) * np.sum(w * f(x_transformed))
    
    return result
```

### 2. Gauss-Chebyshev Quadrature

- **Formula**: $\int_{-1}^{1} \frac{f(x)}{\sqrt{1-x^2}} dx \approx \frac{\pi}{n} \sum_{i=1}^{n} f(x_i)$
- **Nodes**: x_i = cos((2i-1)π/(2n)), i = 1, 2, ..., n
- **Weights**: All weights are π/n
- **Applications**: Integrals with weight function 1/√(1-x²)

### 3. Gauss-Laguerre Quadrature

- **Formula**: $\int_{0}^{\infty} e^{-x} f(x) dx \approx \sum_{i=1}^{n} w_i f(x_i)$
- **Applications**: Infinite intervals with exponential decay

### 4. Gauss-Hermite Quadrature

- **Formula**: $\int_{-\infty}^{\infty} e^{-x^2} f(x) dx \approx \sum_{i=1}^{n} w_i f(x_i)$
- **Applications**: Infinite intervals with Gaussian weight

### 5. General Gauss Quadrature

- **Form**: $\int_{a}^{b} w(x) f(x) dx \approx \sum_{i=1}^{n} w_i f(x_i)$
- **Nodes**: Roots of orthogonal polynomials w.r.t. weight function w(x)
- **Applications**: Integrals with specific weight functions

## Adaptive Quadrature

Adaptive quadrature adjusts the integration method based on the function's behavior.

### 1. Adaptive Simpson's Rule

- **Approach**: Recursively subdivide intervals until desired accuracy
- **Error Estimation**: Compare with lower-order method
- **Efficiency**: Concentrates effort where function varies rapidly

#### Adaptive Simpson's Algorithm

1. Apply Simpson's rule to the entire interval [a,b] to get I₁
2. Divide the interval in half and apply Simpson's rule to [a,(a+b)/2] and [(a+b)/2,b] to get I₂
3. If |I₁ - I₂| < tolerance, accept I₂
4. Otherwise, recursively apply steps 1-3 to each subinterval

#### Python Implementation
```python
def adaptive_simpson(f, a, b, tol=1e-6, max_depth=20):
    """
    Adaptively integrate f from a to b using Simpson's rule.
    
    Parameters:
    - f: Function to integrate
    - a, b: Integration limits
    - tol: Error tolerance
    - max_depth: Maximum recursion depth
    
    Returns:
    - Approximate value of the integral
    """
    def simpson(a, b):
        c = (a + b) / 2
        h = b - a
        return h * (f(a) + 4 * f(c) + f(b)) / 6
    
    def adaptive_simpson_recursive(a, b, tol, depth):
        c = (a + b) / 2
        left = simpson(a, c)
        right = simpson(c, b)
        whole = simpson(a, b)
        
        if depth >= max_depth:
            return left + right
        
        if abs(left + right - whole) < 15 * tol:
            return left + right
        else:
            return (adaptive_simpson_recursive(a, c, tol/2, depth+1) + 
                   adaptive_simpson_recursive(c, b, tol/2, depth+1))
    
    return adaptive_simpson_recursive(a, b, tol, 0)
```

### 2. Adaptive Gauss-Kronrod Quadrature

- **Approach**: Combines Gaussian quadrature with Kronrod extension
- **Advantage**: Efficient error estimation without extra function evaluations
- **Applications**: Standard in many numerical libraries

### 3. Error Estimation Techniques

- **Richardson Extrapolation**: Improve accuracy by combining results at different step sizes
- **Embedded Rules**: Compare results from methods of different orders
- **Statistical Error**: Monte Carlo methods with variance estimation

## Special Integration Techniques

### 1. Improper Integrals

- **Types**:
  - Infinite intervals: $\int_{a}^{\infty} f(x) dx$ or $\int_{-\infty}^{b} f(x) dx$
  - Discontinuous integrands: $\int_{a}^{b} f(x) dx$ where f is singular at some point in [a,b]
- **Approach**: Transform to proper integral or use special quadrature

#### Handling Infinite Intervals

For integrals of the form $\int_{a}^{\infty} f(x) dx$:
1. Choose a large value M and compute $\int_{a}^{M} f(x) dx$ using standard methods
2. Use variable transformation x = 1/t to transform to a proper integral
3. Use specialized quadrature like Gauss-Laguerre

#### Example: Computing $\int_{0}^{\infty} e^{-x} dx$

**Method 1**: Truncate at a large value M
$\int_{0}^{\infty} e^{-x} dx \approx \int_{0}^{M} e^{-x} dx = 1 - e^{-M}$

For M = 10, this gives 1 - e^(-10) ≈ 0.9999546

**Method 2**: Use Gauss-Laguerre quadrature
$\int_{0}^{\infty} e^{-x} \cdot 1 \cdot dx$ can be computed exactly even with n = 1

The exact result is 1.

### 2. Multidimensional Integration

- **Tensor Product**: Extend 1D rules to multiple dimensions
- **Monte Carlo Method**: Statistical sampling for high dimensions
- **Sparse Grids**: Reduce curse of dimensionality

#### Double Integral Example

To approximate $\int_{a}^{b} \int_{c}^{d} f(x,y) dy dx$:

Using tensor product of trapezoidal rule:
$\int_{a}^{b} \int_{c}^{d} f(x,y) dy dx \approx \frac{(b-a)(d-c)}{4} [f(a,c) + f(a,d) + f(b,c) + f(b,d)]$

#### Python Implementation for 2D Integration
```python
def double_integral_trapezoidal(f, a, b, c, d, nx, ny):
    """
    Compute the double integral of f(x,y) over [a,b]×[c,d].
    
    Parameters:
    - f: Function of two variables to integrate
    - a, b: x-limits
    - c, d: y-limits
    - nx, ny: Number of subintervals in x and y directions
    
    Returns:
    - Approximate value of the double integral
    """
    hx = (b - a) / nx
    hy = (d - c) / ny
    
    x = np.linspace(a, b, nx+1)
    y = np.linspace(c, d, ny+1)
    
    result = 0.0
    
    # Corner points
    result += f(a, c) + f(a, d) + f(b, c) + f(b, d)
    
    # Edge points
    for i in range(1, nx):
        result += 2 * (f(x[i], c) + f(x[i], d))
    
    for j in range(1, ny):
        result += 2 * (f(a, y[j]) + f(b, y[j]))
    
    # Interior points
    for i in range(1, nx):
        for j in range(1, ny):
            result += 4 * f(x[i], y[j])
    
    return result * hx * hy / 4
```

### 3. Monte Carlo Integration

- **Formula**: $\int_{a}^{b} f(x) dx \approx (b-a) \frac{1}{N} \sum_{i=1}^{N} f(x_i)$, where x_i are random points
- **Error**: Proportional to 1/√N, regardless of dimensionality
- **Applications**: High-dimensional integrals, complex regions

#### Monte Carlo Integration Example

To approximate $\int_{0}^{1} \sin(\pi x) dx$ using N = 1000 random points:

1. Generate 1000 random points x₁, x₂, ..., x₁₀₀₀ in [0,1]
2. Compute the average of sin(πx_i)
3. Multiply by (b-a) = 1

The exact result is $\int_{0}^{1} \sin(\pi x) dx = 2/\pi \approx 0.6366$

#### Python Implementation
```python
def monte_carlo_integration(f, a, b, n_samples):
    """
    Monte Carlo integration of f(x) over [a, b].
    
    Parameters:
    - f: Function to integrate
    - a, b: Integration limits
    - n_samples: Number of random samples
    
    Returns:
    - Approximate value of the integral
    - Estimated error (standard deviation)
    """
    # Generate random points
    x = np.random.uniform(a, b, n_samples)
    
    # Evaluate function at random points
    y = f(x)
    
    # Compute mean and standard deviation
    mean = np.mean(y)
    std_dev = np.std(y) / np.sqrt(n_samples)
    
    # Scale by interval width
    integral = (b - a) * mean
    error = (b - a) * std_dev
    
    return integral, error
```

## Numerical Integration of ODEs

Numerical integration is also used to solve ordinary differential equations (ODEs).

### 1. Initial Value Problems

- **Form**: $\frac{dy}{dt} = f(t, y)$, $y(t_0) = y_0$
- **Goal**: Find y(t) for t > t₀

### 2. Euler's Method

- **Formula**: $y_{n+1} = y_n + h f(t_n, y_n)$
- **Order**: O(h)
- **Stability**: Conditionally stable

#### Euler's Method Derivation

Starting with the Taylor series expansion:
$y(t+h) = y(t) + h y'(t) + \frac{h^2}{2} y''(t) + \dots$

Truncating after the first-order term and using $y'(t) = f(t, y(t))$:
$y(t+h) \approx y(t) + h f(t, y(t))$

which gives Euler's method.

#### Worked Example: Euler's Method

Solve $\frac{dy}{dt} = y$, $y(0) = 1$ for t ∈ [0, 1] using h = 0.1.

**Solution**:
Starting with y₀ = 1 at t₀ = 0:

y₁ = y₀ + h·f(t₀, y₀) = 1 + 0.1·(1) = 1.1
y₂ = y₁ + h·f(t₁, y₁) = 1.1 + 0.1·(1.1) = 1.21
y₃ = y₂ + h·f(t₂, y₂) = 1.21 + 0.1·(1.21) = 1.331
...

After 10 steps, we get y₁₀ ≈ 2.5937.

The exact solution is y(t) = e^t, so y(1) = e ≈ 2.7183.
Error ≈ 0.1246 or about 4.6%.

#### Python Implementation
```python
def euler_method(f, t0, y0, t_end, h):
    """
    Solve an initial value problem using Euler's method.
    
    Parameters:
    - f: Function of the form f(t, y)
    - t0, y0: Initial conditions
    - t_end: End time
    - h: Step size
    
    Returns:
    - Arrays of time points and solution values
    """
    n_steps = int((t_end - t0) / h)
    t = np.zeros(n_steps + 1)
    y = np.zeros(n_steps + 1)
    
    t[0] = t0
    y[0] = y0
    
    for i in range(n_steps):
        t[i+1] = t[i] + h
        y[i+1] = y[i] + h * f(t[i], y[i])
    
    return t, y
```

### 3. Runge-Kutta Methods

#### Fourth-Order Runge-Kutta (RK4)

- **Formula**:
  - $k_1 = f(t_n, y_n)$
  - $k_2 = f(t_n + \frac{h}{2}, y_n + \frac{h}{2}k_1)$
  - $k_3 = f(t_n + \frac{h}{2}, y_n + \frac{h}{2}k_2)$
  - $k_4 = f(t_n + h, y_n + hk_3)$
  - $y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$
- **Order**: O(h⁴)
- **Stability**: Larger stability region than Euler's method

#### Worked Example: RK4

Solve $\frac{dy}{dt} = y$, $y(0) = 1$ for t = 1 using a single step of RK4 with h = 1.

**Solution**:
k₁ = f(t₀, y₀) = f(0, 1) = 1
k₂ = f(t₀ + h/2, y₀ + (h/2)k₁) = f(0.5, 1 + 0.5·1) = f(0.5, 1.5) = 1.5
k₃ = f(t₀ + h/2, y₀ + (h/2)k₂) = f(0.5, 1 + 0.5·1.5) = f(0.5, 1.75) = 1.75
k₄ = f(t₀ + h, y₀ + h·k₃) = f(1, 1 + 1·1.75) = f(1, 2.75) = 2.75

y₁ = y₀ + (h/6)(k₁ + 2k₂ + 2k₃ + k₄) = 1 + (1/6)(1 + 2·1.5 + 2·1.75 + 2.75) = 1 + (1/6)(10.25) ≈ 2.7083

Compared to the exact solution y(1) = e ≈ 2.7183, the error is only about 0.01, which is much better than Euler's method.

#### Python Implementation
```python
def rk4_method(f, t0, y0, t_end, h):
    """
    Solve an initial value problem using the fourth-order Runge-Kutta method.
    
    Parameters:
    - f: Function of the form f(t, y)
    - t0, y0: Initial conditions
    - t_end: End time
    - h: Step size
    
    Returns:
    - Arrays of time points and solution values
    """
    n_steps = int((t_end - t0) / h)
    t = np.zeros(n_steps + 1)
    y = np.zeros(n_steps + 1)
    
    t[0] = t0
    y[0] = y0
    
    for i in range(n_steps):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + 0.5*h, y[i] + 0.5*h*k1)
        k3 = f(t[i] + 0.5*h, y[i] + 0.5*h*k2)
        k4 = f(t[i] + h, y[i] + h*k3)
        
        t[i+1] = t[i] + h
        y[i+1] = y[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, y
```

### 4. Multi-step Methods

- **Adams-Bashforth**: Explicit method using previous steps
- **Adams-Moulton**: Implicit method requiring solving for y_{n+1}
- **Predictor-Corrector**: Combines explicit and implicit methods

### 5. Adaptive Step Size Control

- **Approach**: Adjust h based on local error estimate
- **Implementation**: Compare solutions with different step sizes
- **Efficiency**: Small steps where needed, large steps where possible

#### Adaptive RK45 (Dormand-Prince) Algorithm

1. Take a step with RK4 to get y₁
2. Take a step with RK5 to get y₁*
3. Estimate error as |y₁ - y₁*|
4. If error > tolerance, reduce h and repeat
5. If error < tolerance, accept step and possibly increase h

### 6. Stability Analysis

- **Stability Region**: Set of hλ values for which the method is stable
- **Absolute Stability**: Method remains bounded for Re(λ) < 0
- **Stiff Problems**: Problems requiring very small step sizes for stability

## Exam Focus Areas

1. **Method Selection**: Choosing appropriate integration methods
2. **Error Analysis**: Understanding and calculating integration errors
3. **Algorithm Implementation**: Steps for implementing integration methods
4. **Stability Analysis**: Understanding stability regions and conditions
5. **Adaptive Techniques**: Understanding when and how to apply adaptive methods

### Sample Exam Questions

1. Derive the error term for the trapezoidal rule.
2. Compare the accuracy of different quadrature methods for a specific function.
3. Implement an adaptive integration algorithm and analyze its performance.
4. Analyze the stability of Euler's method for a given differential equation.
5. Apply Gaussian quadrature to evaluate a challenging integral.

## Practice Problems

1. Calculate $\int_{0}^{\pi} \sin(x) dx$ using:
   - Trapezoidal rule with n = 4
   - Simpson's rule with n = 4
   - Gauss-Legendre quadrature with n = 2
   Compare with the exact value of 2.

2. Solve the initial value problem $\frac{dy}{dt} = -y$, $y(0) = 1$ on [0, 1] using:
   - Euler's method with h = 0.1
   - RK4 with h = 0.1
   Compare with the exact solution y(t) = e^(-t).

3. Calculate $\int_{0}^{1} \frac{1}{1+x^2} dx$ using:
   - Simpson's rule with n = 10
   - Adaptive Simpson's rule with tolerance 10^(-6)
   - Monte Carlo integration with 10,000 samples
   Compare with the exact value of π/4 ≈ 0.7854.

### Challenge Problem

Consider the integral:
$I = \int_{0}^{1} \frac{\sin(10x)}{1+x^2} dx$

1. Calculate this integral using different methods and compare their accuracy:
   - Composite trapezoidal rule with n = 20
   - Composite Simpson's rule with n = 20
   - Adaptive quadrature with tolerance 10^(-6)
   - Gauss-Legendre quadrature with n = 10

2. Analyze the number of function evaluations required by each method.

3. Discuss which method is most efficient for this particular integral and why.

**Solution Approach**:
The function sin(10x)/(1+x²) oscillates rapidly, making it challenging for uniform-grid methods. Adaptive methods or higher-order Gaussian quadrature are likely to perform better. The exact value can be computed with high precision using adaptive methods as a reference.

Original lecture notes are available at: `/files/CE7453/CE7453/06-numerical_integration-4slides1page(1).pdf` 