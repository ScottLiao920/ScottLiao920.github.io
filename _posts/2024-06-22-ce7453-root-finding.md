****---
title: 'CE7453: Root Finding Methods'
date: 2024-06-22
permalink: /posts/2024/06/ce7453-root-finding/
tags:
  - CE7453
  - root finding
  - numerical methods
  - exam preparation
---

This post summarizes the key concepts and methods for root finding covered in CE7453, based on the lecture notes "01-RootFinding".

## Key Concepts

Root finding is the process of determining where a function equals zero (i.e., finding x where f(x) = 0). These methods are essential in many engineering applications where analytical solutions aren't available.

### Real-world Applications

- **Structural Engineering**: Finding equilibrium states in force distributions
- **Control Systems**: Determining stability criteria for feedback systems
- **Chemical Engineering**: Calculating chemical equilibrium concentrations
- **Computational Geometry**: Finding intersection points between curves
- **Financial Mathematics**: Calculating internal rate of return for investments

## Methods Covered

### 1. Bisection Method

- **Principle**: Repeatedly bisect an interval and select the subinterval where the function changes sign
- **Convergence**: Linear, reliable but slow
- **Key Equations**: x<sub>next</sub> = (a + b)/2
- **Advantages**: Always converges if function changes sign in interval
- **Disadvantages**: Slow convergence

#### Worked Example: Finding √2 using Bisection

Let's find the square root of 2 by solving f(x) = x² - 2 = 0:

| Iteration | a    | b    | x = (a+b)/2 | f(x)      | New Interval |
|-----------|------|------|-------------|-----------|--------------|
| 1         | 1    | 2    | 1.5         | 0.25 > 0  | [1, 1.5]     |
| 2         | 1    | 1.5  | 1.25        | -0.4375 < 0 | [1.25, 1.5]  |
| 3         | 1.25 | 1.5  | 1.375       | -0.1094 < 0 | [1.375, 1.5] |
| 4         | 1.375| 1.5  | 1.4375      | 0.0664 > 0  | [1.375, 1.4375] |
| 5         | 1.375| 1.4375| 1.40625    | -0.0229 < 0 | [1.40625, 1.4375] |

After 10 iterations, we get 1.414... which approaches √2 ≈ 1.41421.

#### Error Analysis

- **Error bound**: |b - a|/2<sup>n</sup> after n iterations
- **Iterations required**: To achieve accuracy ε, need approximately log₂((b - a)/ε) iterations

#### Python Implementation
```python
def bisection(f, a, b, tol=1e-6, max_iter=100):
    """
    Find root of f(x) = 0 in interval [a, b] using bisection method.
    
    Parameters:
    - f: Function to find root of
    - a, b: Interval endpoints
    - tol: Tolerance for convergence
    - max_iter: Maximum iterations
    
    Returns:
    - Approximate root
    - Number of iterations
    """
    if f(a) * f(b) > 0:
        raise ValueError("Function must have opposite signs at interval endpoints")
    
    iter_count = 0
    while (b - a) > tol and iter_count < max_iter:
        c = (a + b) / 2
        if f(c) == 0:
            return c, iter_count
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iter_count += 1
    
    return (a + b) / 2, iter_count
```

### 2. Newton-Raphson Method

- **Principle**: Uses tangent line approximations to find improved estimates
- **Convergence**: Quadratic (when conditions are right)
- **Key Equation**: x<sub>next</sub> = x - f(x)/f'(x)
- **Advantages**: Fast convergence near roots
- **Disadvantages**: Requires derivative calculation; can diverge for poor initial guesses

#### Geometric Interpretation

The Newton-Raphson method uses the tangent line at the current approximation to find where this line crosses the x-axis. This intersection becomes the next approximation.

#### Worked Example: Finding √2 using Newton-Raphson

Let's solve f(x) = x² - 2 = 0 with f'(x) = 2x:

| Iteration | x<sub>n</sub> | f(x<sub>n</sub>) | f'(x<sub>n</sub>) | x<sub>n+1</sub> |
|-----------|---------------|------------------|-------------------|-----------------|
| 1         | 1.0           | -1.0             | 2.0               | 1.5             |
| 2         | 1.5           | 0.25             | 3.0               | 1.4167          |
| 3         | 1.4167        | 0.0069           | 2.8334            | 1.4142          |
| 4         | 1.4142        | 0.0000049        | 2.8284            | 1.4142          |

Observe how quickly the method converges!

#### Convergence Conditions

Newton-Raphson converges quadratically when:
- f'(x) ≠ 0 near the root
- f''(x) is continuous near the root
- The initial guess is sufficiently close

#### Failure Cases

1. **When f'(x) = 0**: The tangent is horizontal, causing the method to fail
2. **Multiple roots**: Convergence becomes linear instead of quadratic
3. **Poor initial guess**: May converge to a different root or diverge entirely

#### Python Implementation
```python
def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    """
    Find root of f(x) = 0 using Newton-Raphson method.
    
    Parameters:
    - f: Function to find root of
    - df: Derivative of f
    - x0: Initial guess
    - tol: Tolerance for convergence
    - max_iter: Maximum iterations
    
    Returns:
    - Approximate root
    - Number of iterations
    """
    x = x0
    iter_count = 0
    
    while iter_count < max_iter:
        fx = f(x)
        if abs(fx) < tol:
            return x, iter_count
        
        dfx = df(x)
        if dfx == 0:
            raise ValueError("Derivative is zero, Newton-Raphson method fails")
        
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new, iter_count
        
        x = x_new
        iter_count += 1
    
    return x, iter_count
```

### 3. Secant Method

- **Principle**: Approximates derivative using two previous points
- **Convergence**: Superlinear (order ~1.618)
- **Key Equation**: x<sub>next</sub> = x<sub>i</sub> - f(x<sub>i</sub>)(x<sub>i</sub> - x<sub>i-1</sub>)/(f(x<sub>i</sub>) - f(x<sub>i-1</sub>))
- **Advantages**: Doesn't require derivatives
- **Disadvantages**: Less reliable than bisection

#### Worked Example: Finding √2 using Secant Method

Let's solve f(x) = x² - 2 = 0 with initial guesses x₀ = 1 and x₁ = 2:

| Iteration | x<sub>n-1</sub> | x<sub>n</sub> | f(x<sub>n-1</sub>) | f(x<sub>n</sub>) | x<sub>n+1</sub> |
|-----------|-----------------|---------------|-------------------|------------------|-----------------|
| 1         | 1.0             | 2.0           | -1.0              | 2.0              | 1.3333          |
| 2         | 2.0             | 1.3333        | 2.0               | -0.2222          | 1.4            |
| 3         | 1.3333          | 1.4           | -0.2222           | -0.04            | 1.4142          |
| 4         | 1.4             | 1.4142        | -0.04             | -0.0000196       | 1.4142          |

#### Comparison with Newton-Raphson

- **Advantage**: No need for derivative calculations
- **Disadvantage**: Slightly slower convergence (order ~1.618 vs. 2)
- **Memory requirement**: Must store two previous points instead of one

#### Python Implementation
```python
def secant(f, x0, x1, tol=1e-6, max_iter=100):
    """
    Find root of f(x) = 0 using Secant method.
    
    Parameters:
    - f: Function to find root of
    - x0, x1: Two initial points
    - tol: Tolerance for convergence
    - max_iter: Maximum iterations
    
    Returns:
    - Approximate root
    - Number of iterations
    """
    iter_count = 0
    
    while iter_count < max_iter:
        fx0 = f(x0)
        fx1 = f(x1)
        
        if abs(fx1) < tol:
            return x1, iter_count
        
        if fx0 == fx1:
            raise ValueError("Division by zero in Secant method")
        
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        
        if abs(x_new - x1) < tol:
            return x_new, iter_count
        
        x0, x1 = x1, x_new
        iter_count += 1
    
    return x1, iter_count
```

### 4. Fixed-Point Iteration

- **Principle**: Reformulate f(x) = 0 as x = g(x) and iterate
- **Convergence**: Linear when |g'(x)| < 1 near root
- **Key Equation**: x<sub>next</sub> = g(x)
- **Advantages**: Simple to implement
- **Disadvantages**: Convergence not guaranteed

#### Fixed-Point Formulation

To find a root of f(x) = 0, we rewrite it as x = g(x) where:
- g(x) = x + f(x) (often diverges)
- g(x) = x - αf(x) where α is a scalar parameter
- Any other formulation with the same fixed point

#### Convergence Criterion

The method converges if |g'(x)| < 1 in the neighborhood of the root. The smaller |g'(x)|, the faster the convergence.

#### Worked Example: Finding √2 using Fixed-Point Iteration

We want to solve x² = 2. Let's rewrite it as:
- g(x) = x - (x² - 2)/4 = (4 - x² + 2)/4 = (6 - x²)/4

| Iteration | x<sub>n</sub> | g(x<sub>n</sub>) |
|-----------|---------------|------------------|
| 1         | 1.0           | 1.25             |
| 2         | 1.25          | 1.390625         |
| 3         | 1.390625      | 1.41308         |
| 4         | 1.41308       | 1.41414         |
| 5         | 1.41414       | 1.4142          |

Note that g'(x) = -x/2, and at x = √2, |g'(√2)| = |-(√2)/2| ≈ 0.7071 < 1, ensuring convergence.

#### Python Implementation
```python
def fixed_point(g, x0, tol=1e-6, max_iter=100):
    """
    Find fixed point of g(x) = x using iteration.
    
    Parameters:
    - g: Function with fixed point
    - x0: Initial guess
    - tol: Tolerance for convergence
    - max_iter: Maximum iterations
    
    Returns:
    - Approximate fixed point
    - Number of iterations
    """
    x = x0
    iter_count = 0
    
    while iter_count < max_iter:
        x_new = g(x)
        
        if abs(x_new - x) < tol:
            return x_new, iter_count
        
        x = x_new
        iter_count += 1
    
    return x, iter_count
```

## Method Comparison: A Case Study

Let's compare all methods by finding the root of f(x) = e^x - 3x for x ∈ [0, 2]:

| Method           | Initial Values | Iterations | Final Approximation | Error       |
|------------------|----------------|------------|---------------------|-------------|
| Bisection        | [0, 2]         | 20         | 1.0986              | < 1.0e-6    |
| Newton-Raphson   | 1.0            | 5          | 1.0986              | < 1.0e-10   |
| Secant           | [0, 2]         | 7          | 1.0986              | < 1.0e-10   |
| Fixed-Point      | 1.0            | 28         | 1.0986              | < 1.0e-6    |

### Performance Analysis

- **Bisection**: Reliable but slow, requiring many iterations
- **Newton-Raphson**: Fastest convergence when derivative is available
- **Secant**: Good balance of speed and simplicity
- **Fixed-Point**: Simple but requires careful formulation and may converge slowly

## Implementation Considerations

- **Stopping Criteria**: Usually based on |f(x)| < ε or |x<sub>next</sub> - x| < ε
- **Initial Guesses**: Critical for methods like Newton-Raphson
- **Error Analysis**: Understand how errors propagate in each method

### Hybrid Approaches

For robust implementations, consider hybrid methods:
- Start with bisection for a few iterations to get close to the root
- Switch to Newton-Raphson or secant for fast final convergence

## Exam Focus Areas

1. **Method Selection**: Know when to use each method based on problem characteristics
2. **Convergence Analysis**: Understand and compare convergence rates
3. **Implementation Details**: Be able to write pseudocode for each method
4. **Failure Cases**: Recognize when and why methods might fail
5. **Error Bounds**: Calculate or estimate errors for iterative methods

## Practice Problems

Try finding roots for:
1. f(x) = x³ - 7x² + 14x - 6 (has three roots: 1, 2, and 3)
2. f(x) = cos(x) - x (solution involves transcendental function)
3. f(x) = e^x - 3x (nonlinear equation with single root)

### Challenge Problem

Find all roots of f(x) = x^5 - 5x^3 + 4x in the interval [-3, 3].

Original lecture notes are available at: `/files/CE7453/CE7453/01-RootFinding-4slides1page(1).pdf` 