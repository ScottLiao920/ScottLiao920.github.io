---
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

## Methods Covered

### 1. Bisection Method

- **Principle**: Repeatedly bisect an interval and select the subinterval where the function changes sign
- **Convergence**: Linear, reliable but slow
- **Key Equations**: x<sub>next</sub> = (a + b)/2
- **Advantages**: Always converges if function changes sign in interval
- **Disadvantages**: Slow convergence

### 2. Newton-Raphson Method

- **Principle**: Uses tangent line approximations to find improved estimates
- **Convergence**: Quadratic (when conditions are right)
- **Key Equation**: x<sub>next</sub> = x - f(x)/f'(x)
- **Advantages**: Fast convergence near roots
- **Disadvantages**: Requires derivative calculation; can diverge for poor initial guesses

### 3. Secant Method

- **Principle**: Approximates derivative using two previous points
- **Convergence**: Superlinear (order ~1.618)
- **Key Equation**: x<sub>next</sub> = x<sub>i</sub> - f(x<sub>i</sub>)(x<sub>i</sub> - x<sub>i-1</sub>)/(f(x<sub>i</sub>) - f(x<sub>i-1</sub>))
- **Advantages**: Doesn't require derivatives
- **Disadvantages**: Less reliable than bisection

### 4. Fixed-Point Iteration

- **Principle**: Reformulate f(x) = 0 as x = g(x) and iterate
- **Convergence**: Linear when |g'(x)| < 1 near root
- **Key Equation**: x<sub>next</sub> = g(x)
- **Advantages**: Simple to implement
- **Disadvantages**: Convergence not guaranteed

## Implementation Considerations

- **Stopping Criteria**: Usually based on |f(x)| < ε or |x<sub>next</sub> - x| < ε
- **Initial Guesses**: Critical for methods like Newton-Raphson
- **Error Analysis**: Understand how errors propagate in each method

## Exam Focus Areas

1. **Method Selection**: Know when to use each method based on problem characteristics
2. **Convergence Analysis**: Understand and compare convergence rates
3. **Implementation Details**: Be able to write pseudocode for each method
4. **Failure Cases**: Recognize when and why methods might fail
5. **Error Bounds**: Calculate or estimate errors for iterative methods

## Practice Problems

Try finding roots for:
1. f(x) = x³ - 7x² + 14x - 6
2. f(x) = cos(x) - x
3. f(x) = e^x - 3x

Original lecture notes are available at: `/files/CE7453/CE7453/01-RootFinding-4slides1page(1).pdf` 