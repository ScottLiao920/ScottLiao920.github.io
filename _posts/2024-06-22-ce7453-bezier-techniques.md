---
title: 'CE7453: Bezier Techniques'
date: 2024-06-22
permalink: /posts/2024/06/ce7453-bezier-techniques/
tags:
  - CE7453
  - Bezier curves
  - computer graphics
  - geometric modeling
  - exam preparation
---

This post summarizes key concepts of Bezier techniques covered in CE7453, based on the lecture notes "03-BeizerTechniques".

## Key Concepts

Bezier curves and surfaces are parametric representations used extensively in computer graphics, CAD systems, and geometric modeling. They provide intuitive control over curve shapes through control points.

### Historical Context

Bezier curves were developed independently by Pierre Bézier (at Renault) and Paul de Casteljau (at Citroën) in the 1960s for automotive design. The mathematical foundation, however, dates back to the Bernstein polynomials developed in the early 20th century.

### Real-world Applications

- **Vehicle Design**: Automobile body shapes and aerodynamic profiles
- **Font Design**: TrueType and PostScript font outlines
- **Animation**: Smooth motion paths and transitions
- **CAD/CAM**: Industrial part design and manufacturing
- **Computer Graphics**: Representing curves and surfaces in rendering

## Bezier Curves

### 1. Mathematical Foundation

- **Definition**: A Bezier curve of degree n is defined by n+1 control points
- **Bernstein Basis**: The mathematical foundation of Bezier curves
  - B<sub>i,n</sub>(t) = (n choose i) t<sup>i</sup> (1-t)<sup>n-i</sup>
- **Parametric Equation**: P(t) = Σ<sub>i=0</sub><sup>n</sup> P<sub>i</sub> B<sub>i,n</sub>(t), t ∈ [0,1]

#### Bernstein Polynomials Derivation

The Bernstein basis polynomials of degree n are:

B<sub>i,n</sub>(t) = (n choose i) t<sup>i</sup> (1-t)<sup>n-i</sup> for i = 0, 1, ..., n

Where (n choose i) is the binomial coefficient:

(n choose i) = n! / (i! × (n-i)!)

#### Common Bezier Curve Forms

1. **Linear Bezier Curve** (n=1, 2 control points):
   - P(t) = (1-t)P<sub>0</sub> + tP<sub>1</sub>
   - Simple linear interpolation between two points

2. **Quadratic Bezier Curve** (n=2, 3 control points):
   - P(t) = (1-t)²P<sub>0</sub> + 2(1-t)tP<sub>1</sub> + t²P<sub>2</sub>
   - Creates a parabolic arc

3. **Cubic Bezier Curve** (n=3, 4 control points):
   - P(t) = (1-t)³P<sub>0</sub> + 3(1-t)²tP<sub>1</sub> + 3(1-t)t²P<sub>2</sub> + t³P<sub>3</sub>
   - Most commonly used form, offering good balance between flexibility and computational efficiency

#### Worked Example: Cubic Bezier Curve

Given control points in 2D:
- P<sub>0</sub> = (0, 0)
- P<sub>1</sub> = (1, 3)
- P<sub>2</sub> = (3, 3)
- P<sub>3</sub> = (4, 0)

Let's compute points on the curve for t = 0, 0.25, 0.5, 0.75, 1:

| t    | (1-t)³ | 3(1-t)²t | 3(1-t)t² | t³   | P(t)          |
|------|--------|----------|----------|------|---------------|
| 0    | 1      | 0        | 0        | 0    | (0, 0)        |
| 0.25 | 0.422  | 0.422    | 0.141    | 0.016| (1.31, 2.34)  |
| 0.5  | 0.125  | 0.375    | 0.375    | 0.125| (2.5, 2.25)   |
| 0.75 | 0.016  | 0.141    | 0.422    | 0.422| (3.44, 1.22)  |
| 1    | 0      | 0        | 0        | 1    | (4, 0)        |

This creates a smooth curve that starts at P<sub>0</sub>, is pulled toward P<sub>1</sub> and P<sub>2</sub>, and ends at P<sub>3</sub>.

### 2. Properties

- **Endpoint Interpolation**: Curve passes through first and last control points
- **Convex Hull**: Entire curve lies within convex hull of control points
- **Affine Invariance**: Transforming control points transforms curve predictably
- **Variation Diminishing**: Curve doesn't oscillate more than control polygon
- **Differentiability**: C<sup>∞</sup> continuous within segments

#### Mathematical Proof of Properties

**Endpoint Interpolation**:
- At t=0: P(0) = B<sub>0,n</sub>(0)P<sub>0</sub> + ... + B<sub>n,n</sub>(0)P<sub>n</sub> = P<sub>0</sub>
  - Because B<sub>0,n</sub>(0) = 1 and B<sub>i,n</sub>(0) = 0 for i > 0
- At t=1: P(1) = B<sub>0,n</sub>(1)P<sub>0</sub> + ... + B<sub>n,n</sub>(1)P<sub>n</sub> = P<sub>n</sub>
  - Because B<sub>n,n</sub>(1) = 1 and B<sub>i,n</sub>(1) = 0 for i < n

**Convex Hull Property**:
- Since Bernstein polynomials are non-negative and sum to 1 (partition of unity), the Bezier curve is a convex combination of its control points.

#### Derivatives of Bezier Curves

The derivative of a Bezier curve of degree n is:

P'(t) = n Σ<sub>i=0</sub><sup>n-1</sup> (P<sub>i+1</sub> - P<sub>i</sub>) B<sub>i,n-1</sub>(t)

For a cubic Bezier curve:
P'(t) = 3[(P<sub>1</sub> - P<sub>0</sub>)(1-t)² + (P<sub>2</sub> - P<sub>1</sub>)2(1-t)t + (P<sub>3</sub> - P<sub>2</sub>)t²]

The second derivative is:
P''(t) = n(n-1) Σ<sub>i=0</sub><sup>n-2</sup> (P<sub>i+2</sub> - 2P<sub>i+1</sub> + P<sub>i</sub>) B<sub>i,n-2</sub>(t)

### 3. De Casteljau Algorithm

- **Purpose**: Efficient and numerically stable way to evaluate Bezier curves
- **Process**: Recursive linear interpolation between control points
- **Advantages**: 
  - Geometrically intuitive
  - Allows curve subdivision at any parameter value
  - Numerically stable

#### Step-by-step De Casteljau Algorithm

1. Start with n+1 control points: P<sub>0</sub>, P<sub>1</sub>, ..., P<sub>n</sub>
2. For r from 1 to n:
   - For i from 0 to n-r:
     - P<sub>i</sub><sup>r</sup> = (1-t)P<sub>i</sub><sup>r-1</sup> + tP<sub>i+1</sub><sup>r-1</sup>
3. P<sub>0</sub><sup>n</sup> is the point on the curve at parameter t

#### Worked Example: De Casteljau Algorithm

For a cubic Bezier curve with control points P<sub>0</sub>, P<sub>1</sub>, P<sub>2</sub>, P<sub>3</sub>, let's evaluate at t = 0.5:

**First level** (r = 1):
- P<sub>0</sub><sup>1</sup> = (1-0.5)P<sub>0</sub> + 0.5P<sub>1</sub> = 0.5P<sub>0</sub> + 0.5P<sub>1</sub>
- P<sub>1</sub><sup>1</sup> = (1-0.5)P<sub>1</sub> + 0.5P<sub>2</sub> = 0.5P<sub>1</sub> + 0.5P<sub>2</sub>
- P<sub>2</sub><sup>1</sup> = (1-0.5)P<sub>2</sub> + 0.5P<sub>3</sub> = 0.5P<sub>2</sub> + 0.5P<sub>3</sub>

**Second level** (r = 2):
- P<sub>0</sub><sup>2</sup> = (1-0.5)P<sub>0</sub><sup>1</sup> + 0.5P<sub>1</sub><sup>1</sup> = 0.5(0.5P<sub>0</sub> + 0.5P<sub>1</sub>) + 0.5(0.5P<sub>1</sub> + 0.5P<sub>2</sub>)
  = 0.25P<sub>0</sub> + 0.5P<sub>1</sub> + 0.25P<sub>2</sub>
- P<sub>1</sub><sup>2</sup> = (1-0.5)P<sub>1</sub><sup>1</sup> + 0.5P<sub>2</sub><sup>1</sup> = 0.5(0.5P<sub>1</sub> + 0.5P<sub>2</sub>) + 0.5(0.5P<sub>2</sub> + 0.5P<sub>3</sub>)
  = 0.25P<sub>1</sub> + 0.5P<sub>2</sub> + 0.25P<sub>3</sub>

**Third level** (r = 3):
- P<sub>0</sub><sup>3</sup> = (1-0.5)P<sub>0</sub><sup>2</sup> + 0.5P<sub>1</sub><sup>2</sup>
  = 0.5(0.25P<sub>0</sub> + 0.5P<sub>1</sub> + 0.25P<sub>2</sub>) + 0.5(0.25P<sub>1</sub> + 0.5P<sub>2</sub> + 0.25P<sub>3</sub>)
  = 0.125P<sub>0</sub> + 0.375P<sub>1</sub> + 0.375P<sub>2</sub> + 0.125P<sub>3</sub>

This gives us P(0.5) = 0.125P<sub>0</sub> + 0.375P<sub>1</sub> + 0.375P<sub>2</sub> + 0.125P<sub>3</sub>, which matches the cubic Bernstein polynomial evaluation.

#### Python Implementation for De Casteljau
```python
import numpy as np

def de_casteljau(control_points, t):
    """
    Evaluate a Bezier curve at parameter t using De Casteljau algorithm.
    
    Parameters:
    - control_points: List of control points (can be 2D or 3D)
    - t: Parameter value in range [0, 1]
    
    Returns:
    - Point on the Bezier curve at parameter t
    """
    points = np.copy(control_points)
    n = len(points) - 1  # Degree of the curve
    
    for r in range(1, n + 1):
        for i in range(n - r + 1):
            points[i] = (1 - t) * points[i] + t * points[i + 1]
    
    return points[0]  # Final point is P₀ⁿ
```

#### Curve Subdivision

One powerful application of the De Casteljau algorithm is curve subdivision. At parameter t, the algorithm computes intermediate points that can be used to represent two new Bezier curves of the same degree, together forming the original curve.

For a cubic curve at t = 0.5, the subdivided control points are:
- Left half: [P<sub>0</sub>, P<sub>0</sub><sup>1</sup>, P<sub>0</sub><sup>2</sup>, P<sub>0</sub><sup>3</sup>]
- Right half: [P<sub>0</sub><sup>3</sup>, P<sub>1</sub><sup>2</sup>, P<sub>2</sub><sup>1</sup>, P<sub>3</sub>]

This subdivision property is useful for rendering, intersection calculations, and adaptive approximation of Bezier curves.

## Bezier Surfaces

- **Construction**: Tensor product of Bezier curves
- **Control Net**: Array of control points that define the surface
- **Equation**: P(u,v) = Σ<sub>i=0</sub><sup>n</sup> Σ<sub>j=0</sub><sup>m</sup> P<sub>i,j</sub> B<sub>i,n</sub>(u) B<sub>j,m</sub>(v)

### Tensor Product Construction

A Bezier surface is created by taking a bidirectional net of control points P<sub>i,j</sub> and applying the Bernstein basis in both parametric directions (u and v).

For a bicubic Bezier surface (n=3, m=3), we have a 4×4 grid of 16 control points and:

P(u,v) = Σ<sub>i=0</sub><sup>3</sup> Σ<sub>j=0</sub><sup>3</sup> P<sub>i,j</sub> B<sub>i,3</sub>(u) B<sub>j,3</sub>(v)

### Evaluating Bezier Surfaces

Bezier surfaces can be evaluated using a bivariate extension of the De Casteljau algorithm:
1. Apply the De Casteljau algorithm to each row of control points using parameter u
2. Apply the De Casteljau algorithm to the resulting points using parameter v

Alternatively:
1. Apply the De Casteljau algorithm to each column of control points using parameter v
2. Apply the De Casteljau algorithm to the resulting points using parameter u

Either approach yields the same result due to the tensor product structure.

#### Python Implementation for Bezier Surface Evaluation
```python
def bezier_surface(control_points, u, v):
    """
    Evaluate a Bezier surface at parameters (u, v).
    
    Parameters:
    - control_points: 2D array of control points
    - u, v: Parameter values in range [0, 1]
    
    Returns:
    - Point on the Bezier surface at (u, v)
    """
    # First pass: evaluate along rows (u parameter)
    row_points = []
    for row in control_points:
        point = de_casteljau(row, u)
        row_points.append(point)
    
    # Second pass: evaluate along result (v parameter)
    return de_casteljau(row_points, v)
```

## Practical Applications

### 1. Curve Design

- **Path Design**: Creating smooth trajectories for animation or motion planning
- **Font Design**: Representing letter outlines in digital typography
- **Boundary Representation**: Defining object boundaries in modeling

### 2. Surface Design

- **Product Design**: Creating smooth surfaces for industrial design
- **Character Modeling**: Defining smooth surfaces for animation characters
- **Architectural Design**: Creating complex curved structures

#### Interactive Design Tools

Modern design software uses Bezier curves and surfaces as fundamental building blocks:
- **Adobe Illustrator/Photoshop**: Pen tool creates Bezier paths
- **Blender/Maya**: Curve and surface tools based on Bezier formulations
- **AutoCAD/SolidWorks**: Curve and surface design tools

### Rendering Bezier Curves

While Bezier curves are defined mathematically as continuous functions, for rendering purposes they are typically approximated using line segments or other primitives.

The common approach is:
1. Evaluate the curve at a set of parameter values: t = 0, Δt, 2Δt, ..., 1
2. Connect the resulting points with line segments

More sophisticated approaches include:
- Adaptive subdivision based on curvature
- Conversion to other primitive types supported by graphics hardware

## Limitations and Solutions

- **Local Control**: Lack of local control with pure Bezier curves
  - Solution: Use piecewise Bezier curves
- **Degree Elevation**: Increasing the degree of a curve without changing its shape
- **Degree Reduction**: Approximating a high-degree curve with a lower-degree one

### Piecewise Bezier Curves

To achieve local control, multiple Bezier curve segments can be joined together. For smoothness, continuity conditions must be enforced at join points:

- **C<sup>0</sup> continuity**: Endpoint of one segment equals the start point of the next
- **C<sup>1</sup> continuity**: First derivatives are equal at join points
- **C<sup>2</sup> continuity**: Second derivatives are equal at join points

For cubic Bezier curves with control points [P<sub>i,0</sub>, P<sub>i,1</sub>, P<sub>i,2</sub>, P<sub>i,3</sub>] and [P<sub>i+1,0</sub>, P<sub>i+1,1</sub>, P<sub>i+1,2</sub>, P<sub>i+1,3</sub>]:

- C<sup>0</sup> continuity: P<sub>i,3</sub> = P<sub>i+1,0</sub>
- C<sup>1</sup> continuity: P<sub>i,3</sub> - P<sub>i,2</sub> = α(P<sub>i+1,1</sub> - P<sub>i+1,0</sub>), where α > 0
- C<sup>2</sup> continuity: Additional constraints on control points

### Degree Elevation

Degree elevation allows representing a Bezier curve of degree n as a Bezier curve of degree n+1 without changing its shape. The new control points are:

P<sub>i</sub>'= (i/(n+1))P<sub>i-1</sub> + (1-i/(n+1))P<sub>i</sub> for i = 0, 1, ..., n+1

Where P<sub>-1</sub> and P<sub>n+1</sub> are defined as P<sub>0</sub> and P<sub>n</sub> respectively.

#### Example: Elevating a Quadratic to Cubic

For a quadratic Bezier curve with control points [P<sub>0</sub>, P<sub>1</sub>, P<sub>2</sub>], the elevated cubic control points are:

- P<sub>0</sub>' = P<sub>0</sub>
- P<sub>1</sub>' = (1/3)P<sub>0</sub> + (2/3)P<sub>1</sub>
- P<sub>2</sub>' = (2/3)P<sub>1</sub> + (1/3)P<sub>2</sub>
- P<sub>3</sub>' = P<sub>2</sub>

## Exam Focus Areas

1. **Bernstein Basis**: Understanding and evaluating the basis functions
2. **De Casteljau Algorithm**: Implementation and geometric interpretation
3. **Curve Properties**: Understanding and applying the key properties
4. **Curve Manipulation**: Degree elevation, subdivision, joining curves
5. **Surface Construction**: Building and evaluating Bezier surfaces

### Common Exam Questions

1. **Analytical**: Derive the Bernstein basis functions for degree n
2. **Computational**: Evaluate a Bezier curve at a specific parameter value
3. **Geometric**: Describe the effect of moving control points
4. **Algorithmic**: Trace through the De Casteljau algorithm steps
5. **Application**: Design a piecewise Bezier curve with specific continuity

## Practice Problems

1. Implement the De Casteljau algorithm for a cubic Bezier curve
2. Find the derivative of a Bezier curve and evaluate it at a specific parameter
3. Perform degree elevation on a quadratic Bezier curve
4. Create a C<sup>1</sup> continuous piecewise Bezier curve from given control points

### Challenge Problem

Design a piecewise cubic Bezier curve that passes through the points (0,0), (2,3), (5,1), and (7,4) with C<sup>1</sup> continuity, and derive the necessary control points.

Original lecture notes are available at: `/files/CE7453/CE7453/03-BeizerTechniques-4slides1page(1).pdf` 