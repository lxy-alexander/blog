---
title: "SVD"
published: 2026-05-13
description: "SVD"
image: ""
tags: ["llm","SVD"]
category: llm
draft: false
lang: ""
createdAt: "2026-05-13T20:41:45.112.999669998Z"
---

# Singular Value Decomposition

SVD breaks down a matrix into: directions, the importance of each direction, and finally get new coordinate directions

First, rotate the coordinates, then stretch/compress in different directions, and finally rotate to a new direction.

| Matrix Type             | Meaning                            | Common Applications                |
| ----------------------- | ---------------------------------- | ---------------------------------- |
| Data Matrix             | Rows = samples, columns = features | Machine learning, PCA              |
| User-Item Rating Matrix | User preferences or ratings        | Recommendation systems             |
| Image Matrix            | Pixel intensity values             | Image compression, computer vision |
| Signal Matrix           | Time-series or sensor signals      | Signal processing, denoising       |
| Weight Matrix           | Neural network parameters          | Deep learning, model compression   |
| System Matrix           | Represents linear systems (Ax=b)   | Control systems, optimization      |
| Covariance Matrix       | Measures feature correlations      | Statistics, PCA                    |
| Transformation Matrix   | Geometric mappings and rotations   | Robotics, graphics, vision         |

## 1. Core Definition

SVD decomposes a matrix $A$ into three matrices: left singular vectors (左奇异向量), singular values (奇异值), and right singular vectors (右奇异向量).
$$
A = U \Sigma V^T
$$
Where:

1.  $U$ is an orthogonal matrix (正交矩阵) whose columns are left singular vectors.
2.  $\Sigma$ is a diagonal matrix (对角矩阵) containing non-negative singular values.
3.  $V^T$ is the transpose of an orthogonal matrix whose columns are right singular vectors.

<br>

## 2. Intuition

SVD means that a linear transformation first rotates the input, then scales it, and finally rotates it again.
$$
x \xrightarrow{V^T} \text{rotate} \xrightarrow{\Sigma} \text{scale} \xrightarrow{U} \text{rotate}
$$
Interview memory sentence: SVD separates a matrix into direction, importance, and direction again.

<br>

## 3. Matrix Shapes

For a matrix $A \in \mathbb{R}^{m \times n}$, the full SVD shape is:
$$
A_{m \times n} = U_{m \times m}\Sigma_{m \times n}V^T_{n \times n}
$$
The compact SVD (紧凑奇异值分解) keeps only non-zero singular values.
$$
A_{m \times n} = U_{m \times r}\Sigma_{r \times r}V^T_{r \times n}
$$
Here, $r$ is the rank (秩) of matrix $A$.

<br>

## 4. How to Compute SVD Manually

The standard manual method uses eigenvalue decomposition (特征值分解) of $A^TA$ and $AA^T$.
$$
A^TA = V\Lambda V^T
$$
Singular values are the square roots of the non-negative eigenvalues (特征值).
$$
\sigma_i = \sqrt{\lambda_i}
$$

## 5. Complete Numerical Example Step by Step

Let:
$$
A =
\begin{bmatrix}
3 & 0 \\
0 & 2
\end{bmatrix}
$$
This matrix is already diagonal, so the SVD is easy to verify step by step.

### 1) Compute $A^TA$

$$
A^T =
\begin{bmatrix}
3 & 0 \\
0 & 2
\end{bmatrix}
$$

### 2) Find Eigenvalues of $A^TA$

Solve:
$$
\det(A^TA - \lambda I)=0
$$
Therefore:
$$
\lambda_1 = 9,\quad \lambda_2 = 4
$$

### 3) Compute Singular Values

Singular values are square roots of eigenvalues.
$$
\sigma_1 = \sqrt{9}=3
$$
So:
$$
\Sigma =
\begin{bmatrix}
3 & 0 \\
0 & 2
\end{bmatrix}
$$

### 4) Compute Right Singular Vectors $V$

For $\lambda_1 = 9$:
$$
(A^TA - 9I)v = 0
$$
So:
$$
y = 0
$$
Choose normalized vector:
$$
v_1 =
\begin{bmatrix}
1 \\
0
\end{bmatrix}
$$
For $\lambda_2 = 4$:
$$
(A^TA - 4I)v = 0
$$
So:
$$
x = 0
$$
Choose normalized vector:
$$
v_2 =
\begin{bmatrix}
0 \\
1
\end{bmatrix}
$$
Therefore:
$$
V =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$
And:
$$
V^T =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

### 5) Compute Left Singular Vectors $U$

Use:
$$
u_i = \frac{Av_i}{\sigma_i}
$$
For $v_1$:
$$
Av_1 =
\begin{bmatrix}
3 & 0 \\
0 & 2
\end{bmatrix}
\begin{bmatrix}
1 \\
0
\end{bmatrix}
=
\begin{bmatrix}
3 \\
0
\end{bmatrix}
$$
For $v_2$:
$$
Av_2 =
\begin{bmatrix}
3 & 0 \\
0 & 2
\end{bmatrix}
\begin{bmatrix}
0 \\
1
\end{bmatrix}
=
\begin{bmatrix}
0 \\
2
\end{bmatrix}
$$
Therefore:
$$
U =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

### 6) Verify the Decomposition

$$
U\Sigma V^T =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
3 & 0 \\
0 & 2
\end{bmatrix}
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

So:
$$
A = U\Sigma V^T
$$

## 6. Non-Diagonal Numerical Example

Let:
$$
A =
\begin{bmatrix}
1 & 1 \\
0 & 0
\end{bmatrix}
$$
This example shows how SVD handles rank-deficient matrices (秩亏矩阵).

### 1) Compute $A^TA$

$$
A^T =
\begin{bmatrix}
1 & 0 \\
1 & 0
\end{bmatrix}
$$

### 2) Find Eigenvalues

$$
\det(A^TA-\lambda I)=0
$$

Therefore:
$$
\lambda_1 = 2,\quad \lambda_2 = 0
$$

### 3) Compute Singular Values

$$
\sigma_1 = \sqrt{2}
$$

So:
$$
\Sigma =
\begin{bmatrix}
\sqrt{2} & 0 \\
0 & 0
\end{bmatrix}
$$

### 4) Compute Right Singular Vectors

For $\lambda_1 = 2$:
$$
(A^TA - 2I)v=0
$$
This gives:
$$
x = y
$$
Choose and normalize:
$$
v_1 =
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 \\
1
\end{bmatrix}
$$
For $\lambda_2 = 0$:
$$
A^TAv=0
$$
This gives:
$$
x + y = 0
$$
Choose and normalize:
$$
v_2 =
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 \\
-1
\end{bmatrix}
$$
Therefore:
$$
V =
\begin{bmatrix}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
\end{bmatrix}
$$

### 5) Compute Left Singular Vector $u_1$

Use:
$$
u_1 = \frac{Av_1}{\sigma_1}
$$
Since $\sigma_2 = 0$, $u_2$ can be any unit vector orthogonal to $u_1$.
$$
u_2 =
\begin{bmatrix}
0 \\
1
\end{bmatrix}
$$
Therefore:
$$
U =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

### 6) Verify the Decomposition

$$
U\Sigma V^T
=
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
\sqrt{2} & 0 \\
0 & 0
\end{bmatrix}
\begin{bmatrix}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
\end{bmatrix}
$$

First multiply $U\Sigma$:
$$
U\Sigma =
\begin{bmatrix}
\sqrt{2} & 0 \\
0 & 0
\end{bmatrix}
$$
Then multiply by $V^T$:
$$
\begin{bmatrix}
\sqrt{2} & 0 \\
0 & 0
\end{bmatrix}
\begin{bmatrix}
\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
\end{bmatrix}
=
\begin{bmatrix}
1 & 1 \\
0 & 0
\end{bmatrix}
$$
So:
$$
A = U\Sigma V^T
$$

## 7. Python Example

This code can be executed independently and verifies $A = U\Sigma V^T$.

```
import numpy as np

A = np.array([
    [1, 1],
    [0, 0]
], dtype=float)

U, singular_values, VT = np.linalg.svd(A)

Sigma = np.zeros_like(A, dtype=float)
np.fill_diagonal(Sigma, singular_values)

A_reconstructed = U @ Sigma @ VT

print("A:")
print(A)

print("\nU:")
print(U)

print("\nSingular values:")
print(singular_values)

print("\nSigma:")
print(Sigma)

print("\nVT:")
print(VT)

print("\nU @ Sigma @ VT:")
print(A_reconstructed)

print("\nIs reconstruction correct?")
print(np.allclose(A, A_reconstructed))

# Expected output:
# A:
# [[1. 1.]
#  [0. 0.]]
#
# U:
# [[1. 0.]
#  [0. 1.]]
#
# Singular values:
# [1.41421356 0.        ]
#
# Sigma:
# [[1.41421356 0.        ]
#  [0.         0.        ]]
#
# VT:
# [[ 0.70710678  0.70710678]
#  [-0.70710678  0.70710678]]
#
# U @ Sigma @ VT:
# [[1. 1.]
#  [0. 0.]]
#
# Is reconstruction correct?
# True
```

Note: The signs of singular vectors may differ because eigenvectors can be multiplied by $-1$ without changing the decomposition.

<br>



