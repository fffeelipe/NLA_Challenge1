## Project Overview

This project focuses on applying **Singular Value Decomposition (SVD)** for image compression and noise reduction. Using the SVD, we decompose the matrix representation of an image to perform compression, filter noise, and reconstruct the image with varying levels of detail. The techniques demonstrated here highlight the power of SVD in reducing data dimensionality and optimizing storage while preserving image quality.

## Challenge Description

We work with a grayscale image of size \( m \times n \), represented as a matrix \( A \). The SVD of \( A \) is given by:

\[ A = U \Sigma V^T \]

where \( U \) and \( V \) are orthogonal matrices, and \( \Sigma \) is a diagonal matrix of singular values. By truncating the SVD to a lower rank, we can compress the image, reducing the storage size while maintaining quality.

### Goals
1. Perform SVD to compress the image.
2. Apply noise reduction techniques.
3. Explore image compression efficiency using varying ranks of approximation.

## Tasks

### 1. **Load Image and Compute \( A^T A \)**
   - Load the image as an Eigen matrix \( A \).
   - Compute the product \( A^T A \) and report its Euclidean norm.

### 2. **Solve Eigenvalue Problem \( A^T A x = \lambda x \)**
   - Solve the eigenvalue problem using Eigen's solver.
   - Report the two largest computed singular values.

### 3. **Export \( A^T A \) in Matrix Market Format**
   - Save the matrix \( A^T A \) in Matrix Market (.mtx) format for use with external solvers like LIS.

### 4. **Find Shift for Eigenvalue Solver**
   - Determine a shift \( \mu \) that accelerates convergence of the eigensolver.
   - Report \( \mu \) and the number of iterations required.

### 5. **Perform SVD of \( A \)**
   - Perform SVD using Eigen’s module and report the Euclidean norm of the diagonal matrix \( \Sigma \).

### 6. **Compute Matrices \( C \) and \( D \) for \( k = 40 \) and \( k = 80 \)**
   - Using truncated SVD, compute the matrices \( C \) and \( D \).
   - Report the number of non-zero entries in both matrices.

### 7. **Compress the Image using \( C D^T \)**
   - Compute compressed images for \( k = 40 \) and \( k = 80 \).
   - Export the results as .png files.

### 8. **Create a Checkerboard Image**
   - Generate a black-and-white checkerboard image of size 200x200 pixels.
   - Report the Euclidean norm of the corresponding matrix.

### 9. **Introduce Noise into the Checkerboard Image**
   - Add noise by altering the pixel values with random fluctuations in the range [-50, 50].
   - Export the noisy image as a .png file.

### 10. **Perform SVD on the Noisy Image**
   - Perform SVD on the noisy image and report the two largest singular values.

### 11. **Compute \( C \) and \( D \) for \( k = 5 \) and \( k = 10 \)**
   - Calculate the matrices \( C \) and \( D \) for the noisy image using \( k = 5 \) and \( k = 10 \).
   - Report the sizes of both matrices.

### 12. **Compress the Noisy Image using \( C D^T \)**
   - Reconstruct the noisy image using \( C D^T \) for both \( k = 5 \) and \( k = 10 \).
   - Save the compressed images as .png files.

### 13. **Compare the Compressed Images**
   - Compare the compressed images to both the original and noisy versions.
   - Provide commentary on the results.

## Tools & Libraries

- **Eigen**: A C++ library for linear algebra, used to perform SVD, matrix operations, and solve eigenvalue problems.
- **stb_image**: A library to load and save images in various formats (e.g., .png).
- **LIS Library**: A library used to solve large sparse systems, helpful for iterative solvers.