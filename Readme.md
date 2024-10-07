# Tasks

1. Load the image as an Eigen matrix with size m × n. Each entry in the matrix corresponds to a pixel on the screen and takes a value somewhere between 0 (black) and 255 (white). Report the size of the matrix.
2. Introduce a noise signal into the loaded image by adding random fluctuations of color ranging between [−50, 50] to each pixel. Export the resulting image in .png and upload it.
3. Reshape the original and noisy images as vectors v and w, respectively. Verify that each vector has m n components. Report here the Euclidean norm of v.
4. Write the convolution operation corresponding to the smoothing kernel Hav2 as a matrix vector multiplication between a matrix A1 having size mn × mn and the image vector. Report the number of non-zero entries in A1.
5. Apply the previous smoothing filter to the noisy image by performing the matrix vector multiplication A1w. Export and upload the resulting image.
6. Write the convolution operation corresponding to the sharpening kernel Hsh2 as a matrix vector multiplication by a matrix A2 having size mn × mn. Report the number of non-zero entries in A2. Is A2 symmetric?
7. Apply the previous sharpening filter to the original image by performing the matrix vector multiplication A2v. Export and upload the resulting image.
8. Export the Eigen matrix A2 and vector w in the .mtx format. Using a suitable iterative solver and preconditioner technique available in the LIS library compute the approximate solution to the linear system A2x = w prescribing a tolerance of 10−9. Report here the iteration count and the final residual.
9. Import the previous approximate solution vector x in Eigen and then convert it into a .png image. Upload the resulting file here.
10. Write the convolution operation corresponding to the detection kernel Hlap as a matrix vector multiplication by a matrix A3 having size mn × mn. Is matrix A3 symmetric?
11. Apply the previous edge detection filter to the original image by performing the matrix vector multiplication A3 v. Export and upload the resulting image.
12. Using a suitable iterative solver available in the Eigen library compute the approximate solution of the linear system (I+A3)y = w, where I denotes the identity matrix, prescribing a tolerance of 10−10. Report here the iteration count and the final residual.
13. Convert the image stored in the vector y into a .png image and upload it.
14. Comment the obtained results.

