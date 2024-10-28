#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <fstream>
#define STB_IMAGE_IMPLEMENTATION
#include "Library/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Library/stb_image_write.h"

#include <Eigen/SVD>

using namespace Eigen;
using namespace std;

// Some useful alias
using spMatrix = Eigen::SparseMatrix<double, RowMajor>;
using spVector = Eigen::VectorXd;

const std::string IMAGE_PATH = "Input/Albert_Einstein_Head.jpg";

void save_img(MatrixXd m, int width, int height, std::string outputName)
{
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> img(height, width);
    img = m.unaryExpr([](double val) -> unsigned char
                      { return static_cast<unsigned char>(std::min(255.0, std::max(0.0, val))); });

    // Save the image using stb_image_write
    if (stbi_write_png(outputName.c_str(), width, height, 1, img.data(), width) == 0)
    {
        std::cerr << "\nError: Could not save " + outputName + " image" << std::endl;
        return;
    }

    std::cout << "\nimage saved to " << outputName << std::endl;
}



// Calculate the truncated matrices Ck and Dk.
std::tuple<MatrixXd, MatrixXd> calculate_truncateSVD(const  BDCSVD<MatrixXd>& svd, const VectorXd& singularValue, int k) {
    MatrixXd C_k = svd.matrixU().leftCols(k);
    MatrixXd V_k = svd.matrixV().leftCols(k);
    MatrixXd Sigma_k = singularValue.head(k).asDiagonal();

    MatrixXd D_k = Sigma_k * V_k.transpose();
    std::cout << "\nFor k = " << k << ", the number of nonzero entries in the matrix C and D are: " << C_k.nonZeros() << " and " << D_k.nonZeros() << std::endl;

    return std::make_tuple(C_k, D_k);

}



int main(int argc, char *argv[])
{
    // TASK 1: Load the image as Eigen matrix A and compute A^T * A
    // Load the image into an Eigen matrix. Each pixel value is represented as an entry in matrix A.
    // Then compute the matrix product A^T * A to analyze the correlation between image columns.
    // Finally, compute and print the Euclidean norm of A^T * A.

    int n, m, channels;
    unsigned char *image_data = stbi_load(IMAGE_PATH.c_str(), &n, &m, &channels, 1); // Force 1 channel

    if (!image_data)
    {
        std::cerr << "\nError: Could not load image " << IMAGE_PATH << std::endl;
        return 1;
    }
    MatrixXd A(m, n);

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            A(i, j) = static_cast<double>(image_data[(i * n + j)]);
        }
    }
    stbi_image_free(image_data);

    auto At_A  = A.transpose() * A;

    auto norm_At_A = At_A.norm();
  
    // printf("Task1: The norm of A^T * A is: %f\n", norm_At_A);   
    std:cout << "\nTask1: The norm of A^T * A is: " << norm_At_A << std::endl;  // 1.05041e+09


    // TASK 2: Solve the eigenvalue problem A^T A x = λ x
    // Use Eigen's solver to compute the eigenvalues of A^T A. Report the two largest singular values of matrix A.
    /* Use SelfAdjointEigenSolver() to solve eigenvalue problem for symmetric matrices.(A^TT is symmetric)*/
    SelfAdjointEigenSolver<MatrixXd> saeigensolver(At_A);
    if (saeigensolver.info() != Eigen::Success) 
    {
        printf("\nCouldn't calculate the eigenpairs, sorry. \n");
        abort();
    } 
    /* Eigenvalues are sorted in ascending order, last element is the largest.*/
    // printf("Task2: The two largest eigenvalues are: %f and %f\n", saeigensolver.eigenvalues()(n-1), saeigensolver.eigenvalues()(n-2));
    std::cout << "\nTask2: The two largest eigenvalues are: " << saeigensolver.eigenvalues()(n-1) << " and " << saeigensolver.eigenvalues()(n-2) << std::endl;


    // TASK 3: Export matrix A^T A in Matrix Market format
    // Save A^T A in the Matrix Market format (.mtx) to be used with external solvers such as LIS.

    if (saveMarket(At_A, "output/At_A.mtx"))
    {
        printf("\nAt_A.mtx succesfully saved.\n");
    }
    else
    {
        printf("\nError: At_A couldn't be saved.\n");
        return 1;
    }
    /* Task3:
    root@36d8968386ff test # cp /shared-folder/NLA_Challenge1/Challenge_2/output/At_A.mtx .
    root@36d8968386ff test # mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis etest1.c -o eigen1
    root@36d8968386ff test # mpirun -n 4 ./eigen1 At_A.mtx eigvec.txt hist.txt -e pi -etol 1e-8   

    number of processes = 4
    matrix size = 256 x 256 (65536 nonzero entries)

    initial vector x      : all components set to 1
    precision             : double
    eigensolver           : Power
    convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-08 * ||lx||_2
    matrix storage format : CSR
    shift                 : 0.000000e+00
    eigensolver status    : normal end

    Power: mode number          = 0
    Power: eigenvalue           = 1.045818e+09
    Power: number of iterations = 8
    Power: elapsed time         = 8.532920e-04 sec.
    Power:   preconditioner     = 0.000000e+00 sec.
    Power:     matrix creation  = 0.000000e+00 sec.
    Power:   linear solver      = 0.000000e+00 sec.
    Power: relative residual    = 1.866013e-09

    // The maximum eigenvalue from task2 is 1.04582e+09, while in task3, the maximum eigenvalue is 1.045818e+09.
    // Since these two eigenvalues are close enough(the difference is less than 10e-8), they can be considered equivalent.
    // Therefore, for task3, the result calculated by the LIS solver is in agreement with the one obtained in task2.

    */
   std::cout << "\nTask3: The maximum eigenvalue calculated by LIS is: 1.045818e+09, and is in agreement with task2." << std::endl; 


    // TASK 4: Find a shift μ to accelerate the eigenvalue solver
    // Identify an optimal shift μ that can improve the convergence of the iterative eigensolver.
    // Report μ and the number of iterations required for convergence.

    /*
    root@36d8968386ff test # mpirun -n 4 ./eigen1 At_A.mtx eigvec.txt hist.txt -e pi 4.63e7

    number of processes = 4
    matrix size = 256 x 256 (65536 nonzero entries)

    initial vector x      : all components set to 1
    precision             : double
    eigensolver           : Power
    convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-12 * ||lx||_2
    matrix storage format : CSR
    shift                 : 0.000000e+00
    eigensolver status    : normal end

    Power: mode number          = 0
    Power: eigenvalue           = 1.045818e+09
    Power: number of iterations = 12
    Power: elapsed time         = 9.940840e-04 sec.
    Power:   preconditioner     = 0.000000e+00 sec.
    Power:     matrix creation  = 0.000000e+00 sec.
    Power:   linear solver      = 0.000000e+00 sec.
    Power: relative residual    = 1.054496e-13

    // Based on the concept of the shift in eigenvalue solvers, I selected μ(shift) that is approximately equal to (λ2+λmin)/2, 
    // which resulted in a shift value close to 4.53e7. I then tested slight variations around this shift,
    // specifically μ+100(4.63e7) and μ-100(4.43e7). All these three shifts produced the same number of iterations and relative residuals, 
    // indicating stable convergence behavior. The minimum elapsed time was achieved with 4.63e7, at 9.940840e-04 sec, while
    // the other two were on the order of e-3. Thus, I finalized this shift value as a good choice for the eigensolver.
    
    // Using 4.63e7 as the shift required 12 iterations to converge, which is the minimum number of iterations observed among the tested shifts.
    // The relative residual is 1.054496e-13, which is smaller than the tolerance of 10e-8, meeting the accuracy requirement.
    // Shift 4.63e7 offers a good balance between minimizing the number of iterations and achiveing a small relative residual, prove both speed and precision.
    */
   std::cout << "\nTask4: The most effective shift for accelerating the eigensolver is 4.63e7." << std::endl;


    // TASK 5: Perform SVD on matrix A
    // Perform Singular Value Decomposition (SVD) on matrix A using Eigen's SVD module.
    // Report the Euclidean norm of the singular values (Σ) matrix.
    BDCSVD<MatrixXd> svdA(A, ComputeThinU | ComputeThinV);  // Use BDCSVD for large matrix
    VectorXd singularValuesA = svdA.singularValues();
    auto norm_singularValueA = singularValuesA.norm();
    std::cout << "\nTask5: The Euclidean norm of the singular value is: " << norm_singularValueA << std::endl;


    // TASK 6: Compute matrices C and D for k = 40 and k = 80
    // Compute C and D matrices using k = 40 and k = 80 based on the SVD results.
    // Report the number of non-zero entries in C and D.
    int k1 = 40, k2 = 80;
    std::cout << "\nTask6 && Task7: " << std::endl;

    // TASK 7: Compress image using C * D^T for k = 40 and k = 80
    // Reconstruct the compressed images from the matrices C and D.
    // Export the images as .png files and save them.
    auto [C_k1, D_k1] = calculate_truncateSVD(svdA, singularValuesA, k1);
    MatrixXd compressed_matrix_k1 = C_k1 * D_k1;
    save_img(compressed_matrix_k1, n, m, "Assets/task7_k40.png");

    auto [C_k2, D_k2] = calculate_truncateSVD(svdA, singularValuesA, k2);
    MatrixXd compressed_matrix_k2 = C_k2 * D_k2;
    save_img(compressed_matrix_k2, n, m, "Assets/task7_k80.png");



    // TASK 8: Create a black-and-white checkerboard image
    // Generate a checkerboard pattern image of 200x200 pixels. Each pixel alternates between black and white.
    // Report the Euclidean norm of the resulting matrix representing the image.
    MatrixXd BW_Matrix(200, 200);
    // Fill the matrix with color data
    for (int i = 0; i < 200; ++i)
        for (int j = 0; j < 200; ++j)
        BW_Matrix(i,j) = (i/25 + j/25)%2?255:0;
        
    
    std::cout << "\nTask8: " << std::endl;
    save_img(BW_Matrix, 200, 200, "Assets/BW_Matrix.png");
    auto BW_Matrix_Vector = BW_Matrix.transpose().reshaped();

    double BW_Matrix_norm = std::sqrt(BW_Matrix_Vector.dot(BW_Matrix_Vector));
    printf("\nThe norm of BW_Matrix is: %f\n", BW_Matrix_norm );


    // TASK 9: Introduce noise into the checkerboard image
    // Add random noise to the checkerboard image by altering pixel values within the range [-50, 50].
    // Save and export the noisy image as a .png file.

    MatrixXd BW_noisy = BW_Matrix; // Start with the original checkerboard

    // Random number generator for noise in the range [-50, 50]
    srand(777); // Seed rand() for reproducible random numbers

    // Add noise to each pixel of BW_noisy
    BW_noisy = BW_noisy.unaryExpr([&](double val) -> double {
        int noise = rand() % 101 - 50;  // Generate noise in the range [-50, 50]
        double noisy_val = val + noise;  // Add noise to the original pixel value
        return std::clamp(noisy_val, 0.0, 255.0);  // Clamp result to range [0, 255]
    });

    save_img(BW_noisy, 200, 200, "Assets/BW_Noisy.png");


    // TASK 10: Perform SVD on the noisy image
    // Use SVD to analyze the noisy image and report the two largest singular values.
    BDCSVD<MatrixXd> svdNoisy(BW_noisy, ComputeThinU | ComputeThinV);
    auto svdNoisySingularValues = svdNoisy.singularValues();
    printf("the two largest singular values are: %f and %f\n", svdNoisySingularValues(0), svdNoisySingularValues(1));


    // TASK 11: Compute C and D for k = 5 and k = 10
    // Based on the SVD of the noisy image, compute C and D matrices for k = 5 and k = 10.
    // Report the size of the C and D matrices.

    //just for fun:
    auto [C2_BW_noisy, D2_BW_noisy] = calculate_truncateSVD(svdNoisy, svdNoisySingularValues, 2);
    save_img(C2_BW_noisy * D2_BW_noisy, 200, 200, "Assets/SVD_2_BW_Noisy.png");


    auto [C5_BW_noisy, D5_BW_noisy] = calculate_truncateSVD(svdNoisy, svdNoisySingularValues, 5);
    auto [C10_BW_noisy, D10_BW_noisy] = calculate_truncateSVD(svdNoisy, svdNoisySingularValues, 10);

    printf("Size for C and D for svd with k=5: %dx%d %dx%d\n", C5_BW_noisy.rows(), C5_BW_noisy.cols(), D5_BW_noisy.rows(), D5_BW_noisy.cols());
    printf("Size for C and D for svd with k=10: %dx%d %dx%d\n", C10_BW_noisy.rows(), C10_BW_noisy.cols(), D10_BW_noisy.rows(), D10_BW_noisy.cols());

    // TASK 12: Compress the noisy image using C * D^T for k = 5 and k = 10
    // Reconstruct the compressed versions of the noisy image using k = 5 and k = 10.
    // Save the compressed images as .png files.

    save_img(C5_BW_noisy * D5_BW_noisy, 200, 200, "Assets/SVD_5_BW_Noisy.png");
    save_img(C10_BW_noisy * D10_BW_noisy, 200, 200, "Assets/SVD_10_BW_Noisy.png");

    // TASK 13: Compare compressed images with the original and noisy images
    // Compare the quality of the compressed images with both the original and noisy versions.
    // Report the results and observations.

    /*The compressed versions of the checkerboard image (using SVD) effectively reduced noise while maintaining image structure, depending on k. 
    Lower k values worked well for denoising but blurred details, while higher k kept more detail, including some noise. */

    return 0;
}