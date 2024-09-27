#include <eigen3/Eigen/Dense>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;

const std::string IMAGE_NAME = "Albert_Einstein_Head.jpg";


void save_img(MatrixXi m, int width, int height, std::string outputName){
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> img(height, width);
            img = m.unaryExpr([](int val) -> unsigned char {
                return static_cast<unsigned char>(val);
            });

            // Save the grayscale image using stb_image_write
            if (stbi_write_png(outputName.c_str(), width, height, 1,
                               img.data(), width) == 0) {
                std::cerr << "Error: Could not save " + outputName + " image" << std::endl;

                return ;
            }

            std::cout << "image saved to " << outputName << std::endl;

}

int main(int argc, char* argv[]) {
            srand(777); //repetible randoms

            // Load the image using stb_image
            int width, height, channels;
            unsigned char* image_data = stbi_load(IMAGE_NAME.c_str(), &width, &height, &channels, 1);  // Force 1 channel

            if (!image_data) {
                std::cerr << "Error: Could not load image " << IMAGE_NAME << std::endl;
                return 1;
            }

            std::cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << std::endl;

            MatrixXi gray(height, width);

            // Fill the matrix with image data    
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    gray(i, j) = static_cast<int>(image_data[(i * width + j)]);
                }
            }
            // Free memory!!!
            stbi_image_free(image_data);

            save_img(gray, width, height, "initial.png");
            
            MatrixXi noisy = gray.unaryExpr([](int val) -> int {
                return std::min(255, std::max(0, val + rand()%101 - 50));
            });

            save_img(noisy, width, height, "noisy.png");

    return 0;           
        }