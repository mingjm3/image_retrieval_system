#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "vlad_process.cpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{   
    if (argc<4) {
        std::cout << "Usage: [path_to_images_dataset] [path_to_save_visual_vocabulary] [path_to_save_vlad_vector_model]" << std::endl;
        std::cout << "Example: ./build ~/images ~/visual_vocabulary.yaml ~/vlad_vector.yaml " << endl;
        return 1;
    }

    string datasetPath = argv[1];
    string vocabularyPath = argv[2];
    string vladVectorPath = argv[3];
    VladProcess vladProcess(datasetPath, vocabularyPath, vladVectorPath);
    vladProcess.run();
    return 0;
}