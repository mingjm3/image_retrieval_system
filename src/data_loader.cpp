#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

class DataLoader {
public:
    DataLoader(const string& datasetPath) : datasetPath_(datasetPath) {
        datasetPath_ = datasetPath;
    }

    vector<Mat> loadImages()
    {
        vector<Mat> images;
        vector<string> imagePaths;
        unordered_map<int, string> indexToImagePath;

        // get all images from the dataset folder
        glob(datasetPath_, imagePaths);
        int index = 0;
        for (const auto& imagePath : imagePaths) {
            Mat image = imread(imagePath);
            if (!image.empty()) {
                images.push_back(image);
                indexToImagePath[index++] = imagePath;
                std::cout << "loading image: " << imagePath << endl;
            }
            else {
                std::cout << "Can't load image: " << imagePath << endl;
            }
        }

        // save the index to images path file in build directory / execute directory
        fs::path currentPath = fs::current_path();
        fs::path indexPath = currentPath / "index_to_images";
        string indexPathStr = indexPath.string();
        saveImageIndexMap(indexPathStr, indexToImagePath);
        return images;
    }

private:
    void saveImageIndexMap(const string& filePath, const unordered_map<int, string>& indexToImagePath)
    {
        ofstream file(filePath);
        if (file.is_open()) {
            for (const auto& pair : indexToImagePath) {
                file << pair.first << "," << pair.second << endl;
            }
            file.close();
            std::cout << "Image index map saved" << endl;
        }
        else {
            std::cout << "Failed to save image index map" << endl;
        }
    }

    string datasetPath_;
};