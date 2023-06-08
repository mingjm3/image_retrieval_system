#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "vlad_process.cpp"

using namespace cv;
using namespace std;

/**
 * Load vocabulary from disk
 * @mingjm3
 * @param filename
 * @return vocabulary
*/
Mat loadVisualVocabulary(const string& filePath)
{
    Mat visualVocabulary;
    FileStorage fs(filePath, FileStorage::READ);
    if (fs.isOpened()) {
        fs["data"] >> visualVocabulary;
        fs.release();
    }
    else {
        std::cout << "Failed to open vocabulary file: " << filePath << std::endl;
    }
    return visualVocabulary;
}

/**
 * Load VLAD model from disk
 * @mingjm3
 * @param filename
 * @return VLAD vector model
*/
vector<Mat> loadVladModel(const string& filePath)
{
    vector<Mat> vladVectors;
    FileStorage fs(filePath, FileStorage::READ);
    if (fs.isOpened()) {
        fs["data"] >> vladVectors;
        fs.release();
    }
    else {
        std::cout << "Failed to open vlad vector file: " << filePath << std::endl;
    }
    return vladVectors;
}

/**
 * Load index to image map file from disk
 * @mingjm3
 * @return index to image map
*/
unordered_map<int, string> loadImageIndexMap()
{
    unordered_map<int, string> indexToImage;

    fs::path currentPath = fs::current_path();
    fs::path indexPath = currentPath / "index_to_images";
    string indexPathStr = indexPath.string();

    ifstream file(indexPathStr);
    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            size_t commaPos = line.find(',');
            if (commaPos != string::npos) {
                int index = stoi(line.substr(0, commaPos));
                string imagePath = line.substr(commaPos + 1);
                indexToImage[index] = imagePath;
            }
        }
        file.close();
        std::cout << "Image index map loaded" << endl;
    }
    else {
        std::cout << "Failed to load image index map" << endl;
    }

    return indexToImage;
}


double calculateDistance(const Mat& queryVector, const Mat& vladVector)
{
    return norm(queryVector, vladVector, NORM_L2);
}

/**
 * Find the most similar image in the images set
 * @author @mingjm3
 * @param queryImage
 * @param vladVectors
 * @return the most similar image index
*/
int findMostSimilarImage(const Mat& queryImage, const vector<Mat>& vladVectors, const Mat vocabulary)
{
    Ptr<Feature2D> detector = SIFT::create();
    vector<KeyPoint> keypoints;
    Mat descriptors;
    detector->detectAndCompute(queryImage, noArray(), keypoints, descriptors);

    VladProcess vladProcess;
    Mat queryVector = vladProcess.computeVLAD(descriptors, vocabulary);

    double minDistance = numeric_limits<double>::max();
    int mostSimilarImageIndex = -1;
    for (int i = 0; i < vladVectors.size(); i++) {
        double distance = calculateDistance(queryVector, vladVectors[i]);
        if (distance < minDistance) {
            minDistance = distance;
            mostSimilarImageIndex = i;
        }
    }

    return mostSimilarImageIndex;
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::cout << "Usage: [path_to_query_image] [path_to_visual_vocabulary] [path_to_vlad_vector_model]" << endl;
        std::cout << "Example: ./app ~/query.jpg ~/visual_vocabulary.yaml ~/vlad_vector.yaml " << endl;
        std::cout << "Shoud run build.cpp to train a VLAD vector model and visual vocabulary before run app.cpp" << endl;
        return 1;
    }

    string queryImagePath = argv[1];
    string vocabularyPath = argv[2];
    string vladModelPath = argv[3];

    Mat queryImage = imread(queryImagePath, IMREAD_COLOR);
    if (queryImage.empty()) {
        std::cout << "Failed to load query image." << endl;
        return 1;
    }
    imshow("query image", queryImage);

    Mat visualVocabulary = loadVisualVocabulary(vocabularyPath);
    if (visualVocabulary.empty()) {
        std::cout << "Failed to load visual vocabulary." << endl;
        return 1;
    }

    vector<Mat> vladVectors = loadVladModel(vladModelPath);
    if (vladVectors.empty()) {
        std::cout << "Failed to load VLAD vector model." << endl;
        return 1;
    }

    int mostSimilarImageIndex = findMostSimilarImage(queryImage, vladVectors, visualVocabulary);
    if (mostSimilarImageIndex == -1) {
        std::cout << "Failed to find the most similar image." << endl;
        return 1;
    }

    std::cout << "The most similar image is: " << mostSimilarImageIndex << endl;
    
    // get index to image path from indexPathStr
    unordered_map<int, string> indexToImagePath = loadImageIndexMap();
    Mat mostSimilarImage = imread(indexToImagePath[mostSimilarImageIndex]);
    imshow("most similar image", mostSimilarImage);
    waitKey(0);
    return 0;
}