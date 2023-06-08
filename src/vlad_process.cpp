#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "data_loader.cpp"

using namespace cv;
using namespace std;

class VladProcess {
public:
    VladProcess(const string& datasetPath, const string& vocabularyPath, const string& vladVectorPath) : datasetPath_(datasetPath), vocabularyPath_(vocabularyPath), vladVectorPath_(vladVectorPath) {
        datasetPath_ = datasetPath;
        vocabularyPath_ = vocabularyPath;
        vladVectorPath_ = vladVectorPath;
    }

    VladProcess() {}

    /**
     * Run the VLAD training process
     * @author @mingjm3
    */
    void run()
    {
        std::cout<< "\n Loading images \n" << std::endl;
        DataLoader dataLoader(datasetPath_);
        vector<Mat> images = dataLoader.loadImages();

        std::cout<< "\n Extracting features \n" << std::endl;
        vector<Mat> features = extractFeatures(images);

        std::cout<< "\n Building visual vocabulary \n" << std::endl;
        Mat vocabulary = buildVisualVocabulary(features, 100);
        saveData(vocabulary, vocabularyPath_);

        std::cout<< "\n Calculating VLAD vector \n" << std::endl;
        std::vector<Mat> vladVectors;
        int cnt = 0;
        for (const auto& feature : features) {
            Mat vladVector = computeVLAD(feature, vocabulary);
            vladVectors.push_back(vladVector);
            std::cout << "VLAD vector for image " << cnt++ << " processed" << endl;
        }

        std::cout<< "\n Saving vlad vector model \n" << std::endl;
        saveData(vladVectors, vladVectorPath_);
    }

     /**
     * Generate VLAD vector according to image descriptors and visual vocabulary
     * @author @mingjm3
     * @param descriptors Image descriptors generated from SIFT algorithm
     * @param vocabulary VLAD visual vocabulary
     * @return VLAD vector model
    */
    Mat computeVLAD(const Mat& descriptors, const Mat& vocabulary)
    {
        int numClusters = vocabulary.rows;
        int descriptorSize = descriptors.cols;
        int numDescriptors = descriptors.rows;
       
        Mat descriptors32F;
        descriptors.convertTo(descriptors32F, CV_32F);
       
        Mat vocabulary32F;
        vocabulary.convertTo(vocabulary32F, CV_32F);

        Mat vladVector = Mat::zeros(numClusters * descriptorSize, 1, CV_32F);

        Mat indices, dists;
        flann::Index index(vocabulary32F, flann::KDTreeIndexParams());
    
        index.knnSearch(descriptors32F, indices, dists, 1, flann::SearchParams());

        for (int i = 0; i < numDescriptors; i++) {
            int clusterIdx = indices.at<int>(i, 0);
            Mat residual = descriptors32F.row(i) - vocabulary32F.row(clusterIdx);
            for (int j = 0; j < descriptorSize; j++) {
                vladVector.at<float>(clusterIdx * descriptorSize + j) += residual.at<float>(0, j);
            }
        }

        // Normalize
        std::cout << "Normalizing..." << endl;
        normalize(vladVector, vladVector, NORM_L2);
        return vladVector;
    }

private:
    /**
     * Extract features using SIFT algorithm
     * @author @mingjm3
     * @param images
     * @return extracted features as vector
    */
    vector<Mat> extractFeatures(const vector<Mat>& images)
    {
        vector<Mat> features;
        Ptr<Feature2D> detector = SIFT::create();
        int cnt = 0;
        for (const auto& image : images) {
            vector<KeyPoint> keypoints;
            Mat descriptors;
            detector->detectAndCompute(image, noArray(), keypoints, descriptors);
            features.push_back(descriptors);
            std::cout<< "Extracted " << (cnt++) << " image features successfully!" << std::endl;
        }
        return features;
    }

    /**
     * Build visual vocabulary using K-Means algorithm
     * @author @mingjm3
     * @param features
     * @param vocabularySize
     * @return visual vocabulary
    */
    Mat buildVisualVocabulary(const vector<Mat>& features, int vocabularySize)
    {
        // Convert the descriptors vector to a single matrix
        cv::Mat visualVocabulary;
        cv::vconcat(features, visualVocabulary);

        // Specify the number of clusters (visual words)
        int numClusters = vocabularySize;

        // Perform clustering to obtain the visual words
        cv::TermCriteria termCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.01);
        cv::Mat labels, centers;
        cv::kmeans(visualVocabulary, numClusters, labels, termCriteria, 1, cv::KMEANS_PP_CENTERS, centers);
        std::cout<< "K-means executed successfully" << std::endl;
        std::cout<< "Visual vocabulary generated" << std::endl;
        return centers;
    }

    template<typename T>
    void saveData(const T& data, string path)
    {
        FileStorage fs(path, FileStorage::WRITE);
        fs << "data" << data;
        fs.release();
        std::cout << "Data saved" << std::endl;
    }

    string datasetPath_;
    string vocabularyPath_;
    string vladVectorPath_;
};