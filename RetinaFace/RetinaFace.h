#ifndef RETINAFACE_TRT_RETINAFACE_H
#define RETINAFACE_TRT_RETINAFACE_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"

class RetinaFace{
    struct FaceBox{
        float x;
        float y;
        float w;
        float h;
    };

    struct FaceRes{
        float confidence;
        FaceBox face_box;
        std::vector<cv::Point2f> keypoints;
        bool has_mask = false;
    };

public:
    explicit RetinaFace(const std::string &config_file);
    ~RetinaFace();
    void LoadEngine();
    bool InferenceFolder(const std::string &folder_name);
private:
    void EngineInference(const std::string image_name, const int &outSize,void **buffers,
                         const std::vector<int64_t> &bufferSize, cudaStream_t stream);
    void GenerateAnchors();
    std::vector<float> prepareImage(cv::Mat src_img);
    std::vector<std::vector<FaceRes>> postProcess(const cv::Mat src_img, float *output, const int &outSize);
    void NmsDetect(std::vector<FaceRes> &detections);
    static float IOUCalculate(const FaceBox &det_a, const FaceBox &det_b);

    std::string onnx_file;
    std::string engine_file;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    float obj_threshold;
    float nms_threshold;
    bool detect_mask;
    float mask_thresh;
    float landmark_std;

    cv::Mat refer_matrix;
    int anchor_num = 2;
    int bbox_head = 4;
    int landmark_head = 10;
    std::vector<int> feature_sizes;
    std::vector<int> feature_steps;
    std::vector<int> feature_maps;
    std::vector<std::vector<int>> anchor_sizes;
    int sum_of_feature;
};

#endif //RETINAFACE_TRT_RETINAFACE_H
