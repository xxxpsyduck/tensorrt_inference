#include "RetinaFace.h"
#include "yaml-cpp/yaml.h"
#include "common.hpp"

RetinaFace::RetinaFace(const std::string &config_file) {
    YAML::Node root = YAML::LoadFile(config_file);
    YAML::Node config = root["RetinaFace"];
    onnx_file = config["onnx_file"].as<std::string>();
    engine_file = config["engine_file"].as<std::string>();
    BATCH_SIZE = config["BATCH_SIZE"].as<int>();
    INPUT_CHANNEL = config["INPUT_CHANNEL"].as<int>();
    IMAGE_WIDTH = config["IMAGE_WIDTH"].as<int>();
    IMAGE_HEIGHT = config["IMAGE_HEIGHT"].as<int>();
    obj_threshold = config["obj_threshold"].as<float>();
    nms_threshold = config["nms_threshold"].as<float>();
    detect_mask = config["detect_mask"].as<bool>();
    mask_thresh = config["mask_thresh"].as<float>();
    landmark_std = config["landmark_std"].as<float>();
    feature_steps = config["feature_steps"].as<std::vector<int>>();
    for (const int step:feature_steps) {
        assert(step != 0);
        int feature_map = IMAGE_HEIGHT / step;
        feature_maps.push_back(feature_map);
        int feature_size = feature_map * feature_map;
        feature_sizes.push_back(feature_size);
    }
    anchor_sizes = config["anchor_sizes"].as<std::vector<std::vector<int>>>();
    sum_of_feature = std::accumulate(feature_sizes.begin(), feature_sizes.end(), 0) * anchor_num;
    GenerateAnchors();
}

RetinaFace::~RetinaFace() = default;

void RetinaFace::LoadEngine() {
    std::fstream existEngine;
    existEngine.open(engine_file, std::ios::in);
    if (existEngine) {
        readTrtFile(engine_file, engine);
        assert(engine != nullptr);
    } else {
        onnxToTRTModel(onnx_file, engine_file, engine, BATCH_SIZE);
        assert(engine != nullptr);
    }
}

bool RetinaFace::InferenceFolder(const std::string &folder_name) {
    // std::vector<std::string> sample_images = readFolder(folder_name);
    std::string image_name = "/home/khanhtq/tensorrt_inference/RetinaFace/samples/test.jpg";
    //get context
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);

    //get buffers
    assert(engine->getNbBindings() == 2);
    void *buffers[2];
    std::vector<int64_t> bufferSize;
    int nbBindings = engine->getNbBindings();
    bufferSize.resize(nbBindings);

    for (int i = 0; i < nbBindings; ++i) {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
        bufferSize[i] = totalSize;
        cudaMalloc(&buffers[i], totalSize);
    }

    //get stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int outSize = int(bufferSize[1] / sizeof(float) / BATCH_SIZE);

    EngineInference(image_name, outSize, buffers, bufferSize, stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    // destroy the engine
    context->destroy();
    engine->destroy();
}

void RetinaFace::EngineInference(const std::string image_name, const int &outSize, void **buffers, const std::vector<int64_t> &bufferSize, cudaStream_t stream) {
    cv::Mat src_img = cv::imread(image_name);

    auto org_img = src_img.clone();

    std::vector<float>curInput = prepareImage(org_img);

    cudaMemcpyAsync(buffers[0], curInput.data(), bufferSize[0], cudaMemcpyHostToDevice, stream);

    context->execute(BATCH_SIZE, buffers);

    auto *out = new float[outSize * BATCH_SIZE];
    cudaMemcpyAsync(out, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    auto faces = postProcess(org_img, out, outSize);
    delete[] out;

    auto rects = faces[0];
    std::cout << "rects size: " << rects.size() << std::endl;
    for(const auto &rect : rects)
    {
        cv::Rect box(rect.face_box.x - rect.face_box.w / 2, rect.face_box.y - rect.face_box.h / 2, rect.face_box.w, rect.face_box.h);
        cv::rectangle(org_img, box, cv::Scalar(255, 0, 0), 2, cv::LINE_8, 0);

    }
    cv::imwrite("result.jpg", org_img);

}

void RetinaFace::GenerateAnchors() {
    float base_cx = 7.5;
    float base_cy = 7.5;

    refer_matrix = cv::Mat(sum_of_feature, bbox_head, CV_32FC1);
    int line = 0;
    for(size_t feature_map = 0; feature_map < feature_maps.size(); feature_map++) {
        for (int height = 0; height < feature_maps[feature_map]; ++height) {
            for (int width = 0; width < feature_maps[feature_map]; ++width) {
                for (int anchor = 0; anchor < anchor_sizes[feature_map].size(); ++anchor) {
                    auto *row = refer_matrix.ptr<float>(line);
                    row[0] = base_cx + (float)width * feature_steps[feature_map];
                    row[1] = base_cy + (float)height * feature_steps[feature_map];
                    row[2] = anchor_sizes[feature_map][anchor];
                    row[3] = anchor_sizes[feature_map][anchor];
                    line++;
                }
            }
        }
    }
}

std::vector<float> RetinaFace::prepareImage(cv::Mat src_img) {
    std::vector<float> result(BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * INPUT_CHANNEL);
    float *data = result.data();
    int index = 0;
    float ratio = float(IMAGE_WIDTH) / float(src_img.cols) < float(IMAGE_HEIGHT) / float(src_img.rows) ? float(IMAGE_WIDTH) / float(src_img.cols) : float(IMAGE_HEIGHT) / float(src_img.rows);
    cv::Mat flt_img = cv::Mat::zeros(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3);
    cv::Mat rsz_img;
    cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);
    rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
    flt_img.convertTo(flt_img, CV_32FC3);

    int channelLength = IMAGE_WIDTH * IMAGE_HEIGHT;
    std::vector<cv::Mat> split_img = {
            cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC1, data + channelLength * (index + 2)),
            cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC1, data + channelLength * (index + 1)),
            cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_32FC1, data + channelLength * index)
    };
    cv::split(flt_img, split_img);

    return result;
}

std::vector<std::vector<RetinaFace::FaceRes>> RetinaFace::postProcess(const cv::Mat src_img, float *output, const int &outSize) {
    std::vector<std::vector<FaceRes>> vec_result;
    int index = 0;

    std::vector<FaceRes> result;
    float *out = output + index * outSize;
    float ratio = float(src_img.cols) / float(IMAGE_WIDTH) > float(src_img.rows) / float(IMAGE_HEIGHT)  ? float(src_img.cols) / float(IMAGE_WIDTH) : float(src_img.rows) / float(IMAGE_HEIGHT);

    int result_cols = (detect_mask ? 2 : 1) + bbox_head + landmark_head;
    cv::Mat result_matrix = cv::Mat(sum_of_feature, result_cols, CV_32FC1, out);

    for (int item = 0; item < result_matrix.rows; ++item) {
        auto *current_row = result_matrix.ptr<float>(item);
        if(current_row[0] > obj_threshold){
            FaceRes headbox;
            headbox.confidence = current_row[0];
            auto *anchor = refer_matrix.ptr<float>(item);
            auto *bbox = current_row + 1;
            auto *keyp = current_row + 1 + bbox_head;
            auto *mask = current_row + 1 + bbox_head + landmark_head;

            headbox.face_box.x = (anchor[0] + bbox[0] * anchor[2]) * ratio;
            headbox.face_box.y = (anchor[1] + bbox[1] * anchor[3]) * ratio;
            headbox.face_box.w = anchor[2] * exp(bbox[2]) * ratio;
            headbox.face_box.h = anchor[3] * exp(bbox[3]) * ratio;

            headbox.keypoints = {
                    cv::Point(int((anchor[0] + keyp[0] * anchor[2] * landmark_std) * ratio), int((anchor[1] + keyp[1] * anchor[3] * landmark_std) * ratio)),
                    cv::Point(int((anchor[0] + keyp[2] * anchor[2] * landmark_std) * ratio), int((anchor[1] + keyp[3] * anchor[3] * landmark_std) * ratio)),
                    cv::Point(int((anchor[0] + keyp[4] * anchor[2] * landmark_std) * ratio), int((anchor[1] + keyp[5] * anchor[3] * landmark_std) * ratio)),
                    cv::Point(int((anchor[0] + keyp[6] * anchor[2] * landmark_std) * ratio), int((anchor[1] + keyp[7] * anchor[3] * landmark_std) * ratio)),
                    cv::Point(int((anchor[0] + keyp[8] * anchor[2] * landmark_std) * ratio), int((anchor[1] + keyp[9] * anchor[3] * landmark_std) * ratio))
            };

            if (detect_mask and mask[0] > mask_thresh)
                headbox.has_mask = true;
            result.push_back(headbox);
        }
    }
    NmsDetect(result);
    vec_result.push_back(result);
    return vec_result;
}

void RetinaFace::NmsDetect(std::vector<FaceRes> &detections) {
    sort(detections.begin(), detections.end(), [=](const FaceRes &left, const FaceRes &right) {
        return left.confidence > right.confidence;
    });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            float iou = IOUCalculate(detections[i].face_box, detections[j].face_box);
            if (iou > nms_threshold)
                detections[j].confidence = 0;
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const FaceRes &det)
    { return det.confidence == 0; }), detections.end());
}

float RetinaFace::IOUCalculate(const RetinaFace::FaceBox &det_a, const RetinaFace::FaceBox &det_b) {
    cv::Point2f center_a(det_a.x + det_a.w / 2, det_a.y + det_a.h / 2);
    cv::Point2f center_b(det_b.x + det_b.w / 2, det_b.y + det_b.h / 2);
    cv::Point2f left_up(std::min(det_a.x, det_b.x),std::min(det_a.y, det_b.y));
    cv::Point2f right_down(std::max(det_a.x + det_a.w, det_b.x + det_b.w),std::max(det_a.y + det_a.h, det_b.y + det_b.h));
    float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
    float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
    float inter_l = det_a.x > det_b.x ? det_a.x : det_b.x;
    float inter_t = det_a.y > det_b.y ? det_a.y : det_b.y;
    float inter_r = det_a.x + det_a.w < det_b.x + det_b.w ? det_a.x + det_a.w : det_b.x + det_b.w;
    float inter_b = det_a.y + det_a.h < det_b.y + det_b.h ? det_a.y + det_a.h : det_b.y + det_b.h;
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else
        return inter_area / union_area - distance_d / distance_c;
}
