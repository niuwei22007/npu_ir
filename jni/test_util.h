#ifndef BUILD_IR_MODEL_TEST_UTIL_H
#define BUILD_IR_MODEL_TEST_UTIL_H

#include <cstdlib>
#include <fstream>
#include <memory>
#include <unistd.h>
#include <algorithm>
#include <iomanip>
#include <string>
#include <vector>
#include <iostream>
#include <random>
#include <android/log.h>

#include "hiai_ir_build.h"
#include "HiAiModelManagerService.h"
#include "graph/buffer.h"
#include "graph/graph.h"
#include "graph/model.h"
#include "graph/op/all_ops.h"
#include "graph/operator_hiai_reg.h"
#include "graph/compatible/operator_reg.h"
#include "graph/compatible/all_ops.h"


#define LOG_TAG "NNN_TEST"
#define ALOGE(...) \
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__); \
    printf(__VA_ARGS__)

#define ALOGI(...) \
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__); \
    printf(__VA_ARGS__)


typedef bool(*TestFunc)(ge::Graph &graph);

struct TestCase {
    std::string caseName;
    TestFunc func;
    bool inputFromFile;
    uint32_t inputH = 0;
    uint32_t inputW = 0;
};

bool WriteFile(const void *data, size_t size, const std::string &path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        ALOGE("%s open failed.\n", path.c_str());
        return false;
    }
    file.write((const char *) data, size);
    file.flush();
    file.close();
    return true;
}


template<typename T>
bool ReadFile(const std::string &path, std::vector<T> *data) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        ALOGE("%s open failed.\n", path.c_str());
        return false;
    }
    const auto fbegin = file.tellg();
    file.seekg(0, std::ios::end);
    const auto fend = file.tellg();
    size_t size = fend - fbegin;
    file.seekg(0, std::ios::beg);
    data->resize(size / sizeof(T));
    file.read((char *) data->data(), size);
    file.close();
    return true;
}


void PrintTensorInfo(const string &msg, const std::shared_ptr<hiai::AiTensor> &tensor) {
    auto dims = tensor->GetTensorDimension();
    std::cout << msg << " NCHW: " << dims.GetNumber() << ", " << dims.GetChannel() << ", " << dims.GetHeight() << ", "
              << dims.GetWidth() << std::endl;
}


template<typename T>
void SaveTensorData(const std::shared_ptr<hiai::AiTensor> &tensor, const std::string &path) {
    auto size = tensor->GetSize();
    auto num = size / sizeof(T);
    auto ptr = (T *) tensor->GetBuffer();
    WriteFile(ptr, size, path);
    ALOGI("save tensor %p size: %d, num: %d\n", tensor, size, num);
}

template<typename T>
void PrintTensorData(const std::shared_ptr<hiai::AiTensor> &tensor, int start = 0, int end = -1) {
    const int printNumOnEachLine = 16;
    auto size = tensor->GetSize();
    int num = size / sizeof(T);
    auto ptr = (T *) tensor->GetBuffer();
    if (end < 0) {
        end += num;
    }
    std::cout << "Print tensor " << tensor << " size: " << size << " num: " << num;
    for (int i = start; i < std::min(num, end); i++) {
        if (i % printNumOnEachLine == 0) {
            std::cout << std::endl
                      << "[" << std::setfill('0') << std::setw(5) << i << ", "
                      << std::setfill('0') << std::setw(5)
                      << std::min(i + printNumOnEachLine - 1, end) << "]";
        }
        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3) << (float) ptr[i] << " ";
    }
    std::cout << std::endl;
}

template<typename T>
void FillTensorWithData(std::shared_ptr<hiai::AiTensor> &tensor, const std::vector<T> &data) {
    auto size = tensor->GetSize();
    auto num = size / sizeof(T);
    if (num != data.size()) {
        ALOGE("tensor size(%lu) != data.size(%lu)\n", num, data.size());
        return;
    }
    (void) memcpy(tensor->GetBuffer(), data.data(), data.size() * sizeof(T));
}

template<typename T>
void FillTensorWithData(std::shared_ptr<hiai::AiTensor> &tensor) {
    std::default_random_engine engine(time(nullptr));
    std::uniform_real_distribution<float> uniform(-1, 1);
    auto size = tensor->GetSize();
    ALOGE("fill tensor size: %d\n", size);
    auto num = size / sizeof(T);
    std::vector<T> data(num, 0);
    for (int i = 0; i < num; i++) {
        data[i] = uniform(engine);
    }
    FillTensorWithData<T>(tensor, data);
}

template<typename T>
void FillTensorFromFile(std::shared_ptr<hiai::AiTensor> &tensor, const std::string &path) {
    std::vector<T> value;
    if (!ReadFile<T>(path, &value)) {
        ALOGE("%s open failed.\n", path.c_str());
    }
    return FillTensorWithData<T>(tensor, value);
}


bool RunModel(const std::shared_ptr<hiai::AiModelMngerClient> &client, const std::string &modelName,
              std::vector<std::shared_ptr<hiai::AiTensor>> *inputTensors,
              std::vector<std::shared_ptr<hiai::AiTensor>> *outputTensors,
              int repeats = 1, float sleepMSAfterProcess = 0) {
    hiai::AiContext context;
    string key = "model_name";
    const string &value = modelName;
    context.AddPara(key, value);

    vector<double> inferenceTime(repeats, 0);
    struct timeval tpstart, tpend;
    int istamp;
    struct timeval delay;
    delay.tv_sec = 0;
    delay.tv_usec = sleepMSAfterProcess * 1000; // ms

    for (int i = 0; i < repeats; i++) {
        gettimeofday(&tpstart, nullptr);
        int retCode = client->Process(context, *inputTensors, *outputTensors, 1000, istamp);
        if (retCode) {
            ALOGE("Run model failed. retCode=%d\n", retCode);
            return false;
        }
        gettimeofday(&tpend, nullptr);
        double timeUse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
        inferenceTime[i] = timeUse;
        if (sleepMSAfterProcess > 0) {
            select(0, nullptr, nullptr, nullptr, &delay);
        }
    }

    ALOGI("Show inference time start:\n");
    double count = 0;
    double min = 99999;
    double max = 0;
    for (int i = 0; i < inferenceTime.size(); ++i) {
        inferenceTime[i] /= 1000;
        count += inferenceTime[i];
        if (inferenceTime[i] > max) {
            max = inferenceTime[i];
        } else if (inferenceTime[i] < min) {
            min = inferenceTime[i];
        }
        ALOGI("index: %d, time: %.3f\n", i, inferenceTime[i]);
    }
    ALOGI("[total] %lu, [avg] %.3f, [max] %.3f, [min] %.3f\nShow inference time end\n ", inferenceTime.size(),
          count / inferenceTime.size(), max, min);
    return true;
}

std::shared_ptr<hiai::AiModelMngerClient> Build(
        const std::string &modelName,
        ge::Model &irModel,
        std::vector<std::shared_ptr<hiai::AiTensor>> *inputTensors,
        std::vector<std::shared_ptr<hiai::AiTensor>> *outputTensors) {
    domi::HiaiIrBuild irBuild;
    domi::ModelBufferData omModelBuf;

    ge::Buffer buffer;
    irModel.Save(buffer);
    WriteFile(buffer.GetData(), buffer.GetSize(), modelName + ".irpb");
    if (!irBuild.CreateModelBuff(irModel, omModelBuf)) {
        ALOGE("ERROR: build alloc om failed.\n");
        return nullptr;
    }
    if (!irBuild.BuildIRModel(irModel, omModelBuf)) {
        irBuild.ReleaseModelBuff(omModelBuf);
        ALOGE("ERROR: build ir model failed.\n");
        return nullptr;
    }
    auto client = std::make_shared<hiai::AiModelMngerClient>();
    int retCode = client->Init(nullptr);
    if (retCode != hiai::AI_SUCCESS) {
        ALOGE("ERROR: build init hiai::AiModelManagerClient failed(retCode=%d)\n", retCode);
        return nullptr;
    }
    auto modelDesc = std::make_shared<hiai::AiModelDescription>(modelName, 3, 0, 0, 0);
    modelDesc->SetModelBuffer(omModelBuf.data, omModelBuf.length);
    std::vector<std::shared_ptr<hiai::AiModelDescription>> modelDescs;
    modelDescs.push_back(modelDesc);
    retCode = client->Load(modelDescs);
    if (retCode != 0) {
        ALOGE("ERROR: hiai::AiModelMngerClient load model failed.\n");
        return nullptr;
    }
    std::vector<hiai::TensorDimension> inputDims;
    std::vector<hiai::TensorDimension> outputDims;
    retCode = client->GetModelIOTensorDim(modelName, inputDims, outputDims);
    if (retCode != 0) {
        ALOGE("ERROR: get IO tensor failed retCode=%d.\n", retCode);
        return nullptr;
    }
    inputTensors->clear();
    outputTensors->clear();
    for (int i = 0; i < inputDims.size(); i++) {
        std::shared_ptr<hiai::AiTensor> inputTensor = std::make_shared<hiai::AiTensor>();
        inputTensor->Init(&inputDims[i]);
        inputTensors->push_back(inputTensor);
        PrintTensorInfo("input_tensor_" + std::to_string(i), inputTensor);
    }
    for (int i = 0; i < outputDims.size(); i++) {
        std::shared_ptr<hiai::AiTensor> outputTensor = std::make_shared<hiai::AiTensor>();
        outputTensor->Init(&outputDims[i]);
        outputTensors->push_back(outputTensor);
        PrintTensorInfo("output_tensor_" + std::to_string(i), outputTensor);
    }
    if (client != nullptr) {
        if (!WriteFile(omModelBuf.data, omModelBuf.length, modelName)) {
            ALOGE("ERROR: save om model failed.\n");
        }
    }
    irBuild.ReleaseModelBuff(omModelBuf);
    return client;
}

#endif //BUILD_IR_MODEL_TEST_UTIL_H
