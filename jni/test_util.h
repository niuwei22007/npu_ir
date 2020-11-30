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
#include <unordered_map>
#include <array>
#include <sstream>
#include <android/log.h>
#include <sys/system_properties.h>

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

static const int SUCCESS = 0;
static const int FAILED = -1;

namespace test_case {
typedef bool(* TestFunc)(ge::Graph& graph);

struct TestCase {
    std::string caseName;
    TestFunc func;
    bool inputFromFile;
    uint32_t inputH;
    uint32_t inputW;

    TestCase(const string& name, TestFunc f, bool input, uint32_t h = 0, uint32_t w = 0) {
        this->caseName = name;
        this->func = f;
        this->inputFromFile = input;
        this->inputH = h;
        this->inputW = w;
    }
};
}

namespace test_util {
bool WriteFile(const void* data, size_t size, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        ALOGE("%s open failed.\n", path.c_str());
        return false;
    }
    file.write((const char*)data, size);
    file.flush();
    file.close();
    return true;
}

template<typename T>
bool ReadFile(const std::string& path, std::vector<T>* data) {
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
    file.read((char*)data->data(), size);
    file.close();
    return true;
}

void PrintTensorInfo(const string& msg, const std::shared_ptr<hiai::AiTensor>& tensor) {
    auto dims = tensor->GetTensorDimension();
    std::cout << msg << " NCHW: " << dims.GetNumber() << ", " << dims.GetChannel() << ", " << dims.GetHeight() << ", "
              << dims.GetWidth() << std::endl;
}

template<typename T>
void SaveTensorData(const std::shared_ptr<hiai::AiTensor>& tensor, const std::string& path) {
    auto size = tensor->GetSize();
    auto num = size / sizeof(T);
    auto ptr = (T*)tensor->GetBuffer();
    WriteFile(ptr, size, path);
    ALOGI("save tensor size: %u, num: %lu\n", size, num);
}

template<typename T>
void PrintTensorData(const std::shared_ptr<hiai::AiTensor>& tensor, int start = 0, int end = -1) {
    const int printNumOnEachLine = 16;
    auto size = tensor->GetSize();
    int num = size / sizeof(T);
    auto ptr = (T*)tensor->GetBuffer();
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
        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3) << (float)ptr[i] << " ";
    }
    std::cout << std::endl;
}

template<typename T>
void FillTensorWithData(std::shared_ptr<hiai::AiTensor>& tensor, const std::vector<T>& data) {
    auto size = tensor->GetSize();
    auto num = size / sizeof(T);
    if (num != data.size()) {
        ALOGE("tensor size(%lu) != data.size(%lu)\n", num, data.size());
        return;
    }
    (void)memcpy(tensor->GetBuffer(), data.data(), data.size() * sizeof(T));
}

template<typename T>
void FillTensorWithData(std::shared_ptr<hiai::AiTensor>& tensor) {
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
void FillTensorFromFile(std::shared_ptr<hiai::AiTensor>& tensor, const std::string& path) {
    std::vector<T> value;
    if (!ReadFile<T>(path, &value)) {
        ALOGE("%s open failed.\n", path.c_str());
    }
    return FillTensorWithData<T>(tensor, value);
}

int Prod(const std::vector<int64_t>& list) {
    int prod = 1;
    for (auto s: list) {
        prod *= s;
    }
    return prod;
}

void SetConstData(hiai::op::Const& constOp, const hiai::TensorDesc& wDesc, uint8_t* data, size_t dataSize) {
    hiai::TensorPtr weight = std::make_shared<hiai::Tensor>();
    weight->SetTensorDesc(wDesc);
    weight->SetData(data, dataSize);
    constOp.set_attr_value(weight);
}

void SetConvTranspose(string& name, hiai::op::ConvTranspose& deconv, hiai::op::Const& filter,
                      const hiai::Shape& filterShape, hiai::op::Const& bias, const hiai::Shape& biasShape) {
    hiai::TensorDesc convWeightDesc(filterShape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    vector<float> convWeightValue(Prod(filterShape.GetDims()), 1);
    SetConstData(filter, convWeightDesc, (uint8_t*)convWeightValue.data(), convWeightValue.size() * sizeof(float));

    hiai::TensorDesc convBiasDesc(biasShape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    vector<float> convBiasValue(Prod(biasShape.GetDims()), 1);
    SetConstData(bias, convBiasDesc, (uint8_t*)convBiasValue.data(), convBiasValue.size() * sizeof(float));

    deconv.set_input_filter(filter)
        .set_input_bias(bias)
        .set_attr_dilations({1, 1})
        .set_attr_strides({2, 2})
        .set_attr_groups(1)
        .set_attr_pad_mode("SPECIFIC")
        .set_attr_pads({0, 0, 0, 0});
}
}

namespace ir_model {
bool RunModel(const std::shared_ptr<hiai::AiModelMngerClient>& client,
              const std::string& modelName,
              std::vector<std::shared_ptr<hiai::AiTensor>>* inputTensors,
              std::vector<std::shared_ptr<hiai::AiTensor>>* outputTensors,
              int repeats = 1, float sleepMSAfterProcess = 0) {
    hiai::AiContext context;
    string key = "model_name";
    const string& value = modelName;
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
        }
        if (inferenceTime[i] < min) {
            min = inferenceTime[i];
        }
        ALOGI("index: %d, time: %.3f ms\n", i, inferenceTime[i]);
    }
    ALOGI("[total] %lu, [avg] %.3f ms, [max] %.3f ms, [min] %.3f ms\nShow inference time end.\n ", inferenceTime.size(),
          count / inferenceTime.size(), max, min);
    return true;
}

std::shared_ptr<hiai::AiModelMngerClient> Build(
    const std::string& modelName,
    ge::Model& irModel,
    std::vector<std::shared_ptr<hiai::AiTensor>>* inputTensors,
    std::vector<std::shared_ptr<hiai::AiTensor>>* outputTensors) {
    domi::HiaiIrBuild irBuild;
    domi::ModelBufferData omModelBuf;

    ge::Buffer buffer;
    irModel.Save(buffer);
    test_util::WriteFile(buffer.GetData(), buffer.GetSize(), modelName + ".irpb");
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
        test_util::PrintTensorInfo("input_tensor_" + std::to_string(i), inputTensor);
    }
    for (int i = 0; i < outputDims.size(); i++) {
        std::shared_ptr<hiai::AiTensor> outputTensor = std::make_shared<hiai::AiTensor>();
        outputTensor->Init(&outputDims[i]);
        outputTensors->push_back(outputTensor);
        test_util::PrintTensorInfo("output_tensor_" + std::to_string(i), outputTensor);
    }
    if (client != nullptr) {
        if (!test_util::WriteFile(omModelBuf.data, omModelBuf.length, modelName)) {
            ALOGE("ERROR: save om model failed.\n");
        }
    }
    irBuild.ReleaseModelBuff(omModelBuf);
    return client;
}
}

namespace om_model {
static std::map<std::string, int> g_modelNameMap;
using VecAiTensor = std::vector<std::shared_ptr<hiai::AiTensor>>;
using VecTensorDim = std::vector<hiai::TensorDimension>;
using VecVecAiTensor = std::vector<VecAiTensor>;

int UpdateTensorVec(std::string& modelName, VecVecAiTensor& modelTensors, VecTensorDim& modelDims, bool useAipp) {
    VecAiTensor tensors;
    int ret;
    for (const auto& inDim : modelDims) {
        std::shared_ptr<hiai::AiTensor> input = std::make_shared<hiai::AiTensor>();
        if (useAipp) {
            ret = input->Init(inDim.GetNumber(), inDim.GetHeight(), inDim.GetWidth(), hiai::AiTensorImage_YUV420SP_U8);
            ALOGI("[HIAI_DEMO_SYNC] model %s uses AIPP(input).", modelName.c_str());
        } else {
            ret = input->Init(&inDim);
            ALOGI("[HIAI_DEMO_SYNC] model %s does not use AIPP(input).", modelName.c_str());
        }
        if (ret != SUCCESS) {
            ALOGE("[HIAI_DEMO_SYNC] model %s AiTensor Init failed(input).", modelName.c_str());
            return FAILED;
        }
        tensors.push_back(input);
    }
    modelTensors.push_back(tensors);
    if (modelTensors.empty()) {
        ALOGE("[HIAI_DEMO_SYNC] input_tensor is empty");
        return FAILED;
    }
    return SUCCESS;
}

void ResourceDestroy(std::shared_ptr<hiai::AiModelBuilder>& modelBuilder, std::vector<hiai::MemBuffer*>& memBuffers) {
    if (modelBuilder == nullptr) {
        ALOGE("[HIAI_DEMO_SYNC] modelBuilder is null.");
        return;
    }

    for (auto tmpBuffer : memBuffers) {
        modelBuilder->MemBufferDestroy(tmpBuffer);
    }
}

int LoadSync(std::vector<std::string>& names,
             std::vector<std::string>& modelPaths,
             std::shared_ptr<hiai::AiModelMngerClient>& client) {
    std::vector<std::shared_ptr<hiai::AiModelDescription>> modelDescs;
    std::vector<hiai::MemBuffer*> memBuffers;
    std::shared_ptr<hiai::AiModelBuilder> modelBuilder = std::make_shared<hiai::AiModelBuilder>(client);
    if (modelBuilder == nullptr) {
        ALOGI("[HIAI_DEMO_SYNC] creat modelBuilder failed.");
        return FAILED;
    }

    for (size_t i = 0; i < modelPaths.size(); ++i) {
        std::string modelPath = modelPaths[i];
        std::string modelName = names[i];
        g_modelNameMap[modelName] = i;

        // We can achieve the optimization by loading model from OM file.
        ALOGI("[HIAI_DEMO_SYNC] modelpath is %s\n.", modelPath.c_str());
        hiai::MemBuffer* buffer = modelBuilder->InputMemBufferCreate(modelPath);
        if (buffer == nullptr) {
            ALOGE("[HIAI_DEMO_SYNC] cannot find the model file.");
            return FAILED;
        }
        memBuffers.push_back(buffer);

        std::string modelNameFull = std::string(modelName) + std::string(".om");
        std::shared_ptr<hiai::AiModelDescription> desc =
            std::make_shared<hiai::AiModelDescription>(modelNameFull,
                                                       hiai::AiModelDescription_Frequency_HIGH,
                                                       hiai::HIAI_FRAMEWORK_NONE,
                                                       hiai::HIAI_MODELTYPE_ONLINE,
                                                       hiai::AiModelDescription_DeviceType_NPU);
        if (desc == nullptr) {
            ALOGE("[HIAI_DEMO_SYNC] LoadModelSync: desc make_shared error.");
            ResourceDestroy(modelBuilder, memBuffers);
            return FAILED;
        }
        desc->SetModelBuffer(buffer->GetMemBufferData(), buffer->GetMemBufferSize());

        ALOGE("[HIAI_DEMO_SYNC] loadModel %s IO Tensor.", desc->GetName().c_str());
        modelDescs.push_back(desc);
    }

    int ret = client->Load(modelDescs);
    ResourceDestroy(modelBuilder, memBuffers);
    if (ret != SUCCESS) {
        ALOGE("[HIAI_DEMO_SYNC] Model Load Failed.");
        return FAILED;
    }
    return SUCCESS;
}

std::shared_ptr<hiai::AiModelMngerClient> LoadModelSync(std::vector<std::string>& names,
                                                        std::vector<std::string>& modelPaths,
                                                        VecVecAiTensor& modelsInputs,
                                                        VecVecAiTensor& modelsOutputs,
                                                        std::vector<bool>& aipps) {
    std::shared_ptr<hiai::AiModelMngerClient> clientSync = std::make_shared<hiai::AiModelMngerClient>();
    if (clientSync == nullptr) {
        ALOGE("[HIAI_DEMO_SYNC] Model Manager Client make_shared error.");
        return nullptr;
    }
    int ret = clientSync->Init(nullptr);
    if (ret != SUCCESS) {
        ALOGE("[HIAI_DEMO_SYNC] Model Manager Init Failed.");
        return nullptr;
    }
    ret = LoadSync(names, modelPaths, clientSync);
    if (ret != SUCCESS) {
        ALOGE("[HIAI_DEMO_ASYNC] LoadSync Failed.");
        return nullptr;
    }

    modelsInputs.clear();
    modelsOutputs.clear();
    for (size_t i = 0; i < names.size(); ++i) {
        std::string modelName = names[i];
        bool isUseAipp = !aipps.empty() && aipps[i];
        ALOGI("[HIAI_DEMO_SYNC] Get model %s IO Tensor. Use AIPP %d", modelName.c_str(), isUseAipp);
        std::vector<hiai::TensorDimension> inputDims, outputDims;
        ret = clientSync->GetModelIOTensorDim(std::string(modelName) + std::string(".om"), inputDims, outputDims);
        if (ret != SUCCESS) {
            ALOGE("[HIAI_DEMO_SYNC] Get Model IO Tensor Dimension failed,ret is %d.", ret);
            return nullptr;
        }
        if (inputDims.empty()) {
            ALOGE("[HIAI_DEMO_SYNC] inputDims is empty.");
            return nullptr;
        }
        if (UpdateTensorVec(modelName, modelsInputs, inputDims, isUseAipp) != SUCCESS) {
            return nullptr;
        }
        if (UpdateTensorVec(modelName, modelsOutputs, outputDims, false) != SUCCESS) {
            return nullptr;
        }
    }
    return clientSync;
}

hiai::AIStatus Process(std::shared_ptr<hiai::AiModelMngerClient>& client,
                       std::string& name,
                       VecAiTensor& inputs,
                       VecAiTensor& outputs) {
    hiai::AiContext context;
    std::string key = "model_name";
    std::string value = name;
    value += ".om";
    context.AddPara(key, value);
    ALOGI("[HIAI_DEMO_SYNC] runModel modelname: %s", value.c_str());
    // before process
    struct timeval tpstart, tpend;
    gettimeofday(&tpstart, nullptr);
    int istamp;
    int ret = client->Process(context, inputs, outputs, 1000, istamp);
    if (ret != SUCCESS) {
        ALOGE("[HIAI_DEMO_SYNC] Runmodel Failed!, ret=%d\n", ret);
        return ret;
    }
    // after process
    gettimeofday(&tpend, nullptr);
    float timeUse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
    ALOGI("[HIAI_DEMO_SYNC] inference time %f ms.\n", timeUse / 1000);
    return hiai::AI_SUCCESS;
}
}

#endif //BUILD_IR_MODEL_TEST_UTIL_H
