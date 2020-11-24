#include "test_util.h"

using namespace std;
namespace testcase {
bool CheckResult() {
    return true;
}

void Test(const TestCase& test) {
    string modelName = "/data/local/tmp/output/" + test.caseName + ".om";
    ge::Graph irGraph = ge::Graph("ir_graph");
    cout << "============= CaseName: " << test.caseName << endl;
    test.func(irGraph);
    vector<shared_ptr<hiai::AiTensor>> inputTensors;
    vector<shared_ptr<hiai::AiTensor>> outputTensors;

    ge::Model irModel("model", modelName);
    irModel.SetGraph(irGraph);
    auto client = Build(modelName, irModel, &inputTensors, &outputTensors);
    if (client == nullptr) {
        cerr << "ERROR: build " << modelName << " failed." << endl;
        return;
    }
    if (test.inputFromFile) {
        FillTensorFromFile<float>(inputTensors[0], test.caseName + ".bin");
    } else {
        FillTensorWithData<float>(inputTensors[0]);
    }
    if (!RunModel(client, modelName, &inputTensors, &outputTensors)) {
        cerr << "ERROR: run " << modelName << " failed." << endl;
        return;
    }
    PrintTensorData<float>(inputTensors[0], 0, 32);
    int i = 0;
    for (const shared_ptr<hiai::AiTensor>& tensor : outputTensors) {
        PrintTensorData<float>(tensor, 0, 32);
        SaveTensorData<float>(tensor, "/data/local/tmp/output/output_" + to_string(i++) + ".bin");
    }
    cout << "-------------" << test.caseName << " -------- " << CheckResult() << endl;
}

bool BuildSqrtGraph(ge::Graph& graph) {
    auto data = ge::op::Data("data");
    ge::TensorDesc inputDesc(ge::Shape({16, 512, 1, 1}));
    data.update_input_desc_x(inputDesc);

    auto sqrt = ge::op::Sqrt("sqrt")
        .set_input_x(data);

    vector<ge::Operator> inputs;
    inputs.push_back(data);
    vector<ge::Operator> outputs;
    outputs.push_back(sqrt);
    graph.SetInputs(inputs).SetOutputs(outputs);
    return true;
}

bool BuildConvTransposeGraph(ge::Graph& graph) {
    auto data = ge::op::Data("data");
    ge::TensorDesc inputDesc(ge::Shape({1, 8, 864, 480}));
    data.update_input_desc_x(inputDesc);

    string deconvName = "deconvolution";
    auto deconvOp = hiai::op::ConvTranspose(deconvName);
    auto filter = hiai::op::Const(deconvName + "_filter");
    auto bias = hiai::op::Const(deconvName + "_bias");
    auto outputShape = hiai::op::Const(deconvName + "_output");
    {
        hiai::TensorDesc outDesc(ge::Shape({4}), ge::FORMAT_NCHW, ge::DT_INT32);
        std::vector<int32_t> outShapeValue{
            (int32_t)inputDesc.GetShape().GetDim(0),
            (int32_t)inputDesc.GetShape().GetDim(1),
            (int32_t)inputDesc.GetShape().GetDim(2) * 2,
            (int32_t)inputDesc.GetShape().GetDim(3) * 2,
        };
        SetConstData(outputShape, outDesc, (uint8_t*)outShapeValue.data(), outShapeValue.size() * sizeof(uint32_t));

        hiai::Shape filterShape = ge::Shape({8, 1, 4, 4});
        hiai::Shape biasShape = ge::Shape({1, 1, 1, 1});

        hiai::TensorDesc convWeightDesc(filterShape, ge::FORMAT_NCHW, ge::DT_FLOAT);
        vector<float> convWeightValue(Prod(filterShape.GetDims()), 1);
        SetConstData(filter, convWeightDesc, (uint8_t*)convWeightValue.data(),
                     convWeightValue.size() * sizeof(float));

        hiai::TensorDesc convBiasDesc(biasShape, ge::FORMAT_NCHW, ge::DT_FLOAT);
        vector<float> convBiasValue(Prod(biasShape.GetDims()), 1);
        SetConstData(bias, convBiasDesc, (uint8_t*)convBiasValue.data(), convBiasValue.size() * sizeof(float));

        deconvOp.set_input_output_shape(outputShape)
            .set_input_filter(filter)
            .set_input_x(data)
            .set_attr_dilations({1, 1})
            .set_attr_strides({2, 2})
            .set_attr_groups(1)
            .set_attr_pad_mode("SAME")
            .set_attr_pads({0, 0, 0, 0});
    }

    std::vector<ge::Operator> inputs{data};
    std::vector<ge::Operator> outputs{deconvOp};
    graph.SetInputs(inputs).SetOutputs(outputs);
    return true;
}

bool BuildResizeBilinearGraph(ge::Graph& graph) {
    auto data = ge::op::Data("data");
    ge::TensorDesc inputDesc(ge::Shape({1, 32, 192, 192}));
    data.update_input_desc_x(inputDesc);

    auto outputShape = hiai::op::Const("resize_v2_output");
    {
        hiai::TensorDesc outDesc(ge::Shape({2}), ge::FORMAT_NCHW, ge::DT_INT32);
        std::vector<int32_t> outShapeValue{384, 384};
        SetConstData(outputShape, outDesc, (uint8_t*)outShapeValue.data(), outShapeValue.size() * sizeof(uint32_t));
    }
    auto resize = hiai::op::ResizeBilinearV2("resize_v2")
        .set_input_x(data)
        .set_input_size(outputShape)
        .set_attr_align_corners(false)
        .set_attr_half_pixel_centers(true);

    std::vector<ge::Operator> inputs{data};
    std::vector<ge::Operator> outputs{resize};
    graph.SetInputs(inputs).SetOutputs(outputs);
    return true;
}
}

namespace testcheck {
static const int VERSION_LEN = 4;
static unordered_map<string, array<int, VERSION_LEN>> g_supportedEMUI{
    {"TAS", {11, 0, 0, 145}}, // M30
    {"LIO", {11, 0, 0, 145}}, // M30 P
    {"ANA", {11, 0, 0, 145}}, // P40
    {"ELS", {11, 0, 0, 145}}, // P40 P
};

static array<char, PROP_VALUE_MAX> g_board{0};
static array<char, PROP_VALUE_MAX> g_baseVersion{0};

void ShowInfo() {
    string board(g_board.data());
    string baseVersion(g_baseVersion.data());
    cout << "Board=" << board << endl;
    cout << "BaseVersion=" << baseVersion << endl;
}

bool GetEMUIInfo() {
    if (g_board[0] == 0) {
        if (__system_property_get("ro.product.board", g_board.data()) <= 0) {
            return false;
        }
    }
    if (g_baseVersion[0] == 0) {
        if (__system_property_get("persist.sys.hiview.base_version", g_baseVersion.data()) <= 0) {
            return false;
        }
    }
    return true;
}

bool CheckResizeBilinearHalfPixel(const string& board, string& baseVersion) {
    if (g_supportedEMUI.count(board) == 0) {
        return false;
    }
    // 去掉首尾空格
    baseVersion.erase(0, baseVersion.find_first_not_of(' '));
    baseVersion.erase(baseVersion.find_last_not_of(' ') + 1);
    // 剩余字符串应该只有1个
    int loc = baseVersion.find_first_of(' ');
    if (loc < 0 || loc != baseVersion.find_last_of(' ')) {
        return false;
    }
    // 去掉空格之前的
    istringstream emuiStream(baseVersion.substr(loc + 1));
    array<int, VERSION_LEN> version{0};
    array<char, VERSION_LEN - 1> point{0};
    size_t index = 0;
    for (; index < point.size(); index++) {
        emuiStream >> version[index] >> point[index];
        if (version[index] < g_supportedEMUI[board][index]) {
            return false;
        }
        if (point[index] != '.') {
            return false;
        }
    }
    emuiStream >> version[index];
    return version[index] >= g_supportedEMUI[board][index];
}

bool SupportResizeBilinearHalfPixel() {
    if (!GetEMUIInfo()) {
        return false;
    }
    string board(g_board.data());
    string baseVersion(g_baseVersion.data());
    return CheckResizeBilinearHalfPixel(board, baseVersion);
}
}

using namespace testcase;

int main(int argc, char* argv[]) {
    ALOGE("=========== RUN TestCase ===========\n");
    TestCase caseList[] = {
        // {"sqrt_ir",             BuildSqrtGraph,           false},
        // {"convtranspose_ir",    BuildConvTransposeGraph,  false},
        {"resizebilinearv2_ir", BuildResizeBilinearGraph, false},
    };
    for (const TestCase& tc : caseList) {
        Test(tc);
    }
    ALOGE("=========== RUN Check ===========\n");
    bool supportResize = testcheck::SupportResizeBilinearHalfPixel();
    ALOGE("Current Device %s support high performance ResizeBilinear with half_pixel!",
          (supportResize ? "" : "not"));
    ALOGE("=========== ALL DONE ===========\n");
    return 0;
}

