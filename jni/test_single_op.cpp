#include "test_util.h"

using namespace std;
namespace testcase {
bool CheckResult() {
    return true;
}

void Test(const TestCase &test) {
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
    for (const shared_ptr<hiai::AiTensor> &tensor : outputTensors) {
        PrintTensorData<float>(tensor, 0, 32);
        SaveTensorData<float>(tensor, "/data/local/tmp/output/output_" + to_string(i++) + ".bin");
    }
    cout << "-------------" << test.caseName << " -------- " << CheckResult() << endl;
}

bool BuildSqrtGraph(ge::Graph &graph) {
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

bool BuildConvTransposeGraph(ge::Graph &graph) {
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
                (int32_t) inputDesc.GetShape().GetDim(0),
                (int32_t) inputDesc.GetShape().GetDim(1),
                (int32_t) inputDesc.GetShape().GetDim(2) * 2,
                (int32_t) inputDesc.GetShape().GetDim(3) * 2,
        };
        SetConstData(outputShape, outDesc, (uint8_t *) outShapeValue.data(), outShapeValue.size() * sizeof(uint32_t));

        hiai::Shape filterShape = ge::Shape({8, 1, 4, 4});
        hiai::Shape biasShape = ge::Shape({1, 1, 1, 1});

        hiai::TensorDesc convWeightDesc(filterShape, ge::FORMAT_NCHW, ge::DT_FLOAT);
        vector<float> convWeightValue(Prod(filterShape.GetDims()), 1);
        SetConstData(filter, convWeightDesc, (uint8_t *) convWeightValue.data(),
                     convWeightValue.size() * sizeof(float));

        hiai::TensorDesc convBiasDesc(biasShape, ge::FORMAT_NCHW, ge::DT_FLOAT);
        vector<float> convBiasValue(Prod(biasShape.GetDims()), 1);
        SetConstData(bias, convBiasDesc, (uint8_t *) convBiasValue.data(), convBiasValue.size() * sizeof(float));

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
}

using namespace testcase;

int main(int argc, char *argv[]) {
    TestCase caseList[] = {
//            {"sqrt_ir",          BuildSqrtGraph,          false},
            {"convtranspose_ir", BuildConvTransposeGraph, false},
    };
    for (const TestCase &tc : caseList) {
        Test(tc);
    }
    ALOGI("=========== ALL DONE ===========\n");
    return 0;
}

