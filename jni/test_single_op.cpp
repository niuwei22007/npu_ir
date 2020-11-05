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
//    int i = 0;
//    for (const shared_ptr<hiai::AiTensor> &tensor : outputTensors) {
//        PrintTensorData<float>(tensor, 0, 32);
//        SaveTensorData<float>(tensor, "/data/local/tmp/output/output_" + to_string(i++) + ".bin");
//    }
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
}

using namespace testcase;

int main(int argc, char *argv[]) {
    TestCase caseList[] = {
            {"bytenn_arch29", BuildSqrtGraph, false},
    };
    for (const TestCase &tc : caseList) {
        Test(tc);
    }
    ALOGI("=========== ALL DONE ===========\n");
    return 0;
}

