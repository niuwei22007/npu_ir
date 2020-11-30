#include "test_util.h"
#include "check.h"

using namespace std;
using namespace test_case;
using namespace test_util;
using namespace om_model;
namespace test_case {
bool CheckResult() {
    return true;
}

void Test(const TestCase& test) {
    cout << "============= CaseName: " << test.caseName << endl;
    std::vector<std::string> names{test.caseName};
    std::vector<std::string> modelPaths{test.caseName + ".om"};
    VecVecAiTensor modelsInputs;
    VecVecAiTensor modelsOutputs;
    std::vector<bool> useAipps{false};

    auto client = LoadModelSync(names, modelPaths, modelsInputs, modelsOutputs, useAipps);
    if (client == nullptr) {
        cerr << "ERROR: Load " << test.caseName << " failed." << endl;
        return;
    }
    if (test.inputFromFile) {
        FillTensorFromFile<float>(modelsInputs[0][0], test.caseName + ".bin");
    } else {
        FillTensorWithData<float>(modelsInputs[0][0]);
    }
    if (Process(client, names[0], modelsInputs[0], modelsOutputs[0]) != SUCCESS) {
        cerr << "ERROR: run " << names[0] << " failed." << endl;
        return;
    }
    PrintTensorData<float>(modelsInputs[0][0], 0, 32);
    int i = 0;
    for (const shared_ptr<hiai::AiTensor>& tensor : modelsOutputs[0]) {
        PrintTensorData<float>(tensor, 0, 32);
        SaveTensorData<float>(tensor, "/data/local/tmp/output/output_" + to_string(i++) + ".bin");
    }
    cout << "-------------" << test.caseName << " -------- " << CheckResult() << endl;
}
}

int main(int argc, char* argv[]) {
    ALOGE("=========== RUN TestCase ===========\n");
    TestCase caseList[] = {
        {"sqrt_ir", nullptr, false},
    };
    for (const TestCase& tc : caseList) {
        Test(tc);
    }
    ALOGE("=========== ALL DONE ===========\n");
    return 0;
}

