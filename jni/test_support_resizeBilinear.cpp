#include <istream>
#include <sys/system_properties.h>
#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <sstream>
#include <unordered_map>

using namespace std;
namespace hiai_check {
static const int VERSION_LEN = 4;
static unordered_map<string, array<int, VERSION_LEN>> g_supportedEMUI{
    {"TAS", {11, 0, 0, 145}}, // M30
    {"LIO", {11, 0, 0, 145}}, // M30 P
    {"ANA", {11, 0, 0, 145}}, // P40
    {"ELS", {11, 0, 0, 145}}, // P40 P
};

static array<char, PROP_VALUE_MAX> g_board{0};
static array<char, PROP_VALUE_MAX> g_baseVersion{0};
static array<char, PROP_VALUE_MAX> g_hiaiversion{0};

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

bool GetHiaiVersion() {
    if (g_hiaiversion[0] == 0) {
        if (__system_property_get("ro.vendor.hiaiversion", g_hiaiversion.data()) <= 0) {
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
    // 剩余字符串应该只有1个空格
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

vector<string> Split(const string& s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

bool Check() {
    if (!GetHiaiVersion()) {
        cout << "Get HiaiVersion failed." << endl;
        return false;
    }
    string hiaiversion(g_hiaiversion.data());
    vector<string> versionSplit = Split(hiaiversion, '.');
    bool torchSupport = false;
    // 若100.320开头，则
    //     若第3段 == 010，则第4段需要 ≥ 024
    //     若第3段 == 011，则第4段需要 ≥ 020
    //     若第3段 == 012，则第4段需要 ≥ 012
    // 若100.330或100.500开头，则只需要判断第4段需要 ≥ 012
    // 其他是不含NPU的场景。解释：含NPU的版本号是100.320、100.330、100.500开头的
    if (versionSplit[0].compare("100") == 0 && versionSplit[1].compare("320") == 0) {
        if (versionSplit[2].compare("010") == 0) {
            if (versionSplit[3].compare("024") >= 0) {
                torchSupport = true;
            }
        } else if (versionSplit[2].compare("011") == 0) {
            if (versionSplit[3].compare("020") >= 0) {
                torchSupport = true;
            }
        } else if (versionSplit[2].compare("012") == 0) {
            if (versionSplit[3].compare("012") >= 0) {
                torchSupport = true;
            }
        }
    } else if ((versionSplit[0].compare("100") == 0 && versionSplit[1].compare("330") == 0) ||
               (versionSplit[0].compare("100") == 0 && versionSplit[1].compare("500") == 0)) {
        if (versionSplit[3].compare("012") >= 0) {
            torchSupport = true;
        }
    }
    if (!torchSupport) {
        //for another try for specific EMUI
        torchSupport = SupportResizeBilinearHalfPixel();
    }
    return torchSupport;
}
}

// int main(int argc, char* argv[]) {
//     bool supported = hiaicheck::Check();
//     cout << "This device"
//          << (supported ? " " : " not ")
//          << "support high performance ResizeBilinear with half_pixel!"
//          << endl;
// }