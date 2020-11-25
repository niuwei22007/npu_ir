#include <istream>
#include <sys/system_properties.h>
#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <sstream>
#include <unordered_map>

using namespace std;
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

int main(int argc, char* argv[]) {
    bool supported = SupportResizeBilinearHalfPixel();
    cout << "This device"
         << (supported ? " " : " not ")
         << "support high performance ResizeBilinear with half_pixel!"
         << endl;
}