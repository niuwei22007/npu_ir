#include <istream>
#include <sys/system_properties.h>
#include <string>
#include <vector>
#include <array>
#include <iostream>

static std::vector<std::string> g_npuList{"kirin990", "kirin810", "kirin820", "kirin985", "kirin9000"};
static std::array<char, PROP_VALUE_MAX> g_bufVersion{0};

bool GetCpuInfo() {
    if (g_bufVersion[0] == 0) {
        if (__system_property_get("ro.product.vendor.device", g_bufVersion.data()) <= 0) {
            return false;
        }
    }
    return true;
}

bool ContainsNpu() {
    if (!GetCpuInfo()) {
        return false;
    }
    std::string version(g_bufVersion.data());
    for (const auto &npu: g_npuList) {
        if (version == npu) {
            return true;
        }
    }
    return false;
}

int main(int argc, char *argv[]) {
    bool contains = ContainsNpu();
    if (contains) {
        std::cout << "This device contains NPU" << std::endl;
    } else {
        std::cout << "This device not contains NPU" << std::endl;
    }
}