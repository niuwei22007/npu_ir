#include <istream>
#include <sys/system_properties.h>
#include <string>
#include <vector>
#include <array>
#include <iostream>
#include "EGL/egl.h"
#include "EGL/eglext.h"
#include <native_handle.h>
#include "dlfcn.h"
#include "android/hardware_buffer.h"
#include "asm-generic/mman-common.h"
#include "sys/mman.h"

using GetHardwareBufferNativeHandlePtr = const native_handle_t* (*) (const AHardwareBuffer* buffer);
void Test()
{
    AHardwareBuffer* graphicBuf = nullptr; // 流程一致
    void* libnativewindow = dlopen("libnativewindow.so", RTLD_LAZY);
    void *funcPtr = dlsym(libnativewindow, "AHardwareBuffer_getNativeHandle");
    GetHardwareBufferNativeHandlePtr getNativeHandle = reinterpret_cast<GetHardwareBufferNativeHandlePtr>(funcPtr);
    const native_handle_t* handle = getNativeHandle(graphicBuf);
    buffer_handle_t bufferHandle = static_cast<buffer_handle_t>(handle);
}
