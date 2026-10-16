#include "stack_resolver.hpp"

#include <cstdio>
#include <cstdlib>
#include <cxxabi.h>
#include <dlfcn.h>

namespace {

struct CategoryRule {
    const char* substr;
    FrameCategory category;
};

constexpr CategoryRule kCategoryRules[] = {
    {"posix_signal_handler",    FrameCategory::DEBUG},
    {"get_stack_callback",      FrameCategory::DEBUG},
    {"backtrace",               FrameCategory::DEBUG},
    {"resolve_stack_to_string", FrameCategory::DEBUG},
    {"CudaProfiler::",          FrameCategory::INTERNAL},
    {"GpuProfiler::",           FrameCategory::INTERNAL},
    {"cupti",                   FrameCategory::INTERNAL},
    {"CUpti",                   FrameCategory::INTERNAL},
    {"init_trace",              FrameCategory::INTERNAL},
    {"finalize_trace",          FrameCategory::INTERNAL},

    {"cudaMemcpy",              FrameCategory::CUDA_API},
    {"cudaMalloc",              FrameCategory::CUDA_API},
    {"cudaFree",                FrameCategory::CUDA_API},
    {"cudaLaunchKernel",        FrameCategory::CUDA_API},

    {"libcuda",                 FrameCategory::CUDA_DRIVER},
    {"libcudart",               FrameCategory::CUDA_RUNTIME},
    {"cudart",                  FrameCategory::CUDA_RUNTIME},

    {"cuDriver",                FrameCategory::CUDA_DRIVER},
    {"cuDevice",                FrameCategory::CUDA_DRIVER},
    {"nvaci",                   FrameCategory::CUDA_DRIVER},

    {"__libc",                  FrameCategory::SYSTEM},
    {"pthread",                 FrameCategory::SYSTEM},
    {"_dl_",                    FrameCategory::SYSTEM},
    {"_IO_",                    FrameCategory::SYSTEM},
    {"ioctl",                   FrameCategory::SYSTEM},
    {"D3DKMT",                  FrameCategory::SYSTEM},
    {"_start",                  FrameCategory::SYSTEM},
    {"unknown",                 FrameCategory::UNKNOWN}
};

}  // namespace

void StackResolver::configure(uint32_t filter_mask, bool debug) {
    filter_mask_ = filter_mask;
    debug_ = debug;
}

std::string StackResolver::clean_name(const char* mangled_name) const {
    if (!mangled_name) return "unknown";

    int status;
    char* demangled = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);
    std::string name = (status == 0) ? demangled : mangled_name;
    if (status == 0) free(demangled);

    const size_t paren = name.find('(');
    if (paren != std::string::npos) name = name.substr(0, paren);

    return name;
}

FrameCategory StackResolver::get_frame_category(const std::string& name) const {
    if (name.empty()) return FrameCategory::APP;

    for (const auto& rule : kCategoryRules) {
        if (name.find(rule.substr) != std::string::npos) {
            return rule.category;
        }
    }

    return FrameCategory::APP;
}

std::string StackResolver::resolve(void* const* callstack, int frames,
                                   const std::string& kernel_name) {
    std::string full_path;

    for (int i = frames - 1; i >= 0; --i) {
        void* addr = callstack[i];

        if (symbol_cache_.find(addr) == symbol_cache_.end()) {
            Dl_info info{};
            if (dladdr(addr, &info) && info.dli_sname) {
                std::string name = clean_name(info.dli_sname);
                symbol_cache_[addr] = {name, get_frame_category(name)};
            } else if (info.dli_fname) {
                std::string fname = info.dli_fname;
                const size_t last_slash = fname.find_last_of('/');
                if (last_slash != std::string::npos) fname = fname.substr(last_slash + 1);

                const uintptr_t offset = reinterpret_cast<uintptr_t>(addr) -
                                         reinterpret_cast<uintptr_t>(info.dli_fbase);
                char buffer[128];
                snprintf(buffer, sizeof(buffer), "%s[+0x%lx]", fname.c_str(), offset);

                FrameCategory category = FrameCategory::UNKNOWN;
                std::string name = buffer;

                if (name.find("libcuda") != std::string::npos ||
                    name.find("nvaci") != std::string::npos) {
                    category = FrameCategory::CUDA_DRIVER;
                } else if (name.find("cudart") != std::string::npos) {
                    category = FrameCategory::CUDA_RUNTIME;
                } else if (name.find("libc.so") != std::string::npos ||
                           name.find("pthread") != std::string::npos) {
                    category = FrameCategory::SYSTEM;
                }

                symbol_cache_[addr] = {name, category};
            } else {
                symbol_cache_[addr] = {"unknown", FrameCategory::UNKNOWN};
            }
        }

        const CachedFrame& frame = symbol_cache_[addr];

        if (!(filter_mask_ & (1 << static_cast<int>(frame.category)))) continue;
        if (frame.category == FrameCategory::DEBUG && !debug_) continue;
        if (!kernel_name.empty() && kernel_name == frame.name) continue;

        if (!frame.name.empty()) full_path += frame.name + ";";
    }

    if (!kernel_name.empty()) {
        full_path += kernel_name;
    } else if (!full_path.empty() && full_path.back() == ';') {
        full_path.pop_back();
    }

    return full_path;
}
