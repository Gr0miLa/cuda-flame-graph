#include "profiler.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>

CudaProfiler& CudaProfiler::instance() {
    static CudaProfiler inst;
    return inst;
}

void CudaProfiler::set_frequency(int freq) {
    if (freq > 0) {
        frequency = freq;
    }
}

void CudaProfiler::set_filter(FrameCategory category) {
    if (category == FrameCategory::INTERNAL) settings.show_internal = true;
    if (category == FrameCategory::SYSTEM) settings.show_system = true;
    if (category == FrameCategory::UNKNOWN) settings.show_unknown = true;
}

void CudaProfiler::setup_cpu_timer() {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = posix_signal_handler;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigfillset(&sa.sa_mask);
    sigaction(SIGALRM, &sa, NULL);

    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGALRM);
    pthread_sigmask(SIG_UNBLOCK, &set, NULL);

    struct itimerval timer;
    long interval_us = 1000000 / frequency;
    timer.it_interval.tv_sec = 0;
    timer.it_interval.tv_usec = (interval_us > 0) ? interval_us : 1;
    timer.it_value = timer.it_interval;
    setitimer(ITIMER_REAL, &timer, NULL);
}

void CudaProfiler::init() {
    const char* pti_env = std::getenv("PTI_ENABLE");
    const char* freq_env = std::getenv("PTI_FREQ");
    const char* show_all_env = std::getenv("PTI_SHOW_ALL");
    const char* show_debug_env = std::getenv("PTI_SHOW_DEBUG");
    const char* show_internal_env = std::getenv("PTI_SHOW_INTERNAL");
    const char* show_system_env = std::getenv("PTI_SHOW_SYSTEM");
    const char* show_cuda_env = std::getenv("PTI_SHOW_CUDA");
    const char* show_unknown_env = std::getenv("PTI_SHOW_UNKNOWN");

    if (pti_env == nullptr || std::string(pti_env) != "1") {
        return; 
    }
    if (freq_env) {
        frequency = std::atoi(freq_env);
    }
    if (show_all_env || show_internal_env) settings.show_internal = true;
    if (show_all_env || show_system_env) settings.show_system = true;
    if (show_all_env || show_cuda_env) settings.show_cuda = true;
    if (show_unknown_env) settings.show_unknown = true;
    if (show_debug_env) settings.show_debug = true;

    setup_cpu_timer();

    cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)get_stack_callback, NULL);
    
    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, 
                        CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);

    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    
    auto alloc_buf = [](uint8_t **buf, size_t *size, size_t *maxNumRecords) {
        *size = 64 * 1024; 
        *buf = (uint8_t *)malloc(*size); 
        *maxNumRecords = 0;
    };

    cuptiActivityRegisterCallbacks(alloc_buf, buffer_completed_callback);
}

void CudaProfiler::finalize() {
    struct itimerval stop_timer = {};
    setitimer(ITIMER_REAL, &stop_timer, NULL);
    
    cuptiActivityFlushAll(0);

    uint64_t ns_per_sample = 1000000000ULL / frequency;

    std::unordered_map<std::string, uint64_t> aggregated;

    int gpu_total = std::min((int)gpu_sample_count, (int)kernel_activity_count);

    for (int i = 0; i < gpu_total; ++i) {
        std::string kName = clean_name(gpu_samples[i].kernel_name);
        std::string path = resolve_stack_to_string(gpu_samples[i].frames, gpu_samples[i].depth, kName);
        if (!path.empty()) {
            if (path.back() == ';') path.pop_back();
            aggregated[path] += kernel_activities[i].duration;
        }
    }

    int cpu_total = std::min((int)cpu_sample_count, MAX_SAMPLES_COUNT);
    for (int i = 0; i < cpu_total; ++i) {
        std::string path = resolve_stack_to_string(cpu_samples[i].frames, cpu_samples[i].depth);
        if (!path.empty()) {
            if (path.back() == ';') path.pop_back();
            aggregated[path] += ns_per_sample;
        }
    }

    for (auto const& [stack, count] : aggregated) {
        printf("%s %lu\n", stack.c_str(), count);
    }
}

std::string CudaProfiler::clean_name(const char* mangled_name) {
    if (!mangled_name) return "unknown";
    
    int status;
    char* demangled = abi::__cxa_demangle(mangled_name, NULL, NULL, &status);
    std::string name = (status == 0) ? demangled : mangled_name;
    if (status == 0) free(demangled);
    
    size_t paren = name.find('(');
    if (paren != std::string::npos) name = name.substr(0, paren);
    
    return name;
}

FrameCategory CudaProfiler::getFrameCategory(const std::string& name) {
    if (name.empty()) return FrameCategory::APP;

    if (name.find("posix_signal_handler") != std::string::npos || 
        name.find("get_stack_callback") != std::string::npos ||
        name.find("backtrace") != std::string::npos || 
        name.find("resolve_stack_to_string") != std::string::npos) {
        return FrameCategory::DEBUG;
    }

    if (name.find("CudaProfiler::") != std::string::npos || 
        name.find("cupti") != std::string::npos || 
        name.find("CUpti") != std::string::npos ||
        name.find("init_trace") != std::string::npos ||
        name.find("finalize_trace") != std::string::npos) {
        return FrameCategory::INTERNAL;
    }

    if (name.find("cuDriver") != std::string::npos || 
        name.find("cuDevice") != std::string::npos ||
        name.find("nvaci") != std::string::npos ||
        name.find("libcuda") != std::string::npos) {
        return FrameCategory::CUDA;
    }

    if (name.find("__libc") != std::string::npos || 
        name.find("pthread") != std::string::npos || 
        name.find("_dl_") != std::string::npos ||
        name.find("_IO_") != std::string::npos ||
        name.find("ioctl") != std::string::npos ||
        name.find("D3DKMT") != std::string::npos ||
        name.find("_start") != std::string::npos) {
        return FrameCategory::SYSTEM;
    }

    if (name.find("unknown") != std::string::npos) return FrameCategory::UNKNOWN;

    return FrameCategory::APP;
}

std::string CudaProfiler::resolve_stack_to_string(void** callstack, int frames, const std::string& kernelName) {
    std::string full_path = "";
    bool show_debug = (std::getenv("PTI_DEBUG") != nullptr);

    for (int i = frames - 1; i >= 0; --i) {
        void* addr = callstack[i];

        if (symbol_cache.find(addr) == symbol_cache.end()) {
            Dl_info info;
            if (dladdr(addr, &info) && info.dli_sname) {
                std::string name = clean_name(info.dli_sname);
                symbol_cache[addr] = { name, getFrameCategory(name) };
            } 
            else if (info.dli_fname) {
                std::string fname = info.dli_fname;
                size_t last_slash = fname.find_last_of('/');
                if (last_slash != std::string::npos) fname = fname.substr(last_slash + 1);

                uintptr_t offset = (uintptr_t)addr - (uintptr_t)info.dli_fbase;
                char buffer[128];
                snprintf(buffer, sizeof(buffer), "%s[+0x%lx]", fname.c_str(), offset);

                // Автоматическое определение категории по имени файла
                FrameCategory cat = FrameCategory::UNKNOWN;
                std::string s_name = buffer;
                
                if (s_name.find("libcuda") != std::string::npos || 
                    s_name.find("nvaci") != std::string::npos ||
                    s_name.find("cudart") != std::string::npos) {
                    cat = FrameCategory::CUDA;
                } else if (s_name.find("libc.so") != std::string::npos || 
                           s_name.find("pthread") != std::string::npos) {
                    cat = FrameCategory::SYSTEM;
                }

                symbol_cache[addr] = { s_name, cat };
            } 
            else {
                symbol_cache[addr] = { "unknown", FrameCategory::UNKNOWN };
            }
        }

        const CachedFrame& frame = symbol_cache[addr];

        if (frame.category == FrameCategory::INTERNAL && !settings.show_internal) continue;
        if (frame.category == FrameCategory::SYSTEM && !settings.show_system) continue;
        if (frame.category == FrameCategory::CUDA && !settings.show_cuda) continue;
        if (frame.category == FrameCategory::UNKNOWN && !settings.show_unknown) continue;
        if (frame.category == FrameCategory::DEBUG && !show_debug) continue;

        if (!kernelName.empty() && kernelName == frame.name) continue;

        if (!frame.name.empty()) {
            full_path += frame.name + ";";
        }
    }

    if (!kernelName.empty()) {
        full_path += "[compute] " + kernelName;
    } else if (!full_path.empty() && full_path.back() == ';') {
        full_path.pop_back();
    }

    return full_path;
}

void CudaProfiler::posix_signal_handler(int sig, siginfo_t* info, void* context) {
    auto& self = instance();
    int idx = self.cpu_sample_count.fetch_add(1);
    if (idx < MAX_SAMPLES_COUNT) {
        self.cpu_samples[idx].depth = backtrace(self.cpu_samples[idx].frames, MAX_STACK_DEPTH);
    }
}

void CudaProfiler::get_stack_callback(void* userdata, CUpti_CallbackDomain domain, 
                                      CUpti_CallbackId cbid, const CUpti_CallbackData* cbInfo) {
    if (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 || 
        cbInfo->callbackSite != CUPTI_API_ENTER) return;

    auto& self = instance();
    int idx = self.gpu_sample_count.fetch_add(1);
    if (idx < MAX_SAMPLES_COUNT) {
        self.gpu_samples[idx].depth = backtrace(self.gpu_samples[idx].frames, MAX_STACK_DEPTH);
        strncpy(self.gpu_samples[idx].kernel_name, cbInfo->symbolName, 127);
        self.gpu_samples[idx].kernel_name[127] = '\0';
    }
}

void CudaProfiler::buffer_completed_callback(CUcontext ctx, uint32_t streamId, uint8_t* buffer, 
                                             size_t size, size_t validSize) {
    CUpti_Activity *record = NULL;
    auto& self = instance();
    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            auto *k = (CUpti_ActivityKernel4 *)record;
            int idx = self.kernel_activity_count.fetch_add(1);
            if (idx < MAX_SAMPLES_COUNT) {
                self.kernel_activities[idx] = {k->correlationId, (uint64_t)(k->end - k->start)};
            }
        }
    }
    free(buffer);
}