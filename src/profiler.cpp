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

void CudaProfiler::set_filter(bool enable) {
    filter_internals = enable;
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
    const char* show_env = std::getenv("PTI_SHOW");

    if (pti_env == nullptr || std::string(pti_env) != "1") {
        return; 
    }
    if (freq_env) {
        frequency = std::atoi(freq_env);
    }
    if (show_env) {
        filter_internals = false;
    }

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

    for (const auto& rec : kernel_activities) {
        if (gpu_launch_stacks.count(rec.correlationId)) {
            std::string kName = clean_name(mangled_kernel_names[rec.correlationId].c_str());
            std::string path = resolve_stack_to_string(gpu_launch_stacks[rec.correlationId].frames, 
                                                       gpu_launch_stacks[rec.correlationId].depth, kName);
            printf("%s%s %lu\n", path.c_str(), kName.c_str(), (unsigned long)rec.duration);
        }
    }

    for (int i = 0; i < std::min((int)sample_count, MAX_SAMPLES_COUNT); i++) {
        std::string path = resolve_stack_to_string(cpu_samples[i].frames, cpu_samples[i].depth);
        if (!path.empty()) {
            if (path.back() == ';') path.pop_back();
            aggregated[path]++;
        }
    }

    for (auto const& [stack, count] : aggregated) {
        printf("%s %lu\n", stack.c_str(), count * ns_per_sample);
    }
}

std::string CudaProfiler::clean_name(const char* mangled_name) {
    int status;
    char* demangled = abi::__cxa_demangle(mangled_name, NULL, NULL, &status);
    std::string name = (status == 0) ? demangled : mangled_name;
    if (status == 0) free(demangled);
    size_t paren = name.find('(');
    if (paren != std::string::npos) name = name.substr(0, paren);
    return name;
}

std::string CudaProfiler::resolve_stack_to_string(void** callstack, int frames, const std::string& kernelName) {
    char** strs = backtrace_symbols(callstack, frames);
    std::string full_path = "";

    for (int i = frames - 1; i >= 0; i--) {
        char* begin_name = strchr(strs[i], '(');
        char* begin_offset = strchr(strs[i], '+');
        if (!begin_name || !begin_offset || begin_name >= begin_offset) continue;

        *begin_name++ = '\0';
        *begin_offset = '\0';

        int status;
        char* demangled = abi::__cxa_demangle(begin_name, NULL, NULL, &status);
        char* name = (status == 0) ? demangled : begin_name;
        char* paren = strchr(name, '(');
        if (paren) *paren = '\0';
        
        bool show_debug = (std::getenv("PTI_DEBUG") != nullptr);
        if (!show_debug) {
            if (strstr(name, "posix_signal_handler") || 
                strstr(name, "get_stack_callback") ||
                strstr(name, "__restore_rt") || 
                strstr(name, "backtrace") ||
                strstr(name, "resolve_stack_to_string")) {
                if (status == 0) free(demangled);
                continue;
            }
        }

        bool is_profiler_logic = (strstr(name, "CudaProfiler::") || 
                                  strstr(name, "init_trace") ||
                                  strstr(name, "finalize_trace") ||
                                  strstr(name, "cupti") ||
                                  strstr(name, "CUpti"));

        if (is_profiler_logic && filter_internals) {
            if (status == 0) free(demangled);
            continue; 
        }

        if (strstr(name, "resolve_stack_to_string")) {
            if (status == 0) free(demangled);
            free(strs);
            return ""; 
        }
        
        if (!kernelName.empty() && kernelName == name) {
            if (status == 0) free(demangled);
            continue;
        }

        if (strlen(name) > 0) full_path += std::string(name) + ";";
        if (status == 0) free(demangled);
    }
    free(strs);
    return full_path;
}

void CudaProfiler::posix_signal_handler(int sig, siginfo_t* info, void* context) {
    auto& self = instance();
    int idx = __sync_fetch_and_add(&self.sample_count, 1);
    if (idx < MAX_SAMPLES_COUNT) {
        self.cpu_samples[idx].depth = backtrace(self.cpu_samples[idx].frames, MAX_STACK_DEPTH);
    }
}

void CudaProfiler::get_stack_callback(void* userdata, CUpti_CallbackDomain domain, 
                                      CUpti_CallbackId cbid, const CUpti_CallbackData* cbInfo) {
    if (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 || 
        cbInfo->callbackSite != CUPTI_API_ENTER) return;

    auto& self = instance();
    RawSample s;
    s.depth = backtrace(s.frames, MAX_STACK_DEPTH);
    self.gpu_launch_stacks[cbInfo->correlationId] = s;
    self.mangled_kernel_names[cbInfo->correlationId] = cbInfo->symbolName;
}

void CudaProfiler::buffer_completed_callback(CUcontext ctx, uint32_t streamId, uint8_t* buffer, 
                                             size_t size, size_t validSize) {
    CUpti_Activity *record = NULL;
    auto& self = instance();
    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            auto *k = (CUpti_ActivityKernel4 *)record;
            self.kernel_activities.push_back({k->correlationId, k->end - k->start});
        }
    }
    free(buffer);
}