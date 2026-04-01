#include <cstdio>
#include <cuda.h>
#include <cupti.h>
#include <execinfo.h>
#include <cxxabi.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <signal.h>
#include <sys/time.h>
#include <cstring>

extern "C" {
    void Usage() {
        printf("CUDA Hybrid Profiler Loaded\n");
        printf("Usage: PROFILER_FREQ=999 ./pti_loader <app> [args]\n");
    }

    int ParseArgs(int argc, char* argv[]) {
        return 1; 
    }

    void SetToolEnv() {
    }
}

#define MAX_SAMPLES 50000
#define MAX_DEPTH 128

struct RawSample {
    void* frames[MAX_DEPTH];
    int depth;
};

struct ProfilerConfig {
    int perf_freq = 999;
    bool filter_cuda = false;
};

ProfilerConfig g_config;
std::unordered_map<uint32_t, std::string> g_stack_storage;
RawSample g_samples[MAX_SAMPLES];
volatile int g_sample_count = 0;

std::string clean_name(const char* mangled_name) {
    int status;
    char* demangled = abi::__cxa_demangle(mangled_name, NULL, NULL, &status);
    std::string name = (status == 0) ? demangled : mangled_name;
    if (status == 0) free(demangled);
    size_t paren = name.find('(');
    if (paren != std::string::npos) name = name.substr(0, paren);
    return name;
}

void posix_signal_handler(int sig, siginfo_t* info, void* context) {
    int idx = __sync_fetch_and_add(&g_sample_count, 1);
    if (idx < MAX_SAMPLES) {
        g_samples[idx].depth = backtrace(g_samples[idx].frames, MAX_DEPTH);
    }
}

std::string resolve_stack_to_string(void** callstack, int frames, const std::string& kernelName = "") {
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

        if (strstr(name, "posix_signal_handler") || strstr(name, "get_stack_callback")) {
            if (status == 0) free(demangled);
            continue; 
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

void CUPTIAPI get_stack_callback(void *userdata, CUpti_CallbackDomain domain,
                                 CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo) {
    if (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 || 
        cbInfo->callbackSite != CUPTI_API_ENTER) return;

    void* callstack[MAX_DEPTH];
    int frames = backtrace(callstack, MAX_DEPTH);

    std::string kernelName = clean_name(cbInfo->symbolName);
    std::string path = resolve_stack_to_string(callstack, frames, kernelName);
    g_stack_storage[cbInfo->correlationId] = path + kernelName;
}

void CUPTIAPI buffer_completed_callback(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
    CUpti_Activity *record = NULL;
    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            auto *kernel = (CUpti_ActivityKernel4 *)record;
            if (g_stack_storage.count(kernel->correlationId)) {
                printf("%s %lu\n", g_stack_storage[kernel->correlationId].c_str(), (unsigned long)(kernel->end - kernel->start));
            }
        }
    }
    free(buffer);
}

void setup_cpu_timer() {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = posix_signal_handler;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGALRM, &sa, NULL);

    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGALRM);
    pthread_sigmask(SIG_UNBLOCK, &set, NULL);

    struct itimerval timer;
    long interval_us = 1000000 / g_config.perf_freq;
    timer.it_interval.tv_sec = 0;
    timer.it_interval.tv_usec = interval_us;
    timer.it_value = timer.it_interval;
    setitimer(ITIMER_REAL, &timer, NULL);
}

__attribute__((constructor))
void init_trace() {
    if (const char* f = std::getenv("PROFILER_FREQ")) g_config.perf_freq = std::atoi(f);
    if (const char* filter_env = std::getenv("PROFILER_FILTER")) g_config.filter_cuda = (std::string(filter_env) == "1");

    CUpti_SubscriberHandle subscriber;
    cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)get_stack_callback, NULL);
    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);

    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    auto alloc_buf = [](uint8_t **buf, size_t *size, size_t *maxNumRecords) {
        *size = 64 * 1024; *buf = (uint8_t *)malloc(*size); *maxNumRecords = 0;
    };
    cuptiActivityRegisterCallbacks(alloc_buf, buffer_completed_callback);

    setup_cpu_timer();
}

__attribute__((destructor))
void finalize_trace() {
    struct itimerval stop_timer = {};
    setitimer(ITIMER_REAL, &stop_timer, NULL);
    
    cuptiActivityFlushAll(0);

    uint64_t ns_per_sample = 1000000000ULL / g_config.perf_freq;

    std::unordered_map<std::string, uint64_t> aggregated;
    for (int i = 0; i < std::min((int)g_sample_count, MAX_SAMPLES); i++) {
        std::string path = resolve_stack_to_string(g_samples[i].frames, g_samples[i].depth);
        
        if (!path.empty()) {
            if (path.back() == ';') path.pop_back();
            aggregated[path]++;
        }
    }

    for (auto const& [stack, count] : aggregated) {
        printf("%s %lu\n", stack.c_str(), count * ns_per_sample);
    }
}