#ifndef CUDA_PROFILER_HPP
#define CUDA_PROFILER_HPP

#include <atomic>
#include <cstring>
#include <cupti.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <fstream>
#include <mutex>
#include <signal.h>
#include <string>
#include <sys/time.h>
#include <unordered_map>
#include <vector>

const int MAX_STACK_DEPTH = 128;
const int MAX_SAMPLES_COUNT = 50000;

struct RawSample {
    void* frames[MAX_STACK_DEPTH];
    int depth;
};

struct GpuSample {
    void* frames[MAX_STACK_DEPTH];
    int depth;
    char kernel_name[128];
};

struct KernelRecord {
    uint32_t correlationId;
    uint64_t duration;
};

class CudaProfiler {
private:
    uint32_t frequency = 99;
    bool filter_internals = true;
    bool is_running = false;

    CUpti_SubscriberHandle subscriber;

    KernelRecord kernel_activities[MAX_SAMPLES_COUNT];
    std::atomic<int> kernel_activity_count{0};

    RawSample cpu_samples[MAX_SAMPLES_COUNT];
    std::atomic<int> cpu_sample_count{0};

    GpuSample gpu_samples[MAX_SAMPLES_COUNT];
    std::atomic<int> gpu_sample_count{0};
    
    std::unordered_map<void*, std::string> symbol_cache;

    CudaProfiler() = default;
    ~CudaProfiler() = default;

    void setup_cpu_timer();

    std::string resolve_stack_to_string(void** frames, int depth, const std::string& kernelName = "");
    std::string clean_name(const char* mangled_name);

    static void CUPTIAPI get_stack_callback(void* userdata, CUpti_CallbackDomain domain, 
                                            CUpti_CallbackId cbid, const CUpti_CallbackData* cbInfo);
    static void CUPTIAPI buffer_completed_callback(CUcontext ctx, uint32_t streamId, uint8_t* buffer, 
                                                   size_t size, size_t validSize);
    static void posix_signal_handler(int sig, siginfo_t* info, void* context);

public:
    static CudaProfiler& instance();
    
    void set_frequency(int freq);
    void set_filter(bool enable);

    void init();
    void finalize();

};

#endif // CUDA_PROFILER_HPP
