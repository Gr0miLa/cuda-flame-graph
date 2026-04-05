#ifndef CUDA_PROFILER_HPP
#define CUDA_PROFILER_HPP

#include <vector>
#include <signal.h>
#include <unordered_map>
#include <string>
#include <mutex>
#include <cupti.h>
#include <sys/time.h>
#include <execinfo.h>
#include <cxxabi.h>

const int MAX_STACK_DEPTH = 128;
const int MAX_SAMPLES_COUNT = 50000;

struct RawSample {
    void* frames[MAX_STACK_DEPTH];
    int depth;
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
    RawSample cpu_samples[MAX_SAMPLES_COUNT];
    volatile int sample_count = 0;

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

    std::unordered_map<uint32_t, RawSample> gpu_launch_stacks;
    std::unordered_map<uint32_t, std::string> mangled_kernel_names;
    std::vector<KernelRecord> kernel_activities;

public:
    static CudaProfiler& instance();
    
    void set_frequency(int freq);
    void set_filter(bool enable);

    void init();
    void finalize();

};

#endif // CUDA_PROFILER_HPP
