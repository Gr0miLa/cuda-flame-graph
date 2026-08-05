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

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <unistd.h>

const int MAX_STACK_DEPTH = 64;
const int MAX_SAMPLES_COUNT = 50000;

enum class FrameCategory {
    APP,
    CUDA_API,
    CUDA_DRIVER,
    CUDA_RUNTIME,
    SYSTEM,
    INTERNAL,
    UNKNOWN,
    DEBUG
};

enum FrameFlags : uint32_t {
    FLAG_APP          = 1 << 0,
    FLAG_CUDA_API     = 1 << 1,
    FLAG_CUDA_DRIVER  = 1 << 2,
    FLAG_CUDA_RUNTIME = 1 << 3,
    FLAG_SYSTEM       = 1 << 4,
    FLAG_INTERNAL     = 1 << 5,
    FLAG_UNKNOWN      = 1 << 6,
    FLAG_DEBUG        = 1 << 7,
    
    FLAG_CUDA         = FLAG_CUDA_API,
    FLAG_CUDA_ALL     = FLAG_CUDA_API | FLAG_CUDA_DRIVER | FLAG_CUDA_RUNTIME,
    FLAG_DEFAULT      = FLAG_APP | FLAG_CUDA_API
};

struct ProfilerSettings {
    uint32_t filter_mask = FLAG_DEFAULT;
};

struct GpuSample {
    void* frames[MAX_STACK_DEPTH];
    int depth;
    char kernel_name[128];
    uint32_t correlationId;
};

struct KernelRecord {
    uint32_t correlationId;
    uint64_t duration;
    char name[128];
};

struct CachedFrame {
    std::string name;
    FrameCategory category;
};

class CudaProfiler {
private:
    uint32_t frequency = 99;
    ProfilerSettings settings;
    bool is_running = false;
    int perf_fd = -1;
    struct perf_event_mmap_page *perf_page = nullptr;

    CUpti_SubscriberHandle subscriber;

    KernelRecord kernel_activities[MAX_SAMPLES_COUNT];
    std::atomic<int> kernel_activity_count{0};

    GpuSample gpu_samples[MAX_SAMPLES_COUNT];
    std::atomic<int> gpu_sample_count{0};
    
    std::unordered_map<void*, CachedFrame> symbol_cache;

    CudaProfiler() = default;
    ~CudaProfiler() = default;

    void setup_perf_events();

    std::string resolve_stack_to_string(void** frames, int depth, const std::string& kernelName = "");
    std::string clean_name(const char* mangled_name);

    void process_gpu_samples(std::unordered_map<std::string, uint64_t>& aggregated);
    void process_cpu_samples(std::unordered_map<std::string, uint64_t>& aggregated);

    static void CUPTIAPI get_stack_callback(void* userdata, CUpti_CallbackDomain domain, 
                                            CUpti_CallbackId cbid, const CUpti_CallbackData* cbInfo);
    static void CUPTIAPI buffer_completed_callback(CUcontext ctx, uint32_t streamId, uint8_t* buffer, 
                                                   size_t size, size_t validSize);

    FrameCategory getFrameCategory(const std::string& name);

public:
    static CudaProfiler& instance();
    
    void set_frequency(int freq);

    void init();
    void finalize();

};

#endif  // CUDA_PROFILER_HPP
