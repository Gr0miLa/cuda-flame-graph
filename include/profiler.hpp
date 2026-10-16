#ifndef CUDA_PROFILER_HPP
#define CUDA_PROFILER_HPP

#include "cpu_profiler.hpp"
#include "stack_resolver.hpp"

#include <atomic>
#include <cstring>
#include <cupti.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <fstream>
#include <memory>
#include <mutex>
#include <signal.h>
#include <string>
#include <sys/time.h>
#include <unordered_map>
#include <vector>

const int MAX_STACK_DEPTH = 64;

struct ProfilerConfig {
    uint32_t frequency = 99;
    uint32_t filter_mask = FLAG_DEFAULT;
    bool stall_mode = false;
    bool show_all_stalls = false;
    bool debug = false;
    CUpti_ActivityPCSamplingPeriod sampling_period = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_LOW;
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

struct PCSamplingRecord {
    uint32_t correlationId;
    uint64_t pcOffset;
    uint32_t functionIndex;
    char functionName[128];
    CUpti_ActivityPCSamplingStallReason stallReason;
    uint32_t samples;
    uint32_t latencySamples;
};

struct PCSamplingData {
    std::vector<PCSamplingRecord> records;
    std::mutex records_mutex;
    CUpti_ActivityPCSamplingPeriod samplingPeriod;
    uint64_t total_samples = 0;
    uint64_t dropped_samples = 0;
    uint64_t sampling_period_cycles = 0;
    uint64_t record_info_count = 0;
    
    PCSamplingData() {
        records.reserve(10000);
    }
};

class GpuProfiler;

class CudaProfiler {
private:
    ProfilerConfig config;
    bool initialized_ = false;
    bool finalized_ = false;
    CpuProfiler cpu_profiler;
    std::unique_ptr<GpuProfiler> gpu_profiler;
    
    StackResolver stack_resolver;

    CudaProfiler();
    ~CudaProfiler();

    void process_gpu_samples(std::unordered_map<std::string, uint64_t>& aggregated);
    uint64_t process_pc_sampling_records(
        std::unordered_map<std::string, uint64_t>& aggregated);
    
    std::string get_stall_reason_name(CUpti_ActivityPCSamplingStallReason reason);

public:
    static CudaProfiler& instance();
    
    void set_frequency(int freq);

    void init();
    void finalize();

};

#endif  // CUDA_PROFILER_HPP
