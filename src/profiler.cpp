#include "profiler.hpp"
#include "gpu_profiler.hpp"
#include <algorithm>
#include <charconv>
#include <cstring>
#include <iostream>
#include <map>

namespace {

std::string format_pc_offset(uint64_t pc_offset) {
    char buffer[2 + 16];
    buffer[0] = '0';
    buffer[1] = 'x';
    const auto result = std::to_chars(
        buffer + 2, buffer + sizeof(buffer), pc_offset, 16);
    return std::string(buffer, result.ptr);
}

}  // namespace

CudaProfiler::CudaProfiler() : gpu_profiler(std::make_unique<GpuProfiler>()) {}

CudaProfiler::~CudaProfiler() = default;

CudaProfiler& CudaProfiler::instance() {
    static CudaProfiler inst;
    return inst;
}

void CudaProfiler::set_frequency(int freq) {
    if (freq > 0) {
        config.frequency = freq;
    }
}

std::string CudaProfiler::get_stall_reason_name(CUpti_ActivityPCSamplingStallReason reason) {
    switch (reason) {
        case CUPTI_ACTIVITY_PC_SAMPLING_STALL_NONE:
            return "No_Stall";
        case CUPTI_ACTIVITY_PC_SAMPLING_STALL_INST_FETCH:
            return "Instruction_Fetch";
        case CUPTI_ACTIVITY_PC_SAMPLING_STALL_EXEC_DEPENDENCY:
            return "Execution_Dependency";
        case CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_DEPENDENCY:
            return "Memory_Dependency";
        case CUPTI_ACTIVITY_PC_SAMPLING_STALL_TEXTURE:
            return "Texture_Access";
        case CUPTI_ACTIVITY_PC_SAMPLING_STALL_SYNC:
            return "Synchronization";
        case CUPTI_ACTIVITY_PC_SAMPLING_STALL_CONSTANT_MEMORY_DEPENDENCY:
            return "Constant_Memory";
        case CUPTI_ACTIVITY_PC_SAMPLING_STALL_PIPE_BUSY:
            return "Pipeline_Busy";
        case CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_THROTTLE:
            return "Memory_Throttle";
        case CUPTI_ACTIVITY_PC_SAMPLING_STALL_NOT_SELECTED:
            return "Not_Selected";
        case CUPTI_ACTIVITY_PC_SAMPLING_STALL_OTHER:
            return "Other";
        case CUPTI_ACTIVITY_PC_SAMPLING_STALL_SLEEPING:
            return "Sleeping";
        default:
            return "Unknown_Stall_" + std::to_string(reason);
    }
}

void CudaProfiler::init() {
    if (initialized_) return;

    const char* pti_env = std::getenv("PTI_ENABLE");
    if (pti_env == nullptr || std::string(pti_env) != "1") {
        return; 
    }
    
    if (const char* freq_env = std::getenv("PTI_FREQ")) {
        int freq = std::atoi(freq_env);
        if (freq <= 0 || freq > 10000) {
            fprintf(stderr, "Warning: Invalid frequency %d. Using default 99 Hz\n", freq);
            config.frequency = 99;
        } else {
            config.frequency = freq;
        }
    }

    uint32_t mask = FLAG_DEFAULT;
    if (const char* profile = std::getenv("PTI_PROFILE")) {
        std::string prof(profile);
        if (prof == "minimal") {
            mask = FLAG_APP | FLAG_CUDA_API;
        } else if (prof == "standard") {
            mask = FLAG_APP | FLAG_CUDA_API | FLAG_SYSTEM;
        } else if (prof == "full") {
            mask = FLAG_APP | FLAG_CUDA_ALL | FLAG_SYSTEM | FLAG_UNKNOWN;
        } else if (prof == "debug") {
            mask = 0xFFFFFFFF;
        }
    }
    config.filter_mask = mask;

    if (const char* stall_mode_env = std::getenv("PTI_STALL_MODE")) {
        config.stall_mode = (std::string(stall_mode_env) == "1");
    }
    
    if (const char* show_all_env = std::getenv("PTI_SHOW_ALL_STALLS")) {
        config.show_all_stalls = (std::string(show_all_env) == "1");
    }

    config.debug = (std::getenv("PTI_DEBUG") != nullptr);
    
    if (const char* period_env = std::getenv("PTI_SAMPLING_PERIOD")) {
        std::string period(period_env);
        if (period == "min") {
            config.sampling_period = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN;
        } else if (period == "low") {
            config.sampling_period = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_LOW;
        } else if (period == "mid") {
            config.sampling_period = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MID;
        } else if (period == "high") {
            config.sampling_period = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_HIGH;
        } else if (period == "max") {
            config.sampling_period = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MAX;
        }
    }

    stack_resolver.configure(config.filter_mask, config.debug);

    if (!config.stall_mode) {
        cpu_profiler.start(config.frequency);
    }
    if (!gpu_profiler->start(config)) {
        cpu_profiler.release();
        return;
    }

    initialized_ = true;
}

void CudaProfiler::process_gpu_samples(std::unordered_map<std::string, uint64_t>& aggregated) {
    std::unordered_map<uint32_t, uint64_t> activity_durations;

    for (const auto& activity : gpu_profiler->kernel_activities()) {
        activity_durations[activity.correlationId] = activity.duration;
    }
    
    uint64_t ns_per_sample = 1000000000ULL / config.frequency;
    
    for (const auto& sample : gpu_profiler->gpu_samples()) {
        std::string kName = stack_resolver.clean_name(sample.kernel_name);
        std::string path = stack_resolver.resolve(sample.frames, sample.depth, kName);
            
        if (!path.empty()) {
            if (path.back() == ';') path.pop_back();
                
            uint64_t duration = 0;
            auto it = activity_durations.find(sample.correlationId);
            if (it != activity_durations.end()) {
                duration = it->second;
            }
                
            if (duration > 0) {
                uint64_t equivalent_samples = duration / ns_per_sample;
                if (equivalent_samples == 0) equivalent_samples = 1;
                    
                if (config.debug) {
                    fprintf(stderr, "  [DEBUG] path='%s' corrId=%u duration=%lu samples=%lu\n",
                            path.c_str(), sample.correlationId, duration, equivalent_samples);
                }
                aggregated[path] += equivalent_samples;
            }
        }
    }
}
uint64_t CudaProfiler::process_pc_sampling_records(
    std::unordered_map<std::string, uint64_t>& aggregated) {
    if (!gpu_profiler->pc_sampling_enabled()) return 0;
    
    std::unordered_map<uint32_t, std::string> kernel_names;
    for (const auto& activity : gpu_profiler->kernel_activities()) {
        kernel_names[activity.correlationId] = stack_resolver.clean_name(activity.name);
    }
    
    std::multimap<uint32_t, std::string> gpu_stacks;
    for (const auto& sample : gpu_profiler->gpu_samples()) {
        std::string kernel_name = "unknown_kernel";
        auto it_kn = kernel_names.find(sample.correlationId);
        if (it_kn != kernel_names.end()) {
            kernel_name = it_kn->second;
        }
        std::string stack_str = stack_resolver.resolve(sample.frames, sample.depth, kernel_name);
        if (!stack_str.empty() && stack_str.back() == ';') {
            stack_str.pop_back();
        }
        gpu_stacks.insert({sample.correlationId, stack_str});
    }
    
    uint32_t unmatched_count = 0;
    
    for (const auto& rec : gpu_profiler->pc_sampling_records()) {
        if (rec.latencySamples == 0) continue;
        
        std::string kernel_name = "unknown_kernel";
        auto it_kernel = kernel_names.find(rec.correlationId);
        if (it_kernel != kernel_names.end()) {
            kernel_name = it_kernel->second;
        }
        
        std::string cpu_stack = "";
        auto it_stack_range = gpu_stacks.equal_range(rec.correlationId);
        if (it_stack_range.first != it_stack_range.second) {
            cpu_stack = it_stack_range.first->second;
        } else {
            unmatched_count++;
        }
        
        std::string stall_name = get_stall_reason_name(rec.stallReason);
        
        std::string path;
        const std::string pc_frame = format_pc_offset(rec.pcOffset);
        if (!cpu_stack.empty()) {
            path = cpu_stack + ";" + stall_name + ";" + pc_frame;
        } else {
            path = kernel_name + ";" + stall_name + ";" + pc_frame;
        }
        
        aggregated[path] += rec.latencySamples;
    }
    
    if (unmatched_count > 0) {
        fprintf(stderr, "Warning: %u PC sampling records had no matching CPU stack (correlationId mismatch)\n", 
                unmatched_count);
    }
    return unmatched_count;
}

void CudaProfiler::finalize() {
    if (!initialized_ || finalized_) return;
    finalized_ = true;

    cpu_profiler.stop();
    gpu_profiler->stop();

    std::unordered_map<std::string, uint64_t> aggregated;
    uint64_t unmatched_pc_records = 0;

    if (config.stall_mode) {
        if (gpu_profiler->pc_sampling_enabled()) {
            unmatched_pc_records = process_pc_sampling_records(aggregated);
        }
    } else {
        cpu_profiler.process_samples(stack_resolver, aggregated);
        process_gpu_samples(aggregated);
    }

    fprintf(stderr, "\n=== Profiler Statistics ===\n");
    fprintf(stderr, "GPU samples captured: %zu\n", gpu_profiler->gpu_samples().size());
    fprintf(stderr, "GPU activities recorded: %zu\n",
            gpu_profiler->kernel_activities().size());
    fprintf(stderr, "Unique stacks aggregated: %zu\n", aggregated.size());
    
    if (config.stall_mode) {
        fprintf(stderr, "CPU profiling: DISABLED (stall mode)\n");
    } else if (cpu_profiler.enabled()) {
        fprintf(stderr, "CPU profiling: ENABLED\n");
    } else {
        fprintf(stderr, "CPU profiling: DISABLED (run with sudo)\n");
    }
    
    if (gpu_profiler->pc_sampling_enabled()) {
        fprintf(stderr, "PC Sampling records: %zu\n",
                gpu_profiler->pc_sampling_records().size());
        fprintf(stderr, "PC Sampling period: %d\n",
                static_cast<int>(gpu_profiler->sampling_period()));
        fprintf(stderr, "PC Sampling unmatched records: %lu\n", unmatched_pc_records);

        const uint64_t info_count = gpu_profiler->pc_sampling_record_info_count();
        fprintf(stderr, "PC Sampling record-info entries: %lu\n", info_count);
        if (info_count > 0) {
            const uint64_t total_samples = gpu_profiler->pc_sampling_total_samples();
            const uint64_t dropped_samples = gpu_profiler->pc_sampling_dropped_samples();
            const double drop_rate = total_samples == 0
                ? 0.0
                : 100.0 * static_cast<double>(dropped_samples) /
                      static_cast<double>(total_samples);
            fprintf(stderr, "PC Sampling total samples: %lu\n", total_samples);
            fprintf(stderr, "PC Sampling dropped samples: %lu (%.3f%%)\n",
                    dropped_samples, drop_rate);
            fprintf(stderr, "PC Sampling period cycles: %lu\n",
                    gpu_profiler->pc_sampling_period_cycles());
        } else {
            fprintf(stderr, "PC Sampling total/dropped samples: unavailable\n");
        }
    }
    
    fprintf(stderr, "Filter profile: standard (APP + CUDA_API + SYSTEM)\n");
    fprintf(stderr, "===========================\n\n");
    
    for (auto const& [stack, count] : aggregated) {
        printf("%s %lu\n", stack.c_str(), count);
    }
    
    cpu_profiler.release();
}
