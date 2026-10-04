#include "profiler.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <map>

struct CategoryRule {
    const char* substr;
    FrameCategory category;
};

static const CategoryRule kCategoryRules[] = {
    {"posix_signal_handler",    FrameCategory::DEBUG},
    {"get_stack_callback",      FrameCategory::DEBUG},
    {"backtrace",               FrameCategory::DEBUG},
    {"resolve_stack_to_string", FrameCategory::DEBUG},
    {"CudaProfiler::",          FrameCategory::INTERNAL},
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

CudaProfiler& CudaProfiler::instance() {
    static CudaProfiler inst;
    return inst;
}

void CudaProfiler::set_frequency(int freq) {
    if (freq > 0) {
        frequency = freq;
    }
}

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

void CudaProfiler::setup_perf_events() {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(struct perf_event_attr));

    pe.type = PERF_TYPE_SOFTWARE;
    pe.size = sizeof(struct perf_event_attr);
    pe.config = PERF_COUNT_SW_CPU_CLOCK;
    pe.sample_freq = frequency;
    pe.freq = 1;
    
    pe.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_TIME | PERF_SAMPLE_CALLCHAIN;
    pe.exclude_idle = 1;
    pe.disabled = 1;

    perf_fd = perf_event_open(&pe, 0, -1, -1, 0);
    if (perf_fd == -1) {
        fprintf(stderr, "Error: CPU profiling requires root privileges.\n");
        fprintf(stderr, "Solution: Run with 'sudo' or set:\n");
        fprintf(stderr, "  sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'\n");
        return;
    }

    // Выделение памяти под кольцевой буфер ядра: 1 страница метаданных + 128 страниц для самих сэмплов
    size_t page_size = getpagesize();
    size_t mmap_size = page_size * (1 + 128);
    
    void* base = mmap(NULL, mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED, perf_fd, 0);
    if (base == MAP_FAILED) {
        fprintf(stderr, "Error mmaping perf buffer\n");
        close(perf_fd);
        perf_fd = -1;
        perf_page = nullptr;
        return;
    }

    perf_page = static_cast<struct perf_event_mmap_page*>(base);

    ioctl(perf_fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(perf_fd, PERF_EVENT_IOC_ENABLE, 0);
}

void CudaProfiler::setup_pc_sampling(CUpti_ActivityPCSamplingPeriod period) {
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    
    if (!ctx) {
        fprintf(stderr, "Warning: No CUDA context for PC Sampling\n");
        return;
    }
    
    CUdevice device;
    CUresult res_cu = cuCtxGetDevice(&device);
    if (res_cu != CUDA_SUCCESS) {
        fprintf(stderr, "Warning: Failed to get CUDA device\n");
        return;
    }
    
    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    
    if (major < 5 || (major == 5 && minor < 2)) {
        fprintf(stderr, "Warning: PC Sampling requires CC 5.2+ (current: %d.%d)\n", major, minor);
        return;
    }
    
    CUpti_ActivityPCSamplingConfig config = {};
    config.size = sizeof(CUpti_ActivityPCSamplingConfig);
    config.samplingPeriod = period;
    config.samplingPeriod2 = 0;
    
    CUptiResult res = cuptiActivityConfigurePCSampling(ctx, &config);
    if (res != CUPTI_SUCCESS) {
        fprintf(stderr, "Warning: cuptiActivityConfigurePCSampling failed (error %d)\n", res);
        return;
    }
    
    pc_sampling_data.samplingPeriod = config.samplingPeriod;
    
    res = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING);
    if (res != CUPTI_SUCCESS) {
        fprintf(stderr, "Warning: cuptiActivityEnable(PC_SAMPLING) failed (error %d)\n", res);
        return;
    }
    
    pc_sampling_enabled = true;
    fprintf(stderr, "PC Sampling enabled (period: %d, CC %d.%d)\n", (int)period, major, minor);
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
    const char* pti_env = std::getenv("PTI_ENABLE");
    if (pti_env == nullptr || std::string(pti_env) != "1") {
        return; 
    }
    
    if (const char* freq_env = std::getenv("PTI_FREQ")) {
        int freq = std::atoi(freq_env);
        if (freq <= 0 || freq > 10000) {
            fprintf(stderr, "Warning: Invalid frequency %d. Using default 99 Hz\n", freq);
            frequency = 99;
        } else {
            frequency = freq;
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
    settings.filter_mask = mask;

    if (const char* stall_mode_env = std::getenv("PTI_STALL_MODE")) {
        stall_mode = (std::string(stall_mode_env) == "1");
    }
    
    if (const char* show_all_env = std::getenv("PTI_SHOW_ALL_STALLS")) {
        show_all_stalls = (std::string(show_all_env) == "1");
    }
    
    CUpti_ActivityPCSamplingPeriod sampling_period = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_LOW;
    if (const char* period_env = std::getenv("PTI_SAMPLING_PERIOD")) {
        std::string period(period_env);
        if (period == "min") {
            sampling_period = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN;
        } else if (period == "low") {
            sampling_period = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_LOW;
        } else if (period == "mid") {
            sampling_period = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MID;
        } else if (period == "high") {
            sampling_period = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_HIGH;
        } else if (period == "max") {
            sampling_period = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MAX;
        }
    }

    gpu_samples.reserve(50000);
    kernel_activities.reserve(50000);
    
    setup_perf_events();

    CUptiResult res;
    res = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)get_stack_callback, NULL);
    if (res != CUPTI_SUCCESS) {
        fprintf(stderr, "Error: cuptiSubscribe failed\n");
        return;
    }
    
    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, 
                        CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);

    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020);
    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                        CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020);
    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                        CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020);
    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
                        CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020);

    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    
    auto alloc_buf = [](uint8_t **buf, size_t *size, size_t *maxNumRecords) {
        *size = 256 * 1024; 
        *buf = (uint8_t *)malloc(*size); 
        *maxNumRecords = 0;
    };

    cuptiActivityRegisterCallbacks(alloc_buf, buffer_completed_callback);
}

void CudaProfiler::process_gpu_samples(std::unordered_map<std::string, uint64_t>& aggregated) {
    std::unordered_map<uint32_t, uint64_t> activity_durations;
    
    {
        std::lock_guard<std::mutex> lock(kernel_activities_mutex);
        for (const auto& activity : kernel_activities) {
            activity_durations[activity.correlationId] = activity.duration;
        }
    }
    
    uint64_t ns_per_sample = 1000000000ULL / frequency;
    
    {
        std::lock_guard<std::mutex> lock(gpu_samples_mutex);
        for (const auto& sample : gpu_samples) {
            std::string kName = clean_name(sample.kernel_name);
            std::string path = resolve_stack_to_string(sample.frames, sample.depth, kName, true);
            
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
                    
                    if (std::getenv("PTI_DEBUG")) {
                        fprintf(stderr, "  [DEBUG] path='%s' corrId=%u duration=%lu samples=%lu\n", 
                                path.c_str(), sample.correlationId, duration, equivalent_samples);
                    }
                    aggregated[path] += equivalent_samples;
                }
            }
        }
    }
}

void CudaProfiler::process_cpu_samples(std::unordered_map<std::string, uint64_t>& aggregated) {
    if (perf_fd == -1 || perf_page == nullptr) return;
    
    __sync_synchronize();
    
    uint64_t head = perf_page->data_head;
    uint64_t tail = perf_page->data_tail;
    size_t page_size = getpagesize();
    char* base = (char*)perf_page + page_size;
    uint64_t data_size = page_size * 8;
    
    while (tail < head) {
        uint64_t offset = tail % data_size;
        struct perf_event_header* header = (struct perf_event_header*)(base + offset);
        
        if (header->size == 0 || header->size > data_size) break;
        
        if (header->type == PERF_RECORD_SAMPLE) {
            char* ptr = (char*)header + sizeof(struct perf_event_header);
            
            uint64_t ip = *(uint64_t*)ptr; ptr += sizeof(uint64_t);
            uint32_t pid = *(uint32_t*)ptr; ptr += sizeof(uint32_t);
            uint32_t tid = *(uint32_t*)ptr; ptr += sizeof(uint32_t);
            uint64_t time = *(uint64_t*)ptr; ptr += sizeof(uint64_t);
            uint64_t nr = *(uint64_t*)ptr; ptr += sizeof(uint64_t);
            
            void* callstack[MAX_STACK_DEPTH];
            int depth = 0;
            
            if (ip && depth < MAX_STACK_DEPTH) {
                callstack[depth++] = (void*)ip;
            }
            
            for (uint64_t i = 0; i < nr && depth < MAX_STACK_DEPTH; ++i) {
                uint64_t addr = *(uint64_t*)ptr;
                ptr += sizeof(uint64_t);
                if (addr) {
                    callstack[depth++] = (void*)addr;
                }
            }

            std::string path = resolve_stack_to_string(callstack, depth);
            
            if (path.find("__device_stub__") != std::string::npos) {
                tail += header->size;
                continue;
            }
            
            if (!path.empty()) {
                if (path.back() == ';') path.pop_back();
                aggregated[path] += 1;
            }
        }
        
        tail += header->size;
    }
    
    perf_page->data_tail = tail;
}

void CudaProfiler::process_pc_sampling_records(std::unordered_map<std::string, uint64_t>& aggregated) {
    if (!pc_sampling_enabled) return;
    
    std::unordered_map<uint32_t, std::string> kernel_names;
    {
        std::lock_guard<std::mutex> lock(kernel_activities_mutex);
        for (const auto& activity : kernel_activities) {
            kernel_names[activity.correlationId] = clean_name(activity.name);
        }
    }
    
    std::multimap<uint32_t, std::string> gpu_stacks;
    {
        std::lock_guard<std::mutex> lock(gpu_samples_mutex);
        for (const auto& sample : gpu_samples) {
            std::string kernel_name = "unknown_kernel";
            auto it_kn = kernel_names.find(sample.correlationId);
            if (it_kn != kernel_names.end()) {
                kernel_name = it_kn->second;
            }
            std::string stack_str = resolve_stack_to_string(sample.frames, sample.depth, kernel_name, true);
            if (!stack_str.empty() && stack_str.back() == ';') {
                stack_str.pop_back();
            }
            gpu_stacks.insert({sample.correlationId, stack_str});
        }
    }
    
    uint32_t unmatched_count = 0;
    
    std::lock_guard<std::mutex> lock(pc_sampling_data.records_mutex);
    for (const auto& rec : pc_sampling_data.records) {
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
        if (!cpu_stack.empty()) {
            path = cpu_stack + ";[stall=" + stall_name + "];[pc=0x" + 
                   std::to_string(rec.pcOffset) + "]";
        } else {
            path = kernel_name + ";[stall=" + stall_name + "];[pc=0x" + 
                   std::to_string(rec.pcOffset) + "]";
        }
        
        aggregated[path] += rec.latencySamples;
    }
    
    if (unmatched_count > 0) {
        fprintf(stderr, "Warning: %u PC sampling records had no matching CPU stack (correlationId mismatch)\n", 
                unmatched_count);
    }
}

void CudaProfiler::finalize() {
    if (perf_fd != -1) {
        ioctl(perf_fd, PERF_EVENT_IOC_DISABLE, 0);
    }
    
    cuptiActivityFlushAll(0);

    std::unordered_map<std::string, uint64_t> aggregated;

    if (stall_mode) {
        if (pc_sampling_enabled) {
            process_pc_sampling_records(aggregated);
        }
    } else {
        process_cpu_samples(aggregated);
        process_gpu_samples(aggregated);
    }

    fprintf(stderr, "\n=== Profiler Statistics ===\n");
    {
        std::lock_guard<std::mutex> lock(gpu_samples_mutex);
        fprintf(stderr, "GPU samples captured: %zu\n", gpu_samples.size());
    }
    {
        std::lock_guard<std::mutex> lock(kernel_activities_mutex);
        fprintf(stderr, "GPU activities recorded: %zu\n", kernel_activities.size());
    }
    fprintf(stderr, "Unique stacks aggregated: %zu\n", aggregated.size());
    
    if (perf_fd != -1) {
        fprintf(stderr, "CPU profiling: ENABLED\n");
    } else {
        fprintf(stderr, "CPU profiling: DISABLED (run with sudo)\n");
    }
    
    if (pc_sampling_enabled) {
        std::lock_guard<std::mutex> lock(pc_sampling_data.records_mutex);
        fprintf(stderr, "PC Sampling records: %zu\n", pc_sampling_data.records.size());
        fprintf(stderr, "PC Sampling period: %d\n", (int)pc_sampling_data.samplingPeriod);
    }
    
    fprintf(stderr, "Filter profile: standard (APP + CUDA_API + SYSTEM)\n");
    fprintf(stderr, "===========================\n\n");
    
    for (auto const& [stack, count] : aggregated) {
        printf("%s %lu\n", stack.c_str(), count);
    }
    
    if (perf_fd != -1) {
        close(perf_fd);
        perf_fd = -1;
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

    for (const auto& rule : kCategoryRules) {
        if (name.find(rule.substr) != std::string::npos) {
            return rule.category;
        }
    }

    return FrameCategory::APP;
}

std::string CudaProfiler::resolve_stack_to_string(void* const* callstack, int frames, const std::string& kernelName, bool is_gpu) {
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

                FrameCategory cat = FrameCategory::UNKNOWN;
                std::string s_name = buffer;
                
                if (s_name.find("libcuda") != std::string::npos || 
                    s_name.find("nvaci") != std::string::npos) {
                    cat = FrameCategory::CUDA_DRIVER;
                } else if (s_name.find("cudart") != std::string::npos) {
                    cat = FrameCategory::CUDA_RUNTIME;
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

        if (!(settings.filter_mask & (1 << static_cast<int>(frame.category)))) {
            continue;
        }

        if (frame.category == FrameCategory::DEBUG && !show_debug) continue;

        if (!kernelName.empty() && kernelName == frame.name) continue;

        if (!frame.name.empty()) {
            full_path += frame.name + ";";
        }
    }

    if (!kernelName.empty()) {
        full_path += kernelName;
    } else if (!full_path.empty() && full_path.back() == ';') {
        full_path.pop_back();
    }

    return full_path;
}

void CudaProfiler::get_stack_callback(void* userdata, CUpti_CallbackDomain domain, 
                                      CUpti_CallbackId cbid, const CUpti_CallbackData* cbInfo) {
    if (cbInfo->callbackSite != CUPTI_API_ENTER) return;
    
    auto& self = instance();
    
    if (self.stall_mode && !self.pc_sampling_enabled) {
        CUpti_ActivityPCSamplingPeriod period = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_LOW;
        if (const char* period_env = std::getenv("PTI_SAMPLING_PERIOD")) {
            std::string p(period_env);
            if (p == "min") period = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN;
            else if (p == "low") period = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_LOW;
            else if (p == "mid") period = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MID;
            else if (p == "high") period = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_HIGH;
            else if (p == "max") period = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MAX;
        }
        self.setup_pc_sampling(period);
    }
    
    const char* func_name = nullptr;
    switch (cbid) {
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
            func_name = cbInfo->symbolName;
            break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
            func_name = "cudaMemcpy";
            break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
            func_name = "cudaMemcpyAsync";
            break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
            func_name = "cudaMalloc";
            break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020:
            func_name = "cudaFree";
            break;
        default:
            return;
    }

    GpuSample sample;
    sample.depth = backtrace(sample.frames, MAX_STACK_DEPTH);
    strncpy(sample.kernel_name, func_name, 127);
    sample.kernel_name[127] = '\0';
    sample.correlationId = cbInfo->correlationId;
    
    std::lock_guard<std::mutex> lock(self.gpu_samples_mutex);
    self.gpu_samples.push_back(sample);
}

void CudaProfiler::buffer_completed_callback(CUcontext ctx, uint32_t streamId, uint8_t* buffer, 
                                             size_t size, size_t validSize) {
    CUpti_Activity *record = NULL;
    auto& self = instance();
    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
        switch (record->kind) {
            case CUPTI_ACTIVITY_KIND_KERNEL:
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
                auto *k = (CUpti_ActivityKernel4 *)record;
                KernelRecord kr;
                kr.correlationId = k->correlationId;
                kr.duration = (uint64_t)(k->end - k->start);
                strncpy(kr.name, k->name, 127);
                kr.name[127] = '\0';
                
                std::lock_guard<std::mutex> lock(self.kernel_activities_mutex);
                self.kernel_activities.push_back(kr);
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                auto *m = (CUpti_ActivityMemcpy *)record;
                KernelRecord kr;
                kr.correlationId = m->correlationId;
                kr.duration = (uint64_t)(m->end - m->start);
                kr.name[0] = '\0';
                
                std::lock_guard<std::mutex> lock(self.kernel_activities_mutex);
                self.kernel_activities.push_back(kr);
                break;
            }
            case CUPTI_ACTIVITY_KIND_PC_SAMPLING: {
                auto *pc = (CUpti_ActivityPCSampling3 *)record;
                
                if (!self.show_all_stalls) {
                    if (pc->stallReason == CUPTI_ACTIVITY_PC_SAMPLING_STALL_NONE ||
                        pc->stallReason == CUPTI_ACTIVITY_PC_SAMPLING_STALL_INVALID) {
                        continue;
                    }
                }
                
                PCSamplingRecord rec;
                rec.correlationId = pc->correlationId;
                rec.pcOffset = pc->pcOffset;
                rec.functionIndex = pc->functionId;
                rec.stallReason = pc->stallReason;
                rec.samples = pc->samples;
                rec.latencySamples = pc->latencySamples;
                rec.functionName[0] = '\0';
                
                std::lock_guard<std::mutex> lock(self.pc_sampling_data.records_mutex);
                self.pc_sampling_data.records.push_back(rec);
                break;
            }
            default:
                break;
        }
    }
    free(buffer);
}