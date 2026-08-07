#include "profiler.hpp"
#include <iostream>
#include <algorithm>
#include <cstring>

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

    // Выделение памяти под кольцевой буфер ядра: 1 страница метаданных + 8 страниц для самих сэмплов
    size_t page_size = getpagesize();
    size_t mmap_size = page_size * (1 + 8);
    
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
    
    if (stall_mode) {
        setup_pc_sampling(sampling_period);
    }
    
    auto alloc_buf = [](uint8_t **buf, size_t *size, size_t *maxNumRecords) {
        *size = 64 * 1024; 
        *buf = (uint8_t *)malloc(*size); 
        *maxNumRecords = 0;
    };

    cuptiActivityRegisterCallbacks(alloc_buf, buffer_completed_callback);
}

void CudaProfiler::process_gpu_samples(std::unordered_map<std::string, uint64_t>& aggregated) {
    std::unordered_map<uint32_t, uint64_t> activity_durations;
    for (int i = 0; i < kernel_activity_count; ++i) {
        activity_durations[kernel_activities[i].correlationId] = kernel_activities[i].duration;
    }
    
    int gpu_total = std::min((int)gpu_sample_count, MAX_SAMPLES_COUNT);
    
    for (int i = 0; i < gpu_total; ++i) {
        std::string kName = clean_name(gpu_samples[i].kernel_name);
        std::string path = resolve_stack_to_string(gpu_samples[i].frames, gpu_samples[i].depth, kName);
        
        if (!path.empty()) {
            if (path.back() == ';') path.pop_back();
            
            uint64_t duration = 0;
            auto it = activity_durations.find(gpu_samples[i].correlationId);
            if (it != activity_durations.end()) {
                duration = it->second;
            }
            
            if (duration > 0) {
                if (std::getenv("PTI_DEBUG")) {
                    fprintf(stderr, "  [DEBUG] path='%s' corrId=%u duration=%lu\n", 
                            path.c_str(), gpu_samples[i].correlationId, duration);
                }
                aggregated[path] += duration;
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
    
    uint64_t ns_per_sample = 1000000000ULL / frequency;
    
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
            if (!path.empty()) {
                if (path.back() == ';') path.pop_back();
                aggregated[path] += ns_per_sample;
            }
        }
        
        tail += header->size;
    }
    
    perf_page->data_tail = tail;
}

void CudaProfiler::finalize() {
    if (perf_fd != -1) {
        ioctl(perf_fd, PERF_EVENT_IOC_DISABLE, 0);
    }
    
    cuptiActivityFlushAll(0);

    std::unordered_map<std::string, uint64_t> aggregated;

    process_gpu_samples(aggregated);
    process_cpu_samples(aggregated);

    // Отладочный вывод
    fprintf(stderr, "\n=== Profiler Statistics ===\n");
    fprintf(stderr, "GPU samples captured: %d\n", (int)gpu_sample_count);
    fprintf(stderr, "GPU activities recorded: %d\n", (int)kernel_activity_count);
    fprintf(stderr, "Unique stacks aggregated: %zu\n", aggregated.size());
    
    if (perf_fd != -1) {
        fprintf(stderr, "CPU profiling: ENABLED\n");
    } else {
        fprintf(stderr, "CPU profiling: DISABLED (run with sudo)\n");
    }
    
    if (pc_sampling_enabled) {
        fprintf(stderr, "PC Sampling records: %d\n", (int)pc_sampling_data.record_count);
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

    auto& self = instance();
    int idx = self.gpu_sample_count.fetch_add(1);
    if (idx < MAX_SAMPLES_COUNT) {
        self.gpu_samples[idx].depth = backtrace(self.gpu_samples[idx].frames, MAX_STACK_DEPTH);
        strncpy(self.gpu_samples[idx].kernel_name, func_name, 127);
        self.gpu_samples[idx].kernel_name[127] = '\0';
        self.gpu_samples[idx].correlationId = cbInfo->correlationId;
    }
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
                int idx = self.kernel_activity_count.fetch_add(1);
                if (idx < MAX_SAMPLES_COUNT) {
                    self.kernel_activities[idx].correlationId = k->correlationId;
                    self.kernel_activities[idx].duration = (uint64_t)(k->end - k->start);
                    strncpy(self.kernel_activities[idx].name, k->name, 127);
                    self.kernel_activities[idx].name[127] = '\0';
                }
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                auto *m = (CUpti_ActivityMemcpy *)record;
                int idx = self.kernel_activity_count.fetch_add(1);
                if (idx < MAX_SAMPLES_COUNT) {
                    self.kernel_activities[idx].correlationId = m->correlationId;
                    self.kernel_activities[idx].duration = (uint64_t)(m->end - m->start);
                    self.kernel_activities[idx].name[0] = '\0';
                }
                break;
            }
            default:
                break;
        }
    }
    free(buffer);
}