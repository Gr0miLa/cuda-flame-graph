#include "gpu_profiler.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <execinfo.h>

GpuProfiler* GpuProfiler::active_instance_ = nullptr;

GpuProfiler::~GpuProfiler() {
    stop();
}

bool GpuProfiler::start(const ProfilerConfig& config) {
    if (started_) return true;

    config_ = config;
    gpu_samples_.reserve(50000);
    kernel_activities_.reserve(50000);
    active_instance_ = this;

    const CUptiResult result = cuptiSubscribe(
        &subscriber_, reinterpret_cast<CUpti_CallbackFunc>(runtime_callback), this);
    if (result != CUPTI_SUCCESS) {
        fprintf(stderr, "Error: cuptiSubscribe failed\n");
        active_instance_ = nullptr;
        return false;
    }

    started_ = true;

    cuptiEnableCallback(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API,
                        CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000);

    if (!config_.stall_mode) {
        cuptiEnableCallback(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API,
                            CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020);
        cuptiEnableCallback(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API,
                            CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020);
        cuptiEnableCallback(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API,
                            CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020);
        cuptiEnableCallback(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API,
                            CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020);
    }

    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    if (!config_.stall_mode) {
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    }
    cuptiActivityRegisterCallbacks(request_activity_buffer, complete_activity_buffer);
    return true;
}

void GpuProfiler::flush() {
    if (started_) cuptiActivityFlushAll(0);
}

void GpuProfiler::stop() {
    if (!started_) return;

    flush();

    CUptiResult result = cuptiUnsubscribe(subscriber_);
    if (result != CUPTI_SUCCESS) {
        fprintf(stderr, "Warning: cuptiUnsubscribe failed (error %d)\n", result);
    }
    subscriber_ = {};

    if (pc_sampling_enabled_) {
        result = cuptiActivityDisable(CUPTI_ACTIVITY_KIND_PC_SAMPLING);
        if (result != CUPTI_SUCCESS) {
            fprintf(stderr, "Warning: disabling PC Sampling activity failed (error %d)\n",
                    result);
        }
    }
    if (!config_.stall_mode) {
        const CUpti_ActivityKind non_stall_activity_kinds[] = {
            CUPTI_ACTIVITY_KIND_MEMCPY,
            CUPTI_ACTIVITY_KIND_RUNTIME,
        };
        for (const CUpti_ActivityKind kind : non_stall_activity_kinds) {
            result = cuptiActivityDisable(kind);
            if (result != CUPTI_SUCCESS) {
                fprintf(stderr, "Warning: disabling CUPTI activity %d failed (error %d)\n",
                        static_cast<int>(kind), result);
            }
        }
    }

    result = cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
    if (result != CUPTI_SUCCESS) {
        fprintf(stderr, "Warning: disabling CUPTI activity %d failed (error %d)\n",
                static_cast<int>(CUPTI_ACTIVITY_KIND_KERNEL), result);
    }

    if (active_instance_ == this) active_instance_ = nullptr;
    started_ = false;
}

bool GpuProfiler::pc_sampling_enabled() const noexcept {
    return pc_sampling_enabled_;
}

CUpti_ActivityPCSamplingPeriod GpuProfiler::sampling_period() const noexcept {
    return pc_sampling_data_.samplingPeriod;
}

const std::vector<GpuSample>& GpuProfiler::gpu_samples() const noexcept {
    return gpu_samples_;
}

const std::vector<KernelRecord>& GpuProfiler::kernel_activities() const noexcept {
    return kernel_activities_;
}

const std::vector<PCSamplingRecord>& GpuProfiler::pc_sampling_records() const noexcept {
    return pc_sampling_data_.records;
}

uint64_t GpuProfiler::pc_sampling_total_samples() const noexcept {
    return pc_sampling_data_.total_samples;
}

uint64_t GpuProfiler::pc_sampling_dropped_samples() const noexcept {
    return pc_sampling_data_.dropped_samples;
}

uint64_t GpuProfiler::pc_sampling_period_cycles() const noexcept {
    return pc_sampling_data_.sampling_period_cycles;
}

uint64_t GpuProfiler::pc_sampling_record_info_count() const noexcept {
    return pc_sampling_data_.record_info_count;
}

void GpuProfiler::setup_pc_sampling() {
    CUcontext context = nullptr;
    cuCtxGetCurrent(&context);

    if (!context) {
        fprintf(stderr, "Warning: No CUDA context for PC Sampling\n");
        return;
    }

    CUdevice device;
    if (cuCtxGetDevice(&device) != CUDA_SUCCESS) {
        fprintf(stderr, "Warning: Failed to get CUDA device\n");
        return;
    }

    int major = 0;
    int minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);

    if (major < 5 || (major == 5 && minor < 2)) {
        fprintf(stderr, "Warning: PC Sampling requires CC 5.2+ (current: %d.%d)\n",
                major, minor);
        return;
    }

    CUpti_ActivityPCSamplingConfig sampling_config{};
    sampling_config.size = sizeof(CUpti_ActivityPCSamplingConfig);
    sampling_config.samplingPeriod = config_.sampling_period;
    sampling_config.samplingPeriod2 = config_.sampling_period_exponent;

    CUptiResult result = cuptiActivityConfigurePCSampling(context, &sampling_config);
    if (result != CUPTI_SUCCESS) {
        fprintf(stderr, "Warning: cuptiActivityConfigurePCSampling failed (error %d)\n",
                result);
        return;
    }

    pc_sampling_data_.samplingPeriod = sampling_config.samplingPeriod;
    result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING);
    if (result != CUPTI_SUCCESS) {
        fprintf(stderr, "Warning: cuptiActivityEnable(PC_SAMPLING) failed (error %d)\n",
                result);
        return;
    }

    pc_sampling_enabled_ = true;
    if (config_.sampling_period_exponent != 0) {
        fprintf(stderr, "PC Sampling enabled (period: 2^%u cycles, CC %d.%d)\n",
                config_.sampling_period_exponent, major, minor);
    } else {
        fprintf(stderr, "PC Sampling enabled (period: %d, CC %d.%d)\n",
                static_cast<int>(config_.sampling_period), major, minor);
    }
}

void GpuProfiler::add_gpu_sample(GpuSample sample) {
    std::lock_guard<std::mutex> lock(gpu_samples_mutex_);
    gpu_samples_.push_back(sample);
}

void GpuProfiler::add_kernel_activity(KernelRecord record) {
    std::lock_guard<std::mutex> lock(kernel_activities_mutex_);
    kernel_activities_.push_back(record);
}

void GpuProfiler::add_pc_sampling_record(PCSamplingRecord record) {
    std::lock_guard<std::mutex> lock(pc_sampling_data_.records_mutex);
    pc_sampling_data_.records.push_back(record);
}

void GpuProfiler::add_pc_sampling_record_info(
    const CUpti_ActivityPCSamplingRecordInfo& info) {
    std::lock_guard<std::mutex> lock(pc_sampling_data_.records_mutex);
    pc_sampling_data_.total_samples += info.totalSamples;
    pc_sampling_data_.dropped_samples += info.droppedSamples;
    pc_sampling_data_.sampling_period_cycles = info.samplingPeriodInCycles;
    ++pc_sampling_data_.record_info_count;
}

void CUPTIAPI GpuProfiler::runtime_callback(
    void* userdata, CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid, const CUpti_CallbackData* callback_info) {
    if (callback_info->callbackSite != CUPTI_API_ENTER) return;

    auto* profiler = static_cast<GpuProfiler*>(userdata);
    if (!profiler) return;

    if (profiler->config_.stall_mode && !profiler->pc_sampling_enabled_) {
        profiler->setup_pc_sampling();
    }

    const char* function_name = nullptr;
    switch (cbid) {
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
            function_name = callback_info->symbolName;
            break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
            function_name = "cudaMemcpy";
            break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
            function_name = "cudaMemcpyAsync";
            break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
            function_name = "cudaMalloc";
            break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020:
            function_name = "cudaFree";
            break;
        default:
            return;
    }

    GpuSample sample;
    sample.depth = backtrace(sample.frames, MAX_STACK_DEPTH);
    strncpy(sample.kernel_name, function_name, 127);
    sample.kernel_name[127] = '\0';
    sample.correlationId = callback_info->correlationId;
    profiler->add_gpu_sample(sample);
}

void CUPTIAPI GpuProfiler::request_activity_buffer(
    uint8_t** buffer, size_t* size, size_t* max_records) {
    *size = 1024 * 1024;
    *buffer = static_cast<uint8_t*>(malloc(*size));
    *max_records = 0;
}

void CUPTIAPI GpuProfiler::complete_activity_buffer(
    CUcontext context, uint32_t stream_id, uint8_t* buffer,
    size_t size, size_t valid_size) {
    GpuProfiler* profiler = active_instance_;
    if (!profiler) {
        free(buffer);
        return;
    }

    CUpti_Activity* activity = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &activity) == CUPTI_SUCCESS) {
        switch (activity->kind) {
            case CUPTI_ACTIVITY_KIND_KERNEL:
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
                auto* kernel = reinterpret_cast<CUpti_ActivityKernel4*>(activity);
                KernelRecord record;
                record.correlationId = kernel->correlationId;
                record.duration = static_cast<uint64_t>(kernel->end - kernel->start);
                strncpy(record.name, kernel->name, 127);
                record.name[127] = '\0';
                profiler->add_kernel_activity(record);
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                auto* memcpy = reinterpret_cast<CUpti_ActivityMemcpy*>(activity);
                KernelRecord record;
                record.correlationId = memcpy->correlationId;
                record.duration = static_cast<uint64_t>(memcpy->end - memcpy->start);
                record.name[0] = '\0';
                profiler->add_kernel_activity(record);
                break;
            }
            case CUPTI_ACTIVITY_KIND_PC_SAMPLING: {
                auto* pc = reinterpret_cast<CUpti_ActivityPCSampling3*>(activity);
                if (!profiler->config_.show_all_stalls &&
                    (pc->stallReason == CUPTI_ACTIVITY_PC_SAMPLING_STALL_NONE ||
                     pc->stallReason == CUPTI_ACTIVITY_PC_SAMPLING_STALL_INVALID)) {
                    break;
                }

                PCSamplingRecord record;
                record.correlationId = pc->correlationId;
                record.pcOffset = pc->pcOffset;
                record.functionIndex = pc->functionId;
                record.stallReason = pc->stallReason;
                record.samples = pc->samples;
                record.latencySamples = pc->latencySamples;
                record.functionName[0] = '\0';
                profiler->add_pc_sampling_record(record);
                break;
            }
            case CUPTI_ACTIVITY_KIND_PC_SAMPLING_RECORD_INFO: {
                auto* info =
                    reinterpret_cast<CUpti_ActivityPCSamplingRecordInfo*>(activity);
                profiler->add_pc_sampling_record_info(*info);
                break;
            }
            default:
                break;
        }
    }

    free(buffer);
}
