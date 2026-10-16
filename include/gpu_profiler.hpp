#ifndef GPU_PROFILER_HPP
#define GPU_PROFILER_HPP

#include "profiler.hpp"

class GpuProfiler {
public:
    GpuProfiler() = default;
    ~GpuProfiler();

    GpuProfiler(const GpuProfiler&) = delete;
    GpuProfiler& operator=(const GpuProfiler&) = delete;

    bool start(const ProfilerConfig& config);
    void flush();
    void stop();

    bool pc_sampling_enabled() const noexcept;
    CUpti_ActivityPCSamplingPeriod sampling_period() const noexcept;

    const std::vector<GpuSample>& gpu_samples() const noexcept;
    const std::vector<KernelRecord>& kernel_activities() const noexcept;
    const std::vector<PCSamplingRecord>& pc_sampling_records() const noexcept;
    uint64_t pc_sampling_total_samples() const noexcept;
    uint64_t pc_sampling_dropped_samples() const noexcept;
    uint64_t pc_sampling_period_cycles() const noexcept;
    uint64_t pc_sampling_record_info_count() const noexcept;

private:
    void setup_pc_sampling();
    void add_gpu_sample(GpuSample sample);
    void add_kernel_activity(KernelRecord record);
    void add_pc_sampling_record(PCSamplingRecord record);
    void add_pc_sampling_record_info(const CUpti_ActivityPCSamplingRecordInfo& info);

    static void CUPTIAPI runtime_callback(
        void* userdata, CUpti_CallbackDomain domain,
        CUpti_CallbackId cbid, const CUpti_CallbackData* callback_info);
    static void CUPTIAPI request_activity_buffer(
        uint8_t** buffer, size_t* size, size_t* max_records);
    static void CUPTIAPI complete_activity_buffer(
        CUcontext context, uint32_t stream_id, uint8_t* buffer,
        size_t size, size_t valid_size);

    static GpuProfiler* active_instance_;

    ProfilerConfig config_;
    CUpti_SubscriberHandle subscriber_{};
    bool started_ = false;
    bool pc_sampling_enabled_ = false;

    std::vector<KernelRecord> kernel_activities_;
    std::mutex kernel_activities_mutex_;
    std::vector<GpuSample> gpu_samples_;
    std::mutex gpu_samples_mutex_;
    PCSamplingData pc_sampling_data_;
};

#endif  // GPU_PROFILER_HPP
