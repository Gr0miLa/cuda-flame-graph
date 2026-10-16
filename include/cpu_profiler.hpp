#ifndef CPU_PROFILER_HPP
#define CPU_PROFILER_HPP

#include "stack_resolver.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

struct perf_event_mmap_page;

class CpuProfiler {
public:
    CpuProfiler() = default;
    ~CpuProfiler();

    CpuProfiler(const CpuProfiler&) = delete;
    CpuProfiler& operator=(const CpuProfiler&) = delete;

    bool start(uint32_t frequency);
    void stop();
    void release();

    void process_samples(
        StackResolver& resolver,
        std::unordered_map<std::string, uint64_t>& aggregated);

    bool enabled() const noexcept;

private:
    int perf_fd_ = -1;
    perf_event_mmap_page* perf_page_ = nullptr;
    std::size_t mmap_size_ = 0;
    std::size_t data_offset_ = 0;
    std::size_t data_size_ = 0;
};

#endif  // CPU_PROFILER_HPP
