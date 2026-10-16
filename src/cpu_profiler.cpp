#include "cpu_profiler.hpp"

#include "profiler.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr std::size_t kMappedDataPages = 128;

long perf_event_open(perf_event_attr* event, pid_t pid, int cpu,
                     int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, event, pid, cpu, group_fd, flags);
}

void copy_from_ring(const char* data, std::size_t data_size, uint64_t offset,
                    void* destination, std::size_t length) {
    const std::size_t ring_offset = offset % data_size;
    const std::size_t first_part = std::min(length, data_size - ring_offset);
    memcpy(destination, data + ring_offset, first_part);
    if (first_part < length) {
        memcpy(static_cast<char*>(destination) + first_part,
               data, length - first_part);
    }
}

}  // namespace

CpuProfiler::~CpuProfiler() {
    release();
}

bool CpuProfiler::start(uint32_t frequency) {
    perf_event_attr event{};
    event.type = PERF_TYPE_SOFTWARE;
    event.size = sizeof(perf_event_attr);
    event.config = PERF_COUNT_SW_CPU_CLOCK;
    event.sample_freq = frequency;
    event.freq = 1;
    event.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID |
                        PERF_SAMPLE_TIME | PERF_SAMPLE_CALLCHAIN;
    event.exclude_idle = 1;
    event.disabled = 1;

    perf_fd_ = perf_event_open(&event, 0, -1, -1, 0);
    if (perf_fd_ == -1) {
        fprintf(stderr, "Error: CPU profiling requires root privileges.\n");
        fprintf(stderr, "Solution: Run with 'sudo' or set:\n");
        fprintf(stderr, "  sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'\n");
        return false;
    }

    const std::size_t page_size = getpagesize();
    mmap_size_ = page_size * (1 + kMappedDataPages);

    void* base = mmap(nullptr, mmap_size_, PROT_READ | PROT_WRITE,
                      MAP_SHARED, perf_fd_, 0);
    if (base == MAP_FAILED) {
        fprintf(stderr, "Error mmaping perf buffer\n");
        close(perf_fd_);
        perf_fd_ = -1;
        mmap_size_ = 0;
        data_size_ = 0;
        return false;
    }

    perf_page_ = static_cast<perf_event_mmap_page*>(base);
    data_offset_ = perf_page_->data_offset != 0 ? perf_page_->data_offset : page_size;
    data_size_ = perf_page_->data_size != 0
        ? perf_page_->data_size
        : page_size * kMappedDataPages;

    if (data_offset_ + data_size_ > mmap_size_) {
        fprintf(stderr, "Error: invalid perf ring buffer metadata\n");
        release();
        return false;
    }

    ioctl(perf_fd_, PERF_EVENT_IOC_RESET, 0);
    ioctl(perf_fd_, PERF_EVENT_IOC_ENABLE, 0);
    return true;
}

void CpuProfiler::stop() {
    if (perf_fd_ != -1) {
        ioctl(perf_fd_, PERF_EVENT_IOC_DISABLE, 0);
    }
}

void CpuProfiler::release() {
    stop();

    if (perf_page_ != nullptr) {
        munmap(perf_page_, mmap_size_);
        perf_page_ = nullptr;
    }

    if (perf_fd_ != -1) {
        close(perf_fd_);
        perf_fd_ = -1;
    }

    mmap_size_ = 0;
    data_offset_ = 0;
    data_size_ = 0;
}

bool CpuProfiler::enabled() const noexcept {
    return perf_fd_ != -1 && perf_page_ != nullptr;
}

void CpuProfiler::process_samples(
    StackResolver& resolver,
    std::unordered_map<std::string, uint64_t>& aggregated) {
    if (!enabled()) return;

    const uint64_t head = __atomic_load_n(&perf_page_->data_head, __ATOMIC_ACQUIRE);
    uint64_t tail = perf_page_->data_tail;
    const char* data = reinterpret_cast<const char*>(perf_page_) + data_offset_;
    std::vector<uint8_t> wrapped_record;

    while (tail < head) {
        perf_event_header header;
        copy_from_ring(data, data_size_, tail, &header, sizeof(header));

        if (header.size < sizeof(perf_event_header) || header.size > data_size_ ||
            tail + header.size > head) {
            fprintf(stderr, "Warning: invalid or incomplete perf record\n");
            break;
        }

        const std::size_t ring_offset = tail % data_size_;
        const uint8_t* record = nullptr;
        if (ring_offset + header.size <= data_size_) {
            record = reinterpret_cast<const uint8_t*>(data + ring_offset);
        } else {
            wrapped_record.resize(header.size);
            copy_from_ring(data, data_size_, tail, wrapped_record.data(), header.size);
            record = wrapped_record.data();
        }

        if (header.type == PERF_RECORD_SAMPLE) {
            const uint8_t* cursor = record + sizeof(perf_event_header);
            const uint8_t* record_end = record + header.size;

            auto read_value = [&cursor, record_end](auto& value) {
                if (static_cast<std::size_t>(record_end - cursor) < sizeof(value)) {
                    return false;
                }
                memcpy(&value, cursor, sizeof(value));
                cursor += sizeof(value);
                return true;
            };

            uint64_t ip = 0;
            uint32_t pid = 0;
            uint32_t tid = 0;
            uint64_t time = 0;
            uint64_t frame_count = 0;

            if (!read_value(ip) || !read_value(pid) || !read_value(tid) ||
                !read_value(time) || !read_value(frame_count)) {
                fprintf(stderr, "Warning: truncated perf sample\n");
                tail += header.size;
                continue;
            }

            void* callstack[MAX_STACK_DEPTH];
            int depth = 0;

            if (ip && depth < MAX_STACK_DEPTH) {
                callstack[depth++] = reinterpret_cast<void*>(ip);
            }

            const uint64_t available_frames =
                static_cast<uint64_t>(record_end - cursor) / sizeof(uint64_t);
            const uint64_t readable_frames = std::min(frame_count, available_frames);
            for (uint64_t i = 0; i < readable_frames && depth < MAX_STACK_DEPTH; ++i) {
                uint64_t address = 0;
                read_value(address);
                if (address) {
                    callstack[depth++] = reinterpret_cast<void*>(address);
                }
            }

            std::string path = resolver.resolve(callstack, depth);
            if (path.find("__device_stub__") == std::string::npos && !path.empty()) {
                if (path.back() == ';') path.pop_back();
                aggregated[path] += 1;
            }
        }

        tail += header.size;
    }

    __atomic_store_n(&perf_page_->data_tail, tail, __ATOMIC_RELEASE);
}
