#ifndef STACK_RESOLVER_HPP
#define STACK_RESOLVER_HPP

#include <cstdint>
#include <string>
#include <unordered_map>

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

struct CachedFrame {
    std::string name;
    FrameCategory category;
};

class StackResolver {
public:
    void configure(uint32_t filter_mask, bool debug);

    std::string resolve(void* const* callstack, int frames,
                        const std::string& kernel_name = "");
    std::string clean_name(const char* mangled_name) const;

private:
    FrameCategory get_frame_category(const std::string& name) const;

    uint32_t filter_mask_ = FLAG_DEFAULT;
    bool debug_ = false;
    std::unordered_map<void*, CachedFrame> symbol_cache_;
};

#endif  // STACK_RESOLVER_HPP
