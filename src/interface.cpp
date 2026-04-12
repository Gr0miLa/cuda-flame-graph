#include "profiler.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {

    __attribute__((constructor))
    void init_trace() {
        CudaProfiler::instance().init();
    }

    __attribute__((destructor))
    void finalize_trace() {
        CudaProfiler::instance().finalize();
    }

    void SetToolEnv() {
        setenv("PTI_ENABLE", "1", 1);
    }

    void Usage() {
        printf("CUDA C++ Profiler Loader\n");
        printf("Usage: ./pti_loader [options] <application> [args]\n\n");
        printf("Options:\n");
        printf("  -f, --freq <value>    Set sampling frequency in Hz (default: 99)\n");
        printf("  -s, --show-all        Show profiler internal overhead (disabled by default)\n");
        printf("  -h, --help            Show this help message and exit\n");
        printf("\nExample:\n");
        printf("  ./pti_loader -f 1000 ./my_cuda_app\n");
    }

    int ParseArgs(int argc, char* argv[]) {
        // setenv("PTI_DEBUG", "1", 1);
        int i = 1;
        while (i < argc) {
            const char* arg = argv[i];

            if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
                Usage();
                exit(0);
            } else if (strcmp(arg, "-f") == 0 || strcmp(arg, "--freq") == 0) {
                if (i + 1 < argc) {
                    int f = std::atoi(argv[i + 1]);
                    if (f <= 0) {
                        fprintf(stderr, "Error: Frequency must be a positive integer.\n");
                        exit(1);
                    }
                    setenv("PTI_FREQ", argv[i + 1], 1);
                    i += 2;
                } else {
                    fprintf(stderr, "Error: %s requires a value.\n", arg);
                    Usage();
                    exit(1);
                }
            } else if (strcmp(arg, "-i") == 0 || strcmp(arg, "--show-internal") == 0) {
                setenv("PTI_SHOW_INTERNAL", "1", 1);
                i += 1;
            } else if (strcmp(arg, "-s") == 0 || strcmp(arg, "--show-system") == 0) {
                setenv("PTI_SHOW_SYSTEM", "1", 1);
                i += 1;
            } else if (strcmp(arg, "-u") == 0 || strcmp(arg, "--show-unknown") == 0) {
                setenv("PTI_SHOW_UNKNOWN", "1", 1);
                i += 1;
            } else if (strcmp(arg, "-a") == 0 || strcmp(arg, "--show-all") == 0) {
                setenv("PTI_SHOW_ALL", "1", 1);
                i += 1;
            } else if (arg[0] != '-') {
                break;
            } else {
                fprintf(stderr, "Error: Unknown option %s\n", arg);
                Usage();
                exit(1);
            }
        }

        return i;
    }
}