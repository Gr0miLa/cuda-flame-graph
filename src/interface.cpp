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
        printf("  -f, --freq <value>        Set sampling frequency in Hz (default: 99)\n");
        printf("  -p, --profile <name>      Set filtering profile (default: standard)\n");
        printf("  -h, --help                Show this help message and exit\n");
        printf("\nProfiles:\n");
        printf("  minimal  - App code + CUDA API functions (cudaMalloc, cudaMemcpy, etc.)\n");
        printf("  standard - + System libraries (libc, pthread) [RECOMMENDED]\n");
        printf("  full     - + Driver/Runtime internals + Unknown functions\n");
        printf("  debug    - Everything including profiler overhead\n");
        printf("\nExample:\n");
        printf("  ./pti_loader -f 999 -p standard ./my_cuda_app\n");
    }

    int ParseArgs(int argc, char* argv[]) {
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
            } else if (strcmp(arg, "-p") == 0 || strcmp(arg, "--profile") == 0) {
                if (i + 1 < argc) {
                    std::string prof = argv[i + 1];
                    if (prof != "minimal" && prof != "standard" && prof != "full" && prof != "debug") {
                        fprintf(stderr, "Error: Invalid profile '%s'. Choose from: minimal, standard, full, debug.\n", prof.c_str());
                        exit(1);
                    }
                    setenv("PTI_PROFILE", argv[i + 1], 1);
                    i += 2;
                } else {
                    fprintf(stderr, "Error: %s requires a value.\n", arg);
                    Usage();
                    exit(1);
                }
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