// =====================================================================
//  Case Study 15 - CUDA Vector Addition and Kernel Optimization
//
//  One program that contains:
//    * a sequential CPU baseline              (cpu_add_sequential)
//    * an OpenMP multi-core CPU version       (cpu_add_openmp)
//    * three CUDA kernel strategies           (naive / gridstride / float4)
//    * the full CUDA workflow, timed step by step:
//        host alloc -> device alloc -> H2D copy -> kernel -> D2H copy -> verify
//
//  Modes
//    vector_add info                                   device information
//    vector_add run   --n N [--strategy S] [--block B] [--grid G] [--pageable]
//    vector_add auto  --n N [--table models/recommendation_table.csv]
//    vector_add sweep [--out results] [--min-exp 10] [--max-exp 26] [--reps 20]
//
//  'auto' reads the configuration table produced by the AI model
//  (ai/train_model.py) and runs the configuration the model recommends.
// =====================================================================
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------
//  CUDA kernels (the three parallel strategies)
// ---------------------------------------------------------------------

// Strategy 1 - naive: one thread per element, grid = ceil(N / block).
__global__ void vadd_naive(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Strategy 2 - grid-stride loop: a fixed-size grid (a multiple of the SM
// count) walks over the whole vector; each thread handles several elements.
__global__ void vadd_gridstride(const float* a, const float* b, float* c, int n) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        c[i] = a[i] + b[i];
}

// Strategy 3 - vectorised: each thread loads/stores float4 (16 bytes),
// giving fewer, wider memory transactions. The 0-3 leftover elements are
// handled by the first threads.
__global__ void vadd_float4(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int n4 = n / 4;
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* c4 = reinterpret_cast<float4*>(c);
    for (int i = idx; i < n4; i += stride) {
        float4 x = a4[i], y = b4[i];
        c4[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }
    int t = n4 * 4 + idx;
    if (t < n) c[t] = a[t] + b[t];
}

enum Strategy { NAIVE = 0, GRIDSTRIDE = 1, FLOAT4 = 2 };
static const char* STRATEGY_NAMES[] = {"naive", "gridstride", "float4"};

static Strategy parse_strategy(const std::string& s) {
    if (s == "naive") return NAIVE;
    if (s == "gridstride") return GRIDSTRIDE;
    if (s == "float4") return FLOAT4;
    fprintf(stderr, "Unknown strategy '%s' (use naive|gridstride|float4)\n", s.c_str());
    exit(1);
}

// Default grid for a strategy when the user does not give one.
static int default_grid(Strategy s, int n, int block, int num_sms) {
    switch (s) {
        case NAIVE: return (n + block - 1) / block;
        case GRIDSTRIDE: return num_sms * 8;
        case FLOAT4: return std::max(1, (n / 4 + block - 1) / block);
    }
    return 1;
}

static void launch(Strategy s, int grid, int block, const float* a, const float* b,
                   float* c, int n) {
    switch (s) {
        case NAIVE: vadd_naive<<<grid, block>>>(a, b, c, n); break;
        case GRIDSTRIDE: vadd_gridstride<<<grid, block>>>(a, b, c, n); break;
        case FLOAT4: vadd_float4<<<grid, block>>>(a, b, c, n); break;
    }
}

// Theoretical occupancy (% of the SM's maximum resident threads).
static double occupancy_pct(Strategy s, int block, const cudaDeviceProp& p) {
    int blocks_per_sm = 0;
    const void* k = s == NAIVE ? (const void*)vadd_naive
                  : s == GRIDSTRIDE ? (const void*)vadd_gridstride
                  : (const void*)vadd_float4;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, k, block, 0));
    return 100.0 * blocks_per_sm * block / p.maxThreadsPerMultiProcessor;
}

// ---------------------------------------------------------------------
//  CPU versions
// ---------------------------------------------------------------------
static void cpu_add_sequential(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

static void cpu_add_openmp(const float* a, const float* b, float* c, int n, int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

using Clock = std::chrono::high_resolution_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// Time a CPU function: repeat until >= 30 ms of work (min 3, max 500 runs).
template <typename F>
static double time_cpu(F f) {
    f();  // warm-up (page faults, caches)
    std::vector<double> t;
    double total = 0;
    while ((t.size() < 3 || total < 30.0) && t.size() < 500) {
        auto t0 = Clock::now();
        f();
        double ms = ms_since(t0);
        t.push_back(ms);
        total += ms;
    }
    return median(t);
}

// ---------------------------------------------------------------------
//  GPU timing helpers (CUDA events)
// ---------------------------------------------------------------------
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() { CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { CUDA_CHECK(cudaEventRecord(start)); }
    float end() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

static double time_kernel(Strategy s, int grid, int block, const float* a, const float* b,
                          float* c, int n, int reps) {
    GpuTimer t;
    for (int w = 0; w < 2; ++w) launch(s, grid, block, a, b, c, n);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> v;
    for (int r = 0; r < reps; ++r) {
        t.begin();
        launch(s, grid, block, a, b, c, n);
        v.push_back(t.end());
    }
    CUDA_CHECK(cudaGetLastError());
    return median(v);
}

static double time_copy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, int reps) {
    GpuTimer t;
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));  // warm-up
    std::vector<double> v;
    for (int r = 0; r < reps; ++r) {
        t.begin();
        CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
        v.push_back(t.end());
    }
    return median(v);
}

// Practical peak device-memory bandwidth, measured with a large D2D copy.
static double measure_peak_bw_gbs() {
    size_t bytes = 256u << 20;
    void *x, *y;
    CUDA_CHECK(cudaMalloc(&x, bytes));
    CUDA_CHECK(cudaMalloc(&y, bytes));
    CUDA_CHECK(cudaMemset(x, 1, bytes));
    double ms = time_copy(y, x, bytes, cudaMemcpyDeviceToDevice, 10);
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    return 2.0 * bytes / (ms * 1e6);  // read + write
}

static double theoretical_bw_gbs(int dev) {
    int clk_khz = 0, bus_bits = 0;
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
    return 2.0 * clk_khz * 1e3 * (bus_bits / 8.0) / 1e9;
}

static bool verify(const float* ref, const float* out, int n) {
    for (int i = 0; i < n; ++i)
        if (ref[i] != out[i]) {
            fprintf(stderr, "Mismatch at %d: %f vs %f\n", i, ref[i], out[i]);
            return false;
        }
    return true;
}

static void fill_random(std::vector<float>& a, std::vector<float>& b, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-100.f, 100.f);
    for (size_t i = 0; i < a.size(); ++i) { a[i] = d(rng); b[i] = d(rng); }
}

// ---------------------------------------------------------------------
//  Command-line parsing
// ---------------------------------------------------------------------
struct Args {
    std::string mode = "run";
    long long n = 1 << 24;
    std::string strategy = "naive";
    int block = 256;
    int grid = 0;
    bool pinned = true;
    std::string out = "results";
    std::string table = "models/recommendation_table.csv";
    int min_exp = 10, max_exp = 26, reps = 20;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    if (argc > 1) a.mode = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "Missing value for %s\n", k.c_str()); exit(1); }
            return argv[++i];
        };
        if (k == "--n") a.n = std::stoll(next());
        else if (k == "--strategy") a.strategy = next();
        else if (k == "--block") a.block = std::stoi(next());
        else if (k == "--grid") a.grid = std::stoi(next());
        else if (k == "--pageable") a.pinned = false;
        else if (k == "--out") a.out = next();
        else if (k == "--table") a.table = next();
        else if (k == "--min-exp") a.min_exp = std::stoi(next());
        else if (k == "--max-exp") a.max_exp = std::stoi(next());
        else if (k == "--reps") a.reps = std::stoi(next());
        else { fprintf(stderr, "Unknown option %s\n", k.c_str()); exit(1); }
    }
    if (a.n < 1 || a.n > (1LL << 30)) { fprintf(stderr, "--n must be in [1, 2^30]\n"); exit(1); }
    return a;
}

// ---------------------------------------------------------------------
//  Mode: info
// ---------------------------------------------------------------------
static int mode_info() {
    int dev = 0;
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    printf("GPU                     : %s (compute capability %d.%d)\n", p.name, p.major, p.minor);
    printf("Streaming multiprocessors: %d\n", p.multiProcessorCount);
    printf("Max threads / block     : %d\n", p.maxThreadsPerBlock);
    printf("Max threads / SM        : %d\n", p.maxThreadsPerMultiProcessor);
    printf("Warp size               : %d\n", p.warpSize);
    printf("Global memory           : %.2f GB\n", p.totalGlobalMem / 1e9);
    printf("Theoretical bandwidth   : %.1f GB/s\n", theoretical_bw_gbs(dev));
    printf("Measured D2D bandwidth  : %.1f GB/s\n", measure_peak_bw_gbs());
    printf("CPU threads (OpenMP)    : %d\n", omp_get_max_threads());
    return 0;
}

// ---------------------------------------------------------------------
//  Mode: run  -  one configuration, every step of the CUDA workflow shown
// ---------------------------------------------------------------------
static int run_one(const Args& args, Strategy s, int block, int grid_in) {
    const int n = (int)args.n;
    const size_t bytes = (size_t)n * sizeof(float);
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    if (block < 1 || block > p.maxThreadsPerBlock) {
        fprintf(stderr, "--block must be in [1, %d]\n", p.maxThreadsPerBlock);
        return 1;
    }
    int grid = grid_in > 0 ? grid_in : default_grid(s, n, block, p.multiProcessorCount);
    CUDA_CHECK(cudaFree(0));  // create the CUDA context up-front

    printf("=== CUDA vector addition: N = %d (%.1f MB per vector) ===\n", n, bytes / 1e6);
    printf("GPU %s | strategy=%s block=%d grid=%d | host memory=%s\n", p.name,
           STRATEGY_NAMES[s], block, grid, args.pinned ? "pinned" : "pageable");

    // 1. Host allocation + initialisation
    auto t0 = Clock::now();
    std::vector<float> a(n), b(n), ref(n);
    fill_random(a, b, 42);
    float *ha = a.data(), *hb = b.data(), *hc;
    std::vector<float> c_pageable;
    if (args.pinned) {
        CUDA_CHECK(cudaMallocHost(&ha, bytes));
        CUDA_CHECK(cudaMallocHost(&hb, bytes));
        CUDA_CHECK(cudaMallocHost(&hc, bytes));
        memcpy(ha, a.data(), bytes);
        memcpy(hb, b.data(), bytes);
    } else {
        c_pageable.resize(n);
        hc = c_pageable.data();
    }
    printf("[1] host allocation + init       : %9.3f ms\n", ms_since(t0));

    // 2. CPU sequential baseline
    double cpu_ms = time_cpu([&] { cpu_add_sequential(a.data(), b.data(), ref.data(), n); });
    int threads = omp_get_max_threads();
    std::vector<float> tmp(n);
    double omp_ms = time_cpu([&] { cpu_add_openmp(a.data(), b.data(), tmp.data(), n, threads); });
    printf("[2] CPU sequential               : %9.3f ms\n", cpu_ms);
    printf("    CPU OpenMP (%2d threads)      : %9.3f ms\n", threads, omp_ms);

    // 3. Device allocation
    float *da, *db, *dc;
    t0 = Clock::now();
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    double alloc_ms = ms_since(t0);
    printf("[3] cudaMalloc (3 vectors)       : %9.3f ms\n", alloc_ms);

    // 4. Host -> device
    GpuTimer gt;
    gt.begin();
    CUDA_CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));
    double h2d_ms = gt.end();
    printf("[4] copy host -> device (a, b)   : %9.3f ms  (%.2f GB/s)\n", h2d_ms,
           2.0 * bytes / (h2d_ms * 1e6));

    // 5. Kernel
    double k_ms = time_kernel(s, grid, block, da, db, dc, n, args.reps);
    double bw = 3.0 * bytes / (k_ms * 1e6);
    double peak = measure_peak_bw_gbs();
    printf("[5] kernel (median of %d)        : %9.3f ms  (%.2f GB/s = %.1f%% of measured peak %.1f GB/s)\n",
           args.reps, k_ms, bw, 100.0 * bw / peak, peak);
    printf("    theoretical occupancy        : %9.1f %%\n", occupancy_pct(s, block, p));
    printf("    SMs with work                : %9.1f %%\n",
           100.0 * std::min(grid, p.multiProcessorCount) / p.multiProcessorCount);

    // 6. Device -> host (fresh result)
    CUDA_CHECK(cudaMemset(dc, 0, bytes));
    launch(s, grid, block, da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    gt.begin();
    CUDA_CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
    double d2h_ms = gt.end();
    printf("[6] copy device -> host (c)      : %9.3f ms  (%.2f GB/s)\n", d2h_ms, bytes / (d2h_ms * 1e6));

    // 7. Verify
    bool ok = verify(ref.data(), hc, n);
    printf("[7] verification vs CPU result   : %s\n", ok ? "PASSED" : "FAILED");

    double total = h2d_ms + k_ms + d2h_ms;
    double wait = h2d_ms + d2h_ms;
    printf("--- summary ---\n");
    printf("GPU end-to-end (H2D+kernel+D2H)  : %9.3f ms\n", total);
    printf("waiting time (data transfers)    : %9.3f ms  (%.1f%% of GPU time)\n", wait, 100 * wait / total);
    printf("speedup kernel-only vs CPU seq   : %9.2fx\n", cpu_ms / k_ms);
    printf("speedup end-to-end vs CPU seq    : %9.2fx\n", cpu_ms / total);
    printf("throughput (kernel)              : %9.3f G elements/s\n", n / (k_ms * 1e6));
    printf("RESULT,%d,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n", n, STRATEGY_NAMES[s], block, grid,
           cpu_ms, h2d_ms, k_ms, d2h_ms, total, ok ? 1 : 0);

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    if (args.pinned) {
        CUDA_CHECK(cudaFreeHost(ha));
        CUDA_CHECK(cudaFreeHost(hb));
        CUDA_CHECK(cudaFreeHost(hc));
    }
    return ok ? 0 : 2;
}

// ---------------------------------------------------------------------
//  Mode: auto  -  use the AI model's recommendation table
// ---------------------------------------------------------------------
static int mode_auto(const Args& args) {
    std::ifstream f(args.table);
    if (!f) {
        fprintf(stderr, "Cannot open %s - train the model first (python3 ai/train_model.py)\n",
                args.table.c_str());
        return 1;
    }
    // table columns: n,strategy,block_size,grid_size,predicted_kernel_ms
    std::string line, best_line;
    double best_dist = 1e30, target = std::log2((double)args.n);
    std::getline(f, line);  // header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        double n_row = std::stod(line.substr(0, line.find(',')));
        double d = std::fabs(std::log2(n_row) - target);
        if (d < best_dist) { best_dist = d; best_line = line; }
    }
    if (best_line.empty()) { fprintf(stderr, "Recommendation table is empty\n"); return 1; }
    std::stringstream ss(best_line);
    std::string n_s, strat, blk, grd, pred;
    std::getline(ss, n_s, ',');
    std::getline(ss, strat, ',');
    std::getline(ss, blk, ',');
    std::getline(ss, grd, ',');
    std::getline(ss, pred, ',');
    Strategy s = parse_strategy(strat);
    int block = std::stoi(blk);
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    // The table stores grids for the table's N; rescale size-dependent grids to our N.
    int grid = (s == GRIDSTRIDE) ? std::stoi(grd) : default_grid(s, (int)args.n, block, p.multiProcessorCount);
    printf("AI recommendation (table row N=%s): strategy=%s block=%d grid=%d, predicted kernel %.4f ms\n",
           n_s.c_str(), strat.c_str(), block, grid, std::stod(pred));
    return run_one(args, s, block, grid);
}

// ---------------------------------------------------------------------
//  Mode: sweep  -  collect the experimental dataset
// ---------------------------------------------------------------------
static int mode_sweep(const Args& args) {
    mkdir(args.out.c_str(), 0755);
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    CUDA_CHECK(cudaFree(0));
    const int sms = p.multiProcessorCount;
    const double peak = measure_peak_bw_gbs();
    const int max_threads = omp_get_max_threads();

    {
        std::ofstream info(args.out + "/device_info.csv");
        info << "gpu,compute_capability,num_sms,max_threads_per_sm,global_mem_gb,"
                "theoretical_bw_gbs,measured_peak_bw_gbs,cpu_threads\n";
        info << p.name << "," << p.major << "." << p.minor << "," << sms << ","
             << p.maxThreadsPerMultiProcessor << "," << p.totalGlobalMem / 1e9 << ","
             << theoretical_bw_gbs(0) << "," << peak << "," << max_threads << "\n";
    }
    std::ofstream cfg(args.out + "/gpu_configs.csv");
    cfg << "n,strategy,block_size,grid_size,total_threads,kernel_ms,throughput_gelem_s,"
           "bandwidth_gbs,bw_util_pct,occupancy_pct,sm_coverage_pct,verified\n";
    std::ofstream sum(args.out + "/summary.csv");
    sum << "n,cpu_seq_ms,cpu_omp_ms,cpu_omp_threads,alloc_ms,h2d_pageable_ms,d2h_pageable_ms,"
           "h2d_pinned_ms,d2h_pinned_ms,default_kernel_ms,best_kernel_ms,best_strategy,"
           "best_block,best_grid,worst_kernel_ms\n";
    std::ofstream cores(args.out + "/cpu_cores.csv");
    cores << "n,threads,time_ms,speedup\n";

    // Sizes: every power of two and the midpoint 1.5*2^k between them.
    std::vector<long long> sizes;
    for (int e = args.min_exp; e <= args.max_exp; ++e) {
        sizes.push_back(1LL << e);
        if (e < args.max_exp) sizes.push_back(3LL << (e - 1));
    }
    const int blocks[] = {32, 64, 128, 256, 512, 1024};
    const int sm_mults[] = {1, 2, 4, 8, 16, 32};
    std::vector<int> thread_counts;
    for (int t = 1; t < max_threads; t *= 2) thread_counts.push_back(t);
    thread_counts.push_back(max_threads);

    for (long long nn : sizes) {
        const int n = (int)nn;
        const size_t bytes = (size_t)n * sizeof(float);
        printf("N = %-10d ", n);
        fflush(stdout);

        std::vector<float> a(n), b(n), ref(n), out(n);
        fill_random(a, b, (unsigned)n);
        double cpu_ms = time_cpu([&] { cpu_add_sequential(a.data(), b.data(), ref.data(), n); });
        double best_omp = 1e30;
        int best_t = 1;
        for (int t : thread_counts) {
            double ms = time_cpu([&] { cpu_add_openmp(a.data(), b.data(), out.data(), n, t); });
            cores << n << "," << t << "," << ms << "," << cpu_ms / ms << "\n";
            if (ms < best_omp) { best_omp = ms; best_t = t; }
        }

        float *pa, *pb, *pc;
        CUDA_CHECK(cudaMallocHost(&pa, bytes));
        CUDA_CHECK(cudaMallocHost(&pb, bytes));
        CUDA_CHECK(cudaMallocHost(&pc, bytes));
        memcpy(pa, a.data(), bytes);
        memcpy(pb, b.data(), bytes);

        float *da, *db, *dc;
        auto t0 = Clock::now();
        CUDA_CHECK(cudaMalloc(&da, bytes));
        CUDA_CHECK(cudaMalloc(&db, bytes));
        CUDA_CHECK(cudaMalloc(&dc, bytes));
        double alloc_ms = ms_since(t0);

        int treps = 5;
        double h2d_pg = 2 * time_copy(da, a.data(), bytes, cudaMemcpyHostToDevice, treps);
        double h2d_pn = 2 * time_copy(da, pa, bytes, cudaMemcpyHostToDevice, treps);
        CUDA_CHECK(cudaMemcpy(db, pb, bytes, cudaMemcpyHostToDevice));
        double d2h_pg = time_copy(out.data(), dc, bytes, cudaMemcpyDeviceToHost, treps);
        double d2h_pn = time_copy(pc, dc, bytes, cudaMemcpyDeviceToHost, treps);

        struct Cfg { Strategy s; int block, grid; };
        std::vector<Cfg> cfgs;
        for (int bs : blocks) {
            cfgs.push_back({NAIVE, bs, default_grid(NAIVE, n, bs, sms)});
            cfgs.push_back({FLOAT4, bs, default_grid(FLOAT4, n, bs, sms)});
            for (int m : sm_mults) cfgs.push_back({GRIDSTRIDE, bs, sms * m});
        }

        double best = 1e30, worst = 0, def = 0;
        Cfg best_cfg = cfgs[0];
        bool all_ok = true;
        for (const Cfg& c : cfgs) {
            double ms = time_kernel(c.s, c.grid, c.block, da, db, dc, n, args.reps);
            CUDA_CHECK(cudaMemset(dc, 0, bytes));
            launch(c.s, c.grid, c.block, da, db, dc, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(pc, dc, bytes, cudaMemcpyDeviceToHost));
            bool ok = verify(ref.data(), pc, n);
            all_ok &= ok;
            double bw = 3.0 * bytes / (ms * 1e6);
            cfg << n << "," << STRATEGY_NAMES[c.s] << "," << c.block << "," << c.grid << ","
                << (long long)c.block * c.grid << "," << ms << "," << n / (ms * 1e6) << "," << bw
                << "," << 100.0 * bw / peak << "," << occupancy_pct(c.s, c.block, p) << ","
                << 100.0 * std::min(c.grid, sms) / sms << "," << (ok ? 1 : 0) << "\n";
            if (ms < best) { best = ms; best_cfg = c; }
            worst = std::max(worst, ms);
            if (c.s == NAIVE && c.block == 256) def = ms;
        }
        sum << n << "," << cpu_ms << "," << best_omp << "," << best_t << "," << alloc_ms << ","
            << h2d_pg << "," << d2h_pg << "," << h2d_pn << "," << d2h_pn << "," << def << ","
            << best << "," << STRATEGY_NAMES[best_cfg.s] << "," << best_cfg.block << ","
            << best_cfg.grid << "," << worst << "\n";
        cfg.flush();
        sum.flush();
        cores.flush();
        printf("cpu %8.3f ms | best kernel %7.4f ms (%s, block %d, grid %d) | %s\n", cpu_ms, best,
               STRATEGY_NAMES[best_cfg.s], best_cfg.block, best_cfg.grid,
               all_ok ? "verified" : "VERIFY FAILED");

        CUDA_CHECK(cudaFree(da));
        CUDA_CHECK(cudaFree(db));
        CUDA_CHECK(cudaFree(dc));
        CUDA_CHECK(cudaFreeHost(pa));
        CUDA_CHECK(cudaFreeHost(pb));
        CUDA_CHECK(cudaFreeHost(pc));
        if (!all_ok) return 2;
    }
    printf("Results written to %s/\n", args.out.c_str());
    return 0;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        fprintf(stderr, "No CUDA-capable GPU found. Run this on a machine with an NVIDIA GPU "
                        "(e.g. Google Colab with Runtime > Change runtime type > T4 GPU).\n");
        return 1;
    }
    if (args.mode == "info") return mode_info();
    if (args.mode == "run") return run_one(args, parse_strategy(args.strategy), args.block, args.grid);
    if (args.mode == "auto") return mode_auto(args);
    if (args.mode == "sweep") return mode_sweep(args);
    fprintf(stderr, "Unknown mode '%s' (use info|run|auto|sweep)\n", args.mode.c_str());
    return 1;
}
