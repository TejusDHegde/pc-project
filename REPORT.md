# Case Study 15: CUDA Vector Addition and Kernel Optimization with AI-Based Configuration

This report follows the 10-step project workflow. All numbers come from
`results/RESULTS.md`, which `analysis/plots.py` generates from the measurements
taken on the GPU you run the project on. The charts are in `results/figures/`.

---

## 1. Problem and objectives

**Problem.** Add two single-precision vectors element by element, `C[i] = A[i] + B[i]`,
for sizes from N = 1,024 up to about 67 million elements. Find the CUDA execution
configuration (kernel strategy, threads per block, number of blocks) that runs fastest
for each size.

**Objectives**

1. Write a sequential CPU baseline, plus an OpenMP multi-core CPU version to compare against.
2. Write a CUDA version and study every stage of the workflow: memory allocation,
   host-to-device transfer, kernel execution and the return of results.
3. Try several parallel strategies and many thread, block and grid configurations.
4. Measure execution time, speedup, utilisation, waiting time and throughput for
   different input sizes and numbers of cores.
5. Train an AI/ML model that predicts the most efficient CUDA configuration for a
   given vector size.
6. Use the model's output to configure the CUDA program automatically, and compare
   that choice with the default configuration and with the best configuration found
   by trying every option.

### How the generic workflow terms map to this case study

The common workflow template talks about a "scheduler", "burst time" and "cores".
For a CUDA case study they mean the following:

| Workflow term | Meaning in this project |
|---|---|
| Parallel scheduler | How the work is mapped onto the GPU: the CUDA launch configuration (strategy, block, grid), chosen by the program or by the AI |
| Scheduling strategies | 3 kernel strategies (naive, grid-stride, float4) × 6 block sizes × several grid sizes |
| Number of cores | GPU: threads, blocks and the number of SMs the grid covers. CPU: OpenMP with 1…all cores |
| Burst time | Kernel execution time of one configuration, which the AI model predicts |
| Waiting time | Time the computation waits for data: host-to-device copy plus device-to-host copy |
| Processor utilisation | Achieved memory bandwidth as a % of measured peak, theoretical SM occupancy, and % of SMs given work |
| Throughput | Elements processed per second (G elements/s) and GB/s |

---

## 2. Prompts used (initial and refined)

**Initial prompt** (the case-study text, given as is):

> Implement vector addition using CUDA. Develop a sequential CPU implementation and a
> CUDA implementation using different numbers of threads, blocks, and grids. Investigate
> memory allocation, data transfer between host and device, kernel execution, and
> returning results from the CUDA kernel. Develop an AI model that predicts the most
> efficient CUDA configuration for a given vector size. Keep it small but demonstrable.

**Refinement 1: the environment.** No local GPU or laptop was available, so the
prompt was extended:

> The project must run end to end in Google Colab (free T4 GPU) from one notebook with
> "Run all", and also on any Linux machine with a GPU via a script. The data must be
> measured on the real GPU, never simulated.

**Refinement 2: what to measure.**

> Time each CUDA stage separately with CUDA events: cudaMalloc, H2D, kernel, D2H. Compare
> pageable and pinned host memory. Report execution time, speedup (kernel-only and
> end-to-end), waiting time (transfers), throughput, and utilisation (bandwidth vs
> measured peak, occupancy). Verify every GPU result against the CPU result.

**Refinement 3: strategies and configurations.**

> Use three kernel strategies: one-thread-per-element, grid-stride loop, and vectorised
> float4 loads. Sweep block sizes 32–1024 and grid sizes from 1× to 32× the SM count.
> Sweep vector sizes 2^10–2^26 including midpoints (1.5·2^k), so the model can be tested
> on sizes it has never seen.

**Refinement 4: the AI model.**

> Train a regression model that predicts kernel time (the "burst time") from vector size
> and configuration, then recommend the configuration with the lowest predicted time.
> Compare several models, and evaluate with cross-validation grouped by vector size.
> Report prediction error and recommendation quality (% of optimal speed, top-1, within
> 5%). Feed the recommendation back into the CUDA program (`auto` mode).

---

## 3. Sequential baseline

`cpu_add_sequential()` in `src/vector_add.cu` is a single loop on one core. It is timed
with `std::chrono` as the median of repeated runs (at least 30 ms of total work). Its
output is the reference result that every GPU run is checked against, element by element.

`cpu_add_openmp()` is the same loop with `#pragma omp parallel for`. The sweep runs it with
1, 2, 4, … up to all available threads, to show how the CPU scales with the number of
cores (`results/cpu_cores.csv`, chart `fig7_cpu_cores.png`).

## 4. Parallel CUDA implementation

`run_one()` in `src/vector_add.cu` prints and times every stage:

1. **Host allocation:** pageable memory (`std::vector`) or pinned memory (`cudaMallocHost`).
2. **Device allocation:** three calls to `cudaMalloc`.
3. **Host-to-device transfer:** `cudaMemcpy(..., cudaMemcpyHostToDevice)` for A and B.
4. **Kernel execution:** the median of 20 timed launches after 2 warm-up launches, timed with CUDA events.
5. **Returning results:** `cudaMemcpy(..., cudaMemcpyDeviceToHost)` for C.
6. **Verification** against the CPU result, then `cudaFree` / `cudaFreeHost`.

## 5. Scheduling strategies (kernel configurations)

| Strategy | Kernel | Grid | Idea |
|---|---|---|---|
| `naive` | one thread per element | ceil(N / block) | the textbook mapping |
| `gridstride` | each thread loops with stride `gridDim·blockDim` | SMs × {1,2,4,8,16,32} | a fixed number of blocks, reused for every N |
| `float4` | each thread loads and stores 16 bytes (4 floats) | ceil(N / 4 / block) | fewer, wider memory transactions |

Each strategy is run with block sizes {32, 64, 128, 256, 512, 1024}. That gives
**48 configurations for every vector size.**

## 6. Workloads and numbers of cores

- **Workloads:** 33 vector sizes, from 2^10 to 2^26 elements (4 KB to 256 MB per vector),
  including the 1.5·2^k sizes between powers of two.
- **GPU parallelism:** 32 to 1024 threads per block. Grids range from fewer blocks than
  SMs (where some SMs sit idle, shown by `sm_coverage_pct`) up to tens of thousands of blocks.
- **CPU cores:** OpenMP with 1…all threads.

The sweep produces `results/gpu_configs.csv` (every configuration),
`results/summary.csv` (one row per size), `results/cpu_cores.csv` and
`results/device_info.csv`.

## 7. Performance metrics

| Metric | How it is computed |
|---|---|
| Execution time | CPU: `chrono`, median. GPU: CUDA events, median per stage |
| Speedup | `CPU sequential time / GPU kernel time`, and `CPU sequential time / (H2D + kernel + D2H)` |
| Processor utilisation | `achieved bandwidth / measured peak bandwidth` (peak measured with a large device-to-device copy). Theoretical occupancy from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. % of SMs with work |
| Waiting time | `H2D + D2H` transfer time: the GPU cannot start until the data has arrived, and the host waits for the result. Shown as a share of the total in `fig3_time_breakdown.png` |
| Throughput | `N / kernel time` (G elements/s); bandwidth `= 12·N bytes / kernel time` |

Charts: `fig1_execution_time`, `fig2_speedup`, `fig3_time_breakdown`, `fig4_block_size`,
`fig5_strategies`, `fig6_utilisation`, `fig7_cpu_cores`.

## 8. AI model for burst-time (kernel-time) prediction

- **Features:** log2(N), strategy (one-hot), log2(block), log2(grid), log2(elements per thread).
- **Target:** log10(kernel time / ideal time). The ideal time is `12·N bytes / peak bandwidth`.
  Predicting the slowdown relative to this hardware limit makes the model learn the
  effect of the configuration. A raw-time target would mostly learn that a larger N takes longer.
- **Models compared:** Linear Regression, Decision Tree, Random Forest and Gradient Boosting.
- **Evaluation:** 5-fold GroupKFold over vector sizes, so the test sizes are never seen in training.
  The reported metrics are:
  - R² and MAPE of the predicted time;
  - **recommendation efficiency**, the best possible time divided by the time of the configuration the model picks, averaged over sizes;
  - top-1 accuracy;
  - the % of sizes where the pick is within 5% of the best.

  The model with the highest recommendation efficiency is selected automatically.
  The results go to `results/model_comparison.csv` and charts `fig8` and `fig9`.

The model is a regressor followed by an argmin over the candidate configurations,
rather than a direct classifier. Many configurations are near-ties (within 1–2%), so a
classifier's "class" labels would be noisy. Predicting the time of every option and taking
the fastest copes with ties naturally, and it also gives a time estimate for each choice.

## 9. Integrating the AI with the CUDA program

There are two ways the AI's output drives the program:

1. `python3 ai/predict.py --n N --run` ranks all 48 candidate configurations for any N,
   launches `bin/vector_add` with the predicted best, then runs the default (naive, 256)
   and reports the measured speed-up and the prediction error.
2. `train_model.py` exports `models/recommendation_table.csv` (the best configuration for
   sizes at 2^0.25 steps). `vector_add auto --n N` reads this table inside the CUDA program
   and launches the recommended configuration. The CUDA program does not need Python at run time.

## 10. Comparison, optimisation and conclusions

Read these together with the measured numbers in `results/RESULTS.md`.

1. **Vector addition is memory-bound.** There is one addition for every 12 bytes moved, so
   the best kernels reach a large fraction of the GPU's peak memory bandwidth
   (`fig6_utilisation`). No configuration can go faster than `12·N / bandwidth`.
2. **Transfers dominate end-to-end time** (`fig3`). Kernel-only speedups over the CPU are
   large for big N. Once H2D and D2H copies over PCIe are counted, the end-to-end speedup is
   much smaller and can drop below 1×. Vector addition alone is not worth sending to the GPU;
   it pays off when the data already lives on the GPU or is reused by several kernels.
   Pinned memory makes the transfers clearly faster than pageable memory.
3. **Small vectors do not benefit from the GPU.** Below roughly 10^4–10^5 elements, the
   kernel launch overhead (a few µs) is larger than the CPU loop itself.
4. **Configuration matters, and it depends on size** (`fig4`, `fig5`):
   - Very small blocks (32) and grids that leave SMs idle are slow.
   - float4 or grid-stride kernels with 128–512 threads per block usually win at large N.
   - The best choice changes with N.
   - A single fixed default such as (naive, 256) leaves measurable performance unused at some sizes.
5. **The AI model generalises to unseen sizes.** The cross-validated predicted kernel times
   closely follow the measured times (`fig8`). The configurations it recommends come very
   close to the best possible time, and are on average better than the fixed default (`fig9`).
   The recommendation runs automatically in `vector_add auto`.
6. **CPU multi-core scaling is limited by memory bandwidth** (`fig7`). OpenMP helps, but
   it is far from linear, because the cores share the same memory bus.

**Possible extensions:** CUDA streams to overlap transfers with computation, unified memory,
fusing vector addition with other operations so the data is reused on the GPU, and
retraining the model on several GPU types with GPU properties as extra features.
