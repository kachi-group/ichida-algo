![ichida-logo](/resources/ichidalogo.png)

# ichida-algo
[![made-using-c](https://img.shields.io/badge/Made%20with-C%20/%20CUDA-151822.svg)](https://cplusplus.com/)
[![authors-kachi-group](https://img.shields.io/badge/Authors-kachi--group-ad7916.svg)](https://github.com/kachi-group)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build status](https://github.com/kachi-group/ichida-algo/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/kachi-group/ichida-algo/actions/workflows/ci.yml)

This is the submission repository for **StartHack 2024 (June)**. Our team chose Track 1 - HPC, which was hosted by QDX and involved writing a massively parallel implementation of neural network inference on a small demo model.

**Development involved:**
- Optimising for the compiler (**cache** locality, throughput)
- **SIMD** programming, vector intrinsics and alignment in C
- **Multithreading** and task distribution
- **x86_64 assembly**, in-depth profiling and tuning
- Programming in **CUDA**
- **MPI** (Message-Passing Interface) for multi-GPU utilisation

We decided to go down this path because it sounded like some high risk, high reward excitement. Before starting out, we didn't know almost anything about low level
optimisation & GPU programming, so it has been a lot of active learning on the job!

## Installation
In order to correctly run the code, please ensure that you have an x86_64 CPU if you want to test the CPU implementation (as well as OpenMP for multithreading), and a CUDA compatible GPU to test the GPU implementation (as well as the appropriate version of the CUDA toolkit). Please ensure that if you are using multiple GPUs you have an MPI implementation installed (we have verified OpenMPI as working).

To compile and run on CPU (multithreaded):
- `make`
- Run with the provided script: `./speed_demo_cpu.sh ./weights_and_biases.txt ./tensors <iterations_per_input>`

To compile and run with a non-MPI setup:
- `make build_gpu`
- Run with `./speed_gpu ./weights_and_biases.txt ./tensors <iterations_per_input>`

To compile and run with a MPI (multi-gpu) setup:
- `make`
- Run with the provided script: `./speed_demo_gpu.sh ./weights_and_biases.txt ./tensors <iterations_per_input>`

## Implementation details
- The CPU matmul kernel is written using SIMD intrinsics, entirely in C! It makes heavy use of memory alignment, cache locality with a transposition step,
and is quite fast. In fact, as far as we're aware, it beats the inline asm version provided by `cblas` by a noticeable margin for this usecase!
- To skirt around the problem of each matrix calculation being quite small, we went with a monolithic kernel design, where inferences are run essentially
per thread. It took some wrestling with the way CUDA works, but we also managed to get it running at a satisfing speed. This was an interesting first
CUDA experience, and there really wasn't much material about this approach, but we are happy with how it turned out in the end (especially since one of
us only owns a MacBook).
- We divide work evenly among the available GPUs for a speedup using MPI. After some struggles with setup, we managed to get it working. We want to thank
the QDX team for the convenient test machine that they set up for the competition, as it helped us get the multi-GPU aspect as correct as we could.

## Results
Thes are the best runs that we have achieved **(all categories were tested on 52 inputs)**:

| Hardware Used | Parallelism | Best Run / Nr. of iterations | Throughput (time for 1B) |
|---------------|-------------|-----------------------------:|-------------------------:|
| Ryzen 5600x   | 1 thread    | 6.658s   / 100k per input    | 21 minutes 20.34 seconds |
| Ryzen 5600x   | 12 threads  | 11.631s  / 1M per input      | 3 minutes 43.62 seconds  |
| EPYC 7J13*    | 240 threads | 112.124s / 100M per input    | 21.56 seconds            |
| A100 80GB     | 1 GPU       | 103.833s / 100M per input    | 19.96 seconds            |
| A100 80GB     | 8 GPUs      | 70.388s  / 500M per input    | 2.71 seconds             |

*\* Dual socket system with 2x CPUs each at 64 cores / 128 threads* 

## Team members
All team members are from RMIT.
### Artemis Rosman - **rozukke**
- Project management
- CPU optimisation (AVX2/SIMD, kernel, memory, multithreading, testing/profiling & tuning)
- GPU optimisation (monolithic kernel design & work division, small tweaks)
- MPI optimisation (work division)
- Code rewrites & cleanup, code review/maintenance
- Communication & design
- A lot of textbook reading

### Dan Dang - **nhatdongdang**
- Core implementation in C
- Benchmark implementation
- CPU optimisation (AVX2/SIMD, testing & tuning)
- GPU optimisation (memory, kernel implementation, testing/profiling & tuning)
- MPI optimisation
- Code review & CI pipeline
- A lot of textbook reading

### Johnathan Chan - **Jinxto**
- Core implementation in CUDA
- Core MPI implementation & optimisation (GPU detection)
- Builds & CMake setup
- Teamwork :D
- A lot of textbook reading
