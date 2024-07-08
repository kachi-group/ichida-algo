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
TODO

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
Thes are the best runs that we have achieved for each category:
- CPU single thread: TBD
- CPU multithread (12 threads): TBD
- CPU multithread (256 threads): TBD
- GPU single: TBD
- GPU multi (8 cards): TBD

## Team members
All team members are from RMIT.
### Artemis Rosman - **rozukke**
- Project management
- CPU optimisation (AVX2/SIMD, kernel, asm, memory, multithreading, testing/profiling & tuning)
- GPU optimisation (monolithic kernel design & work division, small tweaks)
- MPI optimisation (work division)
- Code rewrites & cleanup, code review/maintenance
- Communication & design
- A lot of textbook reading

### Dan Dang - **nhatdongdang**
- Core implementation in C
- Benchmark implementation
- CPU optimisation (AVX2/SIMD)
- GPU optimisation (memory, kernel impl & tuning, testing/profiling)
- MPI optimisation
- Code review & CI
- A lot of textbook reading

### Johnathan Chan - **Jinxto**
- Core implementation in CUDA
- Core MPI implementation & optimisation
- Builds & CMake
- Teamwork :D
- A lot of textbook reading
