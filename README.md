![ichida-logo](/resources/ichidalogo.png)

# ichida-algo
[![made-using-c](https://img.shields.io/badge/Made%20with-C%20/%20CUDA-151822.svg)](https://cplusplus.com/)
[![authors-kachi-group](https://img.shields.io/badge/Authors-kachi--group-ad7916.svg)](https://github.com/kachi-group)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build status](https://github.com/kachi-group/ichida-algo/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/kachi-group/ichida-algo/actions/workflows/ci.yml)


This is the submission repository for **StartHack 2024 (June)**. Our team chose Track 1 - HPC, which involves writing a massively parallel implementation of neural network inference on a small demo model.

Development involves:
- Optimising for the compiler (**cache** locality, throughput)
- **SIMD** programming, vector intrinsics and alignment in C
- **Multithreading** and task distribution
- **x86_64 assembly**, in-depth profiling and tuning
- Programming in **CUDA**
- **MPI** (Message-Passing Interface) for multi-GPU utilisation

We decided to go down this path because it sounded like some high risk, high reward excitement. Before starting out, we didn't know almost anything about low level optimisations, so it has been a real 'learning-on-the-job' kind of situation!

## Team members
All team members are from RMIT.
### Artemis Rosman - **rozukke**
- Project management
- CPU optimisation (AVX2/SIMD, asm, memory)
- Cleanup/rewrites/profiling
- Code review
- Communication
- Design
- A lot of textbook reading

### Dan Dang - **nhatdongdang**
- Core implementation in C
- Testing and timing
- CPU optimisation (multithread, AVX2/SIMD)
- GPU optimisation
- Code review
- CI
- A lot of textbook reading

### Johnathan Chan - **Jinxto**
- Core implementation in CUDA
- CPU optimisation (SIMD)
- A lot of textbook reading
