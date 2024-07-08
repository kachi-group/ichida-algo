.PHONY: all clean build build_mpi build_gpu test_cpu test_gpu bench stat

# Default iterations
iterations ?= 100000

all: build

clean:
	rm -f test/results.csv
	rm -f results.csv
	rm -rf build
	rm -f speed_cpu speed_gpu

build_gpu: clean
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	$(MAKE) -C build
	cp -u build/speed_cpu ./
	if [ -f build/speed_gpu ]; then cp -u build/speed_gpu ./; fi

build_mpi: clean
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCOMPILE_MPI=True
	$(MAKE) -C build
	cp -u build/speed_cpu ./
	if [ -f build/speed_gpu ]; then cp -u build/speed_gpu ./; fi

build: build_mpi

run_cpu:
	./speed_cpu ./weights_and_biases.txt ./tensors $(iterations)

run_gpu:
	./speed_gpu ./weights_and_biases.txt ./tensors $(iterations)

run_mpi:
	n_gpus=$(shell nvidia-smi --query-gpu=name --format=csv,noheader | wc -l); \
	mpirun -np $$n_gpus ./speed_gpu ./weights_and_biases.txt ./tensors $(iterations)

test_cpu: build run_cpu
	mv ./results.csv ./test
	python3 ./test/verify_csv.py

test_gpu: build run_mpi
	mv ./results.csv ./test
	python3 ./test/verify_csv.py