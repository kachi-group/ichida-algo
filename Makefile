.PHONY: all clean verify verifygpu run rungpu build buildgpu test testgpu

all: build buildgpu

clean:
	rm -f test/results.csv
	rm -f results.csv
	rm -rf build
	rm -f speed_cpu
	rm -f speed_gpu
	rm -rf build_gpu

build:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_CPU=ON -DBUILD_GPU=OFF
	$(MAKE) -C build
	cp build/speed_cpu ./

buildgpu:
	cmake -S . -B build_gpu -DCMAKE_BUILD_TYPE=Release -DBUILD_CPU=OFF -DBUILD_GPU=ON
	$(MAKE) -C build_gpu
	cp build_gpu/speed_gpu ./

run: build
	./speed_demo_cpu.sh ./weights_and_biases.txt ./tensors

rungpu: buildgpu
	./speed_demo_gpu.sh ./weights_and_biases.txt ./tensors

test: build
	./speed_cpu ./weights_and_biases.txt ./tensors

testgpu: buildgpu
	./speed_gpu ./weights_and_biases.txt ./tensors

verify: build
	./speed_demo_cpu.sh ./weights_and_biases.txt ./tensors
	mv ./results.csv ./test
	python3 ./test/verify_csv.py

verifygpu: buildgpu
	./speed_demo_gpu.sh ./weights_and_biases.txt ./tensors
	mv ./results.csv ./test
	python3 ./test/verify_csv.py