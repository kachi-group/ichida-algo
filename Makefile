.PHONY: all test test_gpu clean run_cpu run_gpu build build_gpu run_test run_test_gpu

all: build build_gpu

clean:
	rm -f test/results.csv
	rm -f results.csv
	rm -rf build
	rm -f speed_cpu
	rm -f speed_gpu

build: 
	cmake -Bbuild -DCMAKE_BUILD_TYPE=Release
	$(MAKE) -C ./build
	cp ./build/speed_cpu ./

build_gpu: 
	cmake -Bbuild_gpu -DCMAKE_BUILD_TYPE=Release
	$(MAKE) -C ./build_gpu
	cp ./build_gpu/speed_gpu ./

run_cpu: build
	./speed_demo_cpu.sh ./weights_and_biases.txt ./tensors

run_gpu: build_gpu
	./speed_demo_gpu.sh ./weights_and_biases.txt ./tensors

run_test: build
	./speed_cpu ./weights_and_biases.txt ./tensors

run_test_gpu: build_gpu
	./speed_gpu ./weights_and_biases.txt ./tensors

test: build
	./speed_demo_cpu.sh ./weights_and_biases.txt ./tensors
	mv ./results.csv ./test
	python3 ./test/verify_csv.py

test_gpu: build_gpu
	./speed_demo_gpu.sh ./weights_and_biases.txt ./tensors
	mv ./results.csv ./test
	python3 ./test/verify_csv.py