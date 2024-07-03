.PHONY: all test clean run build run_test

all: rebuild

clean:
	rm -f test/results.csv
	rm -f results.csv
	rm -rf build
	rm -f speed_cpu

build: clean
	cmake -Bbuild
	$(MAKE) -C ./build
	mv ./build/speed_cpu ./
    
rebuild:
	$(MAKE) -C ./build
	mv ./build/speed_cpu ./

run: build
	./speed_demo_cpu.sh ./weights_and_biases.txt ./tensors

run_test: build
	./speed_cpu ./weights_and_biases.txt ./tensors

test: build
	./speed_cpu ./weights_and_biases.txt ./tensors 1
	mv ./results.csv ./test
	python3 ./test/verify_csv.py

bench: build
	./build/benchmark

stat: build
	python3 ./benchmark/stat.py


