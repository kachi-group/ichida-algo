.PHONY: all clean verify run build test

all: build

clean:
	rm -f test/results.csv
	rm -f results.csv
	rm -rf build
	rm -f speed_gpu

build:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	$(MAKE) -C build
	cp build/speed_gpu ./

run: build
	./script/speed_demo_gpu.sh ./weights_and_biases.txt ./tensors

test: build
	./speed_gpu ./weights_and_biases.txt ./tensors

verify: build
	./script/speed_demo_gpu.sh ./weights_and_biases.txt ./tensors
	mv ./results.csv ./test
	python3 ./test/verify_csv.py