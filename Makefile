
.PHONY: all test clean run build run_test

clean:
	rm -rf build
	rm speed_cpu
build: 
	cmake -Bbuild
	$(MAKE) -C ./build
	mv ./build/speed_cpu ./

run: build
	./speed_demo_cpu.sh ./weights_and_biases.txt ./tensors

run_test: build
	./speed_cpu ./weights_and_biases.txt ./tensors

test: build
	./speed_demo_cpu.sh ./weights_and_biases.txt ./tensors
	mv ./results.csv ./test
	python3 ./test/verify_csv.py


