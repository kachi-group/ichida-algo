run:
	cmake -Bbuild
	$(MAKE) -C ./build
	mv ./build/speed_cpu ./
	./speed_demo_cpu.sh ./weights_and_biases.txt ./tensors

run_test:
	cmake -Bbuild
	$(MAKE) -C ./build
	mv ./build/speed_cpu ./
	./speed_cpu ./weights_and_biases.txt ./tensors
