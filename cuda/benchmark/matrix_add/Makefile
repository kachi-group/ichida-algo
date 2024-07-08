compile = nvcc -O3 -arch=sm_75 --use_fast_math 
SRC_DIR := versions
BIN_DIR := bin
SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
EXECUTABLES := $(patsubst $(SRC_DIR)/%.cu, $(BIN_DIR)/%, $(SRC_FILES))

all: clean $(EXECUTABLES) 

clean:
	rm -f -r bin
	mkdir bin

$(BIN_DIR)/%: $(SRC_DIR)/%.cu
	$(compile) $< benchmark.cu -o $@.exe 

plot: all
	python3 ./plot.py


