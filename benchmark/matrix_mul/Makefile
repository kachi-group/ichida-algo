compile = gcc -O3 -march=native -ffast-math -funroll-loops -Wall -Wextra
SRC_DIR := versions
BIN_DIR := bin
SRC_FILES := $(wildcard $(SRC_DIR)/*.c)
EXECUTABLES := $(patsubst $(SRC_DIR)/%.c, $(BIN_DIR)/%, $(SRC_FILES))

all: clean $(EXECUTABLES) 

clean:
	rm -f -r bin
	mkdir bin

$(BIN_DIR)/%: $(SRC_DIR)/%.c
	$(compile) $< benchmark.c -o $@ 

plot: all
	python3 ./plot.py



