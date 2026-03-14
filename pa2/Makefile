NVCC = nvcc
NVCC_FLAGS = -arch=sm_70 -std=c++11 -lcublas

# Directories
SRC_DIR = src
BIN_DIR = bin

# Target executable
CUDA_SOURCES = $(wildcard $(SRC_DIR)/*.cu)
TARGETS = $(patsubst $(SRC_DIR)/%.cu, $(BIN_DIR)/%, $(CUDA_SOURCES))

# Default target
all: $(BIN_DIR) $(TARGETS) 

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Rule to compile and run CUDA source files, should run even if the file is not modified
$(BIN_DIR)/%: $(SRC_DIR)/%.cu main.cc force | $(BIN_DIR)
	@$(NVCC) -DLARGE_TESTS -DNOT_LEETGPU $(NVCC_FLAGS) -o $@ $< main.cc
	@echo "Running $@..."
	$@
	@echo "Finished running $@.\n\n"

leetgpu:
	@leetgpu run main.cc $(FILE)

# Clean target
clean:
	rm -f $(BIN_DIR)/*

force:

.PHONY: all clean leetgpu force

