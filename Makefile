NVCC = nvcc
NVCC_FLAGS = -arch=sm_70 -std=c++11

# Directories
SRC_DIR = src
BIN_DIR = bin

# Source files
CUDA_SOURCES = $(wildcard $(SRC_DIR)/*.cu)
CUDA_OBJECTS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CUDA_SOURCES))

# Target executable
# for each file in src, have a bin which is the same name as the file in src compiled with main.cc
TARGETS = $(patsubst $(SRC_DIR)/%.cu, $(BIN_DIR)/%, $(CUDA_SOURCES))

# Default target
all: $(TARGETS)

# Rule to compile CUDA source files
$(BIN_DIR)/%: $(SRC_DIR)/%.cu main.cc
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

leetgpu:
	leetgpu run main.cc $(FILE)

# Clean target
clean:
	rm -f $(BIN_DIR)/*

.PHONY: all clean

