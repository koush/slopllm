NVCC ?= nvcc
CUDA_PATH ?= /usr/local/cuda
NVCC_FLAGS = -O2 -Xcompiler -fPIC -Icsrc -gencode arch=compute_120,code=sm_120
FLASHINFER_INC = -Ivendor/flashinfer/include
LIB_NAME = libglm_ops.so

BUILD_DIR := build/Release

.PHONY: all clean test

all: $(BUILD_DIR)/$(LIB_NAME)

$(BUILD_DIR)/$(LIB_NAME): csrc/glm_ops.cu csrc/glm_flash.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ $^ \
		$(FLASHINFER_INC) \
		-I$(CUDA_PATH)/include \
		-L$(CUDA_PATH)/lib64 -lcublas -lcudart

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -f $(BUILD_DIR)/$(LIB_NAME)
