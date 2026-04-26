NVCC ?= nvcc
CUDA_PATH ?= /usr/local/cuda
NVCC_FLAGS = -O2 -std=c++20 -Xcompiler -fPIC -Icsrc -gencode arch=compute_120a,code=sm_120a --expt-relaxed-constexpr --extended-lambda
FLASHINFER_INC = -Ivendor/flashinfer/include
LIB_NAME = libglm_ops.so

BUILD_DIR := build/Release

SRCS_CU := csrc/glm_ops.cu csrc/glm_flash.cu csrc/glm_gemv.cu csrc/glm_gdn.cu csrc/glm_sampling.cu
SRCS_CPP := csrc/glm_nccl.cpp csrc/glm_device.cpp
OBJS := $(SRCS_CU:csrc/%.cu=$(BUILD_DIR)/%.o) $(SRCS_CPP:csrc/%.cpp=$(BUILD_DIR)/%.o)

.PHONY: all clean test

all: $(BUILD_DIR)/$(LIB_NAME)

$(BUILD_DIR)/$(LIB_NAME): $(OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ $^ \
		$(FLASHINFER_INC) \
		-I$(CUDA_PATH)/include \
		-L$(CUDA_PATH)/lib64 -lcublas -lcudart -lnccl

$(BUILD_DIR)/%.o: csrc/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $< \
		$(FLASHINFER_INC) \
		-I$(CUDA_PATH)/include

$(BUILD_DIR)/%.o: csrc/%.cpp | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $< \
		-I$(CUDA_PATH)/include

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -f $(BUILD_DIR)/$(LIB_NAME) $(OBJS)
