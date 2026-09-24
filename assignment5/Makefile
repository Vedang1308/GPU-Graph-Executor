NVCC_PATH := $(shell which nvcc 2>/dev/null)
ifneq ($(NVCC_PATH),)
CUDA_DIR ?= $(patsubst %/bin/nvcc,%,$(NVCC_PATH))
else
CUDA_DIR ?= /usr/local/cuda
endif

CC_SRCS := $(wildcard src/*.cc)
CC_OBJS := ${CC_SRCS:src/%.cc=build/obj/%.o}
CUDA_SRCS := $(wildcard src/*.cu)
CUDA_OBJS := ${CUDA_SRCS:src/%.cu=build/obj/%.o}
OBJS := $(CC_OBJS) $(CUDA_OBJS)

CC = g++
WARNINGS = -Wall -Wfatal-errors -Wno-unused -Wno-unused-result
CC_FLAGS = -std=c++11 -fPIC $(WARNINGS) -I$(CUDA_DIR)/include
LD_FLAGS = -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lcublas

NVCC = nvcc
NVCC_FLAGS = -std=c++11 --compiler-options '-fPIC'
ARCH = -gencode arch=compute_50,code=[sm_50,compute_50] \
       -gencode arch=compute_52,code=[sm_52,compute_52] \
       -gencode arch=compute_60,code=[sm_60,compute_60] \
       -gencode arch=compute_70,code=[sm_70,compute_70] \
       -gencode arch=compute_80,code=[sm_80,compute_80]

all: build/lib/libc_runtime_api.so

build/lib/libc_runtime_api.so: $(OBJS)
	@mkdir -p build/lib
	$(CC) -shared $^ -o $@ $(LD_FLAGS)

build/obj/%.o: src/%.cc
	@mkdir -p build/obj
	$(CC) $(CC_FLAGS) -c $< -o $@

build/obj/%.o: src/%.cu
	@mkdir -p build/obj
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf build

.PHONY: clean
