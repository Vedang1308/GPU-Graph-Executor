#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

__global__ void array_set_kernel(float *arr, float value, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) arr[idx] = value;
}

__global__ void broadcast_to_kernel(const float *input, float *output, int input_size, int output_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < output_size) output[idx] = input[idx % input_size];
}

__global__ void reduce_sum_axis_zero_kernel(const float *input, float *output, int nrow, int ncol) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < ncol) {
    float sum = 0;
    for (int i = 0; i < nrow; ++i) {
      sum += input[i * ncol + idx];
    }
    output[idx] = sum;
  }
}

__global__ void matrix_elementwise_add_kernel(const float *a, const float *b, float *out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) out[idx] = a[idx] + b[idx];
}

__global__ void matrix_elementwise_add_by_const_kernel(const float *a, float b, float *out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) out[idx] = a[idx] + b;
}

__global__ void matrix_elementwise_multiply_kernel(const float *a, const float *b, float *out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) out[idx] = a[idx] * b[idx];
}

__global__ void matrix_multiply_by_const_kernel(const float *a, float b, float *out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) out[idx] = a[idx] * b;
}

__global__ void relu_kernel(const float *a, float *out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) out[idx] = max(a[idx], 0.0f);
}

__global__ void relu_gradient_kernel(const float *a, const float *grad, float *out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) out[idx] = a[idx] > 0.0f ? grad[idx] : 0.0f;
}

__global__ void softmax_kernel(const float *input, float *output, int nrow, int ncol) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= nrow) return;
  input += y * ncol;
  output += y * ncol;
  float maxval = *input;
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input[x]);
  }
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input[x] - maxval);
  }
  for (int x = 0; x < ncol; ++x) {
    output[x] = exp(input[x] - maxval) / sum;
  }
}

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
  int size = 1;
  for (int i = 0; i < arr->ndim; ++i) {
    size *= arr->shape[i];
  }
  int threads = 1024;
  int blocks = (size + threads - 1) / threads;
  array_set_kernel<<<blocks, threads>>>((float *)arr->data, value, size);
  return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int input_size = 1;
  for (int i = 0; i < input->ndim; ++i) {
    input_size *= input->shape[i];
  }
  int output_size = 1;
  for (int i = 0; i < output->ndim; ++i) {
    output_size *= output->shape[i];
  }
  int threads = 1024;
  int blocks = (output_size + threads - 1) / threads;
  broadcast_to_kernel<<<blocks, threads>>>((const float *)input->data, (float *)output->data, input_size, output_size);
  return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int ncol = 1;
  for (int i = 1; i < input->ndim; ++i) {
    ncol *= input->shape[i];
  }
  int nrow = input->shape[0];
  int threads = 1024;
  int blocks = (ncol + threads - 1) / threads;
  reduce_sum_axis_zero_kernel<<<blocks, threads>>>((const float *)input->data, (float *)output->data, nrow, ncol);
  return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
  int size = 1;
  for (int i = 0; i < output->ndim; ++i) {
    size *= output->shape[i];
  }
  int threads = 1024;
  int blocks = (size + threads - 1) / threads;
  matrix_elementwise_add_kernel<<<blocks, threads>>>((const float *)matA->data, (const float *)matB->data, (float *)output->data, size);
  return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
  int size = 1;
  for (int i = 0; i < output->ndim; ++i) {
    size *= output->shape[i];
  }
  int threads = 1024;
  int blocks = (size + threads - 1) / threads;
  matrix_elementwise_add_by_const_kernel<<<blocks, threads>>>((const float *)input->data, val, (float *)output->data, size);
  return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
  int size = 1;
  for (int i = 0; i < output->ndim; ++i) {
    size *= output->shape[i];
  }
  int threads = 1024;
  int blocks = (size + threads - 1) / threads;
  matrix_elementwise_multiply_kernel<<<blocks, threads>>>((const float *)matA->data, (const float *)matB->data, (float *)output->data, size);
  return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  int size = 1;
  for (int i = 0; i < output->ndim; ++i) {
    size *= output->shape[i];
  }
  int threads = 1024;
  int blocks = (size + threads - 1) / threads;
  matrix_multiply_by_const_kernel<<<blocks, threads>>>((const float *)input->data, val, (float *)output->data, size);
  return 0;
}

cublasHandle_t cublas_handle = NULL;
int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  if (cublas_handle == NULL) {
    cublasCreate(&cublas_handle);
  }
  int m = transposeA ? matA->shape[1] : matA->shape[0];
  int k = transposeA ? matA->shape[0] : matA->shape[1];
  int n = transposeB ? matB->shape[1] : matB->shape[0];
  
  float alpha = 1.0f;
  float beta = 0.0f;
  
  cublasOperation_t transb = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transa = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;
  
  cublasSgemm(cublas_handle,
              transa, transb,
              n, m, k,
              &alpha,
              (const float *)matB->data, transposeB ? k : n,
              (const float *)matA->data, transposeA ? m : k,
              &beta,
              (float *)matC->data, n);
  return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int size = 1;
  for (int i = 0; i < output->ndim; ++i) {
    size *= output->shape[i];
  }
  int threads = 1024;
  int blocks = (size + threads - 1) / threads;
  relu_kernel<<<blocks, threads>>>((const float *)input->data, (float *)output->data, size);
  return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  int size = 1;
  for (int i = 0; i < output->ndim; ++i) {
    size *= output->shape[i];
  }
  int threads = 1024;
  int blocks = (size + threads - 1) / threads;
  relu_gradient_kernel<<<blocks, threads>>>((const float *)input->data, (const float *)in_grad->data, (float *)output->data, size);
  return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int nrow = input->shape[0];
  int ncol = input->shape[1];
  int threads = 1024;
  int blocks = (nrow + threads - 1) / threads;
  softmax_kernel<<<blocks, threads>>>((const float *)input->data, (float *)output->data, nrow, ncol);
  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
