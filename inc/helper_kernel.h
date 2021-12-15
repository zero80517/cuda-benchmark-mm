#ifndef HELPER_KERNEL_H_
#define HELPER_KERNEL_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 32

// there is used slow global memory
__global__ void kernel_global(float *a, float *b, int n, float *c) {
  int bx = blockIdx.x;   // block number along x
  int by = blockIdx.y;   // block number along y
  int tx = threadIdx.x;  // thread number along x
  int ty = threadIdx.y;  // thread number along y
  float sum = 0.0f;
  int ia = n * (BLOCK_SIZE * by + ty);  // row number from A'
  int ib = BLOCK_SIZE * bx + tx;        // column number from B'
  int ic = ia + ib;  // element number from C'
  // calculate element of matrix C'
  for (int k = 0; k < n; k++) sum += a[ia + k] * b[ib + k * n];
  c[ic] = sum;
}

// there is a bank conflict of 32nd order
__global__ void kernel_smem_1(float *a, float *b, int n, float *c) {
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int aBegin = n * BLOCK_SIZE * by, aEnd = aBegin + n - 1;
  int bBegin = BLOCK_SIZE * bx, aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * n;
  float sum = 0.0f;
  __shared__ float as[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE];
  for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
    as[tx][ty] = a[ia + n * ty + tx];
    bs[tx][ty] = b[ib + n * ty + tx];
    __syncthreads();
    for (int k = 0; k < BLOCK_SIZE; k++) sum += as[k][ty] * bs[tx][k];
    __syncthreads();
  }
  c[aBegin + bBegin + ty * n + tx] = sum;
}

// there is no bank conflict
__global__ void kernel_smem_2(float *a, float *b, int n, float *c) {
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int aBegin = n * BLOCK_SIZE * by, aEnd = aBegin + n - 1;
  int bBegin = BLOCK_SIZE * bx, aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * n;
  float sum = 0.0f;
  __shared__ float as[BLOCK_SIZE][BLOCK_SIZE + 1];
  __shared__ float bs[BLOCK_SIZE][BLOCK_SIZE + 1];
  for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
    as[tx][ty] = a[ia + n * ty + tx];
    bs[tx][ty] = b[ib + n * ty + tx];
    __syncthreads();
    for (int k = 0; k < BLOCK_SIZE; k++) sum += as[k][ty] * bs[tx][k];
    __syncthreads();
  }
  c[aBegin + bBegin + ty * n + tx] = sum;
}

#endif  // HELPER_KERNEL_H_
