#ifndef HELPER_TEST_H_
#define HELPER_TEST_H_

#include <cublas_v2.h>
#include <cublasXt.h>

#include "reduced_helper_cuda.h"
#include "helper_kernel.h"

#define MAX_ERROR 1e-6

void test_cublas(const int N) {
  printf("\n\tcublas calculation time: \n");

  int m, n, k;
  // create event-variables
  float timerValueGPU;
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  int numBytes = N * N * sizeof(float);
  float *adev, *bdev, *cdev, *a, *b, *c, *cc, *bT;
  // allocate memory on host
  a = (float *)malloc(numBytes);  //A matrix
  b = (float *)malloc(numBytes);  //B matrix
  bT = (float *)malloc(numBytes);  //transposed B matrix
  c = (float *)malloc(numBytes);  //C matrix for GPU
  cc = (float *)malloc(numBytes);  //C matrix for CPU

  // initialize A, B matrices and transposed B matrix (column-major)
  for (n = 0; n < N; n++) {
    for (m = 0; m < N; m++) {
      a[n + m * N] = 2.0f * m + n;
      b[n + m * N] = m - n;
      bT[n + m * N] = n - m;
    }
  }
  // create cuBLAS handle and other parameters for Sgemm
  cublasHandle_t handle;
  checkCudaErrors(cublasCreate(&handle));
  float alpha = 1.0;
  float beta = 0.0;
  // allocate memory on GPU
  checkCudaErrors(cudaMalloc((void **)&adev, numBytes));
  checkCudaErrors(cudaMalloc((void **)&bdev, numBytes));
  checkCudaErrors(cudaMalloc((void **)&cdev, numBytes));

  // ---------------- GPU ------------------------
  // copy matrices A and B from host to device
  checkCudaErrors(cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice));
  // run timer
  checkCudaErrors(cudaEventRecord(start, 0));
  // run cuBLAS function
  checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                              &alpha, adev, N, bdev, N, &beta, cdev, N));
  // evaluate calculation time on GPU
  checkCudaErrors(cudaThreadSynchronize());
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&timerValueGPU, start, stop));
  // copy C matric from device to host
  checkCudaErrors(cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost));

  // -------------------- CPU --------------------
  // calculate C matrix
  for (n = 0; n < N; n++) {
    for (m = 0; m < N; m++) {
      cc[n + m * N] = 0.f;
      for (k = 0; k < N; k++)
        cc[n + m * N] += a[n + k * N] * bT[m + k * N];  // bT !!!
    }
  }

  // compare GPU and CPU evaluations
  bool cmp = true;
  for (n = 0; n < N; n++) {
    for (m = 0; m < N; m++) {
      if (fabs(cc[n + m * N] - c[n + m * N]) > MAX_ERROR) {
        cmp = false;
        goto CMP;
      }
    }
  }

CMP:
  printf("\n\t\tGPU calculation time %f msec, comparison %s", timerValueGPU,
         (cmp == true ? "correct" : "incorrect"));
  if (cmp != true) {
    printf(" on element (%d, %d), error = %f\n", n, m,
           fabs(cc[n + m * N] - c[n + m * N]));
  } else {
    printf("\n");
  }

  // free memory on GPU and CPU
  checkCudaErrors(cudaFree(adev));
  checkCudaErrors(cudaFree(bdev));
  checkCudaErrors(cudaFree(cdev));
  free(a);
  free(b);
  free(bT);
  free(c);
  free(cc);
  // destroy event-variables
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  // destroy cuBLAS handle
  checkCudaErrors(cublasDestroy(handle));
}

void test_cublasXt(const int N) {
  printf("\n\tcublas-Xt calculation time: \n");

  int m, n, k;
  // create event-variables
  float timerValueGPU;
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  int numBytes = N * N * sizeof(float);
  float *adev, *bdev, *cdev, *a, *b, *c, *cc, *bT;
  // allocate memory on host
  a = (float *)malloc(numBytes);  //A matrix
  b = (float *)malloc(numBytes);  //B matrix
  bT = (float *)malloc(numBytes);  //transposed B matrix
  c = (float *)malloc(numBytes);  //C matrix for GPU
  cc = (float *)malloc(numBytes);  //C matrix for CPU

  // initialize A, B matrices and transposed B matrix (column-major)
  for (n = 0; n < N; n++) {
    for (m = 0; m < N; m++) {
      a[n + m * N] = 2.0f * m + n;
      b[n + m * N] = m - n;
      bT[n + m * N] = n - m;
    }
  }
  // create cuBLAS-Xt handle and other parameters for Sgemm
  cublasXtHandle_t handle;
  checkCudaErrors(cublasXtCreate(&handle));
  int devices[1] = {0};
  float alpha = 1.0;
  float beta = 0.0;
  /* chose devices for running CUBLAX-XT functions 
   * between which the load will be distributed */
  checkCudaErrors(cublasXtDeviceSelect(handle, 1, devices));
  /* set the size of the blocks (blockDim x blockDim) into which the matrices 
   * will be divided when distributed between devices */
  checkCudaErrors(cublasXtSetBlockDim(handle, N));
  // allocate memory on GPU
  checkCudaErrors(cudaMalloc((void **)&adev, numBytes));
  checkCudaErrors(cudaMalloc((void **)&bdev, numBytes));
  checkCudaErrors(cudaMalloc((void **)&cdev, numBytes));

  // ---------------- GPU ------------------------
  // copy matrices A and B from host to device
  checkCudaErrors(cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice));
  // run timer
  checkCudaErrors(cudaEventRecord(start, 0));
  // run cuBLAS-Xt function
  checkCudaErrors(cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                &alpha, adev, N, bdev, N, &beta, cdev, N));
  // evaluate calculation time on GPU
  checkCudaErrors(cudaThreadSynchronize());
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&timerValueGPU, start, stop));
  // copy C matric from device to host
  checkCudaErrors(cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost));

  // -------------------- CPU --------------------
  // calculate C matrix
  for (n = 0; n < N; n++) {
    for (m = 0; m < N; m++) {
      cc[n + m * N] = 0.f;
      for (k = 0; k < N; k++)
        cc[n + m * N] += a[n + k * N] * bT[m + k * N];  // bT !!!
    }
  }

  // compare GPU and CPU evaluations
  bool cmp = true;
  for (n = 0; n < N; n++) {
    for (m = 0; m < N; m++) {
      if (fabs(cc[n + m * N] - c[n + m * N]) > MAX_ERROR) {
        cmp = false;
        goto CMP;
      }
    }
  }

CMP:
  printf("\n\t\tGPU calculation time %f msec, comparison %s", timerValueGPU,
         (cmp == true ? "correct" : "incorrect"));
  if (cmp != true) {
    printf(" on element (%d, %d), error = %f\n", n, m,
           fabs(cc[n + m * N] - c[n + m * N]));
  } else {
    printf("\n");
  }

  // free memory on GPU and CPU
  checkCudaErrors(cudaFree(adev));
  checkCudaErrors(cudaFree(bdev));
  checkCudaErrors(cudaFree(cdev));
  free(a);
  free(b);
  free(bT);
  free(c);
  free(cc);
  // destroy event-variables
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  // destroy cuBLAS-Xt handle
  checkCudaErrors(cublasXtDestroy(handle));
}

void test_imp_mat_mul(const int N) {
  printf("\n\tSMEM-2 kernel calculation time: \n");

  int m, n, k;
  // create event-variables
  float timerValueGPU;
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  int numBytes = N * N * sizeof(float);
  float *adev, *bdev, *cdev, *a, *b, *c, *cc, *bT;
  // allocate memory on host
  a = (float *)malloc(numBytes);  //A matrix
  b = (float *)malloc(numBytes);  //B matrix
  bT = (float *)malloc(numBytes);  //transposed B matrix
  c = (float *)malloc(numBytes);  //C matrix for GPU
  cc = (float *)malloc(numBytes);  //C matrix for CPU

  // initialize A, B matrices and transposed B matrix (row-major)
  for (n = 0; n < N; n++) {
    for (m = 0; m < N; m++) {
      a[m + n * N] = 2.0f * m + n;
      b[m + n * N] = m - n;
      bT[m + n * N] = n - m;
    }
  }
  // initialize grid and block dimensions
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks(N / threads.x, N / threads.y);
  // allocate memory on GPU
  checkCudaErrors(cudaMalloc((void **)&adev, numBytes));
  checkCudaErrors(cudaMalloc((void **)&bdev, numBytes));
  checkCudaErrors(cudaMalloc((void **)&cdev, numBytes));

  // ---------------- GPU ------------------------
  // copy matrices A and B from host to device
  checkCudaErrors(cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice));
  // run timer
  checkCudaErrors(cudaEventRecord(start, 0));
  // run kernel
  kernel_smem_2<<<blocks, threads>>>(adev, bdev, N, cdev);
  // evaluate calculation time on GPU
  checkCudaErrors(cudaThreadSynchronize());
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&timerValueGPU, start, stop));
  // copy C matric from device to host
  checkCudaErrors(cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost));

  // -------------------- CPU --------------------
  // calculate C matrix
  for (n = 0; n < N; n++) {
    for (m = 0; m < N; m++) {
      cc[m + n * N] = 0.f;
      for (k = 0; k < N; k++)
        cc[m + n * N] += a[k + n * N] * bT[k + m * N];  // bT !!!
    }
  }

  // compare GPU and CPU evaluations
  bool cmp = true;
  for (n = 0; n < N; n++) {
    for (m = 0; m < N; m++) {
      if (fabs(cc[m + n * N] - c[m + n * N]) > MAX_ERROR) {
        cmp = false;
        goto CMP;
      }
    }
  }

CMP:
  printf("\n\t\tGPU calculation time %f msec, comparison %s", timerValueGPU,
         (cmp == true ? "correct" : "incorrect"));
  if (cmp != true) {
    printf(" on element (%d, %d), error = %f\n", n, m,
           fabs(cc[m + n * N] - c[m + n * N]));
  } else {
    printf("\n");
  }

  // free memory on GPU and CPU
  checkCudaErrors(cudaFree(adev));
  checkCudaErrors(cudaFree(bdev));
  checkCudaErrors(cudaFree(cdev));
  free(a);
  free(b);
  free(bT);
  free(c);
  free(cc);
  // destroy event-variables
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
}

#endif  // HELPER_TEST_H_
