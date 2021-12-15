#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "inc\helper_test.h"

int main() {
  for (int N = 512; N <= 2048; N *= 2) {
    printf("\nN = %d:\n", N);
	test_cublas(N);
	test_cublasXt(N);
	test_imp_mat_mul(N);
  }

  return 0;
}
