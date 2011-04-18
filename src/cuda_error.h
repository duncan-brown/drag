#ifndef CUDA_ERROR
#define CUDA_ERROR

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

void cudaSafeCall(cudaError_t error);
void cufftSafeCall(cufftResult);

#endif /* CUDA_ERROR */
