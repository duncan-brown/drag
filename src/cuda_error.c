#include <cuda_error.h>

void cudaSafeCall(cudaError_t error){
  if ( cudaSuccess != error ){
    fprintf( stderr, "cudaSafeCall() error: %s", cudaGetErrorString( error ) );
    exit(1);
  }
  return;
}

void cufftSafeCall(cufftResult error){
  if ( CUFFT_SUCCESS != error)
    fprintf(stderr, "cufftSafeCall() error : %i",error);
  return;  
}
