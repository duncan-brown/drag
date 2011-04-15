#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "timer.h"
#include <config.h>
#include <cuda.h>
#include <cufft.h>
//#include <fftw3.h>

/*
ideally this constant would be set via ./configure, but right now I
 don't know how to do it it describes if multiply and add can be fused
 together (FUSEMULADD==1) or not (FUSEMULADD==2) To my (carsten)
 knowledge, only SPARC64, PowerPC and Itanium support this right now,
 Opteron will come in 2008/9 with SSE5. So, right now for Xeon and
 Opteron this should be 2
*/
#define FUSEMULADD 2

#define FILENAME "dragfft-gpu-" BUILD_SYSTEM_TYPE

double drag_fft_flops( int size, double *secperfft, 
		double *add, double *mul, double *fma ,int memtransfer);
void printCUFFTError(cufftResult result);
	
int main(int argc, char** argv )
{
  int memtransfer=0;
  if (argc==2) if (!strcmp(argv[1],"-m")){
	  memtransfer=1;
    fprintf(stderr,"memory transfer will be performed\n");
  }
	

  char       hostname[512];
  char 	     filename[4096];
  int        maxloops = 1;
  int        loop;
  const int  maxsize = 1<<26; /* ramped this value up for "real"
				 tests */
  int        size    = 1;
  int        power   = 0;
  double     add, mul, fma;
  char      *hostptr = NULL;
  FILE      *fp, *fp_ops;
  time_t ticks = time(NULL);

  gethostname( hostname, 512 );
  if ( (hostptr = strstr( hostname, "." )) )
    *hostptr = (int)NULL;

  for ( loop = 0, power = 0; loop < maxloops; ++loop, power = 0 )
  {
    snprintf( filename, 4096 * sizeof(char), FILENAME "-%s.%d.out", 
        hostname, loop );
    fp = fopen( filename, "w" );
    if ( !fp )
    {
      fprintf( stderr, "could not open output file %d\n", loop );
      exit( 1 );
    }
    fputs( "% log_2(npts)\tmega flops\tms / fft\n", fp );


    if ( loop == 0 )
    {
      snprintf( filename, 4096 * sizeof(char), FILENAME "-%s.ops",
          hostname );
      fp_ops = fopen( filename, "w" );
      if ( !fp_ops )
      {
        fprintf( stderr, "could not open file ops file\n" );
        exit( 1 );
      }
      fputs( "# add\tmul\tfma\n", fp_ops );
    }


    for( size = 2; size < maxsize; size *= 2)
    {
      double megaflops;
      double secperfft;

      ++power;

      fprintf( stderr, "\rsize = %d", size );

      megaflops = drag_fft_flops( size, &secperfft, &add, &mul, &fma ,memtransfer) / 1.e+6;

      fprintf( fp, "%8d\t%.3e\t%.3e\n", power, megaflops, 1e+3 * secperfft );
      fflush( fp );

      if ( loop == 0 )
      {
        fprintf( fp_ops, "%f\t%f\t%f\n", add, mul, fma );
        fflush( fp_ops );
      }
    }

    fprintf( stderr, " [%d]\n", loop );
    fclose( fp );
    if ( loop == 0 )
      fclose( fp_ops );
  }

  return 0;
}



double drag_fft_flops( int size, double *secperfft, 
		double *add, double *mul, double *fma ,int memtransfer)
{

fprintf(stderr,"\n Starting drag_fft\n");
  const double  tmin     = 1;   /* minimum run time (seconds)              */
  //double        fftflop;        /* ops for one fft */
  //double        maxflops = 0;   /* best performance (flops)                */
  double        minratio = 1e6; /* best performance (second per fft)       */
  int           nreps    = 10;  /* number of repititions to find best time */
  int           nffts    = 1;   /* anticipated number of ffts to take tmin */
  const int  maxsize = 1<<26; /* ramped this value up for "real" tests  */
	
  cufftResult result;
  cufftHandle plan;
  cufftComplex *data;
  cufftComplex *localDataIN;
  cufftComplex *localDataOUT;

	//Allocate memory for the fft on the GPU
	cudaMalloc((void**)&data, maxsize * sizeof( cufftComplex ) );

 
  if (memtransfer){
		localDataIN= malloc(maxsize*sizeof(cufftComplex));
    localDataOUT=malloc(maxsize*sizeof(cufftComplex));
    memset(localDataIN,0,maxsize* sizeof(cufftComplex));
	}  
	else 
   cudaMemset((void**)&data, 0,maxsize*sizeof(cufftComplex));

	printCUFFTError(cufftPlan1d( &plan,size,CUFFT_C2C,nffts ));

  
  // fftwf_flops( plan, add, mul, fma );
  //fftflop = *add + *mul + FUSEMULADD * *fma;



  while ( nreps-- > 0 )
  {
    double duration = 0;
    double flops;
    double ratio;

    while ( 1 )
    {
      cudaThreadSynchronize();
      drag_time begin;
      drag_time end;
      int       iter = nffts;

      begin = drag_get_time();
      while ( iter-- > 0 )
      {


        if (memtransfer){
					//copy memory to gpu
					cudaMemcpy((void**)&data,(void**)&localDataIN,
						size*sizeof(cufftComplex),cudaMemcpyHostToDevice);

					//run fft on data
					cufftExecC2C( plan,data,data, CUFFT_FORWARD);			

		      //copy data from gpu to main memory
					cudaMemcpy((void**)&localDataOUT,(void**)&data,
						size*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
				}			
        else
					//run fft on data
        	cufftExecC2C( plan,data,data, CUFFT_FORWARD);



      }
      cudaThreadSynchronize();
      end = drag_get_time();

      duration = drag_time_to_sec( drag_time_diff( end, begin ) );

      if ( duration < tmin )
        nffts *= 2;
      else
        break;
    }

    ratio    = duration/ (double) nffts;
    //flops    = fftflop/ratio;
    //maxflops = flops > maxflops ? flops : maxflops;
    minratio = ratio < minratio ? ratio : minratio;
  }

  cufftDestroy( plan );
  cudaFree(data);

   if (memtransfer){ 
			free(localDataIN);
			free(localDataOUT);
   }
	 
  *secperfft = minratio;
  //return maxflops;
  return -1;
}
 

/* Reports error message from CUFFT library
*/
void printCUFFTError(cufftResult result)
{
	if (result==CUFFT_SUCCESS)
		fprintf(stderr,"success \n");
  else{ 
    fprintf(stderr,"ERROR :");
		if (result==CUFFT_INVALID_PLAN)
			fprintf(stderr," Invalid Plan \n");
		if (result==CUFFT_INVALID_TYPE)
			fprintf(stderr," Invalid Type \n");
		if (result==CUFFT_ALLOC_FAILED)
			fprintf(stderr," Bad Pointer \n");
		if (result==CUFFT_INVALID_VALUE)
			fprintf(stderr," Invalid Value \n");
		if (result==CUFFT_INTERNAL_ERROR)
			fprintf(stderr," Internal Error \n");
		if (result==CUFFT_EXEC_FAILED)
			fprintf(stderr," Execution Failure \n");
		if (result==CUFFT_SETUP_FAILED)
			fprintf(stderr," Setup Failed \n");
		if (result==CUFFT_INVALID_SIZE)
			fprintf(stderr," Invalid Size \n");
	}
}


