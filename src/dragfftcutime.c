#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <timer.h>
#include <config.h>
#include <getopt.h>

#include <fftw3.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>


/*
ideally this constant would be set via ./configure, but right now I
 don't know how to do it it describes if multiply and add can be fused
 together (FUSEMULADD==1) or not (FUSEMULADD==2) To my (carsten)
 knowledge, only SPARC64, PowerPC and Itanium support this right now,
 Opteron will come in 2008/9 with SSE5. So, right now for Xeon and
 Opteron this should be 2
*/
#define FUSEMULADD 2

#define FILENAME "dragfftcu-" BUILD_SYSTEM_TYPE

int use_fftw = 1;
int gpu_copy = 1;

double drag_fft_flops( int size, double *secperfft, 
		double *add, double *mul, double *fma,
                double *secperfft_cuda, double *maxflops_cuda,
                double *copy_to_time, double *copy_from_time, double *cuda_fft_time );

fftwf_complex *in;
fftwf_complex *out;

cufftComplex  *local_in;
cufftComplex  *local_out;
cufftComplex  *gpu_in;
cufftComplex  *gpu_out;

int main( int argc, char* argv[] )
{
  int        j;
  char       hostname[512];
  char 	     filename[4096];
  int        maxloops = 10;
  int        loop;
  const int  maxsize = 1<<24; /* ramped this value up for "real"
				 tests */
  int        size    = 1;
  int        power   = 0;
  double     add, mul, fma;
  char      *hostptr = NULL;
  FILE      *fp, *fp_ops, *fp_time;
  time_t ticks = time(NULL);
  double copy_to_time, copy_from_time, cuda_fft_time;

  struct option long_options[] =
  {
    {"use-fftw",        no_argument,    &use_fftw,       1 },
    {"use-cuda",        no_argument,    &use_fftw,       0 },
    {"no-gpu-copy",     no_argument,    &gpu_copy,       0 },
    {"help",            no_argument,    0,              'h'},
    {0,0,0,0}
  };
  int c;

  while ( 1 )
  {
    int option_index = 0;
    size_t optarg_len;
    c = getopt_long_only( argc, argv, "h", long_options, &option_index );
    if ( c == -1 ) break;
    switch ( c )
    {
      case 0:
        if ( long_options[option_index].flag != 0 ) break;

      case 'h':
        fprintf( stderr, "useage: %s [--use-fftw|--use-cuda]\n", argv[0] );
        exit( 0 );
        break;

      default:
        fprintf( stderr, "error while parsing options\n" );
        exit( 1 );
    }
  }
  if ( optind < argc )
  {
    fprintf( stderr, "error: extraneous command line arguments\n" );
    exit( 1 );
  }

  gethostname( hostname, 512 );
  if ( (hostptr = strstr( hostname, "." )) )
    *hostptr = (char) NULL;

  /* allocate space for fftws on cpu */
  in  = (fftwf_complex*) fftwf_malloc( maxsize * sizeof( fftwf_complex ) );
  out = (fftwf_complex*) fftwf_malloc( maxsize * sizeof( fftwf_complex ) );
  memset( in,  0, maxsize * sizeof( fftwf_complex ) );
  memset( out, 0, maxsize * sizeof( fftwf_complex ) );

  /* allocate local space for cuda ffts */
  local_in  = malloc( maxsize * sizeof( cufftComplex ) );
  local_out = malloc( maxsize * sizeof( cufftComplex ) );
  memset( local_in,  0, maxsize * sizeof( cufftComplex ) );
  memset( local_out, 0, maxsize * sizeof( cufftComplex ) );

  /* allocate remote space for cuda ffts */
  cudaMalloc( (void**) &gpu_in, maxsize * sizeof( cufftComplex ) );
  cudaMalloc( (void**) &gpu_out, maxsize * sizeof( cufftComplex ) );
  cudaMemset( (void*) gpu_in, 0, maxsize * sizeof( cufftComplex ) );
  cudaMemset( (void*) gpu_out, 0, maxsize * sizeof( cufftComplex ) );

  /* allocate some noise */
  for ( j = 1<<20; j < maxsize; ++j )
  {
    in[j][0] = local_in[j].x = rand() / RAND_MAX;
    in[j][1] = local_in[j].y = rand() / RAND_MAX;
  }

  for ( loop = 0, power = 0; loop < maxloops; ++loop, power = 0 )
  {
    snprintf( filename, 4096 * sizeof(char), FILENAME "-%s-%s-%s.%d.out", 
        hostname, use_fftw ? "fftw" : "cuda", gpu_copy ? "copy" : "nocopy", loop );
    fp = fopen( filename, "w" );
    if ( !fp )
    {
      fprintf( stderr, "could not open output file %d\n", loop );
      exit( 1 );
    }
    fputs( "# log_2(npts)\tmega flops\tms / fft\tmega flops (cuda)\tms / fft (cuda)\n", fp );

    snprintf( filename, 4096 * sizeof(char), FILENAME "-timing-%s-%s-%s.%d.out", 
        hostname, use_fftw ? "fftw" : "cuda", gpu_copy ? "copy" : "nocopy", loop );
    fp_time = fopen( filename, "w" );
    if ( !fp_time )
    {
      fprintf( stderr, "could not open output file %d\n", loop );
      exit( 1 );
    }
    fputs( "# log_2(npts)\tcopy_from_time\tcopy_to_time\tcuda_fft_time\n", fp_time );


    if ( loop == 0 )
    {
      snprintf( filename, 4096 * sizeof(char), FILENAME "-%s-%s-%s.out", 
          hostname, use_fftw ? "fftw" : "cuda", gpu_copy ? "copy" : "nocopy" );
      fp_ops = fopen( filename, "w" );
      if ( !fp_ops )
      {
        fprintf( stderr, "could not open file ops file\n" );
        exit( 1 );
      }
      fputs( "# add\tmul\tfma\n", fp_ops );
    }


    for ( size = 2; size < maxsize; size *= 2)
    {
      double megaflops, megaflops_cuda;
      double secperfft, secperfft_cuda;

      ++power;

      fprintf( stderr, "\rsize = %d", size );

      megaflops = drag_fft_flops( size, &secperfft, &add, &mul, &fma, 
          &secperfft_cuda, &megaflops_cuda,
          &copy_to_time, &copy_from_time, &cuda_fft_time ) / 1.e+6;

      fprintf( fp, "%8d\t%.3e\t%.3e\t%.3e\t%.3e\n", 
          power, megaflops, 1e+3 * secperfft, 
          megaflops_cuda, secperfft_cuda );
      fflush( fp );

      fprintf( fp_time, "%8d\t%.3e\t%.3e\t%.3e\n", 
          power, copy_to_time, copy_from_time, cuda_fft_time );
      fflush( fp_time );

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
		double *add, double *mul, double *fma,
                double *secperfft_cuda, double *maxflops_cuda,
                double *copy_to_time, double *copy_from_time, double *cuda_fft_time )
{
  const double  tmin     = 1;   /* minimum run time (seconds)              */
  double        fftflop;        /* ops for one fft */
  double        maxflops = 0;   /* best performance (flops)                */
  double        minratio = 1e6; /* best performance (second per fft)       */
  int           nreps    = 10;  /* number of repititions to find best time */
  int           nffts    = 1;   /* anticipated number of ffts to take tmin */
  fftwf_plan    plan;

  cudaEvent_t t1, t2, t3, t4;
  float duration_cuda = 0.0f;
  float t2_minus_t1 = 0.0f;
  float t4_minus_t3 = 0.0f;
  double        fftflop_cuda;        /* ops for one fft */
  double        minratio_cuda = 1e6; /* best performance (second per fft)       */

  cufftHandle   cuplan;
  
  /* create the fftw plan and number of operations */
  plan = fftwf_plan_dft_1d( size, in, out, FFTW_FORWARD, FFTW_ESTIMATE );
  fftwf_flops( plan, add, mul, fma );
  fftflop = *add + *mul + FUSEMULADD * *fma;

  /* create the cuda plan and number of operations */
  cufftPlan1d( &cuplan, size, CUFFT_C2C, 1 );
  fftflop_cuda = fftflop;

  *maxflops_cuda = 0;
  cudaEventCreate( &t1 );
  cudaEventCreate( &t2 );
  cudaEventCreate( &t3 );
  cudaEventCreate( &t4 );

  while ( nreps-- > 0 )
  {
    double duration = 0;
    double flops;
    double ratio;
    float  duration_cuda = 0;
    double flops_cuda;
    double ratio_cuda;

    while ( 1 )
    {
      drag_time begin;
      drag_time end;
      int       iter = nffts;

      if ( ! gpu_copy ) cudaMemcpy( (void**) &gpu_in, (void**) &local_in, 
          size * sizeof(cufftComplex), cudaMemcpyHostToDevice );

      /* start drag timer */
      begin = drag_get_time();
      
      /* start cuda timer */
      cudaEventRecord( t1, 0 );

      while ( iter-- > 0 )
      {
        if ( use_fftw )
        {
          fftwf_execute( plan );
        }
        else if ( gpu_copy )
        {
          cudaMemcpy( (void**) &gpu_in, (void**) &local_in, 
              size * sizeof(cufftComplex), cudaMemcpyHostToDevice );
          cudaEventRecord( t2, 0 );
          cufftExecC2C( cuplan, gpu_in, gpu_out, CUFFT_FORWARD );
          cudaEventRecord( t3, 0 );
          cudaMemcpy( (void**) &local_out, (void**) &gpu_out, 
              size * sizeof(cufftComplex), cudaMemcpyDeviceToHost );
        }
        else
        {
          cufftExecC2C( cuplan, gpu_in, gpu_out, CUFFT_FORWARD );
        }
      }

      /* stop cuda timer */
      cudaEventRecord( t4, 0 );
      cudaThreadSynchronize();
      cudaEventElapsedTime( &duration_cuda, t2, t3 );
      cudaEventElapsedTime( &t2_minus_t1, t1, t2 );
      cudaEventElapsedTime( &t4_minus_t3, t3, t4 );

      /* stop drag timer */
      end = drag_get_time();

      /* time in seconds */
      duration = drag_time_to_sec( drag_time_diff( end, begin ) );

      /* cuda time in seconds */
      duration_cuda /= 1.0e-3;
      t2_minus_t1 /= 1.0e-3;
      t4_minus_t3 /= 1.0e-3;
      
      if ( ! gpu_copy ) cudaMemcpy( (void**) &local_out, (void**) &gpu_out, 
          size * sizeof(cufftComplex), cudaMemcpyDeviceToHost );

      break;

      if ( duration < tmin )
        nffts *= 2;
      else
        break;

    }

    ratio    = duration/ (double) nffts;
    flops    = fftflop/ratio;
    maxflops = flops > maxflops ? flops : maxflops;
    minratio = ratio < minratio ? ratio : minratio;

    ratio_cuda    = duration_cuda/ (double) nffts;
    flops_cuda    = fftflop_cuda/ratio_cuda;
    *maxflops_cuda = flops_cuda > *maxflops_cuda ? flops_cuda : *maxflops_cuda;
    minratio_cuda = ratio_cuda < minratio_cuda ? ratio_cuda : minratio_cuda;
  }

  fftwf_destroy_plan( plan );

  cudaEventDestroy(t1);
  cudaEventDestroy(t2);
  cudaEventDestroy(t3);
  cudaEventDestroy(t4);

  *copy_to_time = (double) t2_minus_t1;
  *copy_from_time = (double) t4_minus_t3;
  *cuda_fft_time = (double) duration_cuda;

  *secperfft = minratio;
  *secperfft_cuda = minratio_cuda;

  return maxflops;
}
