#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <timer.h>
#include <config.h>

#include <fftw3.h>

/*
ideally this constant would be set via ./configure, but right now I
 don't know how to do it it describes if multiply and add can be fused
 together (FUSEMULADD==1) or not (FUSEMULADD==2) To my (carsten)
 knowledge, only SPARC64, PowerPC and Itanium support this right now,
 Opteron will come in 2008/9 with SSE5. So, right now for Xeon and
 Opteron this should be 2
*/
#define FUSEMULADD 2

#define FILENAME "dragfft-" BUILD_SYSTEM_TYPE

double drag_fft_flops( int size, double *secperfft, 
		double *add, double *mul, double *fma );
fftwf_complex *in;
fftwf_complex *out;

int main( void )
{
  char       hostname[512];
  char 	     filename[4096];
  int        maxloops = 10000;
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
    *hostptr = NULL;

  in  = (fftwf_complex*) fftwf_malloc( maxsize * sizeof( fftwf_complex ) );
  out = (fftwf_complex*) fftwf_malloc( maxsize * sizeof( fftwf_complex ) );
  memset( in,  0, maxsize * sizeof( fftwf_complex ) );
  memset( out, 0, maxsize * sizeof( fftwf_complex ) );

  for ( loop = 0, power = 0; loop < maxloops; ++loop, power = 0 )
  {
    snprintf( filename, 4096 * sizeof(char), "/dev/null" );
    fp = fopen( filename, "w" );
    if ( !fp )
    {
      fprintf( stderr, "could not open output file %d\n", loop );
      exit( 1 );
    }
    fputs( "# log_2(npts)\tmega flops\tms / fft\n", fp );


    if ( loop == 0 )
    {
      snprintf( filename, 4096 * sizeof(char), "/dev/null" );
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

      megaflops = drag_fft_flops( size, &secperfft, &add, &mul, &fma ) / 1.e+6;

      fprintf( fp, "%8d\t%.3e\t%.3e\n", power, megaflops, 1e+3 * secperfft );
      fflush( fp );

      if ( loop == 0 )
      {
        fprintf( fp_ops, "%f\t%f\t%f\n", add, mul, fma );
        fflush( fp_ops );
      }
    }

    fclose( fp );
    if ( loop == 0 )
      fclose( fp_ops );
  }

  return 0;
}



double drag_fft_flops( int size, double *secperfft, 
		double *add, double *mul, double *fma )
{
  const double  tmin     = 1;   /* minimum run time (seconds)              */
  double        fftflop;        /* ops for one fft */
  double        maxflops = 0;   /* best performance (flops)                */
  double        minratio = 1e6; /* best performance (second per fft)       */
  int           nreps    = 10;  /* number of repititions to find best time */
  int           nffts    = 1;   /* anticipated number of ffts to take tmin */
  fftwf_plan    plan;
  
  plan = fftwf_plan_dft_1d( size, in, out, FFTW_FORWARD, FFTW_ESTIMATE );
  fftwf_flops( plan, add, mul, fma );
  fftflop = *add + *mul + FUSEMULADD * *fma;

  memset( in,  0, size * sizeof( fftwf_complex ) );
  memset( out, 0, size * sizeof( fftwf_complex ) );

  while ( nreps-- > 0 )
  {
    double duration = 0;
    double flops;
    double ratio;

    while ( 1 )
    {

      drag_time begin;
      drag_time end;
      int       iter = nffts;

      begin = drag_get_time();
      while ( iter-- > 0 )
      {
        fftwf_execute( plan );
      }
      end = drag_get_time();

      duration = drag_time_to_sec( drag_time_diff( end, begin ) );

      if ( duration < tmin )
        nffts *= 2;
      else
        break;
    }

    ratio    = duration/ (double) nffts;
    flops    = fftflop/ratio;
    maxflops = flops > maxflops ? flops : maxflops;
    minratio = ratio < minratio ? ratio : minratio;
  }

  fftwf_destroy_plan( plan );

  *secperfft = minratio;
  return maxflops;
}
