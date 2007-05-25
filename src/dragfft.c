#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <timer.h>
#include <config.h>

#include <fftw3.h>

#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634074 /* log_2 e */
#endif

#define FILENAME "dragfft-" BUILD_SYSTEM_TYPE "."

double drag_fft_flops( int size, double *secperfft );
fftwf_complex *in;
fftwf_complex *out;

int main( void )
{
  char 	     filename[4096];
  int        maxloops = 1000;
  int        loop;
  const int  maxsize = 4194304;
  int        size    = 1;
  int        power   = 0;
  FILE      *fp;
  time_t ticks = time(NULL);

  in  = (fftwf_complex*) fftwf_malloc( maxsize * sizeof( fftwf_complex ) );
  out = (fftwf_complex*) fftwf_malloc( maxsize * sizeof( fftwf_complex ) );
  memset( in,  0, maxsize * sizeof( fftwf_complex ) );
  memset( out, 0, maxsize * sizeof( fftwf_complex ) );

  for ( loop = 0, power = 0; loop < maxloops; ++loop )
  {
    snprintf( filename, 4096 * sizeof(char), FILENAME "%d.out", loop );
    fp = fopen( filename, "w" );
    if ( !fp )
    {
      fprintf( stderr, "could not open file " FILENAME "\n" );
      exit( 1 );
    }
 
    fputs( "# log_2(npts)\tmega flops\tms / fft\n", fp );

    for( size = 2; size < maxsize; size *= 2)
    {
      double megaflops;
      double secperfft;

      ++power;

      fprintf( stderr, "\rsize = %d", size );

      megaflops = drag_fft_flops( size, &secperfft ) / 1e+6;

      fprintf( fp, "%8d\t%.3e\t%.3e\n", power, megaflops, 1e+3 * secperfft );
      fflush( fp );
    }

    fprintf( stderr, " [%d]\n", loop );
    fclose( fp );
  }

  return 0;
}



double drag_fft_flops( int size, double *secperfft )
{
  const double  tmin     = 1;   /* minimum run time (seconds)              */
  double        fftflop  = 5*size*log(size)*M_LOG2E;    /* ops for one fft */
  double        maxflops = 0;   /* best performance (flops)                */
  double        minratio = 1e6; /* best performance (second per fft)       */
  int           nreps    = 10;  /* number of repititions to find best time */
  int           nffts    = 1;   /* anticipated number of ffts to take tmin */
  fftwf_plan    plan;
  
  plan = fftwf_plan_dft_1d( size, in, out, FFTW_FORWARD, FFTW_PATIENT );
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

    ratio    = duration/nffts;
    flops    = fftflop/ratio;
    maxflops = flops > maxflops ? flops : maxflops;
    minratio = ratio < minratio ? ratio : minratio;
  }

  fftwf_destroy_plan( plan );

  *secperfft = minratio;
  return maxflops;
}
