#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <timer.h>
#include <config.h>

#ifdef HAVE_SFFTW_H
#include <sfftw.h>
#elif HAVE_FFTW_H
#include <fftw.h>
#else
#error "don't have either sfftw.h or fftw.h"
#endif

#ifndef HAVE_VALLOC
#define valloc malloc
#endif

#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634074 /* log_2 e */
#endif

#define FILENAME "dragfft-" BUILD_SYSTEM_TYPE ".out"

double fftw_flops( int size, double *secperfft );
fftw_complex *in;
fftw_complex *out;

int main( void )
{
  const int  maxsize = 1048576;
  int        size    = 1;
  int        power   = 0;
  FILE      *fp;

  in  = valloc( maxsize * sizeof( fftw_complex ) );
  out = valloc( maxsize * sizeof( fftw_complex ) );
  memset( in,  0, maxsize * sizeof( fftw_complex ) );
  memset( out, 0, maxsize * sizeof( fftw_complex ) );

  fp = fopen( FILENAME , "w" );
  if ( !fp )
  {
    fprintf( stderr, "could not open file " FILENAME "\n" );
    exit( 1 );
  }
 
  fputs( "# log_2(npts)\tmega flops\tms / fft\n", fp );

  do
  {
    double megaflops;
    double secperfft;

    ++power;
    size *= 2;

    fprintf( stderr, "\rsize = %d", size );

    megaflops = fftw_flops( size, &secperfft ) / 1e+6;

    fprintf( fp, "%8d\t%.3e\t%.3e\n", power, megaflops, 1e+3 * secperfft );
    fflush( fp );

  }
  while ( size < maxsize );

  fprintf( stderr, "\n" );
  fclose( fp );

  return 0;
}



double fftw_flops( int size, double *secperfft )
{
  const double  tmin     = 1;   /* minimum run time (seconds)              */
  double        fftflop  = 5*size*log(size)*M_LOG2E;    /* ops for one fft */
  double        maxflops = 0;   /* best performance (flops)                */
  double        minratio = 1e6; /* best performance (second per fft)       */
  int           nreps    = 10;  /* number of repititions to find best time */
  int           nffts    = 1;   /* anticipated number of ffts to take tmin */
  fftw_plan     plan;
  
  plan = fftw_create_plan_specific( size, FFTW_FORWARD,
      FFTW_MEASURE | FFTW_OUT_OF_PLACE | FFTW_USE_WISDOM, in, 1, out, 1 );
  memset( in,  0, size * sizeof( fftw_complex ) );
  memset( out, 0, size * sizeof( fftw_complex ) );

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
        fftw_one( plan, in, out );
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

  fftw_destroy_plan( plan );

  *secperfft = minratio;
  return maxflops;
}
