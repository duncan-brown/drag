#include <math.h>
#include <timer.h>

#ifdef USE_GETTIMEOFDAY

#if defined(HAVE_BSDGETTIMEOFDAY) && ! defined(HAVE_GETTIMEOFDAY)
#define gettimeofday BSDgettimeofday
#endif

drag_time drag_gettimeofday_get_time( void )
{
  struct timeval tv;
  gettimeofday( &tv, 0 );
  return tv;
}

drag_time drag_gettimeofday_time_diff( drag_time t1, drag_time t2 )
{
  drag_time diff;
  diff.tv_sec  = t1.tv_sec  - t2.tv_sec;
  diff.tv_usec = t1.tv_usec - t2.tv_usec;
  while ( diff.tv_usec < 0 )
  {
    diff.tv_usec += 1000000L;
    diff.tv_sec  -= 1;
  }

  return diff;
}

double drag_gettimeofday_time_to_sec( drag_time t )
{
  return (double)t.tv_sec + (double)t.tv_usec * 1e-6;
}

#endif /* USE_GETTIMEOFDAY */
