#ifndef TIMER_H
#define TIMER_H

#include <config.h>

#ifdef __cplusplus
extern "C" {
#endif


/*
 *
 * Timers: adapted from fftw.
 *
 */


/************* solaris ************/

#ifdef HAVE_GETHRTIME /* we use the nanosecond virtual timer */

# ifdef HAVE_SYS_TIME_H
# include <sys/time.h>
# endif

  typedef hrtime_t drag_time;

# define drag_get_time() gethrtime()
# define drag_time_diff(t1,t2) ((t1) - (t2))
# define drag_time_to_sec(t) ((double) t / 1.0e9)

/************* generic systems having gettimeofday ************/

#elif defined(HAVE_GETTIMEOFDAY) || defined(HAVE_BSDGETTIMEOFDAY)

# ifdef HAVE_SYS_TIME_H
# include <sys/time.h>
# endif
# ifdef HAVE_UNISTD_H
# include <unistd.h>
# endif
# define USE_GETTIMEOFDAY

  typedef struct timeval drag_time;

  extern drag_time drag_gettimeofday_get_time( void );
  extern drag_time drag_gettimeofday_time_diff( drag_time t1, drag_time t2 );
  extern double drag_gettimeofday_time_to_sec( drag_time t );
# define drag_get_time() drag_gettimeofday_get_time()
# define drag_time_diff(t1, t2) drag_gettimeofday_time_diff(t1, t2)
# define drag_time_to_sec(t) drag_gettimeofday_time_to_sec(t)

/* last resort: use clock() */

#else

# include <time.h>

  typedef clock_t drag_time;

# ifndef CLOCKS_PER_SEC
#   ifdef sun /* stupid sunos4 prototypes */
#     define CLOCKS_PER_SEC 1000000
      extern long clock(void);
#   else /* not sun, we don't know CLOCKS_PER_SEC */
#     error Please define CLOCKS_PER_SEC
#   endif
# endif

# define drag_get_time() clock()
# define drag_time_diff(t1,t2) ((t1) - (t2))
# define drag_time_to_sec(t) (((double) (t)) / CLOCKS_PER_SEC)

#endif /* clock() */

#ifdef __cplusplus
}      /* extern "C"  */
#endif /* __cplusplus */

#endif /* TIMER_H */
