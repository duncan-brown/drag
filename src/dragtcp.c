#include <stdio.h>
#include <ctype.h>
#include <errno.h>
#include <netdb.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <timer.h>
#include <config.h>

#define NODE_IPADDR_PREFIX "192.168.2."
#define FILENAME "dragtcp-" BUILD_SYSTEM_TYPE ".out"
#define VERBOSE( msg ) ( void )( verbose ? puts( msg ) : 0 )
#define READWRITE( sock, buf, size ) \
  ( recflg ? read( sock, buf, size ) : write( sock, buf, size ) )

int loc( const char *remhost );
int rem( void );
void usage( int error, const char *program );

extern char *optarg;
extern int   optind;
extern int   optopt;
extern int   errno;

enum { buflen = 1024, maxnumbuf = 1024 };
static char buf[buflen];
static struct sockaddr_in locaddr; /* local  */
static struct sockaddr_in remaddr; /* remote */
static socklen_t locaddrsize = sizeof( locaddr );
static short     port        = 2000;
static int       verbose     = 0;
static int       recflg      = 0;

int main( int argc, char **argv )
{
  char  remhost[16] = { 0 };
  char *program     = argv[0];
  int   locflg      = 0;
  int   remflg      = 0;
  int   opt;

  while ( ( opt = getopt( argc, argv, "hvlrRi:n:p:" ) ) != EOF )
  {
    switch ( opt )
    {
      case 'h': /* help message */
        usage( 0, program );

      case 'v': /* verbose */
        verbose = 1;
        break;

      case 'l': /* local */
        if ( remflg )
        {
          fputs( "cannot specify both option -l and -r\n", stderr );
          usage( 1, program );
        }
        locflg = 1;
        break;

      case 'r': /* remote */
        if ( locflg )
        {
          fputs( "cannot specify both option -l and -r\n", stderr );
          usage( 1, program );
        }
        remflg = 1;
        break;

      case 'R': /* receive rather than send */
        recflg = 1;
        break;

      case 'n': /* node */
        if ( *remhost )
        {
          fputs( "remote host specified multiple times\n", stderr );
          usage( 1, program );
        }
        if ( snprintf( remhost, sizeof( remhost ), NODE_IPADDR_PREFIX "%s",
              optarg ) < 0 )
        {
          perror( "snprintf()" );
          return 1;
        }
        break;

      case 'i': /* ip address */
        if ( *remhost )
        {
          fputs( "remote host specified multiple times\n", stderr );
          usage( 1, program );
        }
        if ( snprintf( remhost, sizeof( remhost ), "%s", optarg ) < 0 )
        {
          perror( "snprintf()" );
          return 1;
        }
        break;

      case 'p': /* port */
        port = atoi( optarg );
        break;

      default:
        usage( 1, program );
    }
  }

  if ( locflg )
  {
    if ( ! *remhost )
    {
      fputs( "must specify either node number or ip address of remote host\n",
             stderr );
      usage( 1, program );
    }
    return loc( remhost );
  }

  if ( remflg )
  {
    if ( *remhost )
    {
      fputs( "do not specify node number or ip address for remote host\n",
             stderr );
      usage( 1, program );
    }
    if ( recflg )
    {
      fputs( "do not specify send or receive for remote host\n", stderr );
      usage( 1, program );
    }
    return rem();
  }

  /* error: neither local nor remote */
  fputs( "must specify either local or remote\n", stderr );
  usage( 1, program );
  return 2; /* never gets here */
}


int loc( const char *remhost )
{
  FILE *fp;
  int   s;
  int   numbuf;
  char  sendrecv = recflg ? 'r' : 's' ; /* local sends or receives */

  remaddr.sin_family      = AF_INET;
  remaddr.sin_addr.s_addr = inet_addr( remhost );
  remaddr.sin_port        = htons( port );
  locaddr.sin_port        = 0;

  memset( buf, '*', sizeof( buf ) );

  VERBOSE( "opening socket" );
  if ( ( s = socket( AF_INET, SOCK_STREAM, 0 ) ) < 0 )
  {
    perror( "socket()" );
    return 1;
  }


  VERBOSE( "binding socket" );
  if ( bind( s, (struct sockaddr *) &locaddr, sizeof( locaddr ) ) )
  {
    perror( "bind()" );
    return 1;
  }

  VERBOSE( "connecting" );
  if ( connect( s, (struct sockaddr *) &remaddr, sizeof( remaddr ) ) )
  {
    perror( "connect()" );
    return 1;
  }

  VERBOSE( "timing socket communication" );

  /*
  if ( ! ( fp = fopen( FILENAME, "w" ) ) )
  {
    perror( "fopen()" );
    return 1;
  }
  */
  fp = stdout;

  if ( fprintf( fp, "# bytes\ttime (seconds) \trate (10^6 bytes per second)\n")
       < 0 )
  {
    perror( "fprintf()" );
    return 1;
  }

  /* send send or receive info */
  if ( write( s, &sendrecv, sizeof( sendrecv ) ) < 0 )
  {
    perror( "write()" );
    return 1;
  }

  for ( numbuf = 1; numbuf <= maxnumbuf; numbuf *= 2 )
  {
    int nbuf;
    int nbytes = 0;
    double time;
    drag_time end;
    drag_time begin = drag_get_time();
    for ( nbuf = 0; nbuf < numbuf; ++nbuf )
    {
      int bytes;
      if ( ( bytes = READWRITE( s, buf, sizeof( buf ) ) ) < 0 )
      {
        recflg ?  perror( "read()" ) : perror( "write()" ) ;
        return 1;
      }
      nbytes += bytes;
    }
    end = drag_get_time();
    time = drag_time_to_sec( drag_time_diff( end, begin ) );
    fprintf( fp, "%7d\t%e\t%f\n", nbytes, time, nbytes/(1e6*time) );
  }

  /*
  if ( fclose( fp ) )
  {
    perror( "fclose()" );
    return 1;
  }
  */

  VERBOSE( "closing socket" );
  if ( close( s ) < 0 )
  {
    perror( "close()" );
    return 1;
  }

  return 0;
}


int rem( void )
{
  int  s;
  int  ns;
  int  numbuf;
  int  reuseaddr = 1;
  char sendrecv;

  remaddr.sin_port = htons( port );

  memset( buf, '+', sizeof( buf ) );

  VERBOSE( "opening socket" );
  if ( ( s = socket( AF_INET, SOCK_STREAM, 0 ) ) < 0 )
  {
    perror( "socket()" );
    return 1;
  }

  VERBOSE( "setting socket option: SO_REUSEADDR" );
  if ( setsockopt( s, SOL_SOCKET, SO_REUSEADDR,
                   (void *) &reuseaddr, sizeof( reuseaddr ) ) )
  {
    perror( "setsockopt()" );
    return 1;
  }

  VERBOSE( "binding socket" );
  if ( bind( s, (struct sockaddr *) &remaddr, sizeof( remaddr ) ) )
  {
    perror( "bind()" );
    return 1;
  }

  VERBOSE( "listening" );
  if ( listen( s, 0 ) )
  {
    perror( "listen()" );
    return 1;
  }

  if ( ( ns = accept( s, (struct sockaddr *) &locaddr, &locaddrsize ) ) < 0 )
  {
    perror( "accept()" );
    return 1;
  }
  VERBOSE( "accepted connection" );

  VERBOSE( "communicating" );

  if ( read( ns, &sendrecv, sizeof( sendrecv ) ) < 0 )
  {
    perror( "read()" );
    return 1;
  }
  switch ( sendrecv )
  {
    case 's': /* we are expected to receive */
      recflg = 1;
      break;
    case 'r': /* we are expected to send */
      recflg = 0;
      break;
    default:
      fputs( "could not determine whether to send or receive\n", stderr );
      return 1;
  }

  for ( numbuf = 1; numbuf <= maxnumbuf; numbuf *= 2 )
  {
    int nbuf;
    int nbytes = 0;
    for ( nbuf = 0; nbuf < numbuf; ++nbuf )
    {
      int bytes;
      if ( ( bytes = READWRITE( ns, buf, sizeof( buf ) ) ) < 0 )
      {
        recflg ? perror( "read()" ) : perror( "write()" ) ;
        return 1;
      }
      nbytes += bytes;
    }
  }

  VERBOSE( "closing socket" );
  if ( close( ns ) < 0 )
  {
    perror( "close()" );
    return 1;
  }

  if ( close( s ) < 0 )
  {
    perror( "close()" );
    return 1;
  }

  return 0;
}


void usage( int error, const char *program )
{
  FILE *fp = error ? stderr : stdout ;
  fprintf( fp, "Usage: %s hvlrRi:n:p:\n", program );
  fprintf( fp, "Options:\n" );
  fprintf( fp, "  -h            help: print this message\n" );
  fprintf( fp, "  -v            verbose\n" );
  fprintf( fp, "  -l            local\n" );
  fprintf( fp, "  -r            remote\n" );
  fprintf( fp, "  -R            receive rather than send (local only)\n" );
  fprintf( fp, "  -i remhost    ip address of remote host (local only)\n" );
  fprintf( fp, "  -n node       node number of remote host (local only)\n" );
  fprintf( fp, "  -p port       specify the port (2000 is default)\n" );
  exit( error );
}