dnl acinclude.m4

AC_DEFUN(AC_WITH_EXTRA_CPPFLAGS,
[AC_ARG_WITH(
        extra_cppflags, 
        [  --with-extra-cppflags=CPPFLAGS  additional C preprocessor flags],
        [ if test -n "${with_extra_cppflags}"
          then
            CPPFLAGS="$CPPFLAGS ${with_extra_cppflags}";
          fi
        ],)
])

AC_DEFUN(AC_WITH_EXTRA_CFLAGS,
[AC_ARG_WITH(
        extra_cflags, 
        [  --with-extra-cflags=CFLAGS  additional C compiler flags],
        [ if test -n "${with_extra_cflags}"
          then
            CFLAGS="$CFLAGS ${with_extra_cflags}";
          fi
        ],)
])

AC_DEFUN(AC_WITH_EXTRA_LDFLAGS,
[AC_ARG_WITH(
        extra_ldflags, 
        [  --with-extra-ldflags=LDFLAGS  additional linker flags],
        [ if test -n "${with_extra_ldflags}"
          then
            LDFLAGS="$LDFLAGS ${with_extra_ldflags}";
          fi
        ],)
])

AC_DEFUN(AC_WITH_CC,
[AC_ARG_WITH(
        cc, 
        [  --with-cc=CC            use the CC C compiler],
        [ if test -n "${with_cc}"
          then
            CC="${with_cc}";
          fi
        ],)
])

AC_DEFUN(AC_ENABLE_FFTTEST,
[AC_ARG_ENABLE(
        ffttest,
        [  --enable-ffttest        test fft performance],
        [ case "${enableval}" in
            yes) ffttest=true  ;;
            no)  ffttest=false ;;
            *) AC_MSG_ERROR(bad value ${enableval} for --enable-ffttest) ;;
          esac
        ], [ ffttest=true ] )
])

AC_DEFUN(AC_ENABLE_TCPTEST,
[AC_ARG_ENABLE(
        tcptest,
        [  --enable-tcptest        test tcp performance],
        [ case "${enableval}" in
            yes) tcptest=true  ;;
            no)  tcptest=false ;;
            *) AC_MSG_ERROR(bad value ${enableval} for --enable-tcptest) ;;
          esac
        ], [ tcptest=false ] )
])

AC_DEFUN(AC_TYPE_SOCKLEN_T,
[AC_REQUIRE([AC_HEADER_STDC])dnl
AC_MSG_CHECKING(for socklen_t)
AC_CACHE_VAL(ac_cv_type_socklen_t,
[AC_EGREP_CPP(dnl
changequote(<<,>>)dnl
<<(^|[^a-zA-Z_0-9])socklen_t[^a-zA-Z_0-9]>>dnl
changequote([,]), [#include <sys/socket.h>],
ac_cv_type_socklen_t=yes, ac_cv_type_socklen_t=no)])dnl
AC_MSG_RESULT($ac_cv_type_socklen_t)
if test $ac_cv_type_socklen_t = no; then
  AC_DEFINE(socklen_t, int)
fi
])

AC_DEFUN(AC_SFFTW_WORKS,
[AC_MSG_CHECKING(whether single precison FFTW works)
AC_TRY_RUN([
#include <stdio.h>
#ifdef HAVE_SFFTW_H
#include <sfftw.h>
#elif HAVE_FFTW_H
#include <fftw.h>
#else
#error "don't have either sfftw.h or fftw.h"
#endif
int main() { return sizeof(fftw_real) - 4; } ],
AC_MSG_RESULT(yes),
AC_MSG_RESULT(no)
AC_MSG_ERROR([single precision FFTW must be properly installed.])
)])
