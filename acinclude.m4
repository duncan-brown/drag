dnl acinclude.m4

AC_DEFUN(DRAG_WITH_GCC_FLAGS,
[AC_ARG_WITH(
        gcc_flags,   
        [  --with-gcc-flags        turn on strict gcc warning flags],
        [ if test -n "${with_gcc_flags}"
          then
            drag_gcc_flags="-g3 -O4 -Wall -W -Wmissing-prototypes -Wstrict-prototypes -Wshadow -Wpointer-arith -Wcast-qual -Wcast-align -Wwrite-strings -fshort-enums -fno-common -Wnested-externs -Werror"
          else
            drag_gcc_flags=""
          fi
        ], [ drag_gcc_flags="" ] )
])

AC_DEFUN(DRAG_WITH_EXTRA_CPPFLAGS,
[AC_ARG_WITH(
        extra_cppflags, 
        [  --with-extra-cppflags=CPPFLAGS  additional C preprocessor flags],
        [ if test -n "${with_extra_cppflags}"
          then
            CPPFLAGS="$CPPFLAGS ${with_extra_cppflags}";
          fi
        ],)
])

AC_DEFUN(DRAG_WITH_EXTRA_CFLAGS,
[AC_ARG_WITH(
        extra_cflags, 
        [  --with-extra-cflags=CFLAGS  additional C compiler flags],
        [ if test -n "${with_extra_cflags}"
          then
            CFLAGS="$CFLAGS ${with_extra_cflags}";
          fi
        ],)
])

AC_DEFUN(DRAG_WITH_EXTRA_LDFLAGS,
[AC_ARG_WITH(
        extra_ldflags, 
        [  --with-extra-ldflags=LDFLAGS  additional linker flags],
        [ if test -n "${with_extra_ldflags}"
          then
            LDFLAGS="$LDFLAGS ${with_extra_ldflags}";
          fi
        ],)
])

AC_DEFUN(DRAG_WITH_EXTRA_LIBS,
[AC_ARG_WITH(
        extra_libs, 
        [  --with-extra-libs=LIBS  additional -l and -L linker flags],
        [ if test -n "${with_extra_libs}"
          then
            LIBS="$LIBS ${with_extra_libs}";
          fi
        ],)
])

AC_DEFUN(DRAG_WITH_CC,
[AC_ARG_WITH(
        cc, 
        [  --with-cc=CC            use the CC C compiler],
        [ if test -n "${with_cc}"
          then
            CC="${with_cc}";
          fi
        ],)
])

AC_DEFUN(DRAG_ENABLE_FFTTEST,
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

AC_DEFUN(DRAG_ENABLE_TCPTEST,
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

AC_DEFUN(DRAG_TYPE_SOCKLEN_T,
[AC_REQUIRE([AC_HEADER_STDC])dnl
AC_MSG_CHECKING(for socklen_t)
AC_CACHE_VAL(drag_cv_type_socklen_t,
[AC_EGREP_CPP(dnl
changequote(<<,>>)dnl
<<(^|[^a-zA-Z_0-9])socklen_t[^a-zA-Z_0-9]>>dnl
changequote([,]), [#include <sys/socket.h>],
drag_cv_type_socklen_t=yes, drag_cv_type_socklen_t=no)])dnl
AC_MSG_RESULT($drag_cv_type_socklen_t)
if test $drag_cv_type_socklen_t = no; then
  AC_DEFINE(socklen_t, int)
fi
])

AC_DEFUN(DRAG_SFFTW_WORKS,
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
