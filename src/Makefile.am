noinst_PROGRAMS = dragfft dragtcp
dragfft_SOURCES = dragfft.c timer.c timer.h
dragtcp_SOURCES = dragtcp.c timer.c timer.h

if FFTTCPTEST
TESTS = dragtcp.sh dragfft
else
if FFTTEST
TESTS = dragfft
else
if TCPTEST
TESTS = dragtcp.sh
else
TESTS = notest.sh
endif
endif
endif

CLEANFILES = *.out
EXTRA_DIST = dragtcp.sh notest.sh
