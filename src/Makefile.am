if ENABLECUDA
CUDAPROGS = dragfftcu
else
CUDAPROGS =
endif

bin_PROGRAMS = dragfft dragfftsilent dragtcp $(CUDAPROGS)
dragfft_SOURCES = dragfft.c timer.c timer.h
dragfftsilent_SOURCES = dragfftsilent.c timer.c timer.h
dragtcp_SOURCES = dragtcp.c timer.c timer.h
dragfftcu_SOURCES = dragfftcu.c timer.c timer.h

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
