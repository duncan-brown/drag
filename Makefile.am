SUBDIRS = src
EXTRA_DIST = acconfig.h
CVSCLEANFILES = Makefile.in aclocal.m4 configure                        \
                src/Makefile.in src/config.h.in src/stamp-h.in
cvs-clean: maintainer-clean
	rm -f $(CVSCLEANFILES)
