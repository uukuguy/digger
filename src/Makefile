TARGET = ../bin/udb

OBJS = logger.c.o \
	   filesystem.c.o \
	   utils.c.o \
	   udb.cc.o

include ./Makefile.common

# first try -mavx -maes, if fails, retry with -mavx and/or -maes removed, 
# or with -mavx replaced by -msse4.1 or -msse4.2.
CXXFLAGS += -mavx -maes

all: ${TARGET}

../bin:
	mkdir -p ../bin

${TARGET}: ../bin ${OBJS}
	${CC} -o ${TARGET} ${OBJS} ${LDFLAGS} -lshogun -lleveldb -lstdc++ 

clean:
	rm -f ${TARGET} ${OBJS}

