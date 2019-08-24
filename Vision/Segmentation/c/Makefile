CC  = g++
LIB = -ltensorflow
TGT = test
LINK = ld
OBJ = \
      test.o \
      tiny_deeplab_api.o


%.o : %.cpp
	${CC} -c $< -o $@

% : ${OBJ}
	${CC} ${OBJ} ${LIB} -o $@

all: ${TGT}
.PHONY: all

clean:
	rm ${TGT} ${OBJ}
.PHONY: clean


