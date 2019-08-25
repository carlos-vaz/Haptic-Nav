CC  = g++
LIB = -ltensorflow
TGT = TEST
OBJ = test.o \
      tiny_deeplab_api.o


%.o : %.cpp
	${CC} -c $< -o $@

% : ${OBJ}
	${CC} $^ -o $@ ${LIB}

all: ${TGT}
.PHONY: all

clean:
	rm ${TGT} ${OBJ}
.PHONY: clean

