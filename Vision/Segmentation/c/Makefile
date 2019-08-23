CC  = gcc
TGT = test_tf
LIB = -ltensorflow

% : %.c 
	gcc $< -o $@ ${LIB}

all: ${TGT}

clean:
	rm ${TGT}


