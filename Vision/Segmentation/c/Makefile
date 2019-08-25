CC  = g++
LIB = -ltensorflow
TGT = test
OBJ = tiny_deeplab_api.o

all: $(TGT)
.PHONY : all

%.o : %.cpp
	$(CC) -c $< -o $@

$(TGT) : $(TGT).o $(OBJ)
	$(CC) $^ -o $@ $(LIB)

clean:
	rm $(TGT) $(TGT).o $(OBJ)
.PHONY : clean
