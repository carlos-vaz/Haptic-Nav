CC  = g++
LIB = -ltensorflow
TGT = test
OBJ = tiny_deeplab_api.o

all: $(TGT)
.PHONY : all

%.o : %.cpp
	$(CC) -c $< -o $@

$(TGT) : $(TGT).o $(OBJ)
	$(CC) $^ -o $@ $(LIB) `pkg-config --cflags --libs opencv`

clean:
	rm $(TGT) $(TGT).o $(OBJ)
.PHONY : clean

