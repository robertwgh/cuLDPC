objects=cuLDPC.o cuLDPC_CPU.o

all: $(objects)
	nvcc -arch=sm_35 $(objects) -o app

%.o: %.cu
	nvcc -O2 -x cu -arch=sm_35 -I. -dc $< -o $@

%.o: %.cpp
	nvcc -O2 -x cu -arch=sm_35 -I/usr/local/cuda/include -dc $< -o $@

clean:
	rm -f *.o app

