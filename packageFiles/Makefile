.PHONY: all build test clean

all: build
	CC=g++ LDSHARED='$(shell python scripts/configure.py)' python setup.py build
	python setup.py install

build:
	nvcc -rdc=true --compiler-options '-fPIC' -c -o temp.o green/green.cu
	nvcc -dlink --compiler-options '-fPIC' -o green.o temp.o -lcudart
	rm -f libgreen.a
	ar cru libgreen.a green.o temp.o
	ranlib libgreen.a

test: build
	g++ tests/test.c -L. -lgreen -o main -L${CUDA_PATH}/lib64 -lcudart

clean:
	rm -f libgreen.a *.o main temp.py
	rm -rf build
