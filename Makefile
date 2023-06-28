matmatgpu:	main.cu
	nvcc   -o matmatgpudist main.cu -Xcompiler -fopenmp  -lmpi -I/usr/mpi/gcc/openmpi-4.1.0rc5/include/ -L/usr/mpi/gcc/openmpi-4.1.0rc5/lib64 -O3 

