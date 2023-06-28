#!/bin/bash
#SBATCH --job-name=matmatgpudist
#SBATCH --output=matmatgpudist.out
#SBATCH --error=matmatgpudist.err
#SBATCH --time=00:60:00
#SBATCH --partition=gpus
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1



nvcc  -o matmatgpudist main.cu -Xcompiler -fopenmp  -lmpi -I/usr/mpi/gcc/openmpi-4.1.0rc5/include/ -L/usr/mpi/gcc/openmpi-4.1.0rc5/lib64 -O3

mpirun ./matmatgpudist

