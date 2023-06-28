#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define MATRIX_DIMENSION 2048
#define MIN_RANDOM_NUMBER -1000
#define MAX_RANDOM_NUMBER 1000

double get_cur_time() 
{
  struct timeval   tv;
  struct timezone  tz;
  double cur_time;
  
  gettimeofday(&tv, &tz);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;
  
  return cur_time;
} 

__global__ void kernel1(double *Adev, double *Bdev, double *Cdev, int N, int M, int P)
{
    int idglob_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idglob_y = blockDim.y * blockIdx.y + threadIdx.y;
    double sum = Cdev[idglob_x * P + idglob_y];
    int k;
    for(k = 0; k != M; ++k)
        sum += Adev[idglob_x * M + k] * Bdev[idglob_y + k * P];
    Cdev[idglob_x * P + idglob_y] = sum;
}

__global__ void kernel2(double *Adev, double *Bdev, double *Cdev, int N, int M, int P)
{
    __shared__ double Ashared[32][32];
    __shared__ double Bshared[32][32];
    int idglob_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idglob_y = blockDim.y * blockIdx.y + threadIdx.y;
    double sum = Cdev[idglob_x * P + idglob_y];
    int k;
    for(k = 0; k != M / 32; ++k)
    {
        Ashared[threadIdx.x][threadIdx.y] = Adev[idglob_x * M + threadIdx.y + 32 * k];
        Bshared[threadIdx.x][threadIdx.y] = Bdev[idglob_y * M + (32 * k + threadIdx.x) * P];
        __syncthreads();
        int kk;
        for(kk = 0; kk != 32; ++kk)
            sum += Ashared[threadIdx.x][kk] * Bshared[kk][threadIdx.y];
        __syncthreads();
    }
    Cdev[idglob_x * P + idglob_y] = sum;
}

void matmatgpu1(int lda, int ldb, int ldc, double *A, double *B, double *C, int N, int M, int P)
{
    int max = 0;
    if(N > M)
        max = N;
    else
        max = M;
    if(P > max)
        max = P;
    double *buffer = (double*)malloc(sizeof(double) * max * max);
    double *Adev;
    double *Bdev;
    double *Cdev;
    cudaMalloc((void**)&Adev, sizeof(double) * N * M);
    cudaMalloc((void**)&Bdev, sizeof(double) * N * P);
    cudaMalloc((void**)&Cdev, sizeof(double) * N * P);
    int i, j;
    for(i = 0; i != N; ++i)
        for(j = 0; j != M; ++j)
            buffer[i * M + j] = A[i * lda + j];
    cudaMemcpy(Adev, buffer, sizeof(double) * N * M, cudaMemcpyHostToDevice);
    for(i = 0; i != M; ++i)
        for(j = 0; j != P; ++j)
            buffer[i * P + j] = B[i * ldb + j];
    cudaMemcpy(Bdev, buffer, sizeof(double) * N * P, cudaMemcpyHostToDevice);
    for(i = 0; i != N; ++i)
        for(j = 0; j != P; ++j)
            buffer[i * P + j] = C[i * ldc + j];
    cudaMemcpy(Cdev, buffer, sizeof(double) * N * P, cudaMemcpyHostToDevice);
    int BlockDimRow = 32;
    int BlockDimCol = 32;
    dim3 DimBlock(BlockDimRow, BlockDimCol);
    dim3 DimGrid(N / BlockDimRow, P / BlockDimCol);
    kernel1<<<DimGrid, DimBlock>>>(Adev, Bdev, Cdev, N, M, P);
    cudaDeviceSynchronize();
    cudaMemcpy(buffer, Cdev, sizeof(double) * N * P, cudaMemcpyDeviceToHost);
    for(i = 0; i != N ; ++i)
        for(j = 0; j != P; ++j)
            C[i * ldc + j] = buffer[i * P + j];
    cudaFree(Adev);
    cudaFree(Bdev);
    cudaFree(Cdev);
}

double get_random_number(double min, double max)
{
    double scale = rand() / (double) RAND_MAX;
	return min + scale * (max - (min));
    return rand() % 20;
}

int compute_mcm(int nprow, int npcol)
{
    int x = nprow;
    int y = npcol;
    while(y != 0)
    {
        int division = x / y;
        int temp = x - division * y;
        x = y;
        y = temp;
    }
    return nprow * npcol / x;
}

void print_matrix(double *matrix, int rows, int columns)
{
    int row, column;
    for(row = 0; row != rows; ++row)
    {
        printf("Row: %d\n", row + 1);
        for(column = 0; column != columns; ++column)
            printf("%f ", matrix[row * columns + column]);
        puts("");
    }
}

double* init_matrix(int rows, int columns)
{
    double *matrix = (double*)calloc(rows * columns, sizeof(double));
    int row, column;
    for(row = 0; row != rows; ++row)
        for(column = 0; column != columns; ++column)
            matrix[row * columns + column] = get_random_number(MIN_RANDOM_NUMBER, MAX_RANDOM_NUMBER);
    return matrix;
}

int get_amount_blocks_matrix_A(int columns, int grid_rows, int grid_columns)
{
    int blocks_to_send_columns = columns / compute_mcm(grid_rows, grid_columns);
    int leading_dimension_periodic_matrix = blocks_to_send_columns * grid_columns;
    int amount_blocks = columns / leading_dimension_periodic_matrix;
    return amount_blocks;
}

int get_blocks_columns_matrix_A(int columns, int grid_rows, int grid_columns)
{
    int blocks_to_send_columns = columns / compute_mcm(grid_rows, grid_columns);
    return blocks_to_send_columns;
}

int get_blocks_rows_matrix_A(int rows, int grid_rows)
{
    int blocks_to_send_rows = rows / grid_rows;
    return blocks_to_send_rows;
}

int get_amount_blocks_matrix_B(int rows, int columns, int grid_rows, int grid_columns)
{
    int blocks_to_send_columns = columns / grid_columns;
    int blocks_to_send_rows = rows / compute_mcm(grid_rows, grid_columns);
    int leading_dimension_periodic_matrix = blocks_to_send_columns * grid_columns;
    int periodic_matrix_rows = blocks_to_send_rows * grid_rows;
    int amount_blocks = (rows * columns) / (periodic_matrix_rows * leading_dimension_periodic_matrix);
    return amount_blocks;
}

int get_blocks_columns_matrix_B(int columns, int grid_rows, int grid_columns)
{
    int blocks_to_send_columns = columns / grid_columns;
    return blocks_to_send_columns;
}

int get_blocks_rows_matrix_B(int rows, int grid_rows, int grid_columns)
{
    int blocks_to_send_rows = rows / compute_mcm(grid_rows, grid_columns);
    return blocks_to_send_rows;
}

double** get_buffer_blocks_matrix_A(int rows, int columns, int grid_rows, int grid_columns)
{
    int blocks_to_send_columns = columns / compute_mcm(grid_rows, grid_columns);
    int blocks_to_send_rows = rows / grid_rows;
    int leading_dimension_periodic_matrix = blocks_to_send_columns * grid_columns;
    int amount_blocks = columns / leading_dimension_periodic_matrix;
    double **blocks = (double**)calloc(amount_blocks, sizeof(double*));
    int i;
    for(i = 0; i != amount_blocks; ++i)
        blocks[i] = (double*)calloc(blocks_to_send_rows * blocks_to_send_columns, sizeof(double));
    return blocks;
}

double** get_buffer_blocks_matrix_B(int rows, int columns, int grid_rows, int grid_columns)
{
    int blocks_to_send_columns = columns / grid_columns;
    int blocks_to_send_rows = rows / compute_mcm(grid_rows, grid_columns);
    int leading_dimension_periodic_matrix = blocks_to_send_columns * grid_columns;
    int periodic_matrix_rows = blocks_to_send_rows * grid_rows;
    int amount_blocks = (rows * columns) / (periodic_matrix_rows * leading_dimension_periodic_matrix);
    double **blocks = (double**)calloc(amount_blocks, sizeof(double*));
    int i;
    for(i = 0; i != amount_blocks; ++i)
        blocks[i] = (double*)calloc(blocks_to_send_rows * blocks_to_send_columns, sizeof(double));
    return blocks;
}

double* get_block_matrix_C(int rows, int columns, int grid_rows, int grid_columns)
{
    int blocks_to_send_columns = columns / grid_columns;
    int blocks_to_send_rows = rows / grid_rows;
    double *block = (double*)calloc(blocks_to_send_columns * blocks_to_send_rows, sizeof(double));
    return block;
}

int get_blocks_columns_matrix_C(int columns, int grid_columns)
{
    int blocks_to_send_columns = columns / grid_columns;
    return blocks_to_send_columns;
}

int get_blocks_rows_matrix_C(int rows, int grid_rows)
{
    int blocks_to_send_rows = rows / grid_rows;
    return blocks_to_send_rows;
}

void free_buffer_blocks(double **buffer_blocks, int amount_blocks)
{
    int i;
    for(i = 0; i != amount_blocks; ++i)
        free(buffer_blocks[i]);
    free(buffer_blocks);
}

void send_submatrix(int rank_processor, int processors, int processor_destination, double *matrix_entry_point, int leading_dimension, int submatrix_rows, int submatrix_columns, MPI_Comm comm)
{
    MPI_Datatype submatrix;
    MPI_Type_vector(submatrix_rows, submatrix_columns, leading_dimension, MPI_DOUBLE, &submatrix);
    MPI_Type_commit(&submatrix);
    MPI_Send(matrix_entry_point, 1, submatrix, processor_destination, 10 + processor_destination, comm);
}

int* get_matrix_entry_points(int processors, double *matrix, int leading_dimension, int submatrix_rows, int submatrix_columns, MPI_Comm grid)
{
    int i;
    int *entry_points = (int*)calloc(processors, sizeof(int));
    entry_points[0] = 0;
    int coordinates[2];
    for(i = 1; i != processors; ++i)
    {
        MPI_Cart_coords(grid, i, 2, coordinates);
        int column = coordinates[1] * submatrix_columns;
        int row = coordinates[0] * submatrix_rows;
        entry_points[i] = row * leading_dimension + column;
    }
    return entry_points;
}

void distribute_submatrices(int rank_processor, int processors, double *matrix, int leading_dimension, double *block, int submatrix_rows, int submatrix_columns, MPI_Comm grid)
{
    if(rank_processor == 0)
    {
        int *entry_points = get_matrix_entry_points(processors, matrix, leading_dimension, submatrix_rows, submatrix_columns, grid);
        int i;
        for(i = 1; i != processors; ++i)
            send_submatrix(rank_processor, processors, i, matrix + entry_points[i], leading_dimension, submatrix_rows, submatrix_columns, grid);
        free(entry_points);
    }
    else
        MPI_Recv(block, submatrix_rows * submatrix_columns, MPI_DOUBLE, 0, 10 + rank_processor, grid, MPI_STATUS_IGNORE);
}

double* get_submatrix(double *matrix, int rows, int columns, int submatrix_rows, int submatrix_columns)
{
    double *submatrix = (double*)calloc(submatrix_rows * submatrix_columns, sizeof(double));
    int i, j;
    for(i = 0; i != submatrix_rows; ++i)
        for(j = 0; j != submatrix_columns; ++j)
            submatrix[i * submatrix_columns + j] = matrix[i * columns + j];
    return submatrix;
}

void distribute_matrix_A(MPI_Comm grid, int rank, int processors, double *A, int rows, int columns, int grid_rows, int grid_columns, double **blocks)
{
    int blocks_to_send_columns = columns / compute_mcm(grid_rows, grid_columns);
    int blocks_to_send_rows = rows / grid_rows;
    int leading_dimension_periodic_matrix = blocks_to_send_columns * grid_columns;
    int periodic_matrix_rows = blocks_to_send_rows * grid_rows;
    int i, j;
    for(i = 0, j = 0; i < columns; i += leading_dimension_periodic_matrix, ++j)
    {
        double *submatrix = NULL;
        if(rank == 0)
             submatrix = get_submatrix(&A[i], rows, columns, periodic_matrix_rows, leading_dimension_periodic_matrix);
        distribute_submatrices(rank, processors, submatrix, leading_dimension_periodic_matrix, blocks[j], blocks_to_send_rows, blocks_to_send_columns, grid);
        free(submatrix);
    }
}

void distribute_matrix_B(MPI_Comm grid, int rank, int processors, double *B, int rows, int columns, int grid_rows, int grid_columns, double **blocks)
{
    int blocks_to_send_columns = columns / grid_columns;
    int blocks_to_send_rows = rows / compute_mcm(grid_rows, grid_columns);
    int leading_dimension_periodic_matrix = blocks_to_send_columns * grid_columns;
    int periodic_matrix_rows = blocks_to_send_rows * grid_rows;
    int i, j;
    for(i = 0, j = 0; i < columns * rows; i += leading_dimension_periodic_matrix * periodic_matrix_rows, ++j)
    {
        double *submatrix = NULL;
        if(rank == 0)
            submatrix = get_submatrix(&B[i], rows, columns, periodic_matrix_rows, leading_dimension_periodic_matrix);
        distribute_submatrices(rank, processors, submatrix, leading_dimension_periodic_matrix, blocks[j], blocks_to_send_rows, blocks_to_send_columns, grid);
        free(submatrix);
    }
}

void distribute_matrix_C(MPI_Comm grid, int rank, int processors, double *C, int rows, int columns, int grid_rows, int grid_columns, double *block)
{
    int blocks_to_send_columns = columns / grid_columns;
    int blocks_to_send_rows = rows / grid_rows;
    distribute_submatrices(rank, processors, C, columns, block, blocks_to_send_rows, blocks_to_send_columns, grid);
}

void get_blocks_processor_0_matrix_A(double *A, double **blocks, int columns, int block_rows, int block_columns, int grid_columns)
{
    int stride = block_columns * grid_columns;
    int i, j, y, z;
    for(i = 0, z = 0; i < columns; i += stride, ++z)
        for(j = 0; j != block_rows; ++j)
            for(y = 0; y != block_columns; ++y)
                blocks[z][j * block_columns + y] = A[z * stride + j * columns + y];
}

void get_blocks_processor_0_matrix_B(double *B, double **blocks, int rows, int columns, int block_rows, int block_columns, int grid_columns, int grid_rows)
{
    int stride = block_columns * grid_columns;
    int beginning_submatrix = stride * block_rows * grid_rows;
    int i, j, y, z;
    for(i = 0, z = 0; i < rows * columns; i += beginning_submatrix, ++z)
        for(j = 0; j != block_rows; ++j)
            for(y = 0; y != block_columns; ++y)
                blocks[z][j * block_columns + y] = B[z * beginning_submatrix + j * columns + y];
}

double* get_block_processor_0_matrix_C(double *C, int columns, int block_rows, int block_columns, int grid_columns, int grid_rows)
{
    double *block = (double*)calloc(block_rows * block_columns, sizeof(double));
    int i, j;
    for(i = 0; i != block_rows; ++i)
        for(j = 0; j != block_columns; ++j)
            block[i * block_columns + j] = C[i * columns + j];
    return block;
}

void matmatikj(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N, int M, int P)
{
    int i, k, j;
    for (i = 0; i < N; ++i)
        for (k = 0; k < P; ++k)
            for (j = 0; j < M; ++j)
                C[i * ldC + j] = C[i * ldC + j] + A[i * ldA + k] * B[k * ldB + j];
}

void matmatblock(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N, int M, int P, int dbA, int dbB, int dbC)
{
    int i, k, j;
    for (i = 0; i < N / dbA; ++i)
        for (j = 0; j < M / dbB; ++j)
            for (k = 0; k < P / dbC; ++k)
                matmatikj(ldA, ldB, ldC, &A[i * ldA * dbA + j * dbA], &B[j * ldB * dbB + k * dbB], &C[i * ldC * dbA + k * dbC], dbA, dbB, dbC);
}

void matmatthread(int ldA, int ldB, int ldC, double *A, double *B, double *C, int N, int M, int P, int dbA, int dbB, int dbC, int ntrow, int ntcol)
{
    omp_set_num_threads(ntrow * ntcol);
    int thread_row, thread_column;
    #pragma omp parallel private(thread_row, thread_column)
    {
        thread_row  = omp_get_thread_num() / ntcol;
        thread_column = omp_get_thread_num() % ntcol;
        matmatblock(ldA, ldB, ldC, &A[thread_row * ldA * N / ntrow], &B[thread_column * P / ntcol], &C[thread_row * ldC * N / ntrow + thread_column * P / ntcol], N / ntrow, M, P / ntcol, dbA, dbB, dbC);
    }
}

double* concat_matrices_A(double *A, double *B, int rows, int columns)
{
    double *C = (double*)calloc(rows * columns * 2, sizeof(double));
    int i, j;
    for(i = 0; i != rows; ++i)
        for(j = 0; j != columns * 2; ++j)
        {
            if(j < columns / 2)
                C[i * columns * 2 + j] = A[i * columns + j];
            else
                C[i * columns * 2 + j] = B[i * columns + j];
        }
    return C;
}

double* merge_blocks_matrix_A(double **blocks, int rows, int columns, int amount)
{
    double *merged_blocks = (double*)calloc(rows * columns * amount, sizeof(double));
    int i, j, z;
    for(i = 0; i != amount; ++i)
        for(j = 0; j != rows; ++j)
            for(z = 0; z != columns; ++z)
                merged_blocks[(i * columns) + (j * columns * amount + z)] = blocks[i][j * columns + z];
    return merged_blocks;
}

double* merge_blocks_matrix_B(double **blocks, int rows, int columns, int amount)
{
    double *merged_blocks = (double*)calloc(rows * amount * columns, sizeof(double));
    int i;
    for(i = 0; i != amount; ++i)
        memcpy(&merged_blocks[i * rows * columns], blocks[i], sizeof(double) * rows * columns);
    return merged_blocks;
}

void matmatdist(MPI_Comm grid, int ldA, int ldB, int ldC, double *A, double *B, double *C, int N, int M, int P, int dbN, int dbM, int dbP, int ntrow, int ntcol, int nprow, int npcol)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int coord[2];
    MPI_Cart_coords(grid, rank, 2, coord);
    int mcm = compute_mcm(nprow, npcol);
    int N_local = N / nprow;
    int M_local = M / mcm;
    int P_local = P / npcol;
    int column_A_dimension = N_local * M_local;
    int row_B_dimension = M_local * P_local;
    double *A_column = (double*)malloc(column_A_dimension * sizeof(double));
    double *B_row = (double*)malloc(row_B_dimension * sizeof(double));
    int remain[2] = { 0, 1 };
    MPI_Comm comm_rows;
    MPI_Cart_sub(grid, remain, &comm_rows);
    remain[0] = 1;
    remain[1] = 0;
    MPI_Comm comm_columns;
    MPI_Cart_sub(grid, remain, &comm_columns);
    double *block_A = A;
    double *block_B = B;
    int k;
    for(k = 0; k != mcm; ++k)
    {
        int column_broadcast = k % npcol;
        int row_broadcast = k % nprow;
        if(coord[1] == column_broadcast)
        {
            int index = 0;
            int i, j;
            for(i = 0; i != N_local; ++i)
                for(j = 0; j != M_local; ++j)
                {
                    A_column[index] = block_A[ldA * i + j];
                    ++index;
                }
            block_A = block_A + M_local;     
        }
        if(coord[0] == row_broadcast)
        {
            int index = 0;
            int i, j;
            for(i = 0; i != M_local; ++i)
                for(j = 0; j != P_local; ++j)
                {
                    B_row[index] = block_B[ldB * i + j];
                    ++index;
                }
            block_B = block_B + M_local * ldB;
        }
        MPI_Bcast(A_column, column_A_dimension, MPI_DOUBLE, column_broadcast, comm_rows);
        MPI_Bcast(B_row, row_B_dimension, MPI_DOUBLE, row_broadcast, comm_columns);
        matmatgpu1(M_local, P_local, ldC, A_column, B_row, C, N_local, M_local, P_local);
    }
}

void copy_block(double *C, int columns, double *block, int block_rows, int block_columns, int row_stride, int column_stride)
{
    int i, z, j;
    for(i = row_stride, z = 0; z < block_rows; ++i, ++z)
        for(j = 0; j < block_columns; ++j)
            C[i * columns + j + column_stride] = block[z * block_columns + j];
}

void get_multiplication_product(int rank_processor, int processors, double *C, int columns, double *block, int block_rows, int block_columns, MPI_Comm grid_comm)
{
    if(rank_processor == 0)
    {
        copy_block(C, columns, block, block_rows, block_columns, 0, 0);
        double *received_block = (double*)calloc(block_rows * block_columns, sizeof(double));
        int coordinates[2];
        int i;
        for(i = 1; i != processors; ++i)
        {
            MPI_Cart_coords (grid_comm, i, 2, coordinates);
            MPI_Recv(received_block, block_rows * block_columns, MPI_DOUBLE, i, 100 + i, grid_comm, MPI_STATUS_IGNORE);
            copy_block(C, columns, received_block, block_rows, block_columns, coordinates[0] * block_rows, coordinates[1] * block_columns);
        }
        free(received_block);
    }
    else
        MPI_Send(block, block_rows * block_columns, MPI_DOUBLE, 0, 100 + rank_processor, grid_comm);
}

int main(int argc, char **argv)
{
    double *A = NULL;
    double *B = NULL;
    double *C = NULL;
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int dims[2] = { 2, 2 };
    int period[2] = { 1, 1 };
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 0, &grid_comm);
    int ldA = MATRIX_DIMENSION;
    int N = MATRIX_DIMENSION;
    int ldB = MATRIX_DIMENSION;
    int M = MATRIX_DIMENSION;
    int ldC = MATRIX_DIMENSION;
    int P = MATRIX_DIMENSION;
    if(rank == 0)
    {
        A = init_matrix(N, ldA);
        B = init_matrix(M, ldB);
        C = init_matrix(P, ldC);
    }

    //PRINTING THE MATRICES
    /*if(rank == 0)
    {
        puts("MATRIX A------------------");
        print_matrix(A, N, ldA);
        puts("MATRIX B------------------");
        print_matrix(B, M, ldB);
        puts("MATRIX C------------------");
        print_matrix(C, P, ldC);
    }*/

    //DISTRIBUTION MATRIX A
    double **buffer_blocks_matrix_A = get_buffer_blocks_matrix_A(N, ldA, dims[0], dims[1]);
    int amount_block_matrix_A = get_amount_blocks_matrix_A(ldA, dims[0], dims[1]);
    int blocks_rows_matrix_A = get_blocks_rows_matrix_A(N, dims[0]);
    int blocks_columns_matrix_A = get_blocks_columns_matrix_A(ldA, dims[0], dims[1]);
    distribute_matrix_A(grid_comm, rank, size, A, N, ldA, dims[0], dims[1], buffer_blocks_matrix_A);
    if(rank == 0)
        get_blocks_processor_0_matrix_A(A, buffer_blocks_matrix_A, ldA, blocks_rows_matrix_A, blocks_columns_matrix_A, dims[1]);
    double *block_merged_matrix_A = merge_blocks_matrix_A(buffer_blocks_matrix_A, blocks_rows_matrix_A, blocks_columns_matrix_A, amount_block_matrix_A);
    free_buffer_blocks(buffer_blocks_matrix_A, amount_block_matrix_A);
    
    //DISTRIBUTION MATRIX B
    double **buffer_blocks_matrix_B = get_buffer_blocks_matrix_B(M, ldB, dims[0], dims[1]);
    int amount_block_matrix_B = get_amount_blocks_matrix_B(M, ldB, dims[0], dims[1]);
    int blocks_rows_matrix_B = get_blocks_rows_matrix_B(M, dims[0], dims[1]);
    int blocks_columns_matrix_B = get_blocks_columns_matrix_B(ldB, dims[0], dims[1]);
    distribute_matrix_B(grid_comm, rank, size, B, M, ldB, dims[0], dims[1], buffer_blocks_matrix_B);
    if(rank == 0)
        get_blocks_processor_0_matrix_B(B, buffer_blocks_matrix_B, M, ldB, blocks_rows_matrix_B, blocks_columns_matrix_B, dims[1], dims[0]);
    double *block_merged_matrix_B = merge_blocks_matrix_B(buffer_blocks_matrix_B, blocks_rows_matrix_B, blocks_columns_matrix_B, amount_block_matrix_B);
    free_buffer_blocks(buffer_blocks_matrix_B, amount_block_matrix_B);

    //DISTRIBUTION MATRIX C
    double *block_matrix_C = NULL;
    if(rank != 0)
        block_matrix_C = get_block_matrix_C(P, ldC, dims[0], dims[1]);
    int blocks_rows_matrix_C = get_blocks_rows_matrix_C(P, dims[0]);
    int blocks_columns_matrix_C = get_blocks_columns_matrix_C(ldC, dims[1]);
    distribute_matrix_C(grid_comm, rank, size, C, P, ldC, dims[0], dims[1], block_matrix_C);
    if(rank == 0)
        block_matrix_C = get_block_processor_0_matrix_C(C, ldC, blocks_rows_matrix_C, blocks_columns_matrix_C, dims[1], dims[0]);

    //PRODUCT
    long double time_beginning = MPI_Wtime();
    matmatdist(grid_comm, blocks_columns_matrix_A, blocks_columns_matrix_B, blocks_columns_matrix_C, 
               block_merged_matrix_A, block_merged_matrix_B, block_matrix_C, N, M, P, 
               1, 1, 1, 1, 1, dims[0], dims[1]);
    long double time_end = MPI_Wtime();
    long double elapsed_time = time_end - time_beginning;
    long double max_elapsed_time = 0;
    MPI_Reduce(&elapsed_time, &max_elapsed_time, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    //GATHERING
    get_multiplication_product(rank, size, C, ldC, block_matrix_C, blocks_rows_matrix_C, blocks_columns_matrix_C, grid_comm);

    //PRINTING TIME AND GFlops
    if(rank == 0)
        printf("Dimension: %d; Time elapsed: %Lf; Gflops: %Lf\n", MATRIX_DIMENSION, max_elapsed_time, (2 * pow(MATRIX_DIMENSION, 3)) / max_elapsed_time / pow(10, 9));

    //PRINTING THE RESULT
    /*if(rank == 0)
    {
        puts("RESULT----------------");
        print_matrix(C, P, ldC);
    }*/

    free(block_merged_matrix_B);
    free(block_merged_matrix_A);
    free(block_matrix_C);
    if(rank == 0)
    {
        free(A);
        free(B);
        free(C);
    }
    MPI_Finalize();
    return 0;
}
