#include <cuda_runtime.h>

// Every thread in a block loads one element each of A and B, and computes one element of C. This simple way of tiling has already been discussed in class. 

#define TILE_SIZE 32

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int globalCol = blockIdx.x * blockDim.x + threadIdx.x;
    int globalRow = blockIdx.y * blockDim.y + threadIdx.y;

    int localCol = threadIdx.x;
    int localRow = threadIdx.y;

    float sum = 0.0f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aRow = globalRow;
        int aCol = t * TILE_SIZE + localCol;
        int bRow = t * TILE_SIZE + localRow;
        int bCol = globalCol;

        if (aRow < M && aCol < N) {
            tile_A[localRow * TILE_SIZE + localCol] = A[aRow * N + aCol];
        } else {
            tile_A[localRow * TILE_SIZE + localCol] = 0.0f;
        }

        if (bRow < N && bCol < K) {
            tile_B[localRow * TILE_SIZE + localCol] = B[bRow * K + bCol];
        } else {
            tile_B[localRow * TILE_SIZE + localCol] = 0.0f;
        }

        __syncthreads();
    
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tile_A[localRow * TILE_SIZE + i] * tile_B[i * TILE_SIZE + localCol];
        }

        __syncthreads();
    }

    if (globalRow < M && globalCol < K) {
        C[globalRow * K + globalCol] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void matmul_gpu(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}