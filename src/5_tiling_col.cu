#include <cuda_runtime.h>

// Every thread in a block loads one element each of A and B, and computes one element of C. This simple way of tiling has already been discussed in class. 

#define NELEM 8
#define TILE_SIZE 32

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tile_A[NELEM * TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int globalCol = blockIdx.x * TILE_SIZE + threadIdx.x;
    int globalRow = blockIdx.y * TILE_SIZE*NELEM + threadIdx.y;

    int localCol = threadIdx.x;
    int localRow = threadIdx.y;

    float sum[NELEM] = {0.0f};

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {

        for (int e = 0; e < NELEM; e++){        
            int aRow = globalRow + e * TILE_SIZE;
            int aCol = t * TILE_SIZE + localCol;

            if (aRow < M && aCol < N) {
                tile_A[e * TILE_SIZE + localRow][localCol] = A[aRow * N + aCol];
            } else {
                tile_A[e * TILE_SIZE + localRow][localCol] = 0.0f;
            }
        }

        int bRow = t * TILE_SIZE + localRow;
        int bCol = globalCol;

        if (bRow < N && bCol < K) {
            tile_B[localRow][localCol] = B[bRow * K + bCol];
        } else {
            tile_B[localRow][localCol] = 0.0f;
        }

        __syncthreads();
    
        for (int i = 0; i < TILE_SIZE; i++) {
            float bVal = tile_B[i][localCol];

            for (int e = 0; e < NELEM; e++) {
                sum[e] += tile_A[e * TILE_SIZE + localRow][i] * bVal;
            }
        }

        __syncthreads();
    }

    for (int e = 0; e < NELEM; e++) {
        int cRow = globalRow + e * TILE_SIZE;

        if (globalCol < K && cRow < M) {
            C[cRow * K + globalCol] = sum[e];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void matmul_gpu(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((K + TILE_SIZE - 1) / TILE_SIZE,
                       (M + TILE_SIZE*NELEM - 1) / (TILE_SIZE*NELEM));

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}