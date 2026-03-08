#include <cuda_runtime.h>

// Every thread in a block loads one element each of A and B, and computes one element of C. This simple way of tiling has already been discussed in class. 

#define NELEM 8
#define TILE_SIZE 32

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][NELEM * TILE_SIZE];
    
    int globalCol = blockIdx.x * TILE_SIZE*NELEM + threadIdx.x;
    int globalRow = blockIdx.y * TILE_SIZE + threadIdx.y;

    int localCol = threadIdx.x;
    int localRow = threadIdx.y;

    float sum[NELEM] = {0.0f};

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aRow = globalRow;
        int aCol = t * TILE_SIZE + localCol;

        if (aRow < M && aCol < N) {
            tile_A[localRow][localCol] = A[aRow * N + aCol];
        } else {
            tile_A[localRow][localCol] = 0.0f;
        }

        for (int e = 0; e < NELEM; e++) {
            int bRow = t * TILE_SIZE + localRow;
            int bCol = globalCol + e * TILE_SIZE;

            if (bRow < N && bCol < K) {
                tile_B[localRow][e * TILE_SIZE + localCol] = B[bRow * K + bCol];
            } else {
                tile_B[localRow][e * TILE_SIZE + localCol] = 0.0f;
            }
        }

        __syncthreads();
    
        for (int i = 0; i < TILE_SIZE; i++) {
            float aVal = tile_A[localRow][i];

            for (int e = 0; e < NELEM; e++) {
                sum[e] += aVal * tile_B[i][e * TILE_SIZE + localCol];
            }
        }

        __syncthreads();
    }

    for (int e = 0; e < NELEM; e++) {
        int cCol = globalCol + e * TILE_SIZE;

        if (globalRow < M && cCol < K) {
            C[globalRow * K + cCol] = sum[e];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void matmul_gpu(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((K + TILE_SIZE*NELEM - 1) / (TILE_SIZE*NELEM),
                       (M + TILE_SIZE - 1) / TILE_SIZE);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}