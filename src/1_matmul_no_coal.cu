#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int t = 0; t < N; t++) {
            sum += A[row * N + t] * B[t * K + col];
        }
        C[row * K + col] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void matmul_gpu(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (K + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}