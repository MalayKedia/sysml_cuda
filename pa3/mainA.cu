#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cmath>
#include <chrono>

__global__ void matmul_kernel(float *A, float *B, float *C, int N, int Bt)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Bt && col < N)
    {
        float val = 0.0f;
        for (int k = 0; k < N; k++)
        {
            val += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = val;
    }
}

__global__ void relu_kernel(float *A, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        A[idx] = fmaxf(A[idx], 0.0f);
    }
}

void random_init(float *data, int size)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    for (int i = 0; i < size; i++) data[i] = dist(rng);
}

void gpu_ffnn(float *d_X, float *d_W1, float *d_W2, float *d_Y, float *d_Z, int N, int B, int BLOCK_SIZE)
{

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (B + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel<<<grid, block>>>(d_X, d_W1, d_Y, N, B);

    int relu_size = B * N;
    relu_kernel<<<(relu_size + 255) / 256, 256>>>(d_Y, relu_size);

    matmul_kernel<<<grid, block>>>(d_Y, d_W2, d_Z, N, B);
}

void cpu_ffnn(float *h_X, float *h_W1, float *h_W2, float *h_Y, float *h_Z, int N, int B)
{
    for (int i = 0; i < B; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float val = 0.0f;
            for (int k = 0; k < N; k++)
            {
                val += h_X[i * N + k] * h_W1[k * N + j];
            }
            h_Y[i * N + j] = fmaxf(val, 0.0f);
        }
    }

    for (int i = 0; i < B; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float val = 0.0f;
            for (int k = 0; k < N; k++)
            {
                val += h_Y[i * N + k] * h_W2[k * N + j];
            }
            h_Z[i * N + j] = val;
        }
    }
}

int main()
{
    int N = 4096;
    int B = 1024;
    int BLOCK_SIZE = 16;

    size_t X_size = B * N * sizeof(float);
    size_t W_size = N * N * sizeof(float);

    float *h_X = (float *)malloc(X_size);
    float *h_W1 = (float *)malloc(W_size);
    float *h_W2 = (float *)malloc(W_size);

    random_init(h_X, B * N);
    random_init(h_W1, N * N);
    random_init(h_W2, N * N);

    float *d_X, *d_W1, *d_W2, *d_Y, *d_Z;

    cudaMalloc(&d_X, X_size);
    cudaMalloc(&d_W1, W_size);
    cudaMalloc(&d_W2, W_size);
    cudaMalloc(&d_Y, X_size);
    cudaMalloc(&d_Z, X_size);

    cudaMemcpy(d_X, h_X, X_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, h_W1, W_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, W_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    gpu_ffnn(d_X, d_W1, d_W2, d_Y, d_Z, N, B, BLOCK_SIZE);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Time taken by GPU for %d x %d matrix: %f ms\n", N, B, ms);

    float *h_Y_gpu = (float *)malloc(X_size);
    float *h_Z_gpu = (float *)malloc(X_size);
    cudaMemcpy(h_Y_gpu, d_Y, X_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Z_gpu, d_Z, X_size, cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_Y);
    cudaFree(d_Z);

    // cpu validation to check correctness
    float *h_Y_cpu = (float *)malloc(X_size);
    float *h_Z_cpu = (float *)malloc(X_size);



    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_ffnn(h_X, h_W1, h_W2, h_Y_cpu, h_Z_cpu, N, B);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    printf("Time taken by CPU for %d x %d matrix: %f ms\n", N, B, cpu_duration.count());

    // Compare results
    for (int i = 0; i < B * N; i++)
    {
        if (fabs(h_Y_cpu[i] - h_Y_gpu[i]) > 1e-3)
        {
            printf("Mismatch in Y at index %d: CPU %f, GPU %f\n", i, h_Y_cpu[i], h_Y_gpu[i]);
            break;
        }
    }


    return 0;
}