#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <cmath>
#include <chrono>

#ifdef NOT_LEETGPU
#include <cublas_v2.h>
#endif

extern "C" void matmul_gpu(const float* A, const float* B, float* C, int M, int N, int K);

void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0;
            for (int t = 0; t < N; t++) {
                sum += A[i*N + t] * B[t*K + j];
            }
            C[i*K + j] = sum;
        }
    }
}

void cublas_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    #ifdef NOT_LEETGPU
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_C_cublas;
    cudaMalloc(&d_C_cublas, M*K*sizeof(float));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha, B, K, A, N, &beta, C, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cublasDestroy(handle);

    cudaFree(d_C_cublas);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "cuBLAS Time: " << ms << " ms\n";
    #endif
}

void fill_random(float* A, int size) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    for (int i = 0; i < size; i++) A[i] = dist(rng);
}

void run_test(int M, int N, int K, bool verify) {
    float *h_A = new float[M*N];
    float *h_B = new float[N*K];
    float *h_C = new float[M*K];
    float *h_ref = verify ? new float[M*K] : nullptr;

    fill_random(h_A, M*N);
    fill_random(h_B, N*K);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*N*sizeof(float));
    cudaMalloc(&d_B, N*K*sizeof(float));
    cudaMalloc(&d_C, M*K*sizeof(float));

    cudaMemcpy(d_A, h_A, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*K*sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_gpu(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C, d_C, M*K*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Matrix size: " << M << "x" << N << " * " << N << "x" << K << "\n";
    std::cout << "GPU Time: " << ms << " ms\n";

    if (verify) {
        auto cpu_start = std::chrono::high_resolution_clock::now();

        cpu_matmul(h_A, h_B, h_ref, M, N, K);

        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
        std::cout << "CPU Time: " << cpu_duration.count() << " ms\n";

        bool ok = true;
        for (int i = 0; i < M*K; i++) {
            if (std::fabs(h_ref[i] - h_C[i]) > 1e-4) {
                ok = false;
                break;
            }
        }

        std::cout << "Verification: " << (ok ? "PASSED" : "FAILED") << "\n";
    }

    std::cout << "-------------------------\n";

    cublas_matmul(d_A, d_B, d_C, M, N, K);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    if (verify) delete[] h_ref;
}

int main() {

    std::cout << "Test Case 1 (Correctness Check)\n";
    run_test(3, 3, 3, true);

    std::cout << "Test Case 2 (150 by 200)\n";
    run_test(150, 250, 200, true);

    #ifdef LARGE_TESTS
    std::cout << "Test Case 3 (1024 by 1024)\n";
    run_test(1024, 1024, 1024, true);

    std::cout << "Test Case 4 (2048 by 2048)\n";
    run_test(2048, 2048, 2048, false);

    std::cout << "Test Case 5 (4096 by 4096)\n";
    run_test(4096, 4096, 4096, false);
    #endif

    return 0;
}