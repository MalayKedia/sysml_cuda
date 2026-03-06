#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <cmath>

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K);

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
    solve(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C, d_C, M*K*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Matrix size: " << M << "x" << N << " * " << N << "x" << K << "\n";
    std::cout << "GPU Time: " << ms << " ms\n";

    if (verify) {
        cpu_matmul(h_A, h_B, h_ref, M, N, K);

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

    std::cout << "Test Case 2 (Scaling Test)\n";
    run_test(150, 250, 200, false);

    return 0;
}