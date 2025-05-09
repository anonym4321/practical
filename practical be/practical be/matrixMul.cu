#include <iostream>
#include <cuda_runtime.h>

__global__ void matrixMul(float *A, float *B, float *C, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < N && col < N) {
        float value = 0;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

void printMatrix(const char* name, float* M, int N) {
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << M[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int N = 3;
    size_t size = N * N * sizeof(float);

    // Host memory
    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C = (float*)malloc(size);

    // Initialize A and B
    for (int i = 0; i < N * N; ++i) {
        A[i] = (i % N) + 1;            // 1, 2, 3, 1, 2, 3,...
        B[i] = ((i % N) + 1) * 2;      // 2, 4, 6, 2, 4, 6,...
    }

    // Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print matrices
    printMatrix("Matrix A", A, N);
    printMatrix("Matrix B", B, N);
    printMatrix("Result Matrix C (A x B)", C, N);

    // Free memory
    free(A); free(B); free(C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
