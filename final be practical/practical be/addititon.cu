#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel to add two vectors
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 10; 
    size_t size = N * sizeof(float);

    // Host memory
    float *A, *B, *C;

    // Device memory
    float *d_A, *d_B, *d_C;

    // Allocate host memory
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        A[i] = i + 1;
        B[i] = (i + 1) * 2;
    }

    // Copy host arrays to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Output results
    cout << "Vector A: ";
    for (int i = 0; i < N; i++) {
        cout << A[i] << " ";
    }
    cout << endl;

    cout << "Vector B: ";
    for (int i = 0; i < N; i++) {
        cout << B[i] << " ";
    }
    cout << endl;

    cout << "Calculations (A[i] + B[i]):" << endl;
    for (int i = 0; i < N; i++) {
        cout << "C[" << i << "] = " << A[i] << " + " << B[i] << " = " << C[i] << endl;
    }

    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
