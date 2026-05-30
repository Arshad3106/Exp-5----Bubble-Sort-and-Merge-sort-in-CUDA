# Exp5 Bubble Sort and Merge sort in CUDA
**Objective:**
Implement Bubble Sort and Merge Sort on the GPU using CUDA, analyze the efficiency of this sorting algorithm when parallelized, and explore the limitations of Bubble Sort and Merge Sort for large datasets.
## AIM:
Implement Bubble Sort and Merge Sort on the GPU using CUDA to enhance the performance of sorting tasks by parallelizing comparisons and swaps within the sorting algorithm.

Code Overview:
You will work with the provided CUDA implementation of Bubble Sort and Merge Sort. The code initializes an unsorted array, applies the Bubble Sort, Merge Sort algorithm in parallel on the GPU, and returns the sorted array as output.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC, Google Colab with NVCC Compiler, CUDA Toolkit installed, and sample datasets for testing.

## PROCEDURE:

Tasks:

a. Modify the Kernel:

Implement Bubble Sort and Merge Sort using CUDA by assigning each comparison and swap task to individual threads.
Ensure the kernel checks boundaries to avoid out-of-bounds access, particularly for edge cases.
b. Performance Analysis:

Measure the execution time of the CUDA Bubble Sort with different array sizes (e.g., 512, 1024, 2048 elements).
Experiment with various block sizes (e.g., 16, 32, 64 threads per block) to analyze their effect on execution time and efficiency.
c. Comparison:

Compare the performance of the CUDA-based Bubble Sort and Merge Sort with a CPU-based Bubble Sort and Merge Sort implementation.
Discuss the differences in execution time and explain the limitations of Bubble Sort and Merge Sort when parallelized on the GPU.
## PROGRAM:
```
%%writefile sorting.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

// ==============================
// Bubble Sort Kernel (GPU)
// ==============================
__global__ void bubbleSortKernel(int *d_arr, int n) {

    int idx = threadIdx.x;

    // Bubble sort passes
    for (int i = 0; i < n - 1; i++) {

        // Even phase
        if ((idx % 2 == 0) && (idx < n - 1)) {
            if (d_arr[idx] > d_arr[idx + 1]) {
                int temp = d_arr[idx];
                d_arr[idx] = d_arr[idx + 1];
                d_arr[idx + 1] = temp;
            }
        }

        __syncthreads();

        // Odd phase
        if ((idx % 2 == 1) && (idx < n - 1)) {
            if (d_arr[idx] > d_arr[idx + 1]) {
                int temp = d_arr[idx];
                d_arr[idx] = d_arr[idx + 1];
                d_arr[idx + 1] = temp;
            }
        }

        __syncthreads();
    }
}

// ==============================
// Device Merge Function
// ==============================
__device__ void merge(int *arr, int left, int mid, int right, int *temp) {

    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right) {

        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        }
        else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }

    // Copy back
    for (i = left; i <= right; i++) {
        arr[i] = temp[i];
    }
}

// ==============================
// Merge Sort Kernel (GPU)
// ==============================
__global__ void mergeSortKernel(int *d_arr, int *d_temp, int n) {

    for (int size = 1; size < n; size *= 2) {

        int left = 0;

        while (left + size < n) {

            int mid = left + size - 1;

            int right = min(left + 2 * size - 1, n - 1);

            // FIXED FUNCTION CALL
            merge(d_arr, left, mid, right, d_temp);

            left += 2 * size;
        }

        __syncthreads();
    }
}

// ==============================
// Host Merge Function (CPU)
// ==============================
void mergeHost(int *arr, int left, int mid, int right) {

    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = (int*)malloc(n1 * sizeof(int));
    int *R = (int*)malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];

    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0;
    int j = 0;
    int k = left;

    while (i < n1 && j < n2) {

        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        }
        else {
            arr[k++] = R[j++];
        }
    }

    while (i < n1) {
        arr[k++] = L[i++];
    }

    while (j < n2) {
        arr[k++] = R[j++];
    }

    free(L);
    free(R);
}

// ==============================
// Bubble Sort CPU
// ==============================
void bubbleSortCPU(int *arr, int n) {

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n - 1; i++) {

        for (int j = 0; j < n - i - 1; j++) {

            if (arr[j] > arr[j + 1]) {

                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;

    printf("Bubble Sort (CPU) took %f milliseconds\n", duration.count());
}

// ==============================
// Merge Sort CPU
// ==============================
void mergeSortCPU(int *arr, int n) {

    auto start = std::chrono::high_resolution_clock::now();

    for (int size = 1; size < n; size *= 2) {

        int left = 0;

        while (left + size < n) {

            int mid = left + size - 1;

            int right = min(left + 2 * size - 1, n - 1);

            mergeHost(arr, left, mid, right);

            left += 2 * size;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;

    printf("Merge Sort (CPU) took %f milliseconds\n", duration.count());
}

// ==============================
// Bubble Sort GPU
// ==============================
void bubbleSortGPU(int *arr, int n) {

    int *d_arr;

    cudaMalloc((void**)&d_arr, n * sizeof(int));

    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Max threads per block usually 1024
    bubbleSortKernel<<<1, n>>>(d_arr, n);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);

    printf("Bubble Sort (GPU) took %f milliseconds\n", milliseconds);
}

// ==============================
// Merge Sort GPU
// ==============================
void mergeSortGPU(int *arr, int n) {

    int *d_arr;
    int *d_temp;

    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMalloc((void**)&d_temp, n * sizeof(int));

    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    mergeSortKernel<<<1, 1>>>(d_arr, d_temp, n);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_temp);

    printf("Merge Sort (GPU) took %f milliseconds\n", milliseconds);
}

// ==============================
// Print Array
// ==============================
void printArray(int *arr, int n) {

    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }

    printf("\n");
}

// ==============================
// Main Function
// ==============================
int main() {

    // Keep <= 1024 for single-block GPU bubble sort
    int n = 512;

    int *arr = (int*)malloc(n * sizeof(int));

    // ======================
    // Bubble Sort CPU
    // ======================
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    printf("\nOriginal Array:\n");
    printArray(arr, 20);

    bubbleSortCPU(arr, n);

    printf("\nBubble Sort CPU Output:\n");
    printArray(arr, 20);

    // ======================
    // Bubble Sort GPU
    // ======================
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    bubbleSortGPU(arr, n);

    printf("\nBubble Sort GPU Output:\n");
    printArray(arr, 20);

    // ======================
    // Merge Sort CPU
    // ======================
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    mergeSortCPU(arr, n);

    printf("\nMerge Sort CPU Output:\n");
    printArray(arr, 20);

    // ======================
    // Merge Sort GPU
    // ======================
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    mergeSortGPU(arr, n);

    printf("\nMerge Sort GPU Output:\n");
    printArray(arr, 20);

    free(arr);

    return 0;
}
```

## OUTPUT:
<img width="989" height="625" alt="image" src="https://github.com/user-attachments/assets/707ebebb-b3c9-4661-970f-1ca14085b57e" />


## RESULT:
Thus, the program has been executed using CUDA to perform parallel sorting operations using Bubble Sort and Merge Sort algorithms on the GPU.

