//
// Created by tumbler on 17.03.17.
//
#include "utils.h"

bool init_gpu() {
    /* Инициализация GPU и CUDA. Вроде кроме SetDevice ничего не надо */
    cudaSetDevice(0);
    return true;
}

float* upload_to_gpu(const float *host_pointer, int size) {
    float * gpu_pointer;
    cudacall(cudaMalloc((void** ) &gpu_pointer, size * sizeof(gpu_pointer[0])));
    cudacall(cudaMemcpy(
            gpu_pointer, host_pointer, (size_t) (size * sizeof(gpu_pointer[0])),
            cudaMemcpyHostToDevice));
    return gpu_pointer;
}

float* download_from_gpu(const float *gpu_pointer, int size) {
    float* pointer;
    cudacall(cudaMallocHost((void** ) &pointer, size * sizeof(pointer[0])));
    cudacall(cudaMemcpy(
            pointer, gpu_pointer, (size_t) (size * sizeof(pointer[0])),
            cudaMemcpyDeviceToHost));
    return pointer;
}

float* load_matrix(const char *matrix_file, const size_t matrix_offset, const int width, const int height) {
    float* matrixHostPtr;
    FILE *f = fopen(matrix_file,"rb");
    if (!f) {
        return NULL;
    }
    fseek(f, matrix_offset, 0);
    cudacall(cudaMallocHost((void**) &matrixHostPtr,
                            width * height * sizeof(matrixHostPtr[0])));
    fread(&matrixHostPtr[0], sizeof(float), (size_t) (width * height), f);
    fclose(f);
    return matrixHostPtr;
}

void save_matrix(const char *matrix_file, const float *host_ptr,
                 const int width, const int height, const bool append) {
    FILE * f = fopen(matrix_file, append?"ab":"wb");
    fwrite(host_ptr, sizeof(float), (size_t) (width * height), f);
    fclose(f);
}

void cleanup_gpu(float* host_pointers[], const int host_ptr_count,
                 float* gpu_pointers[], const int gpu_ptr_count,
                 const bool reset_device) {
    for(int i = 0; i < host_ptr_count; i++) {
        cudaFreeHost(host_pointers[i]);
    }
    for(int i = 0; i < gpu_ptr_count; i++) {
        cudaFree(gpu_pointers[i]);
    }
    if (reset_device)
        cudacall(cudaDeviceReset());
}
