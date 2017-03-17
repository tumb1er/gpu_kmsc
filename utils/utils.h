//
// Created by tumbler on 17.03.17.
//

#ifndef GPU_KMSC_UTILS_H
#define GPU_KMSC_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/// Инициализирует библиотеки CUDA
bool init_gpu();

/// Освобождает память по списку указателей на хосте и GPU через cudaFree.
void cleanup_gpu(float* host_pointers[], const int host_ptr_count,
                 float* gpu_pointers[], const int gpu_ptr_count,
                 const bool reset_device);

/// загрузка float-матрицы из файла с учетом offset
/// \param matrix_file путь до файла с данными для матрицы - float32 по строкам
/// \param matrix_offset смещение в matrix_file в байтах
/// \param width число столбцов матрицы
/// \param height число строк матрицы
/// \return указатель на хост-память, аллоцированный через cudaMallocGHost
float* load_matrix(const char *matrix_file, const size_t matrix_offset, const int width, const int height);

/// сохранение float-матрицы в файл
/// \param matrix_file путь до выходного файла
/// \param host_ptr указатель на хост-память матрицы (считается, что матрица расположена по строкам)
/// \param width число столбцов матрицы
/// \param height число строк матрицы
/// \param append дописывать в конец
void save_matrix(const char *matrix_file, const float *host_ptr,
                 const int width, const int height, const bool append);

/// обертка для вывода ошибок от библиотеки CUDA
#define cudacall(call) \
    do\
    {\
	cudaError_t err = (call);\
	if(cudaSuccess != err)\
	    {\
		fprintf(stderr, "CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err));\
		cudaDeviceReset();\
		exit(EXIT_FAILURE);\
	    }\
    }\
    while (0)\

/* аллоцирует память на GPU и загружает туда данные с хоста */
float * upload_to_gpu(const float *host_pointer, int size);

/* аллоцирует память на хосте и выгружает данные с GPU */
float * download_from_gpu(const float *gpu_pointer, int size);

#endif //GPU_KMSC_UTILS_H
