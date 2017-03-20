#include <iostream>
#include <sys/stat.h>

#include "kmcuda.h"
#include "utils/argparse.h"
#include "utils/utils.h"

static const float TOLERANCE = 0.01f;
static const float YINYANG_T = 0.1f;
using namespace std;

off_t fileSize(const char *filename) {
    struct stat st;

    if (stat(filename, &st) == 0)
        return st.st_size;

    return -1;
}

int main(int argc, char **argv) {
    gpu_kmsc_args* args = parse_args(argc, argv);
    cout << "factors: " << args->factors << endl;
    cout << "shard_size: " << args->shard_size << endl;
    cout << "clusters: " << args->clusters << endl;
    cout << "S:" << args->samples_file << endl;
    off_t file_size = fileSize(args->samples_file);
    if(file_size < 0) {
        cout << "samples file not found" << endl;
        return -1;
    }
    size_t size = (size_t) file_size;
    if (size % sizeof(float)) {
        cout << "samples file size not matches float32" << endl;
        return -2;
    }

    size_t elements = size / sizeof(float);

    if (elements % args->factors) {
        cout << "samples file not matches factors" << endl;
        return -3;
    }
    float *gpu_centroids;
    cudacall(cudaMalloc((void**) &gpu_centroids, args->clusters * args->factors * sizeof(float)));
    size_t file_offset = args->shard_size * args->factors * sizeof(float);
    unsigned int assignment_offset = 0;

    for (size_t offset=0; offset < size; offset += file_offset) {
        // Размер текущего фрагмента матрицы в байтах
        size_t chunk_size = min(file_offset, size - offset);

        unsigned int chunk_samples = (unsigned int) (chunk_size / (args->factors * sizeof(float)));
        float *samples = load_matrix(args->samples_file, offset, args->factors, chunk_samples);
        float *gpu_samples = upload_to_gpu(samples, args->factors * chunk_samples);
        unsigned int *gpu_assignments;
        cudacall(cudaMalloc((void **) &gpu_assignments, chunk_samples * sizeof(unsigned int)));

        cout << "Starting KMCUDA..." << endl;
        KMCUDAResult result = kmeans_cuda(
                KMCUDAInitMethod::kmcudaInitMethodPlusPlus, NULL,  // init and it's args
                TOLERANCE, YINYANG_T, // hyperparams
                KMCUDADistanceMetric::kmcudaDistanceMetricL2,
                chunk_samples,
                args->factors,
                args->clusters,
                123,  // random seed
                1,  // device bit mask (0x1 means gpu #0)
                0, // device pointers mode (-1 - all data are host pointers, 0 - device pointers at gpu #0)
                0,  // fp16 mode
                1,  // verbosity
                gpu_samples, gpu_centroids, gpu_assignments, NULL  // data pointers
        );
        if (result != KMCUDAResult::kmcudaSuccess) {
            cout << "KMCUDAResult: " << result << endl;
            return -4;
        }
        cout << "Saving centroids..." << endl;
        float * centroids = download_from_gpu(gpu_centroids, args->factors * args->clusters);
        save_matrix("gpu_centroids.bin", centroids, args->factors, args->clusters, offset!=0);
        cout << "Saving assignments..." << endl;
        unsigned int* assignments = download_from_gpu(gpu_assignments, chunk_samples);
        // unique cluster indices for each shard
        if (assignment_offset) {
            for (int i = 0; i < chunk_samples; i++) {
                assignments[i] += assignment_offset;
            }
        }
        save_matrix("assignments.bin", assignments, 1, chunk_samples, offset!=0);

        cout << "Cleaning up..." << endl;
        cudaFreeHost(samples);
        cudaFree(gpu_samples);
        cudaFreeHost(assignments);
        cudaFree(gpu_assignments);
        cudaFreeHost(centroids);
        assignment_offset += args->clusters;
    }
    cudaFree(gpu_centroids);
    cudacall(cudaDeviceReset());
    return 0;
}