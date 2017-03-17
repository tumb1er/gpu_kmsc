//
// Created by tumbler on 17.03.17.
//

#ifndef GPU_KMSC_ARGPARSE_H
#define GPU_KMSC_ARGPARSE_H

typedef struct _gpu_kmsc_args {
    char* samples_file;
    size_t shard_size;
    unsigned short factors;
    unsigned int clusters;
} gpu_kmsc_args;

gpu_kmsc_args* parse_args(int argc, char **argv);
#endif //GPU_KMSC_ARGPARSE_H
