//
// Created by tumbler on 17.03.17.
//

#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include "argparse.h"


void print_usage(char *prog) {
    printf("Usage: %s [options] <samples_file>\n", prog);
    printf("Options:\n");
    printf("--factors <int>\tnumber of factors\n");
    printf("--clusters <int>\tnumber of cluster per shard\n");
    printf("--shard_size <int>\tshard size\n");
    exit(1);
}


gpu_kmsc_args* parse_args(int argc, char **argv) {
    int c;
    int option_index = 0;

    gpu_kmsc_args* args = (gpu_kmsc_args*)malloc(sizeof(gpu_kmsc_args));
    args->samples_file = NULL;
    args->clusters = 1000;
    args->factors = 200;
    args->shard_size = 1000000;
    if (argc < 2) {
        print_usage(argv[0]);
        return NULL;
    }


    while (1) {
        static struct option long_options[] =
                {
                        {"factors",  required_argument, 0, 'f'},
                        {"clusters", required_argument, 0, 'c'},
                        {"shards",   required_argument, 0, 's'},
                        {0, 0,                          0, 0}
                };
        c = getopt_long(argc, argv, "f:c:s:",
                        long_options, &option_index);
        if (c == -1) break;

        switch (c) {
            case 'f':
                args->factors = (unsigned short) atoi(optarg);
                if (args->factors == 0) {
                    print_usage(argv[0]);
                    return NULL;
                }
                break;
            case 'c':
                args->clusters = (unsigned int) atoi(optarg);
                if (args->clusters == 0 || args->clusters > 65536) {
                    print_usage(argv[0]);
                    return NULL;
                }
                break;
            case 's':
                args->shard_size = (size_t)atoi(optarg);
                if (args->shard_size == 0) {
                    print_usage(argv[0]);
                    return NULL;
                }
                break;
            default:
                break;
        }
    }

    if (optind < argc) {
        args->samples_file = argv[optind];
    } else {
        print_usage(argv[0]);
        return NULL;
    }
    return args;
}