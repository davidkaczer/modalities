#!/bin/bash

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 1 --nproc_per_node 1 --rdzv-endpoint=0.0.0.0:29502 src/modalities/__main__.py run --config_file_path config_files/config_example_coca_webdataset.yaml