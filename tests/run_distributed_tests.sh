#!/bin/sh

#######################
### INPUT ARGUMENTS ###
#######################
if [ -z "$1" ] || [ -z "$2" ]  # if one of the two input arguments does not exist
  then
    echo "Need to specify 2 GPU devices as arguments, e.g. bash run_distributed_tests.sh 0 1"
    exit
fi
if [[ $1 =~ [^0-7] ]] || [[ $2 =~ [^0-7] ]]  # if one of the two input arguments is not an integer 0-7
    then
        echo "Need to specify integers 0-7 as arguments, e.g. bash run_distributed_tests.sh 0 1"
        exit 
fi

#################
### VARIABLES ###
#################
DEV0=$1 
DEV1=$2
COVERAGE=$3 # --cov or  --no-cov

#############
### TESTS ###
#############

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR/.."

# test_fsdp_to_disc_checkpointing
COVERAGE_FILE=.coverage/.coverage.part1 CUDA_VISIBLE_DEVICES=$DEV0,$DEV1 coverage run --rcfile=.coveragerc --parallel $(which torchrun) --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) tests/checkpointing/test_fsdp_to_disc_checkpointing.py $COVERAGE

# # test_fsdp_warmstart
COVERAGE_FILE=.coverage/.coverage.part2 CUDA_VISIBLE_DEVICES=$DEV0,$DEV1 coverage run --rcfile=.coveragerc --parallel $(which torchrun) --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) tests/end2end_tests/test_fsdp_warmstart.py -k "test_warm_start" $COVERAGE
COVERAGE_FILE=.coverage/.coverage.part3 CUDA_VISIBLE_DEVICES=$DEV0,$DEV1 coverage run --rcfile=.coveragerc --parallel $(which torchrun) --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) tests/end2end_tests/test_fsdp_warmstart.py -k "test_warmstart_dataloader" $COVERAGE

# # test_distributed_repeating_dataloader
COVERAGE_FILE=.coverage/.coverage.part4 CUDA_VISIBLE_DEVICES=$DEV0,$DEV1 coverage run --rcfile=.coveragerc --parallel $(which torchrun) --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) tests/dataloader/distributed/test_distributed_repeating_dataloader.py -k "test_resumable_dataloader_without_shuffling" $COVERAGE

# # test_distributed_dataloader
COVERAGE_FILE=.coverage/.coverage.part5 CUDA_VISIBLE_DEVICES=$DEV0,$DEV1 coverage run --rcfile=.coveragerc --parallel $(which torchrun) --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) tests/dataloader/distributed/test_distributed_dataloader.py -k "test_resumable_dataloader_without_shuffling" $COVERAGE
COVERAGE_FILE=.coverage/.coverage.part6 CUDA_VISIBLE_DEVICES=$DEV0,$DEV1 coverage run --rcfile=.coveragerc --parallel $(which torchrun) --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) tests/dataloader/distributed/test_distributed_dataloader.py -k "test_resumable_dataloader_with_shuffling_without_skipping" $COVERAGE
COVERAGE_FILE=.coverage/.coverage.part7 CUDA_VISIBLE_DEVICES=$DEV0,$DEV1 coverage run --rcfile=.coveragerc --parallel $(which torchrun) --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 2 $(which pytest) tests/dataloader/distributed/test_distributed_dataloader.py -k "test_resumable_dataloader_with_shuffling_and_skipped_batches" $COVERAGE

# # test optimizer
COVERAGE_FILE=.coverage/.coverage.part8 CUDA_VISIBLE_DEVICES=$DEV0 coverage run --rcfile=.coveragerc --parallel $(which torchrun) --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 1 $(which pytest) tests/test_optimizer_factory.py $COVERAGE

# # test model initialization
COVERAGE_FILE=.coverage/.coverage.part9 CUDA_VISIBLE_DEVICES=$DEV0 coverage run --rcfile=.coveragerc --parallel $(which torchrun) --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 1 $(which pytest) tests/test_initialization.py $COVERAGE

# # test mfu
COVERAGE_FILE=.coverage/.coverage.part10 CUDA_VISIBLE_DEVICES=$DEV0 coverage run --rcfile=.coveragerc --parallel $(which torchrun) --rdzv-endpoint localhost:29502 --nnodes 1 --nproc_per_node 1 $(which pytest) tests/utils/test_mfu.py $COVERAGE