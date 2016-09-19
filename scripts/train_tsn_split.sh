#!/usr/bin/env bash

DATASET=$1
MODALITY=$2
SPLIT=$3

TOOLS=lib/caffe-action/build/install/bin
LOG_FILE=logs/${DATASET}_${MODALITY}_split${SPLIT}.log
N_GPU=3
MPI_BIN_DIR= #/usr/local/openmpi/bin/


echo "logging to ${LOG_FILE}"
echo "Start training on ${DATASET} split ${SPLIT}"

${MPI_BIN_DIR}mpirun -np $N_GPU \
$TOOLS/caffe train --solver=models/${DATASET}/specific_models/tsn_bn_inception_${MODALITY}_solver_split${SPLIT}.prototxt  \
   --weights=models/bn_inception_${MODALITY}_init.caffemodel 2>&1 | tee ${LOG_FILE}
