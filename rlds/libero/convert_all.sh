#!/bin/bash

LIBERO_DIR="path/to/libero/datasets"
OUTPUT_DIR="path/to/libero_rlds"

LIBERO_DATASETS=(
    "libero_10"
    "libero_90"
    "libero_goal"
    "libero_object"
    "libero_spatial"
)

source /path/to/miniconda3/bin/activate
conda activate openx

mkdir -p "$OUTPUT_DIR"

# Loop over all files in the input folder
for DATASET in "${LIBERO_DATASETS[@]}"; do

    echo "Starting ${DATASET}"

    tfds build --manual_dir ${LIBERO_DIR}/${DATASET} --data_dir ${OUTPUT_DIR}/${DATASET}

    # Now move the files to the correct location
    mv ${OUTPUT_DIR}/${DATASET}/libero/1.0.0 ${OUTPUT_DIR}/${DATASET}
    rm -r ${OUTPUT_DIR}/${DATASET}/libero
    rm -r ${OUTPUT_DIR}/${DATASET}/downloads

done

echo "Processing complete."
