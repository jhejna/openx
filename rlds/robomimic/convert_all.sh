#!/bin/bash

ROBOMIMIC_DIR="path/to/robomimic/datasets"
OUTPUT_DIR="path/to/output/dir"

ROBOMIMIC_DATASETS=(
    "lift/ph"
    "lift/mh"
    "can/ph"
    "can/mh"
    "square/ph"
    "square/mh"
    "toolhang/ph"
)

source /path/to/miniconda3/bin/activate
conda activate openx

cd path/to/rlds/robomimic

mkdir -p "$OUTPUT_DIR"

# Loop over all files in the input folder
for DATASET in "${ROBOMIMIC_DATASETS[@]}"; do

    echo "Starting ${DATASET}"

    tfds build --manual_dir ${ROBOMIMIC_DIR}/${DATASET} --data_dir ${OUTPUT_DIR}/${DATASET}

    # Now move the files to the correct location
    mv ${OUTPUT_DIR}/${DATASET}/robo_mimic/1.0.0 ${OUTPUT_DIR}/${DATASET}
    rm -r ${OUTPUT_DIR}/${DATASET}/robo_mimic
    rm -r ${OUTPUT_DIR}/${DATASET}/downloads

done

echo "Processing complete."
