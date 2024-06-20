: '
Script for downloading, cleaning and resizing Open X-Embodiment Dataset (https://robotics-transformer-x.github.io/)

Adapted from https://github.com/kpertsch/rlds_dataset_mod

Performs the preprocessing steps:
  1. Downloads a mixture of Open X-Embodiment datasets
  2. Runs resize function to resize all datasets to 256x256 (if image resolution is larger) and jpeg encoding
  3. Fixes channel flip errors in a few datsets, filters success-only for QT-Opt ("kuka") data

To reduce disk memory usage during conversion, we download the datasets 1-by-1, convert them
and then delete the original.
We specify the number of parallel workers below -- the more parallel workers, the faster data conversion will run.
Adjust workers to fit the available memory of your machine, the more workers + episodes in memory, the faster.
The default values are tested with a server with ~120GB of RAM and 24 cores.
'

DOWNLOAD_DIR=<download_dir>
DATA_DIR=<data_dir>
NUM_WORKERS=20                  # number of workers used for parallel conversion --> adjust based on available RAM
CHUNK_SIZE=500    # number of episodes converted in parallel --> adjust based on available RAM

# increase limit on number of files opened in parallel to 20k --> conversion opens up to 1k temporary files
# in /tmp to store dataset during conversion
ulimit -n 20000

# format: [dataset_name, dataset_version, extra_mods]
DATASET_TRANSFORMS=(
    "fractal20220817_data 0.1.0 ''"
    "kuka 0.1.0 filter_success"
    "taco_play 0.1.0 ''"
    "jaco_play 0.1.0 ''"
    "berkeley_cable_routing 0.1.0 ''"
    "roboturk 0.1.0 ''"
    "nyu_door_opening_surprising_effectiveness 0.1.0 ''"
    "viola 0.1.0 ''"
    "berkeley_autolab_ur5 0.1.0 flip_wrist_image_channels"
    "toto 0.1.0 ''"
    "language_table 0.1.0 ''"
    "nyu_rot_dataset_converted_externally_to_rlds 0.1.0 ''"
    "stanford_hydra_dataset_converted_externally_to_rlds 0.1.0 flip_wrist_image_channels,flip_image_channels"
    "austin_buds_dataset_converted_externally_to_rlds 0.1.0 ''"
    "nyu_franka_play_dataset_converted_externally_to_rlds 0.1.0 ''"
    "furniture_bench_dataset_converted_externally_to_rlds 0.1.0 ''"
    "ucsd_kitchen_dataset_converted_externally_to_rlds 0.1.0 ''"
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds 0.1.0 ''"
    "austin_sailor_dataset_converted_externally_to_rlds 0.1.0 ''"
    "austin_sirius_dataset_converted_externally_to_rlds 0.1.0 ''"
    "bc_z 0.1.0 ''"
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds 0.1.0 ''"
    "dlr_sara_pour_converted_externally_to_rlds 0.1.0 ''"
    "dlr_edan_shared_control_converted_externally_to_rlds 0.1.0 ''"
    "asu_table_top_converted_externally_to_rlds 0.1.0 ''"
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds 0.1.0 ''"
    "utaustin_mutex 0.1.0 flip_wrist_image_channels,flip_image_channels"
    "berkeley_fanuc_manipulation 0.1.0 flip_wrist_image_channels,flip_image_channels"
    "cmu_stretch 0.1.0 ''"
    "cmu_play_fusion 0.1.0 ''"
    "droid 1.0.0 ''"
)

for tuple in "${DATASET_TRANSFORMS[@]}"; do
  # Extract strings from the tuple
  strings=($tuple)
  DATASET=${strings[0]}
  VERSION=${strings[1]}
  MODS=${strings[2]} # TODO: figure out if we can find length of tuple in bash
  # TODO: Note that this job will not work on GCP buckets directly because of TFDS.
  mkdir ${DOWNLOAD_DIR}/${DATASET}
  gsutil -m cp -r gs://gresearch/robotics/${DATASET}/${VERSION} ${DOWNLOAD_DIR}/${DATASET}/${VERSION}
  python3 resize_openx_dataset.py --dataset_path=${DOWNLOAD_DIR}/${DATASET} --data_dir=$DATA_DIR --mods=$MODS --num_workers=$NUM_WORKERS --chunk_size=$CHUNK_SIZE
done
