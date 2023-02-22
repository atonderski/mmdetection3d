#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --time 168:00:00
#SBATCH --mem-per-gpu 64G
#SBATCH --output /workspaces/%u/mmdetection3d/logs/%j.out
#SBATCH --partition zprod
#

singularity exec --nv --bind /workspaces/$USER/mmdetection3d:/mmdetection3d \
  --bind /staging/dataset_donation/round_2:/mmdetection3d/data/zod \
  --pwd /mmdetection3d/ \
  --env PYTHONPATH=/mmdetection3d/ \
  /workspaces/s0000960/mmdetection3d/mmdet3d.sif \
  python3 -u $@
#
#EOF
