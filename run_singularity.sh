#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --time 168:00:00
#SBATCH --mem-per-gpu 64G
#SBATCH --output /workspaces/%u/mmdetection3d/logs/%j.out
#SBATCH --partition zprodlow
#

singularity exec --nv --bind /workspaces/$USER:/workspace \
  --bind /staging/dataset_donation:/staging/dataset_donation \
  --pwd /workspace/mmdetection3d/ \
  --env PYTHONPATH=/workspace/mmdetection3d/:/workspace/mmdetection3d/agp \
  /workspaces/s0000960/mmdetection3d/mmdet3d.sif \
  python3 -u $@
#
#EOF
