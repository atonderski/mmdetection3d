#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 02:00:00
#SBATCH --output /proj/nlp4adas/users/%u/logs/%j.out
#SBATCH -A berzelius-2022-232
#

singularity exec --nv --bind /proj/nlp4adas/users/$USER/mmdetection3d:/mmdetection3d \
  --bind /proj/adas-data/data/zod:/mmdetection3d/data/zod \
  --bind /home/$USER/.local/lib/python3.8/site-packages-mmdet:/home/$USER/.local/lib/python3.8/site-packages \
  --pwd /mmdetection3d/ \
  /proj/nlp4adas/containers/mmdet3d.sif \
  python tools/test.py $@
#
#EOF
