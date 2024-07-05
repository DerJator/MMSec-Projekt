#!/bin/bash -l
#SBATCH --job-name=MMF-Triplet
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --mail-type=end,fail
#SBATCH --time=00:15:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
module purge
module load python
module load cuda
# module load cudnn
# Conda
conda activate MMF-Proj # replace with the name of your conda env
# Copy data to `$TMPDIR` to have faster access, recommended esp. for long trainings
cp -r "$WORK/MMF-Proj/images/CASIA2/CASIA2.0_revised" -d "$TMPDIR"
# in case you have to extract an archive, e.g. a dataset use:
# unzip "$WORK/your-dataset.zip" "$TMPDIR"
cd ${TMPDIR}
# create a temporary job dir on $WORK
mkdir ${WORK}/$SLURM_JOB_ID
# copy input file from location where job was submitted, and run
cp -r ${SLURM_SUBMIT_DIR}/. .
mkdir -p output/logs/
mkdir -p output/checkpoints/
# Run training script (with data copied to node)
srun python model_training.py --dataset-path "$TMPDIR" --workdir "$TMPDIR" # add training parameters
# Create a directory on $HOME and copy the results from our training
mkdir ${WORK}/$SLURM_JOB_ID
cp -r ./output/. ${WORK}/$SLURM_JOB_ID