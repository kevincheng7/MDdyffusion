#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=GPUA800
#SBATCH -J Navier-stokes-interpolation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node or Ngpus per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

nvidia-smi
source /gpfs/share/software/anaconda/3-2023.09-0/etc/profile.d/conda.sh
conda activate /gpfs/share/home/2201111701/miniconda3/envs/dyffusion

# debugging flags (optional)
#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
#export NCCL_SOCKET_IFNAME=^docker0,lo

export WANDB_API_KEY="b9eb5b770cecabc1a1eff9528925e68c0d6797f2"
export WANDB_MODE="offline"

cd ..
srun python run.py \
    experiment=navier_stokes_interpolation \
    trainer.max_epochs=5 \
    datamodule.batch_size=8 \
    datamodule.data_dir="/gpfs/share/home/2201111701/Kaiwencheng/dyffusion/physical-nn-benchmark" \
    logger.wandb.offline=True \
    logger.wandb.mode=offline \
    save_config_to_wandb=False