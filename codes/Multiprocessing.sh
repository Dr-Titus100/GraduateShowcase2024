#!/bin/bash
#Account and Email Information
#SBATCH -A tnde  ## User ID
#SBATCH --mail-type=end
#SBATCH --mail-user=titusnyarkonde@u.boisestate.edu
# Specify parition (queue)
#SBATCH --partition=gpuq
# Join output and errors into output.
#SBATCH -o titus.o%j
#SBATCH -e titus.e%j
# Specify job not to be rerunable.
#SBATCH --no-requeue
# Job Name.
#SBATCH --job-name="mass_calibration"
# Specify walltime.
###SBATCH --time=168:00:00
# Specify number of requested nodes.
#SBATCH -N 4
# Specify the total number of requested procs:
#SBATCH -n 96
# Number of GPUs per node.
#SBATCH --gres=gpu:1
# load all necessary modules.
module load slurm
module load anaconda/anaconda3/5.1.0
# conda activate mass_cal2
# Echo commands to stdout (standard output).
set -x
# Copy your code & data to your R2 Home directory using
# the SFTP (secure file transfer protocol).
# Go to the directory where the actual BATCH file is present.
cd /global/homes/t/titus/Titus/Lensing/codes/GraduateShowcase2024/
# The �python� command runs your python code.
# All output is dumped into titus.o%j with �%j� replaced by the Job ID.
## The file Multiprocessing.py must also 
## be in $/home/tnde/P1_Density_Calibration/Density3D
mpirun -np 8 python3 mcclintockFig9_newdata_sigboost_new_changed_bcov_phys_units.py --redshift 0 --start 0 --end 4 --sys_name _fullrun_mh_no_h1z_in_bmodel_only_rs_old_prior_setup_changed_bov_phys_units >>log.out


