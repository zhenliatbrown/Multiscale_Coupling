#!/bin/bash

# #SBATCH --job-name lammps_coupling
# #SBATCH --nodes 1
# #SBATCH --ntasks-per-node 16
# #SBATCH --cpus-per-task 2
# #SBATCH --mem 16gb
# #SBATCH --time=2:00:00
# #SBATCH --gpus-per-node v100:2

mpirun -np 1 ./lmp_meso -in in.dpd_gpu : -np 1 ./lmp_mpi -in in.dpd > log.out
