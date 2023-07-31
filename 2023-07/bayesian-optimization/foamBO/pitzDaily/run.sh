#!/usr/bin/bash -l
#SBATCH -N 1
#SBATCH -n 148
#SBATCH --output=log.%x
#SBATCH --partition=queue-1
#SBATCH --constraint=c5a.24xlarge
#
cd "${1}" || exit                                # Run from this directory
nproc
hostname
