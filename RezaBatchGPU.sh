#!/bin/sh
#SBATCH -A khangroup_gpu
#SBATCH -p gpu
#SBATCH --job-name=Reza_EffDet_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 3-00:00:00 
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:p100:3
#SBATCH -o %x-%j.out


ipython /cluster/home/t62003uhn/EfficinetNetDFU/Train_Eval_example_Jupyther.py
