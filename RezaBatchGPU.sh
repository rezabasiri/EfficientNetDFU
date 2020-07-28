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

# git clone https://github.com/roboflow-ai/Monk_Object_Detection.git

# cd Monk_Object_Detection/3_mxrcnn/installation && cat requirements_colab.txt | xargs -n 1 -L 1 pip install

# cd ~

# pip install tqdm
# pip install efficientnet_pytorch
# pip install tensorboardX

python /cluster/home/t62003uhn/DFUTry2/Step2/Reza_EffDet_GPU.py
