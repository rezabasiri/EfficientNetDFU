#!/usr/bin/env python
# coding: utf-8

# # Download and Set-Up Paths

# In[2]:


from __future__ import print_function
from IPython import display
from IPython import get_ipython
import tensorflow.compat.v1 as tf
import sys
import os

# Download source code.
if "efficientnet_model.py" not in os.getcwd() or "imagenet.py" not in os.getcwd():
  get_ipython().system('git clone --depth 1 https://github.com/rezabasiri/EfficientNetDFU.git')
  display.clear_output()
  os.chdir('EfficientNetDFU')
  sys.path.append('.')
  os.chdir('common')
  sys.path.append('.')
  os.chdir('..')
else:
  get_ipython().system('git pull')


# In[3]:


MODEL = 'efficientnet-b0'

def download(m):
  if m not in os.listdir():
    get_ipython().system('wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/{m}.tar.gz')
    get_ipython().system('tar xf {m}.tar.gz')
    display.clear_output()
    get_ipython().system('rm -rf {m}.tar.gz')
  ckpt_path = os.path.join(os.getcwd(), m)
  return ckpt_path

# Download checkpoint.
ckpt_path = download(MODEL)
print('Use model checkpoint in {}'.format(ckpt_path))


# 
# # Set-Up Data
# 

# In[ ]:


# Prepare Training Data
get_ipython().system('rm -rf *.zip *.tar tfrecord/ val2017/')
get_ipython().system('mkdir tfrecord')
get_ipython().system('PYTHONPATH=".:$PYTHONPATH"  python dataset/create_coco_tfrecord.py       --image_dir="/cluster/home/t62003uhn/Dataset/Chess/Images"       --object_annotations_file="/cluster/home/t62003uhn/Dataset/Chess/annotations/instances_Images.json"       --output_file_prefix=tfrecord/train       --num_shards=1')


# In[ ]:


# Prepare Validation Data
get_ipython().system('PYTHONPATH=".:$PYTHONPATH"  python dataset/create_coco_tfrecord.py       --image_dir="/cluster/home/t62003uhn/Dataset/ChessVal/ImagesVal"       --object_annotations_file="/cluster/home/t62003uhn/Dataset/ChessVal/annotations/instances_ImagesVal.json"       --output_file_prefix=tfrecord/validation       --num_shards=1')


# In[ ]:


file_pattern = 'train-*-of-00001.tfrecord' # Update to match the number of shards for training set
file_patternVal = 'validation-*-of-00001.tfrecord' # Update to match the number of shards for valid set

images_per_epoch = 57 * len(tf.io.gfile.glob('tfrecord/' + file_pattern))
images_per_epoch = images_per_epoch // 8 * 8  # round to 64.

images_per_epochVal = 57 * len(tf.io.gfile.glob('tfrecord/' + file_patternVal))
images_per_epochVal = images_per_epochVal // 8 * 8  # round to 64.

print('images_per_epoch = {}'.format(images_per_epoch))
print('images_per_epochVal = {}'.format(images_per_epochVal))


# # Create the Model and Run Train-Eval

# In[ ]:

# Hyperparameters
train_img_size = 1601
eval_img_size = 399
train_batch_size = 20
eval_batch_size = 20
Epoch_Size = 50

# (Image_Size/Batch_Size)*Epoch = Train_Steps
train_steps = (train_img_size / float(train_batch_size))*float(Epoch_Size)
steps_per_eval = int(train_steps*0.01)

get_ipython().system('mkdir model_dir/')
get_ipython().system('python main.py --mode=train_and_eval     --data_dir=tfrecord     --model_name={MODEL}     --model_dir=model_dir/{MODEL}-finetune     --train_batch_size={train_batch_size}     --eval_batch_size={eval_batch_size}     --num_eval_images={eval_img_size}     --num_train_images={train_img_size}     --num_label_classes=2     --use_tpu=false     --train_steps={train_steps}     --steps_per_eval={steps_per_eval}     --moving_average_decay=0     --base_learning_rate=0.01     --data_format=channels_first ')


# In[ ]:


#{images_per_epochVal}  
#{images_per_epoch} 
# !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/eval_data/labels_map.txt -O labels_map.txt


# In[ ]:


# import eval_ckpt_main as eval_ckpt
# import tensorflow.compat.v1 as tf

# !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/{model_name}.tar.gz -O {model_name}.tar.gz
# !tar xf {model_name}.tar.gz
# ckpt_dir = model_name
# !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/eval_data/labels_map.txt -O labels_map.txt
# labels_map_file = 'labels_map.txt'


# image_files = [image_file]
# eval_driver = eval_ckpt.get_eval_driver(model_name)
# pred_idx, pred_prob = eval_driver.eval_example_images(
#     ckpt_dir, image_files, labels_map_file)


# In[ ]:


# %load_ext tensorboard
# %tensorboard --logdir model_dir/

