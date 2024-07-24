#!/usr/bin/env python
# coding: utf-8

# # In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# # In[2]:


from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import itertools
from torch import optim

from mamba_ssm import Mamba
# from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer


import torch
import numpy as np
import torch
import torch.nn as nn
from torch import tensor
import scipy as sp

# import torchvision
# import torchvision.transforms as transforms

import torch
from torch.utils.data import DataLoader, TensorDataset
from mamba_ssm.modules.block import Block
from functools import partial


from mamba_model import MambaEEG
from mamba_ssm.models.config_mamba import MambaConfig
from models import *


# In[3]:


from tqdm import tqdm 
import os
import h5py


# In[4]:


def load_matlab_string(matlab_extracted_object):
    """
    Converts a string loaded from h5py into a python string
    :param matlab_extracted_object:     (h5py)  matlab string object
    :return:
        extracted_string    (str)   translated string
    """

    # print((chr(c) for c in matlab_extracted_object))
    extracted_string = u''.join(chr(c) for c in matlab_extracted_object[:].flatten())
    # print(extracted_string)
    return extracted_string


# In[5]:


task = "NR"

rootdir = "/radraid/spanchavati/eegtotext/zuco-benchmark/data/"

print('##############################')
print(f'start processing ZuCo task2-NR-2.0...')

dataset_dict = {}

for file in tqdm(os.listdir(rootdir)[::-1]):
    if file.endswith(task+".mat"):
        print(file)

        file_name = rootdir + file

        # print('file name:', file_name)
        subject = file_name.split("ts")[1].split("_")[0]
        # print('subject: ', subject)

        # exclude YMH due to incomplete data because of dyslexia
        if subject != 'YMH':
            pass

        f = h5py.File(file_name,'r')
        print('keys in f:', list(f.keys()))
        try:
            sentence_data = f['sentenceData']
        except:
            continue

        contents = []
        rawEEG = []
        for i in range(sentence_data['rawData'].len()):
            content = load_matlab_string(f[sentence_data['content'][i][0]])
            raweeg = f[sentence_data['rawData'][i][0]]

            contents.append(content)
            rawEEG.append(np.array(raweeg))

        dataset_dict[subject] = {'content': contents, 'eeg': rawEEG}
        #     break
        # break
        #     # contents.append(sentence_data['content'])
            





from random import shuffle


# In[10]:


keys = list(dataset_dict.keys())
shuffle(keys)
train_keys = keys[:12]
test_keys = keys[12:]

train_ds = EEGTextDataset(dataset_dict, train_keys)
val_ds = EEGTextDataset(dataset_dict, test_keys)


# Create dataloaders
train_dataloader = DataLoader(train_ds, batch_size=8, shuffle=True,num_workers= 0)
val_dataloader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)


model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

encoder = HuggingFaceEncoder(model_name)


mm = MambaConfig(ssm_cfg = {'layer':'Mamba1'}, d_model = 64)

ee = EEGEncoder(n_channels = 105, max_length= 15*500, mamba_config=mm, embedding = 'mean')



model = EEGTextCLIP(
    eeg_encoder=ee,
    text_encoder=encoder,
    text_embedding_dims=768,
    projection_dims=256,
    dropout=0.0,
    temperature=1.0,
    weight_decay=0.0,
    head_lr=1e-3,
    image_encoder_lr=1e-4,
    text_encoder_lr=1e-4,
    lr_scheduler_patience=1.0,
    lr_scheduler_factor=0.8
)


# In[ ]:


# Define callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val/loss',
    dirpath='checkpoints/',
    filename='eeg-text-clip-{epoch:02d}-{val_loss:.2f}',
)

trainer = Trainer(
    max_epochs=10,
    accelerator='gpu',
    devices=[0],
    logger=True,
    # num_sanity_val_steps=10,
    log_every_n_steps=1,
    # flush_logs_every_n_steps=1,
    # fast_dev_run=100,
    callbacks=[checkpoint_callback],
    # log_every_n_steps=50  # Added logging for debugging
)


# Train the model
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


# <!-- # for batch in train_dataloader:
# #     break -->

# In[17]:


# batch = {b:batch[b].to('cuda:1') for b in batch}

# model.to('cuda:1')(batch)


# In[ ]:




