import torch
from pytorch_lightning import LightningModule, LightningDataModule

import numpy as np
import torch
import torch.nn as nn
from torch import tensor
import scipy as sp
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import itertools
from torch import optim
from torch.utils.data import Dataset



# import torchvision
# import torchvision.transforms as transforms

# !pip install mamba-ssm
import torch
from mamba_ssm import Mamba
from torch.utils.data import DataLoader, TensorDataset
from mamba_ssm.modules.block import Block
from functools import partial
from mamba_model import MambaEEG


class EEGTextDataset(Dataset):
    def __init__(self, data_dict, keys, tokenizer_name='bert-base-uncased', maxlen=15*500):
        """
        Args:
            data_dict (dict): Dictionary containing patients' data.
            keys (list): List of keys to be used (e.g., for training/testing).
            tokenizer_name (str): The name of the tokenizer to use.
            maxlen (int): Maximum length for EEG sequences.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.maxlen = maxlen
        self.data = []

        # Load and preprocess data
        self.load_data(data_dict, keys)

    def process_eeg(self, eeg_data, mean, std):
        """
        Normalize EEG by computing total channel mean and std.
        Right pad EEG with 0s to self.maxlen, throw error if eeg_data is longer than maxlen.
        """
        if eeg_data.shape[0] < 100:
            return None, None

        normalized_eeg = (eeg_data - mean) / std
        
        # Check if EEG data length exceeds maxlen
        if normalized_eeg.shape[0] > self.maxlen:
            print(f"EEG data length {normalized_eeg.shape[0]} exceeds maxlen {self.maxlen}")
            return None, None
        
        # Create attention mask
        attention_mask = np.zeros((self.maxlen,))
        attention_mask[:normalized_eeg.shape[0]] = 1
        
        # Right pad EEG data with zeros
        padded_eeg = np.zeros((self.maxlen, normalized_eeg.shape[1]))
        padded_eeg[:normalized_eeg.shape[0], :] = normalized_eeg
        
        return padded_eeg, attention_mask

    def incremental_mean_std(self, data_list):
        """
        Calculate mean and standard deviation incrementally for a list of EEG data arrays.
        """
        n_total = 0
        mean = 0
        M2 = 0
        for data in data_list:
            n = data.shape[0]
            if n < 100:
                continue
            n_total += n
            delta = data - mean
            mean += np.nansum(delta, axis=0) / n_total
            delta2 = data - mean
            M2 += np.nansum(delta * delta2, axis=0)

        variance = M2 / (n_total - 1)
        std = np.sqrt(variance)
        return mean, std

    def load_data(self, data_dict, keys):
        for key in keys:
            patient_data = data_dict[key]
            sentences = np.array(patient_data['content'])
            eeg_data = patient_data['eeg']
            
            # Compute mean and std incrementally for all EEG data of the patient
            mean, std = self.incremental_mean_std(eeg_data)

            # Normalize and pad EEG data using patient's mean and std
            for i, (sentence, eeg) in enumerate(zip(sentences, eeg_data)):
                eeg_processed, attention_mask = self.process_eeg(eeg, mean, std)
                if eeg_processed is not None:
                    self.data.append({
                        'sentence': sentence,
                        'eeg': eeg_processed,
                        'eeg_attention_mask': attention_mask
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize the sentence on-the-fly
        tokenized = self.tokenizer(item['sentence'], return_tensors='pt', padding='max_length', truncation=True)
        
        return {
            'input_ids': tokenized['input_ids'][0],  # Remove the batch dimension
            'attention_mask': tokenized['attention_mask'][0],  # Remove the batch dimension
            'eeg': torch.nan_to_num(torch.tensor(item['eeg'])).float(),
            'eeg_attention_mask': torch.tensor(item['eeg_attention_mask'])
        }







class EEGTextCLIP(LightningModule):
    def __init__(
        self,
        # eeg_encoder_alias: str,
        # text_encoder_alias: str,
        # image_encoder_pretrained: bool = True,
        # image_encoder_trainable: bool = True,
        # text_encoder_trainable: bool = True,
        eeg_encoder,
        text_encoder,
        # image_embedding_dims: int = 2048,
        text_embedding_dims: int = 768,
        projection_dims: int = 256,
        dropout: float = 0.0,
        temperature: float = 1.0,
        weight_decay: float = 0.0,
        head_lr: float = 1e-3,
        image_encoder_lr: float = 1e-4,
        text_encoder_lr: float = 1e-4,
        lr_scheduler_patience: float = 1.0,
        lr_scheduler_factor: float = 0.8,
        # patient_ids = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.text_encoder = text_encoder
        self.eeg_proj = ProjectionHead(eeg_encoder.mamba_config.d_model, projection_dims)
        self.text_proj = ProjectionHead(text_embedding_dims, projection_dims)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.head_lr = head_lr
        self.image_encoder_lr = image_encoder_lr
        self.text_encoder_lr = text_encoder_lr
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.save_hyperparameters()

        


    def _compute_losses(self, eeg_embeddings, text_embeddings):
        logits = (text_embeddings @ eeg_embeddings.T) / self.temperature
        eegs_similarity = eeg_embeddings @ eeg_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (eegs_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        eegs_loss = (-targets.T * self.log_softmax(logits.T)).sum(1)
        texts_loss = (-targets * self.log_softmax(logits)).sum(1)
        # print("Trying Losses")
        return (eegs_loss + texts_loss) / 2.0


    def forward(self, inputs):

        eeg, eeg_mask, pid = inputs['eeg'], inputs['eeg_attention_mask'], inputs['subject_id']
        text = {'input_ids': inputs['input_ids'],'attention_mask': inputs['attention_mask']}
        eeg_features = self.eeg_encoder(eeg, eeg_mask, pid)
        text_features = self.text_encoder(text)
        
        eeg_embeddings = self.eeg_proj(eeg_features)
        text_embeddings = self.text_proj(text_features)


        return eeg_embeddings, text_embeddings


    def configure_optimizers(self):
        parameters = [
            {"params": self.eeg_encoder.parameters(), "lr": self.image_encoder_lr},
            {"params": self.text_encoder.parameters(), "lr": self.text_encoder_lr},
            {
                "params": itertools.chain(
                    self.eeg_proj.parameters(),
                    self.text_proj.parameters(),
                ),
                "lr": self.head_lr,
                "weight_decay": self.weight_decay,
            },
        ]
        optimizer = optim.Adam(parameters, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.lr_scheduler_patience,
            factor=self.lr_scheduler_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss",
        }


    def training_step(self, batch, *args, **kwargs):
        # print("Starting training step")
        eeg_embeddings, text_embeddings = self.forward(batch)
        # print(f"EEG embeddings shape: {eeg_embeddings.shape}")
        # print(f"Text embeddings shape: {text_embeddings.shape}")
        loss = self._compute_losses(eeg_embeddings, text_embeddings).mean()
        # print(f"Computed loss: {loss.item()}")
        train_loss = self.all_gather(loss)
        self.log("train/loss", train_loss.mean())
        return loss


    def validation_step(self, batch, *args, **kwargs):
        eeg_embeddings, text_embeddings = self.forward(batch)
        loss = self._compute_losses(eeg_embeddings, text_embeddings).mean()
        # print(loss)
        val_loss = self.all_gather(loss)
        self.log("val/loss", val_loss.mean())
        return loss



class ProjectionHead(nn.Module):
    #some https://github.com/moein-shariatnia/OpenAI-CLIP
    def __init__(
        self,
        embedding_dim,
        projection_dim=1024,
        dropout=.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class EEGEncoder(nn.Module):
    def __init__(self, n_channels = 22, max_length = 100, n_layers = 16, mamba_config = None, 
                 embedding = 'mean', dropout = .1, patient_ids = None):
        super().__init__()
        self.n_channels = n_channels
        self.max_length = max_length
        self.n_layers = n_layers
        self.mamba_config = mamba_config

        self.project = nn.Linear(n_channels, mamba_config.d_model)
        self.eeg_mamba = MambaEEG(mamba_config)
        ###SIMPLE CHANNEL ATTENTION ISH BLOCK LIKE https://arxiv.org/pdf/2405.20142
        # self.channel_gap = nn.AvgPool1d(max_length)
        self.chan_mha = nn.MultiheadAttention(embed_dim =  1, num_heads = 1, batch_first=True)
        self.embedding_type = embedding #can be mean or last

        self.patient_ids = patient_ids

        self.patient_specific_layers = nn.ModuleDict({
            str(patient_id): nn.Conv1d(self.n_channels, self.n_channels, kernel_size=1)
            for patient_id in self.patient_ids
            })
        


        ###MAMBA SHENANIGANS
      
    def forward(self, x, attention_mask, pid):
        # Attention for channel dimension
        x = x.float()
        chan_X = x.permute(0, 2, 1)  # (batch_size, n_channels, max_length)
        chan_X = chan_X.mean(dim=-1).unsqueeze(-1)  # (batch_size, n_channels, 1)
        attn_output, attn_weights = self.chan_mha(chan_X, chan_X, chan_X)
        x *= attn_output.permute(0, 2, 1)  # (batch_size, max_length, n_channels)

        x = x.permute(0, 2, 1)  # (batch_size, d_model, max_length)

        # Apply patient-specific layer
        x = torch.stack([
            self.patient_specific_layers[str(patient_id.item())](eeg_feature)
            for eeg_feature, patient_id in zip(x, pid)
        ])
        x = x.permute(0, 2, 1)  # (batch_size, d_model, max_length)

        x = self.project(x)
        x = self.eeg_mamba(x)
        x = x.permute(0, 2, 1)  # (batch_size, d_model, max_length)
        

        if self.embedding_type == 'mean':
            # Apply attention mask to ignore padding tokens
            attention_mask_expanded = attention_mask.unsqueeze(1).expand(x.size()).float()
            x = (x * attention_mask_expanded).sum(dim=-1) / attention_mask_expanded.sum(dim=-1)
        elif self.embedding_type == 'last':
            # Select the last non-padded token based on the attention mask
            last_indices = attention_mask.sum(dim=1) - 1
            last_indices = last_indices.int()
            x = x[torch.arange(x.size(0)), :, last_indices]
            
        return x

        

class HuggingFaceEncoder(nn.Module):
    def __init__(self, model_name, freeze):
        super().__init__()
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if freeze:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

    def forward(self, inputs, pooling='last'):
        """
        Get the embedding for the given text using the specified pooling method.

        Args:
        text (str or list of str): The input text (either a sentence or individual words).
        pooling (str): Pooling method to use ('mean', 'max', or 'last').

        Returns:
        torch.Tensor: The computed embedding.
        """
        # inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, is_split_into_words=isinstance(text, list))
        outputs = self.model(**inputs)

        if pooling == 'mean':
            return torch.mean(outputs.last_hidden_state, dim=1)
        elif pooling == 'max':
            return torch.max(outputs.last_hidden_state, dim=1)[0]
        elif pooling == 'last':
            return outputs.last_hidden_state[:, -1, :]
        else:
            raise ValueError("Pooling method not recognized. Choose from 'mean', 'max', or 'last'.")
