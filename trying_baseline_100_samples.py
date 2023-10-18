import pandas as pd
import numpy as np
import torch 
#import wandb
import os
import rasterio
from torch import nn
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from data import GLC23Datasets , GLC23PatchesProviders , GLC23TimeSeriesProviders
from data.GLC23Datasets import RGBNIR_env_Dataset
from models import twoBranchCNN
from util import seed_everything , show_n_sample
import matplotlib.pyplot as plt
from PIL import Image
 

from torch.utils.data import Dataset


df = pd.read_csv('data/sample_data/Presence_only_occurrences/Presences_only_train_sample.csv', sep=';')

seed_everything(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device used: {device}")


data_path = "data/sample_data/"
presence_only_path = data_path + "Presence_only_occurrences/Presences_only_train_sample.csv"
presence_absence_path = data_path + "Presence_Absences_occurrences/Presences_Absences_train_sample.csv"

# J'ai essayé pas mal de batch sizes différents mais pas fameux 
BATCH_SIZE = 32
LEARNING_RATE=1e-3
N_EPOCHS = 10
BIN_TRESH = 0.1
NUM_WORKERS = torch.cuda.device_count()
print(NUM_WORKERS)


run_name = 'First_run_100_samples'
if not os.path.exists(f"models/{run_name}"): 
    os.makedirs(f"models/{run_name}")
    
    
## TRAIN Set
import os

presence_only_df = pd.read_csv(presence_only_path, sep=";", header='infer', low_memory=False)

# Ici je retire les fichiers qui m'ont semblé poser problème : erreur après le message "je bug a cause de..."

presence_only_df = presence_only_df[~presence_only_df['patchID'].isin([3581101 , 3359168, 6023406, 5494000, 5470192, 5439626, 5362196, 5449902, 4169158,3254993, 3901881, 5224976])]

presence_absence_df = pd.read_csv(presence_absence_path, sep=";", header='infer', low_memory=False)

# Erreur à cause des patchs  
train_dataset = RGBNIR_env_Dataset(presence_only_df, env_patch_size=10, rgbnir_patch_size=100)

val_dataset = RGBNIR_env_Dataset(presence_absence_df, species=train_dataset.species, env_patch_size=10, rgbnir_patch_size=100)
n_species = len(train_dataset.species)
print(f"Training set: {len(train_dataset)} sites, {n_species} sites")


# num workers = 1 pour voir à quel moment ça bug
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE , num_workers = 1 ) ## Maybe use a transform argument 
val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE , num_workers = 1 )

#show_n_sample(train_loader,2)

print(f"Dataloaders: {train_loader, val_loader}") 
print(f"Length of train dataloader: {len(train_loader)} batches of {BATCH_SIZE}")
print(f"Length of test dataloader: {len(val_loader)} batches of {BATCH_SIZE}")


# 
model = twoBranchCNN(n_species).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)#, momentum=0.9)

loss_fn = torch.nn.BCEWithLogitsLoss()

val_loss_list, val_precision_list, val_recall_list, val_f1_list = [], [], [], []



model = twoBranchCNN(n_species).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)#, momentum=0.9)

loss_fn = torch.nn.BCEWithLogitsLoss()


# C'est ici que la magie opère, j'ai l'impression que l'erreur survient à la fin du premier batch
#Il arrive pas à passer au batch suivant....
for epoch in range(0, N_EPOCHS):
        print(f"EPOCH {epoch}")

        model.train()
        train_loss_list = []
        for rgb, env, labels in tqdm(train_loader):
            print(f"Shape of the rgb_nir loader : {rgb.shape}")
            print(f"Shape of the env loader : {env.shape}")
            print(f"Output shape : {labels.shape}")
            # if not os.path.isfile(rgb):
            #     raise FileNotFoundError(f"File not found: {rgb}")
            y_pred = model(rgb.to(torch.float32).to(device), env.to(torch.float32).to(device), val=True)
            val_loss = loss_fn(y_pred, labels.to(torch.float32).to(device))
            val_loss_list.append(val_loss)

            y_pred = torch.sigmoid(y_pred).cpu().detach().numpy()
            y_bin = np.where(y_pred > BIN_TRESH, 1, 0)
            
            
            print(f"\tVALIDATION LOSS={val_loss}\t ")
            optimizer.zero_grad()
            val_loss.backward()
            optimizer.step()


