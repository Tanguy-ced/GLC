# Author: Benjamin Deneu <benjamin.deneu@inria.fr>
#         Theo Larcher <theo.larcher@inria.fr>
#
# License: GPLv3
#
# Python version: 3.10.6

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import rasterio


from data.GLC23PatchesProviders import MetaPatchProvider
from data.GLC23TimeSeriesProviders import MetaTimeSeriesProvider

class PatchesDataset(Dataset):
    def __init__(
        self,
        occurrences,
        providers,
        transform=None,
        target_transform=None,
        id_name="glcID",
        label_name="speciesId",
        item_columns=['lat', 'lon', 'patchID'],
    ):
        self.occurences = Path(occurrences)
        self.base_providers = providers
        self.transform = transform
        self.target_transform = target_transform
        self.provider = MetaPatchProvider(self.base_providers, self.transform)

        df = pd.read_csv(self.occurences, sep=";", header='infer', low_memory=False)

        self.observation_ids = df[id_name].values
        self.items = df[item_columns]
        self.targets = df[label_name].values

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, index):
        item = self.items.iloc[index].to_dict()

        patch = self.provider[item]

        target = self.targets[index]

        if self.target_transform:
            target = self.target_transform(target)

        return torch.from_numpy(patch).float(), target
    
    def plot_patch(self, index):
        item = self.items.iloc[index].to_dict()
        self.provider.plot_patch(item)


class PatchesDatasetMultiLabel(PatchesDataset):
    def __init__(self,
        occurrences,
        providers,
        transform=None,
        target_transform=None,
        id_name="glcID",
        label_name="speciesId",
        item_columns=['lat', 'lon', 'patchID']
    ):
        super().__init__(occurrences, providers, transform, target_transform, id_name, label_name, item_columns)
        
    def __getitem__(self, index):
        item = self.items.iloc[index].to_dict()
        patchid_rows_i = self.items[self.items['patchID']==item['patchID']].index
        self.targets_sorted = np.sort(self.targets)

        patch = self.provider[item]

        targets = np.zeros(len(self.targets))
        for idx in patchid_rows_i:
            target = self.targets[idx]
            if self.target_transform:
                target = self.target_transform(target)
            targets[np.where(self.targets_sorted==target)] = 1
        targets = torch.from_numpy(targets)

        return torch.from_numpy(patch).float(), targets

class PatchesDatasetOld(Dataset):
    def __init__(
        self,
        occurrences,
        providers,
        transform=None,
        target_transform=None,
        id_name="glcID",
        label_name="speciesId",
        item_columns=['lat', 'lon', 'patchID'],
    ):
        self.occurences = Path(occurrences)
        self.providers = providers
        self.transform = transform
        self.target_transform = target_transform

        df = pd.read_csv(self.occurences, sep=";", header='infer', low_memory=False)

        self.observation_ids = df[id_name].values
        self.items = df[item_columns]
        self.targets = df[label_name].values

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, index):
        item = self.items.iloc[index].to_dict()

        patches = []
        for provider in self.providers:
            patches.append(provider[item])

        # Concatenate all patches into a single tensor
        if len(patches) == 1:
            patches = patches[0]
        else:
            patches = np.concatenate(patches, axis=0)

        if self.transform:
            patches = self.transform(patches)

        target = self.targets[index]

        if self.target_transform:
            target = self.target_transform(target)

        return torch.from_numpy(patches).float(), target

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        occurrences,
        providers,
        transform=None,
        target_transform=None,
        id_name="glcID",
        label_name="speciesId",
        item_columns=['timeSerieID'],
    ):
        self.occurences = Path(occurrences)
        self.base_providers = providers
        self.transform = transform
        self.target_transform = target_transform
        self.provider = MetaTimeSeriesProvider(self.base_providers, self.transform)

        df = pd.read_csv(self.occurences, sep=";", header='infer', low_memory=False)

        self.observation_ids = df[id_name].values
        self.items = df[item_columns]
        self.targets = df[label_name].values
    
    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, index):
        item = self.items.iloc[index].to_dict()

        patch = self.provider[item]

        target = self.targets[index]

        if self.target_transform:
            target = self.target_transform(target)

        return torch.from_numpy(patch).float(), target

    def plot_ts(self, index):
        item = self.items.iloc[index].to_dict()
        self.provider.plot_ts(item)
        
        
class RGBNIR_env_Dataset(Dataset):
    def __init__(
            self,
            occurrences,
            species=None,
            sites_columns=['patchID','dayOfYear','lat','lon'],
            env_dirs=[
                "data/sample_data/EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/", 
                "data/sample_data/EnvironmentalRasters/HumanFootprint/summarized/"
            ],
            label_col='speciesId',
            rgbnir_dir="data/sample_data/SatelliteImages/",
            env_patch_size=128,
            rgbnir_patch_size=128
    ):
        
        # occurrences
        self.error_path = "data/sample_data/SatelliteImages/rgb/29/99/3949929.jpeg"
        self.template_img = (np.asarray(Image.open(self.error_path)) / 255.0).transpose((2,0,1))
        self.occurrences = occurrences
        self.sites = occurrences[sites_columns].drop_duplicates().reset_index(drop=True)
        self.label_col = label_col
        if species is None: 
            self.species = occurrences['speciesId'].sort_values().unique()
        else: 
            self.species = species

        # satellite images rgbnir
        self.rgbnir_dir = rgbnir_dir
        self.rgbnir_patch_size = rgbnir_patch_size

        # environmental covariates 
        self.env_dirs = env_dirs
        self.env_patch_size = env_patch_size
        self.env_stats = {}
        for dir in env_dirs:
            for file in os.listdir(dir):
                if ".tif" not in file: continue
                with rasterio.open(dir + file) as src:
                    n = src.count
                    assert n == 1, f"number of layers should be 1, for {dir+file} got {n}"
                    nodata_value = src.nodatavals[0]
                    data = src.read().astype(float)
                    data = np.where(data == nodata_value, np.nan, data)
                    self.env_stats[dir + file] = {
                        'mean': np.nanmean(data), 
                        'std': np.nanstd(data), 
                        'min': np.nanmin(data), 
                        'max': np.nanmax(data), 
                        'nodata_value': nodata_value
                    }
            

    def __getitem__(self, index):
        try : 
            item = self.sites.iloc[index].to_dict()
            if (str(int(item['patchID'])) == 3254993) : return None
            item_species = self.occurrences[
                (self.occurrences['patchID'] == item['patchID']) & (self.occurrences['dayOfYear'] == item['dayOfYear'])
            ][self.label_col].values
            labels = 1 * np.isin(self.species, item_species)
            print("Ca galère")
            # satellite images rgbnir
            patch_id = str(int(item['patchID']))
            rgb_path = f"{self.rgbnir_dir}rgb/{patch_id[-2:]}/{patch_id[-4:-2]}/{patch_id}.jpeg"
            print(f'Je vais bugger a cause de toi : {rgb_path} ')
            if rgb_path == "data/sample_data/SatelliteImages/rgb/93/49/3254993.jpeg":
                print("entered")
                rgb_img = (np.asarray(Image.open(self.error_path)) / 255.0).transpose((2,0,1))
            else : 
                rgb_img = (np.array(Image.open(rgb_path)) / 255.0).transpose((2,0,1))
            gray_image = np.mean(rgb_img, axis=2)
            black_pixels = np.sum(gray_image == 0)
            print(f"Percentage of black_pixel {black_pixels / (self.rgbnir_patch_size * self.rgbnir_patch_size)}")
            print('et non')
            nir_path = f"{self.rgbnir_dir}nir/{patch_id[-2:]}/{patch_id[-4:-2]}/{patch_id}.jpeg"
            nir_img = np.expand_dims(np.asarray(Image.open(nir_path)) / 255.0, axis=0)
            print('banana')
            rgbnir = np.concatenate([rgb_img, nir_img])
            if self.rgbnir_patch_size != 128:
                rgbnir = rgbnir[:, round((rgbnir[0].shape[0] - self.rgbnir_patch_size) /2):round((rgbnir[0].shape[0] + self.rgbnir_patch_size) /2),
                                        round((rgbnir[0].shape[1] - self.rgbnir_patch_size) /2):round((rgbnir[0].shape[1] + self.rgbnir_patch_size) /2)]
                
            # environmental covariates 
            patch_list = []
            for rasterpath, stats in self.env_stats.items():
                with rasterio.open(rasterpath) as src:
                    center_x, center_y = src.index(item['lon'], item['lat'])
                    left = center_x - (self.env_patch_size // 2)
                    top = center_y - (self.env_patch_size // 2)
                    patch = src.read(window=rasterio.windows.Window(left, top, self.env_patch_size, self.env_patch_size)).astype(float)
                    # patch = (patch - stats['mean']) / stats['std'] # standard scaling
                    patch = (patch - stats['min']) / (stats['max'] - stats['min']) # min max scaling
                    patch = np.where(patch == stats['nodata_value'], np.nan, patch)
                    patch_list.append(patch)

            env_covs = np.concatenate(patch_list)

            if (rgb_path == "data/sample_data/SatelliteImages/rgb/77/11/5641177.jpeg"):
                print("STOP")
            return rgbnir, env_covs, labels
        except:
            pass
    
    def __len__(self):
        return self.sites.shape[0]