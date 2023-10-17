import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) # Numpy seed also uses by Scikit Learn
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def show_n_sample(train_loader, nb_sample)  :
    rgb_dl , env_dl ,train_label_dl = next(iter(train_loader))
    # Initialize lists to store data for the plots
    nb_sample = 3
    image_rgb_list = []
    image_nir_list = []
    bio_1_list = []
    bio_2_list = []
    bio_3_list = []
    bio_4_list = []

    # Loop through the data
    for j in range(nb_sample):
        rgb, env, labels = rgb_dl[j], env_dl[j], train_label_dl[j]
        image_rgb = rgb[:3].permute(1, 2, 0)
        image_nir = rgb[3]

        image_rgb_list.append(image_rgb)
        image_nir_list.append(image_nir)

        bio_1, bio_2, bio_3, bio_4 = env[0], env[1], env[2], env[3]

        bio_1_list.append(bio_1.numpy())
        bio_2_list.append(bio_2.numpy())
        bio_3_list.append(bio_3.numpy())
        bio_4_list.append(bio_4.numpy())

    # Create subplots and display all the plots at the end
    fig, ax = plt.subplots(nb_sample, 2)
    fig_2, ax_env = plt.subplots(2 * nb_sample, 2)

    for i in range(nb_sample):
        ax[i][0].imshow(image_rgb_list[i])
        ax[i][0].set_title(f"RGB_image of size {image_rgb_list[i].shape[0]} , {image_rgb_list[i].shape[1]}", fontsize=5)
        ax[i][1].imshow(image_nir_list[i])
        ax[i][1].set_title(f"Infrared image of size {image_nir_list[i].shape[0]} , {image_nir_list[i].shape[1]}", fontsize=5)

        ax_env[2 * i][0].imshow(bio_1_list[i])
        ax_env[2 * i][1].imshow(bio_2_list[i])
        ax_env[2 * i + 1][0].imshow(bio_3_list[i])
        ax_env[2 * i + 1][1].imshow(bio_4_list[i])

    # Show the plots
    print("Showing the samples")
    plt.show()

        
