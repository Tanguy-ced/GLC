from torch import nn
import torch

class cnn(nn.Module):
    def __init__(self, n_features, n_species):
        super().__init__()
        self.conv1 = nn.Conv2d(n_features, 32, kernel_size=3)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 8, kernel_size=3)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flat = nn.Flatten()

        self.fc4 = nn.Linear(3200, n_species)
    
    def forward(self, x):
        # input n_featuresx128x128, output 32x126x126
        x = self.act1(self.conv1(x))
        # input 32x126x126, output 32x42x42
        x = self.pool1(x)
        x = self.drop1(x)

        # input 32x42x42, output 8x40x40
        x = self.act2(self.conv2(x))
        # input 8x40x40, output 8x20x20
        x = self.pool2(x)
        # input 8x20x20, output 3200
        x = self.flat(x)
        
        # input 3200, output n_species
        x = self.fc4(x)
        return x

class cnn_batchnorm(nn.Module):
    def __init__(self, n_features, n_species):
        super().__init__()
        self.conv1 = nn.Conv2d(n_features, 32, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 8, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(8)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(3200, 512)
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, n_species)
    
    def forward(self, x):
        # input n_featuresx128x128, output 32x126x126
        x = self.act1(self.batchnorm1(self.conv1(x)))
        # input 32x126x126, output 32x42x42
        x = self.pool1(x)
        x = self.drop1(x)

        # input 32x42x42, output 8x40x40
        x = self.act2(self.batchnorm2(self.conv2(x)))
        # input 8x40x40, output 8x20x20
        x = self.pool2(x)
        # input 8x20x20, output 3200
        x = self.flat(x)

        # input 3200, output 512
        x = self.act3(self.batchnorm3(self.fc3(x)))
        x = self.drop3(x)
        
        # input 512, output n_species
        x = self.fc4(x)
        return x

class cnn_batchnorm_act(nn.Module):
    def __init__(self, n_features, n_species):
        super().__init__()
        self.conv1 = nn.Conv2d(n_features, 32, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 8, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(8)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(3200, 512)
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, n_species)
        self.act4 = nn.ReLU()
    
    def forward(self, x):
        # input n_featuresx128x128, output 32x126x126
        x = self.act1(self.batchnorm1(self.conv1(x)))
        # input 32x126x126, output 32x42x42
        x = self.pool1(x)
        x = self.drop1(x)

        # input 32x42x42, output 8x40x40
        x = self.act2(self.batchnorm2(self.conv2(x)))
        # input 8x40x40, output 8x20x20
        x = self.pool2(x)
        # input 8x20x20, output 3200
        x = self.flat(x)

        # input 3200, output 512
        x = self.act3(self.batchnorm3(self.fc3(x)))
        x = self.drop3(x)
        
        # input 512, output n_species
        x = self.act4(self.fc4(x))
        return x

class cnn_batchnorm_patchsize_20(nn.Module):
    def __init__(self, n_features, n_species, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_features, 16, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(16, 8, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(8)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout(dropout)

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(512, 1024)
        self.batchnorm3 = nn.BatchNorm1d(1024)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(1024, n_species)
    
    def forward(self, x):
        # input n_featuresx20x20, output 16x18x18
        x = self.act1(self.batchnorm1(self.conv1(x)))
        x = self.drop1(x)

        # input 16x18x18, output 8x16x16
        x = self.act2(self.batchnorm2(self.conv2(x)))
        # input 8x16x16, output 8x8x8
        x = self.pool2(x)
        x = self.drop2(x)
        # input 8x8x8, output 512
        x = self.flat(x)

        # input 512, output 1024
        x = self.act3(self.batchnorm3(self.fc3(x)))
        x = self.drop3(x)
        
        # input 1024, output n_species
        x = self.fc4(x)
        return x
    
class MLP(nn.Module):

    def __init__(self, input_size, output_size, num_layers, width, dropout=0.0):
        super(MLP, self).__init__()

        self.output_size = output_size
        self.width = width

        layers = []

        layers.append(nn.Linear(input_size, width))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 1):
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.Linear(width, width))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(p=dropout))
    
        layers.append(nn.Linear(width, output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class twoBranchCNN(nn.Module):

    def __init__(self, n_species):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=5)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(8, 8, kernel_size=5)
        self.batchnorm2 = nn.BatchNorm2d(8)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=4)
        self.flat2 = nn.Flatten()

        self.conv3 = nn.Conv2d(4, 16, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3)
        self.batchnorm4 = nn.BatchNorm2d(16)
        self.act4 = nn.ReLU()
        self.flat4 = nn.Flatten()

        self.fc5 = nn.Linear(1544, 1024)
        self.act5 = nn.ReLU()

        self.fc6 = nn.Linear(1024, n_species)

    def forward(self, rgb_x, env_x, val=False):
        if val: print("entering top network",rgb_x.shape)
        # input 4x100x100 -> output 4x96x96 (k=5)
        rgb_x = self.act1(self.batchnorm1(self.conv1(rgb_x)))
        # input 8x96x96 -> output 8x48x48 (k=2)
        rgb_x = self.pool1(rgb_x)
        # input 8x48x48 -> output 8x44x44 (k=5)
        rgb_x = self.act2(self.batchnorm2(self.conv2(rgb_x)))
        # input 8x44x44 -> output 8x11x11 (k=4)
        rgb_x = self.pool2(rgb_x)
        # input 8x11x11 -> output 968
        rgb_x = self.flat2(rgb_x)
        if val: print("exiting top network", rgb_x.shape)

        if val: print("entering bottom  network",env_x.shape)
        # input 21x10x10 -> output 16x8x8 (k=3)
        env_x = self.act3(self.batchnorm3(self.conv3(env_x)))
        # input 16x8x8 -> output 16x6x6 (k=3)
        env_x = self.act4(self.batchnorm4(self.conv4(env_x)))
        # inpput 16x6x6 -> output 576 (k=2)
        env_x = self.flat4(env_x)
        if val: print("exiting bottom netwrok" , env_x.shape)

        #if val: print(x.shape)
        # 968 + 576 = 1544
        x = torch.cat((rgb_x, env_x), dim=1)
        if val: print(x.shape)
        # input 1544 -> output 1024
        x = self.act5(self.fc5(x))
        # input 1024 -> output n_species
        x = self.fc6(x)
        return x