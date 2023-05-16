from dataLoader.dataSet import UAVDataset
import torch

import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import segmentation_models_pytorch as smp


dataset = UAVDataset('../dataset.csv','../VisDrone2019-DET-val/images/',output_size=256)

proportions = [.75, .10, .15]
lengths = [int(p * len(dataset)) for p in proportions]
lengths[-1] = len(dataset) - sum(lengths[:-1])

train, val, test = torch.utils.data.random_split(
    dataset = dataset, 
    lengths = lengths, 
    generator = torch.Generator().manual_seed(45)
)

batch_size = 4

train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle = True, drop_last = True)
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size)


class TwoConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size = 4, padding = "same")
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.rl = nn.ReLU()
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size = 4, padding = "same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor = 1, mode = "bilinear", align_corners = True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 4, padding = "same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        return x

class UNet_2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock(256, 4, 64)
        self.TCB2 = TwoConvBlock(64, 128, 128)
        self.TCB3 = TwoConvBlock(128, 256, 256)
        self.TCB4 = TwoConvBlock(256, 512, 512)
        self.TCB5 = TwoConvBlock(512, 1024, 1024)
        self.TCB6 = TwoConvBlock(1024, 512, 512)
        self.TCB7 = TwoConvBlock(512, 256, 256)
        self.TCB8 = TwoConvBlock(256, 128, 128)
        self.TCB9 = TwoConvBlock(128, 64, 64)
        self.maxpool = nn.MaxPool2d(1, stride = 1)
        
        self.UC1 = UpConv(1024, 512) 
        self.UC2 = UpConv(512, 256) 
        self.UC3 = UpConv(256, 128) 
        self.UC4= UpConv(128, 64)

        self.conv1 = nn.Conv2d(64, 157, kernel_size = 4, padding = "same")
        self.soft = nn.Softmax(dim = 4)

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)

        x = self.UC1(x)
        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)

        x = self.conv1(x)

        return x


TverskyLoss = smp.losses.TverskyLoss(mode = 'multilabel', log_loss = False)
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
def criterion(pred,target):
    return 0.5 * BCELoss(pred, target) + 0.5 * TverskyLoss(pred, target)



def trainModel():
    device = torch.device("cpu")
    unet = UNet_2D().to(device)
    optimizer = optim.Adam(unet.parameters(), lr = 0.001)

    history = {"train_loss": []}
    n = 0
    m = 0
    val_loss_min = np.Inf
    validation_accuracy = []
    validation_loss = []
    for epoch in range(15):
        train_accuracy = 0
        train_loss = 0
        val_accuracy = 0
        val_loss = 0
        correct_pixels = 0
        total_pixels = 0

    #   training
        unet.train()
        time_init = int(time.time())
        for i, data in enumerate(train_loader):
            inputs, labels = data["img"].to(device), data["label"].to(device)
            optimizer.zero_grad()
            outputs = unet(inputs)

            # pixel accuracy
            thresholded_outputs = (outputs > 0.5).int()
            print(outputs)
            correct_pixels = (thresholded_outputs == labels).sum().item()
            total_pixels += (inputs.size(2) * inputs.size(3)) * inputs.size(0)
            train_acc = correct_pixels / total_pixels
            train_accuracy += train_acc

            # loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            history["train_loss"].append(loss.item())
            n += 1
            if i % ((len(train) // batch_size) // 10) == (len(train) // batch_size) // 10 - 1:
                time_now = int(time.time())
                print(f"epoch:{epoch + 1}  index:{i + 1}  train_accuracy:{train_accuracy / n:.5f}  s_time:{(time_now-time_init)*(len(train_loader)-(i+1))*(15-epoch+1)}")
                print(f"epoch:{epoch + 1}  index:{i + 1}  train_loss:{train_loss / n:.5f}")
                time_init = int(time.time())
                n = 0
                train_accuracy = 0
                train_loss = 0

    #   validation
        unet.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data["img"].to(device), data["label"].to(device)
                outputs = unet(inputs)

                # pixel accuracy
                thresholded_outputs = (outputs > 0.5).int()
                correct_pixels = (thresholded_outputs == labels).sum().item()
                total_pixels += (inputs.size(2) * inputs.size(3)) * inputs.size(0)
                val_acc = correct_pixels / total_pixels
                val_accuracy += val_acc  

                # loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                m += 1
                if i % (len(val) // batch_size) == len(val) // batch_size - 1:
                    print(f"epoch:{epoch + 1}  index:{i + 1}  validation_accuracy:{val_accuracy / m:.5f}")
                    print(f"epoch:{epoch + 1}  index:{i + 1}  validation_loss:{val_loss / m:.5f}")
                    validation_accuracy.append(val_accuracy)
                    validation_loss.append(val_loss)
                    if val_loss < val_loss_min:
                        val_loss_min = val_loss
                        torch.save(unet.state_dict(), 'semantic_segmentation.pt')
                        print('Detected network improvement, saving current model')
        
                    m = 0
                    val_accuracy = 0
                    val_loss = 0
    return history,validation_accuracy,validation_loss

data = trainModel()