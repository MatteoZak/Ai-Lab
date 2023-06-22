from tqdm.auto import tqdm
from dataLoader.dataset import UAVThermicalDataset
import torch
import os
import requests
import zipfile
import cv2
import math
import matplotlib.pyplot as plt
import glob
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import torchvision
from torchvision import transforms
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader

from PIL import Image

ROOT_DIR = 'hit-uav'
train_imgs_dir = f'{ROOT_DIR}/images/train'
train_labels_dir = f'{ROOT_DIR}/labels/train'
val_imgs_dir = f'{ROOT_DIR}/images/val'
val_labels_dir = f'{ROOT_DIR}/labels/val'
test_imgs_dir = f'{ROOT_DIR}/images/test'
test_labels_dir = f'{ROOT_DIR}/labels/test'
classes = ['Person', 'Car', 'Bicycle', 'OtherVechicle', 'DontCare']
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax

def plot_box(image, bboxes, labels, classes=classes, colors=colors, pos='above'):
    # Need the image height and width to denormalize
    # the bounding box coordinates
    height, width, _ = image.shape
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # Line width.
    tf = max(lw - 1, 1) # Font thickness.
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        # denormalize the coordinates
        xmin = int(x1*width)
        ymin = int(y1*height)
        xmax = int(x2*width)
        ymax = int(y2*height)

        p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        
        class_name = classes[int(labels[box_num])]

        color=colors[classes.index(class_name)]
        
        cv2.rectangle(
            image, 
            p1, p2,
            color=color, 
            thickness=lw,
            lineType=cv2.LINE_AA
        ) 

        # For filled rectangle.
        w, h = cv2.getTextSize(
            class_name, 
            0, 
            fontScale=lw / 3, 
            thickness=tf
        )[0]

        outside = p1[1] - h >= 3
        
        if pos == 'above':
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(
                image, 
                p1, p2, 
                color=color, 
                thickness=-1, 
                lineType=cv2.LINE_AA
            )  
            cv2.putText(
                image, 
                class_name, 
                (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=lw/3.5, 
                color=(255, 255, 255), 
                thickness=tf, 
                lineType=cv2.LINE_AA
            )
        else:
            new_p2 = p1[0] + w, p2[1] + h + 3 if outside else p2[1] - h - 3
            cv2.rectangle(
                image, 
                (p1[0], p2[1]), new_p2, 
                color=color, 
                thickness=-1, 
                lineType=cv2.LINE_AA
            )  
            cv2.putText(
                image, 
                class_name, 
                (p1[0], p2[1] + h + 2 if outside else p2[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=lw/3, 
                color=(255, 255, 255), 
                thickness=tf, 
                lineType=cv2.LINE_AA
            )
    return image

def plot(image_path, label_path, num_samples, classes=classes, colors=colors, pos='above'):
    all_training_images = glob.glob(image_path+'/*')
    all_training_labels = glob.glob(label_path+'/*')
    all_training_images.sort()
    all_training_labels.sort()
    
    temp = list(zip(all_training_images, all_training_labels))
    random.shuffle(temp)
    all_training_images, all_training_labels = zip(*temp)
    all_training_images, all_training_labels = list(all_training_images), list(all_training_labels)
    
    num_images = len(all_training_images)
    
    if num_samples == -1:
        num_samples = num_images
    
    num_cols = 2
    num_rows = int(math.ceil(num_samples / num_cols))
        
    plt.figure(figsize=(10 * num_cols, 6 * num_rows))
    for i in range(num_samples):
        image_name = all_training_images[i].split(os.path.sep)[-1]
        image = cv2.imread(all_training_images[i])
        with open(all_training_labels[i], 'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label, x_c, y_c, w, h = label_line.split(' ')
                x_c = float(x_c)
                y_c = float(y_c)
                w = float(w)
                h = float(h)
                bboxes.append([x_c, y_c, w, h])
                labels.append(label)
        result_image = plot_box(image, bboxes, labels, classes, colors, pos)
        plt.subplot(num_rows, num_cols, i+1) # Visualize 2x2 grid of images.
        plt.imshow(image[:, :, ::-1])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# plot(
#     image_path=os.path.join( train_imgs_dir), 
#     label_path=os.path.join( train_labels_dir),
#     num_samples=8
# )
# def collate_fn(batch):
#     images = []
#     targets = []

#     for image, labels in batch:
#         images.append(image)
#         targets.append({
#             'labels': torch.tensor(labels, dtype=torch.int64),
#             'image_id': torch.zeros(1),  # Placeholder for image ID (optional)
#         })

#     images = torch.stack(images)

#     return images, targets

batch_size = 20

train_dataset = UAVThermicalDataset('hit-uav/dataset.yaml','hit-uav',"train")
test_dataset = UAVThermicalDataset('hit-uav/dataset.yaml','hit-uav',"test")
val_dataset = UAVThermicalDataset('hit-uav/dataset.yaml','hit-uav',"val")

train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

# define the device for the computation
#device = "cuda" if torch.cuda.is_available() else "cpu"

# define our CNN
class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()

        # Define your model layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 64 * 64, 256)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 5 * 40 * 4)  # Output size matches the target size
        #self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = x.view(-1, 5, 40, 4)  # Reshape to match the target size
        #x = self.sigm(x)
        return x



# instance of our model
device = "cpu"
model = myCNN().to(device)

# hyperparameter settings
learning_rate = 0.01
epochs = 12#

# loss function definition
loss_fn = nn.CrossEntropyLoss()
#loss_fn = nn.NLLLoss()

TverskyLoss = smp.losses.TverskyLoss(mode = 'multilabel', log_loss = False)
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
def criterion(pred,target):
    return 0.5 * BCELoss(pred, target) + 0.5 * TverskyLoss(pred, target)
# optimizer definition
optimizer = torch.optim.AdamW(model.parameters(),learning_rate) #torch.optim.SGD(model.parameters(), learning_rate)

# defining the training loop
def trainingLoop(train_dataloader, model, loss_fn, optimizer):
    #val_loss_min = np.Inf
    validation_loss = []
    model.train()
    for batch,(x,y) in enumerate(train_dataloader):
        # move data on gpu
        x = x.to(device)
        y = y.to(device)
        # print(y.size())
        # print(x.size())
        #print(batch)
        pred = model(x)
        
        loss = loss_fn(pred,y)

        # backpropagation 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        validation_loss.append(loss.item())
        if batch % 10 == 0:
            loss = loss.item()
            print(f"The loss is {loss}")
    
    return validation_loss
        


def testLoop(test_dataloader, model, loss_fn):
    print_size = len(test_dataloader)
    num_batches = len(test_dataloader)
    test_loss = 0
    correct = 0
    val_loss_min = np.Inf
    validation_accuracy = []
    validation_loss = []
    model.eval()
    with torch.no_grad():
        for batch,(x,y) in enumerate(test_dataloader):
            x,y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred,y).item()
            validation_loss.append(loss)
            test_loss += loss
            corr = (pred == y).type(torch.float).sum().item()
            validation_accuracy .append(corr)
            correct += corr
    test_loss = test_loss/num_batches
    correct = correct / print_size

    print(f"Accuracy: {correct * 100}, Average loss: {test_loss}")

    if test_loss < val_loss_min:
        val_loss_min = test_loss
        torch.save(model.state_dict(), 'semantic_segmentation.pt')
        print("Saving Changes")
    return validation_accuracy,validation_loss


def createBrain():
    history_train = {
            'loss':[]
        }
    history_test = {
        'accuracy':[],
        'loss':[]
    }
    for e in range(epochs):
        loss = trainingLoop(train_dataloader,model,loss_fn,optimizer)
        history_train['loss'].append(loss)
        acc,loss = testLoop(test_dataloader,model,loss_fn)
        history_test['accuracy'].append(acc)
        history_test['loss'].append(loss)

    import json

    with open('savedata_test.json','w') as wr:
        json.dump(history_test,wr,indent=4)

    with open('savedata_train.json','w') as wr:
        json.dump(history_train,wr,indent=4)


if __name__ == "__main__":
    createBrain()