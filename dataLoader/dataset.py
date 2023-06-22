import pandas as pd
import numpy as np
from PIL import Image
from glob import glob
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import segmentation_models_pytorch as smp
import yaml


class UAVThermicalDataset(torch.utils.data.Dataset):
    
    def __init__(self, yamlDataset, rootDir, typeOfRun):
        self.class_list = parseYAML(yamlDataset)
        self.imgpath_list = sorted(glob(f'{rootDir}/images/{typeOfRun}/*.jpg'))
        self.labelpath_list = sorted(glob(f'{rootDir}/labels/{typeOfRun}/*.txt'))

    def __getitem__(self, i):
                
        imgpath = self.imgpath_list[i]
        img = cv2.imread(imgpath)
        img = cv2.resize(img, dsize = (256, 256))
        img = img / 255
        img = torch.from_numpy(img.astype(np.float32)).clone()
        img = img.permute(2, 0, 1)

        labelpath = self.labelpath_list[i]
        # label = Image.open(labelpath)
        # label = np.asarray(label)
        # label = cv2.resize(label, dsize = (256, 256))
        # label = torch.from_numpy(label.astype(np.float32)).clone()
        # label = torch.nn.functional.one_hot(label.long(), num_classes = len(self.class_list))
        # label = label.to(torch.float32)
        # label = label.permute(2, 0, 1)
        rd = open(labelpath).readlines()
        label = []
        label = [[] for _ in self.class_list]
        for r in rd:
            r = r.strip()
            a = r.split(" ")
            position = [float(i) for i in a[1:]]
            label[int(a[0])].append(position)
        # nplabel = np.asarray(label,np.float32)
        # label = torch.from_numpy(nplabel.astype(np.float32)).clone()
        #label = torch.FloatTensor(label)
        #torchvision.transforms.ToTensor(label)
        for k in range(len(label)):
            while len(label[k]) < 40:
                label[k].append([0,0,0,0])
        label = torch.FloatTensor(label)
        return img, label
    def __len__(self):
        return len(self.imgpath_list)



def parseYAML(yamlFile):
    with open(yamlFile) as stream:
        load = yaml.safe_load(stream)
    class_list = [load["names"][key] for key in load["names"]]
    return class_list