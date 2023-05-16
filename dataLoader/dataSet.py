import os
import random
import cv2
from matplotlib import patches, pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


# class UAVDataset(Dataset):
#     def __init__(self,csv_file,root_dir,transform=None,shape=200) -> None:
#         self.points = pd.read_csv(csv_file)
#         #(self.points.iloc[1,1:])
#         self.root_dir = root_dir
#         self.transform = transform
#         self.output_size = shape
    
#     def __len__(self):
#         return len(self.points)
    
#     def random_int(self):
#         return self.__getitem__(random.randint(0,len(self.points)))

#     def __getitem__(self,index):
#         img_path = os.path.join(self.root_dir,self.points.iloc[index,0])
#         image = Image.open(img_path)
#         landmarks = self.points.iloc[index, 1:]
#         landmarks = np.array([landmarks])
#         landmarks = landmarks.astype('int64').reshape(-1, 4)
#         h, w = image.shape[:2]
#         #print(h,w)
#         if h > w:
#             new_h, new_w = self.output_size * h / w, self.output_size
#         else:
#             new_h, new_w = self.output_size, self.output_size * w / h

#         new_h, new_w = int(new_h), int(new_w)
#         #print(new_h,new_w)
#         #if self.transform:
#         image = Image.open(img_path)
#         #print([w // new_w, h // new_h, w // new_w, h // new_h])
#         landmarks = landmarks * [new_w / w, new_h / h,new_w / w, new_h / h]
#         for i in range(len(landmarks)):
#             landmarks[i] = [int(k) for k in landmarks[i]]
#         #print(landmarks)
#         image = self.transform(image)
#         #y_label = torch.tensor(landmarks)
#         y_label = landmarks
#         if self.transform:
#             image = self.transform(image)
#         return (image,landmarks)

class UAVDataset(Dataset):
    
    def __init__(self,csv_file,root_dir,output_size=256):
        self.points = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.output_size = output_size

    

    def __getitem__(self, i):
                
        #imgpath = self.imgpath_list[i]
        img_path = os.path.join(self.root_dir,self.points.iloc[i,0])
        img_prev = cv2.imread(img_path)
        img = cv2.resize(img_prev, dsize = (self.output_size, self.output_size))
        img = img / 255
        img = torch.from_numpy(img.astype(np.float32)).clone()
        #img = img.permute(2, 0, 1)
        label_points = self.points.iloc[i,1:]
        
        label = np.array([label_points])
        label = label.astype('int64').reshape(-1, 4)
        h, w = img_prev.shape[:2]

        new_h, new_w = h / self.output_size, w/ self.output_size

        label = label // [new_w,new_h,new_w,new_h]
        

        data = {"img": img, "label": label}
        return data

    def __len__(self):
        return len(self.points)

def showData():
    dataset = UAVDataset('../dataset.csv','../VisDrone2019-DET-val/images/')
    fig, ax = plt.subplots()
    data = dataset.__getitem__(0)
    #print(data["label"])
    ax.imshow(data["img"].numpy())
    rd = data["label"]
    for i in range(len(rd)):
        rect = patches.Rectangle((rd[i][0], rd[i][1]), rd[i][2]-rd[i][0], rd[i][3]-rd[i][1], linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    plt.show()

def dataTest():
    dataset = UAVDataset('../dataset.csv','../VisDrone2019-DET-val/images/')
    data = dataset.random_int()
    orb = cv2.ORB_create(nfeatures=100)

    def descriptor(image,orb):
        _,des = orb.detectAndCompute(image,None)
        return des

    def cropImage(image,points):
        return image[points[1]-50:points[3]+50,points[0]-50:points[2]+50]

    des1 = descriptor(data[0],orb)
    i = 0
    images = []

    while (data[1][i][0] != 0 and data[1][i][1] != 0) :
        img = cropImage(data[0],data[1][i])
        des2 = descriptor(img, orb)
        if des2 is None:
            i += 1
            continue
        
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(des1,des2,k=2)

        good_matches = []
        for m,n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)
        classId = "None"

        if len(good_matches) > 0:
            classId = "Human"

            cv2.putText(img,classId,(50,50),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
        i += 1
    cv2.imshow('prova1',data[0])
    cv2.waitKey(0)