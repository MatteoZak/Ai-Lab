import cv2
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


dataset = UAVDataset('dataset.csv','image-Dataset/images/',output_size=256)

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
    
    # class OurMLP(nn.Module):
    # def __init__(self):
    #     super().__init__() # inherit from nn.Module
    #     self.mlp = nn.Sequential(
    #         nn.Linear(28 * 28, 20), # input layer
    #         nn.Sigmoid(), # activation function
    #         nn.Linear(20, 50), # hidden layer
    #         nn.Sigmoid(), # activation function
    #         nn.Linear(50, 10), # output layer
    #     )
    #     self.flatten = nn.Flatten() # flatten image


    # def forward(self, x):
    #     x = self.flatten(x) # flatten image
    #     logits = self.mlp(x) # pass through MLP
    #     return logits # return output
 

class UNet_2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256,512,3)
        self.conv2 = nn.Conv2d(512,189,1)
        self.fc1 = nn.Linear(254,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,4)
        self.relu = nn.ReLU()
        # self.mlp = nn.Sequential(
        #     nn.Conv2d(256,256,3),
        #     nn.Linear(768,16 * 16), # input layer
        #     nn.Sigmoid(), # activation function
        #     nn.Linear(16 * 16, 32), # hidden layer
        #     nn.Sigmoid(), # activation function
        #     nn.Linear(32, 189), # output layer
        # )
        # self.flatten = nn.Flatten() # flatten image

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        # second convolution
        x = self.conv2(x)
        x = self.relu(x)
        
        # fully connected
        x = torch.flatten(x,1) # flatten all dimensions except the batch

        # fc1
        x = self.fc1(x)
        x = self.relu(x)

        # fc2
        x = self.fc2(x)
        x = self.relu(x)

        # fc out
        x = self.fc3(x)
        # x = self.flatten(x) # flatten image
        # logits = self.mlp(x) # pass through MLP
        #return logits # return output
        return x


TverskyLoss = smp.losses.TverskyLoss(mode = 'multilabel', log_loss = False)
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
def criterion(pred,target):
    return 0.5 * BCELoss(pred, target) + 0.5 * TverskyLoss(pred, target)



# def trainModel():
#     device = torch.device("cpu")
#     unet = UNet_2D().to(device)
#     optimizer = optim.Adam(unet.parameters(), lr = 0.001)

#     history = {"train_loss": []}
#     n = 0
#     m = 0
#     val_loss_min = np.Inf
#     validation_accuracy = []
#     validation_loss = []
#     for epoch in range(15):
#         train_accuracy = 0
#         train_loss = 0
#         val_accuracy = 0
#         val_loss = 0
#         correct_pixels = 0
#         total_pixels = 0

#     #   training
#         unet.train()
#         time_init = int(time.time())
#         for i, data in enumerate(train_loader):
#             inputs, labels = data["img"].to(device), data["label"].to(device)
#             optimizer.zero_grad()
#             outputs = unet(inputs)

#             # pixel accuracy
#             thresholded_outputs = (outputs > 0.5).int()
#             print(inputs)
#             correct_pixels = (thresholded_outputs == labels).sum().item()
#             total_pixels += (inputs.size(2) * inputs.size(3)) * inputs.size(0)
#             train_acc = correct_pixels / total_pixels
#             train_accuracy += train_acc

#             # loss
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#             history["train_loss"].append(loss.item())
#             n += 1
#             if i % ((len(train) // batch_size) // 10) == (len(train) // batch_size) // 10 - 1:
#                 time_now = int(time.time())
#                 print(f"epoch:{epoch + 1}  index:{i + 1}  train_accuracy:{train_accuracy / n:.5f}  s_time:{(time_now-time_init)*(len(train_loader)-(i+1))*(15-epoch+1)}")
#                 print(f"epoch:{epoch + 1}  index:{i + 1}  train_loss:{train_loss / n:.5f}")
#                 time_init = int(time.time())
#                 n = 0
#                 train_accuracy = 0
#                 train_loss = 0

#     #   validation
#         unet.eval()
#         with torch.no_grad():
#             for i, data in enumerate(val_loader):
#                 inputs, labels = data["img"].to(device), data["label"].to(device)
#                 outputs = unet(inputs)

#                 # pixel accuracy
#                 thresholded_outputs = (outputs > 0.5).int()
#                 correct_pixels = (thresholded_outputs == labels).sum().item()
#                 total_pixels += (inputs.size(2) * inputs.size(3)) * inputs.size(0)
#                 val_acc = correct_pixels / total_pixels
#                 val_accuracy += val_acc  

#                 # loss
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#                 m += 1
#                 if i % (len(val) // batch_size) == len(val) // batch_size - 1:
#                     print(f"epoch:{epoch + 1}  index:{i + 1}  validation_accuracy:{val_accuracy / m:.5f}")
#                     print(f"epoch:{epoch + 1}  index:{i + 1}  validation_loss:{val_loss / m:.5f}")
#                     validation_accuracy.append(val_accuracy)
#                     validation_loss.append(val_loss)
#                     if val_loss < val_loss_min:
#                         val_loss_min = val_loss
#                         torch.save(unet.state_dict(), 'semantic_segmentation.pt')
#                         print('Detected network improvement, saving current model')
        
#                     m = 0
#                     val_accuracy = 0
#                     val_loss = 0
#     return history,validation_accuracy,validation_loss

# data = trainModel()
device = "cpu"

model = UNet_2D().to(device)

# hyperparameter settings
learning_rate = 1e-3
epochs = 12

# loss function definition
loss_fn = nn.CrossEntropyLoss()

# optimizer definition
optimizer = torch.optim.AdamW(model.parameters(),learning_rate) #torch.optim.SGD(model.parameters(), learning_rate)

# defining the training loop
def trainingLoop(train_dataloader, model, loss_fn, optimizer):
    idx = 0
    for batch in train_dataloader:
        # move data on gpu
        x = batch["img"]
        y = batch["label"]
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred,y)

        # backpropagation 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            loss = loss.item()
            print(f"The loss is {loss}")
        idx += 1


def testLoop(test_dataloader, model, loss_fn):
    print_size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss = 0
    correct = 0
    val_loss = 0
    val_loss_min = np.inf

    with torch.no_grad():
        for batch in test_dataloader:
            x = batch["img"]
            y = batch["label"]
            x,y = x.to(device), y.to(device)

            pred = model(x)
            test_loss += loss_fn(pred,y).item()
            # print(pred.argmax(0))
            # print(y)
            val_loss += test_loss
            correct += (pred.argmax(0) == y).type(torch.float).sum().item()
    
    test_loss = test_loss/num_batches
    correct = correct / print_size
    if val_loss < val_loss_min:
        val_loss_min = val_loss
        torch.save(model.state_dict(), 'semantic_segmentation.pt')
        print("Saving Changes")
    print(f"Accuracy: {correct * 100}, Average loss: {test_loss}")




# for e in range(epochs):
#     trainingLoop(train,model,loss_fn,optimizer)
#     testLoop(test,model,loss_fn)
model1 = UNet_2D().to(device)
model1.load_state_dict(torch.load("semantic_segmentation.pt"))
model1.eval()

transform = transforms.ToTensor()

img_path = "image-Dataset/images/9999986_00000_d_0000025.jpg"
img_prev = cv2.imread(img_path)
img = cv2.resize(img_prev, dsize = (256,256))
img = img / 255
img = torch.from_numpy(img.astype(np.float32)).clone()


input = img
batch = {
    'img':input
}
output = model1(input).to('cpu')

sigmoid = nn.Sigmoid()
outputs = sigmoid(output)
print(outputs)
# pred = torch.argmax(outputs, axis = 1)
# pred = torch.nn.functional.one_hot(pred.long(), num_classes = len(class_list)).to(torch.float32)
img = np.asarray(img)
from PIL import Image

a = Image.fromarray((img * 255).astype(np.uint8))
a.save("provaimg.png")