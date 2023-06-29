from termal import myCNN, plot_box
import torch
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import math

model = myCNN()
model.load_state_dict(torch.load("semantic_segmentation.pt"))
model.eval()

transform = transforms.ToTensor()

imgpath = "hit-uav/images/val/0_60_50_0_06537.jpg"
img = cv2.imread(imgpath)
img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
img = Image.fromarray(img)

input = transform(img)
input = input.unsqueeze(0)

output = model(input).to('cpu')



# output -= output.min(1, keepdim=True)[0]
# output /= output.max(1, keepdim=True)[0]
# output = output.view(output.size(0), -1)
# output -= output.min(1, keepdim=True)[0]
# output /= output.max(1, keepdim=True)[0]
# output = output.view(-1, 5, 40, 4)
sigmoid = nn.Sigmoid()
pred = sigmoid(output*(math.log2(output.max()))*5)
print(output)
classes = ['Person', 'Car', 'Bicycle', 'OtherVechicle', 'DontCare']
colors = np.random.uniform(0, 255, size=(len(classes), 3))

bboxes = []
labels = []
image = cv2.imread("hit-uav/images/val/0_60_50_0_06537.jpg")
for i in range(len(output[0])):
    label, x_c, y_c, w, h = i,output[0][i][0][0],output[0][i][0][1],output[0][i][0][2],output[0][i][0][3]
    print(x_c,y_c)
    x_c = float(x_c)
    y_c = float(y_c)
    w = float(w)
    h = float(h)
    bboxes.append([x_c, y_c, w, h])
    labels.append(label)
result_image = plot_box(image, bboxes, labels, classes, colors, 'above')
plt.subplot(1, 1, 1) # Visualize 2x2 grid of images.
plt.imshow(image[:, :, ::-1])
plt.axis('off')
plt.tight_layout()
plt.show()