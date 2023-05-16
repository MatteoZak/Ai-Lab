from dataLoader.dataSet import UAVDataset
import torch


dataset = UAVDataset('dataset.csv','VisDrone2019-DET-val/images/',output_size=256)

train, val, test = torch.utils.data.random_split(
    dataset = dataset, 
    lengths = [320, 40, 40], 
    generator = torch.Generator().manual_seed(45)
)

batch_size = 4

train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle = True, drop_last = True)
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size)
